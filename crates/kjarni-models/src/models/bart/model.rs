// --- Standard Library ---
use std::path::{Path, PathBuf};
use std::sync::Arc;

// --- External Crates ---
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use kjarni_transformers::traits::EncoderDecoderArchitecture;
use ndarray::{Array1, Array3};
use tokenizers::Tokenizer;

// --- Workspace Crates ---
use kjarni_transformers::{
    cache::{Cache, CpuBeamKVCache, GpuBeamKVCache},
    common::{BeamSearchParams, DecodingStrategy, GenerationConfig},
    encoder::{prelude::*, traits::CpuEncoder, CpuEncoderOps, GpuEncoderOps},
    encoder_decoder::traits::{
        CpuCrossDecoder, CpuEncoderDecoderOps, EncoderDecoderLanguageModel, GpuCrossDecoder,
        GpuEncoderDecoderOps,
    },
    gpu_ops::{
        primitives::{add::GpuAdd, broadcast::GpuBroadcast, linear::GpuLinearLayer}, GpuFrameContext,
        GpuTensor,
    },
    linear_layer::LinearLayer,
    models::{download_model_files, ModelType},
    prelude::*,
    traits::{LanguageModelConfig, TransformerModel},
    weights::ModelWeights,
};

// --- Crate-Specific ---
use crate::models::bart::{
    config::{BartConfig, BartLikeConfig},
    cpu_decoder::BartCpuDecoder,
    cpu_encoder::BartCpuEncoder,
    gpu_decoder::BartGpuDecoder,
    gpu_encoder::BartGpuEncoder,
};

pub struct BartModel {
    // CPU components
    cpu_encoder: Option<BartCpuEncoder>,
    cpu_decoder: Option<BartCpuDecoder>,

    // GPU components
    gpu_encoder: Option<BartGpuEncoder>,
    gpu_decoder: Option<BartGpuDecoder>,
    gpu_lm_head: Option<GpuTensor>,
    gpu_final_logits_bias: Option<GpuTensor>,
    gpu_broadcast_kernel: Option<GpuBroadcast>,
    gpu_linear_kernel: Option<GpuLinearLayer>,

    // Shared components
    tokenizer: Tokenizer,
    config: Arc<BartConfig>,
    lm_head: LinearLayer,                   // Still needed for CPU path
    final_logits_bias: Option<Array1<f32>>, // Still needed for CPU path
    device: Device,
    context: Option<Arc<WgpuContext>>,
}

impl BartModel {
    const SUPPORTED_MODELS: &'static [ModelType] =
        &[ModelType::BartLargeCnn, ModelType::DistilBartCnn];
    pub fn bart_cpu_decoder(&self) -> Option<&BartCpuDecoder> {
        self.cpu_decoder.as_ref()
    }

    pub fn bart_gpu_decoder(&self) -> Option<&BartGpuDecoder> {
        self.gpu_decoder.as_ref()
    }
    pub fn bart_cpu_encoder(&self) -> Option<&BartCpuEncoder> {
        self.cpu_encoder.as_ref()
    }

    pub fn bart_gpu_encoder(&self) -> Option<&BartGpuEncoder> {
        self.gpu_encoder.as_ref()
    }
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!("Unsupported BART model type: {:?}", model_type));
        }

        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("kjarni")
        });
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        log::info!("Loading BART model from {:?}", model_dir);
        download_model_files(&model_dir, &model_type.info().paths).await?;

        if device.is_gpu() && context.is_none() {
            Self::from_pretrained(&model_dir, device, Some(WgpuContext::new().await?))
        } else {
            Self::from_pretrained(&model_dir, device, context)
        }
    }

    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let tokenizer =
            Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|e| anyhow!(e))?;

        // let config = Arc::new(BartConfig::from_json(&weights.config_json)?);
        // 1. Detect the correct embedding key
        // Check "model.shared.weight" first (standard), then "model.encoder.embed_tokens.weight" (legacy/fairseq)
        let shared_key = if weights.contains("model.shared.weight") {
            // Assuming ModelWeights has a check method
            "model.shared.weight"
        } else if weights.contains("model.encoder.embed_tokens.weight") {
            "model.encoder.embed_tokens.weight"
        } else if weights.contains("model.decoder.embed_tokens.weight") {
            "model.decoder.embed_tokens.weight"
        } else {
            return Err(anyhow!(
                "Could not find shared embedding weights. Checked 'model.shared.weight' and 'model.encoder.embed_tokens.weight'"
            ));
        };
        println!("Detected shared embedding key: {}", shared_key);

        // 2. Load and Patch Config
        let mut config_obj = BartConfig::from_json(&weights.config_json)?;
        config_obj.shared_embedding_key = Some(shared_key.to_string()); // Store the detected key
        let config = Arc::new(config_obj);
        println!("Using LM Head key: {}", config.get_lm_head_name());
        // Load CPU versions of head/bias regardless, as they are small

        let lm_head = LinearLayer::from_weights(&weights,
                                                config.get_lm_head_name(), None, None, None)?;
        let final_logits_bias = weights.get_array1("final_logits_bias").ok();

        match device {
            Device::Cpu => {
                let cpu_encoder = BartCpuEncoder::new(&weights, config.clone())?;
                let cpu_decoder = BartCpuDecoder::new(&weights, config.clone())?;

                Ok(Self {
                    cpu_encoder: Some(cpu_encoder),
                    cpu_decoder: Some(cpu_decoder),
                    gpu_encoder: None,
                    gpu_decoder: None,
                    gpu_lm_head: None,
                    gpu_final_logits_bias: None,
                    tokenizer,
                    config,
                    lm_head,
                    final_logits_bias,
                    device,
                    context: None,
                    gpu_broadcast_kernel: None,
                    gpu_linear_kernel: None,
                })
            }
            Device::Wgpu => {
                let ctx = context.ok_or_else(|| anyhow!("WGPU device requires a WgpuContext"))?;

                log::info!("Loading BART model to GPU...");
                let gpu_encoder = BartGpuEncoder::new(&ctx, &weights, config.clone())?;
                let gpu_decoder = BartGpuDecoder::new(&ctx, &weights, config.clone())?;

                // Upload LM head and bias to GPU
                let gpu_lm_head = Some(GpuTensor::from_raw(
                    &ctx,
                    &weights.get_raw(config.get_lm_head_name())?,
                    "lm_head",
                )?);
                let gpu_final_logits_bias =
                    if let Ok(raw_bias) = weights.get_raw("final_logits_bias") {
                        Some(GpuTensor::from_raw(&ctx, &raw_bias, "final_logits_bias")?)
                    } else {
                        None
                    };

                Ok(Self {
                    cpu_encoder: None,
                    cpu_decoder: None,
                    gpu_encoder: Some(gpu_encoder),
                    gpu_decoder: Some(gpu_decoder),
                    gpu_lm_head,
                    gpu_final_logits_bias,
                    tokenizer,
                    config,
                    lm_head,
                    final_logits_bias,
                    device,
                    context: Some(ctx.clone()),
                    gpu_broadcast_kernel: Some(GpuBroadcast::new(&ctx.clone())?),
                    gpu_linear_kernel: Some(GpuLinearLayer::new(&ctx)),
                })
            }
        }
    }
}

// --- TransformerModel Implementation ---
impl TransformerModel for BartModel {
    fn device(&self) -> Device {
        self.device
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        self.context.clone()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl CpuEncoderOps for BartModel {
    fn encoder(&self) -> &dyn CpuEncoder {
        self.cpu_encoder
            .as_ref()
            .expect("CPU encoder not initialized")
    }
}

impl GpuEncoderOps for BartModel {
    fn encoder(&self) -> &dyn GpuEncoder {
        self.gpu_encoder
            .as_ref()
            .expect("GPU encoder not initialized")
    }
}

#[async_trait]
impl EncoderLanguageModel for BartModel {
    fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
        if self.device.is_cpu() {
            Some(self)
        } else {
            None
        }
    }
    fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
        if self.device.is_gpu() {
            Some(self)
        } else {
            None
        }
    }
}

// --- LanguageModel Implementation ---
impl LanguageModel for BartModel {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn context_size(&self) -> usize {
        todo!()
    }
    fn forced_bos_token_id(&self) -> Option<u32> {
        todo!()
    }
    fn pad_token_id(&self) -> Option<u32> {
        todo!()
    }
    fn vocab_size(&self) -> usize {
        todo!()
    }
    fn hidden_size(&self) -> usize {
        todo!()
    }
    fn num_heads(&self) -> usize {
        todo!()
    }
    fn num_layers(&self) -> usize {
        todo!()
    }
    fn eos_token_id(&self) -> Option<u32> {
        todo!()
    }
    fn bos_token_id(&self) -> Option<u32> {
        todo!()
    }

    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        num_beams: usize,
    ) -> Result<Box<dyn Cache>> {
        match self.device {
            Device::Cpu => {
                // BART generation relies heavily on Beam Search.
                // If num_beams > 1, the cache must be sized for [num_beams, heads, seq, head_dim]
                let effective_batch = if num_beams > 0 { num_beams } else { batch_size };
                Ok(Box::new(CpuBeamKVCache::new(
                    self.config.decoder_layers,
                    effective_batch,
                    max_len,
                    self.config.d_model,
                )))
            }
            Device::Wgpu => {
                let ctx = self.context.as_ref().unwrap();
                let head_dim = self.config.d_model / self.config.decoder_attention_heads;
                Ok(Box::new(GpuBeamKVCache::new(
                    ctx,
                    self.config.decoder_layers,
                    num_beams,
                    self.config.decoder_attention_heads,
                    head_dim,
                    max_len,
                )?))
            }
        }
    }
}

impl CpuEncoderDecoderOps for BartModel {
    // fn encoder(&self) -> &dyn CpuEncoder {
    //     self.cpu_encoder
    //         .as_ref()
    //         .expect("CPU encoder not initialized")
    // }
    fn decoder(&self) -> &dyn CpuCrossDecoder {
        self.cpu_decoder
            .as_ref()
            .expect("CPU decoder not initialized")
    }

    fn broadcast_encoder_states(
        &self,
        encoder_hidden_states: &Array3<f32>,
        num_beams: usize,
    ) -> Result<Array3<f32>> {
        // This is the exact logic from your old CpuBackend
        Ok(encoder_hidden_states
            .broadcast((
                num_beams,
                encoder_hidden_states.shape()[1],
                encoder_hidden_states.shape()[2],
            ))
            .ok_or_else(|| anyhow!("Failed to broadcast encoder state"))?
            .to_owned())
    }

    // This is the logic moved from the old CpuBackend
    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden) = hidden_states.dim();
        let hidden_2d = hidden_states.view().into_shape((batch * seq, hidden))?;

        // Use the model's own lm_head LinearLayer
        let mut logits_2d = self.lm_head.matmul(&hidden_2d.view());

        if let Some(bias) = &self.final_logits_bias {
            logits_2d += bias;
        }

        logits_2d
            .into_shape((batch, seq, self.vocab_size()))
            .map_err(|e| anyhow!(e))
    }
}

impl GpuEncoderDecoderOps for BartModel {
    // fn encoder(&self) -> &dyn GpuEncoder {
    //     self.gpu_encoder
    //         .as_ref()
    //         .expect("GPU encoder not initialized")
    // }
    fn decoder(&self) -> &dyn GpuCrossDecoder {
        self.gpu_decoder
            .as_ref()
            .expect("GPU decoder not initialized")
    }
    fn broadcast_encoder_states(
        &self,
        frame: &mut GpuFrameContext,
        encoder_hidden_states: &GpuTensor,
        num_beams: usize,
    ) -> Result<GpuTensor> {
        // This is the logic from your old GpuBackend, now correctly encapsulated in the model's ops.
        let (encoder_cmd, _pool) = frame.resources();

        let mut expanded_shape = encoder_hidden_states.shape().to_vec();
        // Ensure the input has a batch size of 1
        if expanded_shape.get(0) != Some(&1) {
            return Err(anyhow!(
                "Cannot broadcast encoder states with batch size != 1"
            ));
        }
        expanded_shape[0] = num_beams;

        let expanded_states = GpuTensor::uninitialized(
            &self.context.as_ref().unwrap(),
            expanded_shape,
            encoder_hidden_states.dtype(),
            "expanded_encoder_states",
        );

        // Get the broadcast kernel from the model struct
        let broadcast_kernel = self
            .gpu_broadcast_kernel
            .as_ref()
            .ok_or_else(|| anyhow!("GpuBroadcast kernel not initialized for this model"))?;

        broadcast_kernel.encode(encoder_cmd, encoder_hidden_states, &expanded_states, 0);

        Ok(expanded_states)
    }

    // This is the logic moved from the old GpuBackend
    fn project_to_logits(
        &self,
        frame: &mut GpuFrameContext,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        let (batch, seq, hidden) = hidden_states.dims3();
        let (encoder_cmd, pool) = frame.resources();

        let lm_head_weights = self.gpu_lm_head.as_ref().unwrap();
        let vocab_size = lm_head_weights.shape()[0];

        let logits = pool.get(vec![batch * seq, vocab_size]);
        let hidden_states_2d = hidden_states.view(vec![batch * seq, hidden]);

        // This kernel should be a shared primitive, not created on the fly.
        // For now, this works. In a future cleanup, you might pass it into the ops from the backend.
        let linear_kernel = self
            .gpu_linear_kernel
            .as_ref()
            .ok_or_else(|| anyhow!("GpuLinear kernel not initialized for this model"))?;
        linear_kernel.encode(encoder_cmd, &hidden_states_2d, lm_head_weights, &logits);

        let final_logits = if let Some(bias) = self.gpu_final_logits_bias.as_ref() {
            let logits_with_bias = pool.get(logits.shape().to_vec());
            let add_kernel = GpuAdd::new(&self.context.as_ref().unwrap());
            add_kernel.encode_broadcast_row(encoder_cmd, &logits, bias, &logits_with_bias);
            logits_with_bias
        } else {
            logits
        };

        Ok(final_logits.view(vec![batch, seq, vocab_size]))
    }
}

#[async_trait]
impl EncoderDecoderLanguageModel for BartModel {
    fn encoder_decoder_cpu_ops(&self) -> Option<&dyn CpuEncoderDecoderOps> {
        if self.device.is_cpu() {
            Some(self)
        } else {
            None
        }
    }
    fn encoder_decoder_gpu_ops(&self) -> Option<&dyn GpuEncoderDecoderOps> {
        if self.device.is_gpu() {
            Some(self)
        } else {
            None
        }
    }

    fn decoder_start_token_id(&self) -> u32 {
        self.config.decoder_start_token_id
    }

    fn get_default_generation_config(&self) -> GenerationConfig {
        // Try to load from task specific params in config
        if let Some(params) = &self.config.task_specific_params() {
            let summary_params = &params.summarization;
            return GenerationConfig {
                max_length: summary_params.max_length,
                min_length: summary_params.min_length,
                no_repeat_ngram_size: summary_params.no_repeat_ngram_size,
                repetition_penalty: 1.0,
                max_new_tokens: None,
                add_bos_token: false,
                strategy: DecodingStrategy::BeamSearch(BeamSearchParams {
                    num_beams: summary_params.num_beams,
                    length_penalty: summary_params.length_penalty,
                    early_stopping: summary_params.early_stopping,
                }),
            };
        }

        // Safe Defaults
        GenerationConfig {
            max_length: self.config.max_position_embeddings(),
            min_length: 0,
            no_repeat_ngram_size: 3,
            repetition_penalty: 1.0,
            max_new_tokens: None,
            add_bos_token: false,
            strategy: DecodingStrategy::BeamSearch(BeamSearchParams {
                num_beams: 4,
                length_penalty: 2.0,
                early_stopping: true,
            }),
        }
    }
}
// #[async_trait]
// impl EncoderDecoderLanguageModel for BartModel {
//     // --- CPU Accessors ---
//     fn encoder(&self) -> Result<&dyn Encoder<Input=Array2<u32>, Output=EncoderOutput>> {
//         // self.cpu_encoder.as_ref().map(|e| e as _).ok_or_else(|| anyhow!("Model not loaded for CPU"))
//         panic!("deprecated! Use `bart_cpu_encoder()` instead.");
//     }
//     fn cpu_encoder(&self) -> Result<&dyn CpuEncoder> {
//         self.cpu_encoder
//             .as_ref()
//             .map(|e| e as _)
//             .ok_or_else(|| anyhow!("Model not loaded for CPU"))
//     }

//     fn cpu_decoder(
//         &self,
//     ) -> Result<
//         &dyn CrossAttentionDecoder<
//             TokenInput=Array2<u32>,
//             EncoderStateInput=Array3<f32>,
//             MaskInput=Array2<f32>,
//             Output=DecoderOutput,
//         >,
//     > {
//         self.cpu_decoder
//             .as_ref()
//             .map(|d| d as _)
//             .ok_or_else(|| anyhow!("Model not loaded for CPU"))
//     }

//     // --- GPU Accessors ---
//     fn gpu_encoder(&self) -> Result<&dyn GpuEncoder> {
//         self.gpu_encoder
//             .as_ref()
//             .map(|e| e as _)
//             .ok_or_else(|| anyhow!("Model not loaded for GPU"))
//     }

//     fn gpu_decoder(&self) -> Result<&dyn GpuCrossAttentionDecoder> {
//         self.gpu_decoder
//             .as_ref()
//             .map(|d| d as _)
//             .ok_or_else(|| anyhow!("Model not loaded for GPU"))
//     }

//     // --- LM Head Accessors ---
//     fn lm_head_layer(&self) -> &LinearLayer {
//         &self.lm_head
//     }

//     fn gpu_lm_head_weights(&self) -> Result<&GpuTensor> {
//         self.gpu_lm_head
//             .as_ref()
//             .ok_or_else(|| anyhow!("GPU LM head not loaded"))
//     }

//     fn final_logits_bias(&self) -> Option<&Array1<f32>> {
//         self.final_logits_bias.as_ref()
//     }

//     fn gpu_final_logits_bias(&self) -> Result<Option<&GpuTensor>> {
//         Ok(self.gpu_final_logits_bias.as_ref())
//     }

//     fn decoder_start_token_id(&self) -> u32 {
//         self.config.decoder_start_token_id
//     }

//     // 6. Generation Config Defaults
//     fn get_default_generation_config(&self) -> GenerationConfig {
//         // Try to load from task specific params in config
//         if let Some(params) = &self.config.task_specific_params() {
//             let summary_params = &params.summarization;
//             return GenerationConfig {
//                 max_length: summary_params.max_length,
//                 min_length: summary_params.min_length,
//                 no_repeat_ngram_size: summary_params.no_repeat_ngram_size,
//                 repetition_penalty: 1.0,
//                 max_new_tokens: None,
//                 add_bos_token: false,
//                 strategy: DecodingStrategy::BeamSearch(BeamSearchParams {
//                     num_beams: summary_params.num_beams,
//                     length_penalty: summary_params.length_penalty,
//                     early_stopping: summary_params.early_stopping,
//                 }),
//             };
//         }

//         // Safe Defaults
//         GenerationConfig {
//             max_length: self.config.max_position_embeddings(),
//             min_length: 0,
//             no_repeat_ngram_size: 3,
//             repetition_penalty: 1.0,
//             max_new_tokens: None,
//             add_bos_token: false,
//             strategy: DecodingStrategy::BeamSearch(BeamSearchParams {
//                 num_beams: 4,
//                 length_penalty: 2.0,
//                 early_stopping: true,
//             }),
//         }
//     }
// }
