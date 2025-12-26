// --- Standard Library ---
use std::path::{Path, PathBuf};
use std::sync::Arc;

// --- External Crates ---
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::traits::{InferenceModel, ModelConfig as _, ModelLayout, ModelMetadata};
use ndarray::{Array1, Array3};
use tokenizers::Tokenizer;

// --- Workspace Crates ---
use kjarni_transformers::{
    cache::{Cache, CpuBeamKVCache, GpuBeamKVCache},
    common::{BeamSearchParams, DecodingStrategy, GenerationConfig},
    encoder::{CpuEncoderOps, GpuEncoderOps, prelude::*, traits::CpuEncoder},
    encoder_decoder::traits::{
        CpuCrossDecoder, CpuEncoderDecoderOps, EncoderDecoderLanguageModel, GpuCrossDecoder,
        GpuEncoderDecoderOps,
    },
    gpu_ops::{
        GpuFrameContext, GpuTensor,
        primitives::{add::GpuAdd, broadcast::GpuBroadcast, linear::GpuLinearLayer},
    },
    linear_layer::LinearLayer,
    models::{ModelType, download_model_files},
    prelude::*,
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
    pub cpu_encoder: Option<BartCpuEncoder>,
    pub cpu_decoder: Option<BartCpuDecoder>,

    // GPU components
    pub gpu_encoder: Option<BartGpuEncoder>,
    pub gpu_decoder: Option<BartGpuDecoder>,
    gpu_lm_head: Option<GpuTensor>,
    gpu_final_logits_bias: Option<GpuTensor>,
    gpu_broadcast_kernel: Option<GpuBroadcast>,
    gpu_linear_kernel: Option<GpuLinearLayer>,

    // Shared components
    tokenizer: Tokenizer,
    config: Arc<BartConfig>,
    lm_head: LinearLayer,
    final_logits_bias: Option<Array1<f32>>,
    device: Device,
    context: Option<Arc<WgpuContext>>,

    // Cached Data-driven structs
    pub meta: ModelMetadata,
    pub layout: ModelLayout,
}

impl BartModel {
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
    const SUPPORTED_MODELS: &'static [ModelType] =
        &[ModelType::BartLargeCnn, ModelType::DistilBartCnn];
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

        // Logic preserved: auto-create context if missing for GPU
        if device.is_gpu() && context.is_none() {
            Self::from_pretrained(&model_dir, device, Some(WgpuContext::new().await?), None)
        } else {
            Self::from_pretrained(&model_dir, device, context, None)
        }
    }
    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>, // Now uses unified load config
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let load_config = load_config.unwrap_or_default();
        let tokenizer =
            Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|e| anyhow!(e))?;

        // 1. Detect the correct embedding key (LOGIC PRESERVED)
        let shared_key = if weights.contains("model.shared.weight") {
            "model.shared.weight"
        } else if weights.contains("model.encoder.embed_tokens.weight") {
            "model.encoder.embed_tokens.weight"
        } else if weights.contains("model.decoder.embed_tokens.weight") {
            "model.decoder.embed_tokens.weight"
        } else {
            return Err(anyhow!("Could not find shared embedding weights."));
        };

        // 2. Initialize Config and Data Structs
        let mut config_obj = BartConfig::from_json(&weights.config_json)?;
        config_obj.shared_embedding_key = Some(shared_key.to_string());
        let config = Arc::new(config_obj);

        let meta = config.metadata();
        let layout = config.layout();

        // 3. Load Heads (Uses layout.lm_head which was determined by shared_key)
        let lm_head = LinearLayer::from_weights(
            &weights,
            &layout.lm_head,
            None,
            load_config.target_dtype,
            None,
        )?;
        let final_logits_bias = weights.get_array1("final_logits_bias").ok();

        match device {
            Device::Cpu => {
                let cpu_encoder = BartCpuEncoder::new(&weights, config.clone(), load_config)?;
                let cpu_decoder = BartCpuDecoder::new(&weights, config.clone(), load_config)?;

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
                    meta,
                    layout,
                })
            }
            Device::Wgpu => {
                let ctx = context.ok_or_else(|| anyhow!("WGPU device requires a WgpuContext"))?;

                let gpu_encoder = BartGpuEncoder::new(&ctx, &weights, config.clone(), load_config)?;
                let gpu_decoder = BartGpuDecoder::new(&ctx, &weights, config.clone(), load_config)?;

                // Upload LM head and bias to GPU
                let gpu_lm_head = Some(GpuTensor::from_raw(
                    &ctx,
                    &weights.get_raw_resolved(&layout.lm_head, load_config.target_dtype)?,
                    "lm_head",
                )?);

                let gpu_final_logits_bias = if weights.contains("final_logits_bias") {
                    Some(GpuTensor::from_raw(
                        &ctx,
                        &weights.get_raw("final_logits_bias")?,
                        "final_logits_bias",
                    )?)
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
                    meta,
                    layout,
                })
            }
        }
    }
}

// --- TransformerModel Implementation ---
impl InferenceModel for BartModel {
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
        self.config.metadata().max_seq_len
    }
    fn forced_eos_token_id(&self) -> Option<u32> {
        // If explicit forced_eos_token_id is set, use it
        if self.config.forced_eos_token_id.is_some() {
            return self.config.forced_eos_token_id;
        }
        None
    }
    fn forced_bos_token_id(&self) -> Option<u32> {
        // If explicit forced_bos_token_id is set, use it
        if self.config.forced_bos_token_id.is_some() {
            return self.config.forced_bos_token_id;
        }

        // HF behavior: if force_bos_token_to_be_generated is true, use bos_token_id
        if self.config.force_bos_token_to_be_generated.unwrap_or(false) {
            return Some(self.config.bos_token_id);
        }

        None
    }
    fn pad_token_id(&self) -> Option<u32> {
        Some(self.config.pad_token_id)
    }
    fn vocab_size(&self) -> usize {
        self.config.metadata().vocab_size
    }
    fn hidden_size(&self) -> usize {
        self.config.metadata().hidden_size
    }
    fn num_heads(&self) -> usize {
        self.config.metadata().num_attention_heads
    }
    fn num_layers(&self) -> usize {
        self.config.metadata().num_layers
    }
    fn eos_token_id(&self) -> Option<u32> {
        Some(self.config.eos_token_id)
    }
    fn bos_token_id(&self) -> Option<u32> {
        Some(self.config.bos_token_id)
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
            max_length: self.meta.max_seq_len,
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
