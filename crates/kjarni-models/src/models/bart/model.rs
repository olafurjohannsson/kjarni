// --- Standard Library ---
use std::path::{Path, PathBuf};
use std::sync::Arc;

// --- External Crates ---
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use kjarni_transformers::common::HFGenerationDefaults;
use kjarni_transformers::encoder_decoder::cpu_encoder::{Seq2SeqCPUEncoder, Seq2SeqEncoderConfig};
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::pipeline::{EncoderDecoderModelFactory, EncoderDecoderPipeline};
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
    // pub cpu_encoder: Option<BartCpuEncoder>,
    // pub cpu_decoder: Option<BartCpuDecoder>,

    // pub cpu_encoder_new: Option<Seq2SeqCPUEncoder>,

    // GPU components
    // pub gpu_encoder: Option<BartGpuEncoder>,
    // pub gpu_decoder: Option<BartGpuDecoder>,
    // gpu_lm_head: Option<GpuTensor>,
    // gpu_final_logits_bias: Option<GpuTensor>,
    // gpu_broadcast_kernel: Option<GpuBroadcast>,
    // gpu_linear_kernel: Option<GpuLinearLayer>,

    // Shared components
    tokenizer: Tokenizer,
    config: Arc<BartConfig>,
    // lm_head: LinearLayer,
    // final_logits_bias: Option<Array1<f32>>,
    // device: Device,
    // context: Option<Arc<WgpuContext>>,

    // Cached Data-driven structs
    // pub meta: ModelMetadata,
    // pub layout: ModelLayout,
    pub pipeline: EncoderDecoderPipeline,
    generation_defaults: Option<HFGenerationDefaults>,
}

impl BartModel {
    pub fn meta(&self) -> ModelMetadata {
        self.config.metadata()
    }
    pub fn layout(&self) -> ModelLayout {
        self.config.layout()
    }
}

impl EncoderDecoderModelFactory for BartModel {
    type Config = BartConfig;

    fn load_config(weights: &ModelWeights) -> Result<Arc<Self::Config>> {
        let mut config = BartConfig::from_json(&weights.config_json)?;
        // Helper to find shared key, same as before
        if config.shared_embedding_key.is_none() {
            let key = if weights.contains("model.shared.weight") {
                "model.shared.weight"
            } else {
                "model.encoder.embed_tokens.weight" // fallback
            };
            config.shared_embedding_key = Some(key.to_string());
        }
        Ok(Arc::new(config))
    }

    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        _layout: &ModelLayout, // Pipeline builder handles generic layout, backends handle specific
        config: &Arc<BartConfig>,
        load_config: &ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
    ) -> Result<(
        Option<Box<dyn CpuEncoder>>,
        Option<Box<dyn GpuEncoder>>,
        Option<Box<dyn CpuCrossDecoder>>,
        Option<Box<dyn GpuCrossDecoder>>,
    )> {
        let mut cpu_enc = None;
        let mut cpu_dec = None;
        let mut gpu_enc = None;
        let mut gpu_dec = None;

        // CPU Backends
        if load_config.gpu_layers.is_none() || load_config.offload_embeddings {
            // Unified Encoder
            let enc_config = Seq2SeqEncoderConfig::bart();
            cpu_enc = Some(Box::new(Seq2SeqCPUEncoder::new(
                weights,
                config.as_ref(), // Pass as &dyn ModelConfig
                enc_config,
                *load_config,
            )?) as Box<dyn CpuEncoder>);

            // Legacy Specific Decoder (until you unify this too)
            cpu_dec = Some(
                Box::new(BartCpuDecoder::new(weights, config.clone(), *load_config)?)
                    as Box<dyn CpuCrossDecoder>,
            );
        }

        // GPU Backends
        if let Some(ctx) = context {
            gpu_enc = Some(Box::new(BartGpuEncoder::new(
                ctx,
                weights,
                config.clone(),
                *load_config,
            )?) as Box<dyn GpuEncoder>);

            gpu_dec = Some(Box::new(BartGpuDecoder::new(
                ctx,
                weights,
                config.clone(),
                *load_config,
            )?) as Box<dyn GpuCrossDecoder>);
        }

        Ok((cpu_enc, gpu_enc, cpu_dec, gpu_dec))
    }

    fn new_from_pipeline(
        pipeline: EncoderDecoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<BartConfig>,
        generation_defaults: Option<HFGenerationDefaults>,
    ) -> Self {
        Self {
            pipeline,
            tokenizer,
            config,
            generation_defaults,
        }
    }
}

impl BartModel {
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        kjarni_transformers::pipeline::Seq2SeqLoader::load_from_registry::<Self>(
            model_type,
            cache_dir,
            device,
            context,
            load_config,
        )
        .await
    }
    pub fn bart_cpu_decoder(&self) -> Option<&BartCpuDecoder> {
        // self.cpu_decoder.as_ref()
        None
    }
    pub fn bart_gpu_decoder(&self) -> Option<&BartGpuDecoder> {
        // self.gpu_decoder.as_ref()
        None
    }

    pub fn bart_gpu_encoder(&self) -> Option<&BartGpuEncoder> {
        // self.gpu_encoder.as_ref()
        None
    }
    const SUPPORTED_MODELS: &'static [ModelType] =
        &[ModelType::BartLargeCnn, ModelType::DistilBartCnn];
    // pub async fn from_registry(
    //     model_type: ModelType,
    //     cache_dir: Option<PathBuf>,
    //     device: Device,
    //     context: Option<Arc<WgpuContext>>,
    // ) -> Result<Self> {
    //     if !Self::SUPPORTED_MODELS.contains(&model_type) {
    //         return Err(anyhow!("Unsupported BART model type: {:?}", model_type));
    //     }

    //     let cache_dir = cache_dir.unwrap_or_else(|| {
    //         dirs::cache_dir()
    //             .expect("No cache directory found")
    //             .join("kjarni")
    //     });
    //     let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

    //     log::info!("Loading BART model from {:?}", model_dir);
    //     download_model_files(
    //         &model_dir,
    //         &model_type.info().paths,
    //         kjarni_transformers::models::registry::WeightsFormat::SafeTensors,
    //     )
    //     .await?;

    //     // Logic preserved: auto-create context if missing for GPU
    //     if device.is_gpu() && context.is_none() {
    //         Self::from_pretrained(&model_dir, device, Some(WgpuContext::new().await?), None)
    //     } else {
    //         Self::from_pretrained(&model_dir, device, context, None)
    //     }
    // }
    // pub fn from_pretrained(
    //     model_path: &Path,
    //     device: Device,
    //     context: Option<Arc<WgpuContext>>,
    //     load_config: Option<ModelLoadConfig>, // Now uses unified load config
    // ) -> Result<Self> {
    //     let weights = ModelWeights::new(model_path)?;
    //     let load_config = load_config.unwrap_or_default();
    //     let tokenizer =
    //         Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|e| anyhow!(e))?;

    //     // 1. Detect the correct embedding key (LOGIC PRESERVED)
    //     let shared_key = if weights.contains("model.shared.weight") {
    //         "model.shared.weight"
    //     } else if weights.contains("model.encoder.embed_tokens.weight") {
    //         "model.encoder.embed_tokens.weight"
    //     } else if weights.contains("model.decoder.embed_tokens.weight") {
    //         "model.decoder.embed_tokens.weight"
    //     } else {
    //         return Err(anyhow!("Could not find shared embedding weights."));
    //     };

    //     // 2. Initialize Config and Data Structs
    //     let mut config_obj = BartConfig::from_json(&weights.config_json)?;
    //     config_obj.shared_embedding_key = Some(shared_key.to_string());
    //     let config = Arc::new(config_obj);

    //     let meta = config.metadata();
    //     let layout = config.layout();

    //     // 3. Load Heads (Uses layout.lm_head which was determined by shared_key)
    //     let lm_head = LinearLayer::builder(&weights, &layout.lm_head)
    //         .with_target_dtype(load_config.target_dtype)
    //         .with_optional_bias(None)
    //         .build()?;
    //     let final_logits_bias = weights.get_array1("final_logits_bias").ok();

    //     match device {
    //         Device::Cpu => {
    //             let cpu_encoder = BartCpuEncoder::new(&weights, config.clone(), load_config)?;
    //             let cpu_decoder = BartCpuDecoder::new(&weights, config.clone(), load_config)?;
    //             let encoder_config = Seq2SeqEncoderConfig::bart();

    //             let cpu_encoder_new = Seq2SeqCPUEncoder::new(
    //                 &weights,
    //                 config.as_ref(), // Pass as &dyn ModelConfig
    //                 encoder_config,
    //                 load_config,
    //             )?;

    //             Ok(Self {
    //                 cpu_encoder_new: Some(cpu_encoder_new),
    //                 cpu_encoder: Some(cpu_encoder),
    //                 cpu_decoder: Some(cpu_decoder),
    //                 gpu_encoder: None,
    //                 gpu_decoder: None,
    //                 gpu_lm_head: None,
    //                 gpu_final_logits_bias: None,
    //                 tokenizer,
    //                 config,
    //                 lm_head,
    //                 final_logits_bias,
    //                 device,
    //                 context: None,
    //                 gpu_broadcast_kernel: None,
    //                 gpu_linear_kernel: None,
    //                 meta,
    //                 layout,
    //             })
    //         }
    //         Device::Wgpu => {
    //             let ctx = context.ok_or_else(|| anyhow!("WGPU device requires a WgpuContext"))?;

    //             let gpu_encoder = BartGpuEncoder::new(&ctx, &weights, config.clone(), load_config)?;
    //             let gpu_decoder = BartGpuDecoder::new(&ctx, &weights, config.clone(), load_config)?;

    //             // Upload LM head and bias to GPU
    //             let gpu_lm_head = Some(GpuTensor::from_model_weights(
    //                 &ctx,
    //                 &weights,
    //                 &layout.lm_head,
    //                 load_config.target_dtype,
    //                 "lm_head",
    //             )?);

    //             let gpu_final_logits_bias = if weights.contains("final_logits_bias") {
    //                 Some(GpuTensor::from_raw(
    //                     &ctx,
    //                     &weights.get_raw("final_logits_bias")?,
    //                     "final_logits_bias",
    //                 )?)
    //             } else {
    //                 None
    //             };

    //             Ok(Self {
    //                 cpu_encoder: None,
    //                 cpu_encoder_new: None,
    //                 cpu_decoder: None,
    //                 gpu_encoder: Some(gpu_encoder),
    //                 gpu_decoder: Some(gpu_decoder),
    //                 gpu_lm_head,
    //                 gpu_final_logits_bias,
    //                 tokenizer,
    //                 config,
    //                 lm_head,
    //                 final_logits_bias,
    //                 device,
    //                 context: Some(ctx.clone()),
    //                 gpu_broadcast_kernel: Some(GpuBroadcast::new(&ctx.clone())?),
    //                 gpu_linear_kernel: Some(GpuLinearLayer::new(&ctx)),
    //                 meta,
    //                 layout,
    //             })
    //         }
    //     }
    // }
}

// --- TransformerModel Implementation ---
// impl InferenceModel for BartModel {
//     fn device(&self) -> Device {
//         self.device
//     }
//     fn context(&self) -> Option<Arc<WgpuContext>> {
//         self.context.clone()
//     }
//     fn as_any(&self) -> &dyn std::any::Any {
//         self
//     }
// }
impl InferenceModel for BartModel {
    fn device(&self) -> Device {
        self.pipeline.plan().layers
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        self.pipeline.context().cloned()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
impl CpuEncoderOps for BartModel {
    // fn encoder(&self) -> &dyn CpuEncoder {
    //     self.cpu_encoder_new
    //         .as_ref()
    //         .expect("CPU encoder not initialized")
    // }
    fn encoder(&self) -> &dyn CpuEncoder {
        self.pipeline.cpu_encoder().expect("CPU Encoder not active")
    }
}

impl GpuEncoderOps for BartModel {
    fn encoder(&self) -> &dyn GpuEncoder {
        self.pipeline.gpu_encoder().expect("GPU Encoder not active")
    }
    // fn encoder(&self) -> &dyn GpuEncoder {
    //     self.gpu_encoder
    //         .as_ref()
    //         .expect("GPU encoder not initialized")
    // }
}

#[async_trait]
impl EncoderLanguageModel for BartModel {
    fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
        if self.pipeline.cpu_encoder().is_some() {
            Some(self)
        } else {
            None
        }
    }
    fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
        if self.pipeline.gpu_encoder().is_some() {
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
        // match self.device {
        match self.pipeline.plan().layers {
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
                // let ctx = self.context.as_ref().unwrap();
                let context = self
                    .context()
                    .ok_or_else(|| anyhow!("GPU cache requires WgpuContext"))?;
                let head_dim = self.config.d_model / self.config.decoder_attention_heads;
                Ok(Box::new(GpuBeamKVCache::new(
                    &context.clone(),
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
        self.pipeline.cpu_decoder().expect("CPU Decoder not active")

        // self.cpu_decoder
        //     .as_ref()
        //     .expect("CPU decoder not initialized")
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
        self.pipeline.lm_head().forward_cpu(hidden_states)
        // let (batch, seq, hidden) = hidden_states.dim();
        // let hidden_2d = hidden_states
        //     .view()
        //     .into_shape_with_order((batch * seq, hidden))?;

        // // Use the model's own lm_head LinearLayer
        // let mut logits_2d = self.lm_head.matmul(&hidden_2d.view());

        // if let Some(bias) = &self.final_logits_bias {
        //     logits_2d += bias;
        // }

        // logits_2d
        //     .into_shape_with_order((batch, seq, self.vocab_size()))
        //     .map_err(|e| anyhow!(e))
    }
}

impl GpuEncoderDecoderOps for BartModel {
    // fn encoder(&self) -> &dyn GpuEncoder {
    //     self.gpu_encoder
    //         .as_ref()
    //         .expect("GPU encoder not initialized")
    // }
    fn decoder(&self) -> &dyn GpuCrossDecoder {
        // self.gpu_decoder
        //     .as_ref()
        //     .expect("GPU decoder not initialized")
        self.pipeline.gpu_decoder().expect("GPU Decoder not active")
    }
    fn broadcast_encoder_states(
        &self,
        frame: &mut GpuFrameContext,
        encoder_hidden_states: &GpuTensor,
        num_beams: usize,
    ) -> Result<GpuTensor> {
        let broadcast = self
            .pipeline
            .gpu_broadcast()
            .ok_or_else(|| anyhow!("No broadcast kernel"))?;
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
            &self.context().as_ref().unwrap(),
            expanded_shape,
            encoder_hidden_states.dtype(),
            "expanded_encoder_states",
        );

        // Get the broadcast kernel from the model struct
        // let broadcast_kernel = self
        //     .gpu_broadcast_kernel
        //     .as_ref()
        //     .ok_or_else(|| anyhow!("GpuBroadcast kernel not initialized for this model"))?;

        broadcast.encode(encoder_cmd, encoder_hidden_states, &expanded_states, 0);

        Ok(expanded_states)
    }

    // This is the logic moved from the old GpuBackend
    fn project_to_logits(
        &self,
        frame: &mut GpuFrameContext,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        // let (batch, seq, hidden) = hidden_states.dims3();
        // let (encoder_cmd, pool) = frame.resources();

        // let lm_head_weights = self.gpu_lm_head.as_ref().unwrap();
        // let vocab_size = lm_head_weights.shape()[0];

        // let logits = pool.get(vec![batch * seq, vocab_size]);
        // let hidden_states_2d = hidden_states.view(vec![batch * seq, hidden]);

        // // This kernel should be a shared primitive, not created on the fly.
        // // For now, this works. In a future cleanup, you might pass it into the ops from the backend.
        // let linear_kernel = self
        //     .gpu_linear_kernel
        //     .as_ref()
        //     .ok_or_else(|| anyhow!("GpuLinear kernel not initialized for this model"))?;
        // linear_kernel.encode(encoder_cmd, &hidden_states_2d, lm_head_weights, &logits);

        // let final_logits = if let Some(bias) = self.gpu_final_logits_bias.as_ref() {
        //     let logits_with_bias = pool.get(logits.shape().to_vec());
        //     let add_kernel = GpuAdd::new(&self.context.as_ref().unwrap());
        //     add_kernel.encode_broadcast_row(encoder_cmd, &logits, bias, &logits_with_bias);
        //     logits_with_bias
        // } else {
        //     logits
        // };

        // Ok(final_logits.view(vec![batch, seq, vocab_size]))

        let (encoder_cmd, pool) = frame.resources();

        self.pipeline
            .lm_head()
            .forward_gpu(encoder_cmd, pool, hidden_states)
    }
}

#[async_trait]
impl EncoderDecoderLanguageModel for BartModel {
    fn encoder_decoder_cpu_ops(&self) -> Option<&dyn CpuEncoderDecoderOps> {
        if self.pipeline.cpu_decoder().is_some() {
            Some(self)
        } else {
            None
        }
        // if self.device.is_cpu() {
        //     Some(self)
        // } else {
        //     None
        // }
    }
    fn encoder_decoder_gpu_ops(&self) -> Option<&dyn GpuEncoderDecoderOps> {
        if self.pipeline.gpu_decoder().is_some() {
            Some(self)
        } else {
            None
        }
        // if self.device.is_gpu() {
        //     Some(self)
        // } else {
        //     None
        // }
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
            max_length: 256, //self.meta.max_seq_len,
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
