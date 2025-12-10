use crate::models::bart::config::{BartConfig, BartLikeConfig};
use crate::models::bart::cpu_decoder::BartCpuDecoder;
use crate::models::bart::cpu_encoder::BartCpuEncoder;
use crate::models::bart::gpu_decoder::BartGpuDecoder;
// NEW
use crate::models::bart::gpu_encoder::BartGpuEncoder;
// NEW

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use edgetransformers::cache::{Cache, CpuBeamKVCache, GpuBeamKVCache};
use edgetransformers::encoder::prelude::*;
use edgetransformers::encoder_decoder::traits::{
    CrossAttentionDecoder, EncoderDecoderLanguageModel, GpuCrossAttentionDecoder,
};
use edgetransformers::gpu_ops::GpuTensor;
use edgetransformers::linear_layer::LinearLayer;
use edgetransformers::models::base::{BeamSearchParams, DecodingStrategy, GenerationConfig};
use edgetransformers::models::download_model_files;
use edgetransformers::models::{ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::traits::{
    CpuEncoder, DecoderOutput, Encoder, EncoderOutput, LanguageModelConfig, TransformerModel,
};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array1, Array2, Array3};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

pub struct BartModel {
    // CPU components
    cpu_encoder: Option<BartCpuEncoder>,
    cpu_decoder: Option<BartCpuDecoder>,

    // GPU components
    gpu_encoder: Option<BartGpuEncoder>,
    gpu_decoder: Option<BartGpuDecoder>,
    gpu_lm_head: Option<GpuTensor>,
    gpu_final_logits_bias: Option<GpuTensor>,

    // Shared components
    tokenizer: Tokenizer,
    config: Arc<BartConfig>,
    lm_head: LinearLayer,                   // Still needed for CPU path
    final_logits_bias: Option<Array1<f32>>, // Still needed for CPU path
    device: Device,
    context: Option<Arc<WgpuContext>>,
}

impl BartModel {
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::BartLargeCnn,
        ModelType::DistilBartCnn,
        ModelType::MarianEnIs,
    ];
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
                .join("edgetransformers")
        });
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        log::info!("Loading BART model from {:?}", model_dir);
        download_model_files(&model_dir, &model_type.info().paths).await?;

        Self::from_pretrained(&model_dir, device, context)
    }

    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let tokenizer =
            Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|e| anyhow!(e))?;
        let config = Arc::new(BartConfig::from_json(&weights.config_json)?);

        // Load CPU versions of head/bias regardless, as they are small
        let lm_head = LinearLayer::from_weights(&weights, "model.shared.weight", None)?;
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
                    &weights.get_raw("model.shared.weight")?,
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
                    context: Some(ctx),
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

// --- LanguageModel Implementation ---
impl LanguageModel for BartModel {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn config(&self) -> &dyn LanguageModelConfig {
        self.config.as_ref()
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

// --- EncoderDecoderLanguageModel Implementation ---
// This is the specific trait that allows Seq2SeqGenerator to work.

#[async_trait]
impl EncoderDecoderLanguageModel for BartModel {
    // --- CPU Accessors ---
    fn encoder(&self) -> Result<&dyn Encoder<Input=Array2<u32>, Output=EncoderOutput>> {
        // self.cpu_encoder.as_ref().map(|e| e as _).ok_or_else(|| anyhow!("Model not loaded for CPU"))
        panic!("deprecated! Use `bart_cpu_encoder()` instead.");
    }
    fn cpu_encoder(&self) -> Result<&dyn CpuEncoder> {
        self.cpu_encoder
            .as_ref()
            .map(|e| e as _)
            .ok_or_else(|| anyhow!("Model not loaded for CPU"))
    }

    fn cpu_decoder(
        &self,
    ) -> Result<
        &dyn CrossAttentionDecoder<
            TokenInput=Array2<u32>,
            EncoderStateInput=Array3<f32>,
            MaskInput=Array2<f32>,
            Output=DecoderOutput,
        >,
    > {
        self.cpu_decoder
            .as_ref()
            .map(|d| d as _)
            .ok_or_else(|| anyhow!("Model not loaded for CPU"))
    }

    // --- GPU Accessors ---
    fn gpu_encoder(&self) -> Result<&dyn GpuEncoder> {
        self.gpu_encoder
            .as_ref()
            .map(|e| e as _)
            .ok_or_else(|| anyhow!("Model not loaded for GPU"))
    }

    fn gpu_decoder(&self) -> Result<&dyn GpuCrossAttentionDecoder> {
        self.gpu_decoder
            .as_ref()
            .map(|d| d as _)
            .ok_or_else(|| anyhow!("Model not loaded for GPU"))
    }

    // --- LM Head Accessors ---
    fn lm_head_layer(&self) -> &LinearLayer {
        &self.lm_head
    }

    fn gpu_lm_head_weights(&self) -> Result<&GpuTensor> {
        self.gpu_lm_head
            .as_ref()
            .ok_or_else(|| anyhow!("GPU LM head not loaded"))
    }

    fn final_logits_bias(&self) -> Option<&Array1<f32>> {
        self.final_logits_bias.as_ref()
    }

    fn gpu_final_logits_bias(&self) -> Result<Option<&GpuTensor>> {
        Ok(self.gpu_final_logits_bias.as_ref())
    }

    fn decoder_start_token_id(&self) -> u32 {
        self.config.decoder_start_token_id
    }

    // 6. Generation Config Defaults
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
