//! LLaMA-style decoder-only language model.
//!
//! This module provides the `LlamaModel`, a model container responsible for loading
//! weights and configuration for Llama and its variants.
//!
//! The actual text generation is handled by the generic `Generator` struct.

use crate::models::llama::config::LlamaConfig;
use crate::models::llama::cpu_decoder::LlamaCpuDecoder;
use crate::models::llama::gpu_decoder::LlamaGpuDecoder;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::gpu_ops::blocks::rope::GpuRoPE;
use edgetransformers::gpu_ops::GpuTensor;
// use edgetransformers::utils::linear_algebra::matmul_2d_faer;
use edgetransformers::linear_layer::LinearLayer;
use edgetransformers::models::base::DecoderLoadConfig;
use edgetransformers::models::base::GpuDecoder;
use edgetransformers::models::base::{AutoregressiveLoop, DecodingStrategy};
use edgetransformers::models::download_model_files;
use edgetransformers::models::{DecoderLanguageModel, LanguageModel, ModelArchitecture, ModelType};
use edgetransformers::rope::RoPE;
use edgetransformers::traits::{Decoder, DecoderArchitecture, DecoderOutput, LanguageModelConfig};
use edgetransformers::utils::linear_algebra::matmul_2d_transposed;
use edgetransformers::weights::DType;
use edgetransformers::weights::ModelWeights;
use edgetransformers::{gpu_context, prelude::*};
use ndarray::{s, Array2, Array3};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// A model container for LLaMA and its variants.
///
/// This struct holds the model's components (decoder, tokenizer, config) but
/// delegates the actual text generation task to the `Generator`.
pub struct LlamaModel {
    decoder: Option<LlamaCpuDecoder>,

    gpu_decoder: Option<LlamaGpuDecoder>, // Use the unified trait

    tokenizer: Tokenizer,
    config: Arc<LlamaConfig>,
    /// The language modeling head, transposed for efficient projection.
    /// Shape: `[hidden_size, vocab_size]`.
    // lm_head: Array2<f32>,
    lm_head: LinearLayer,

    // --- GPU components (will be Some if loaded on GPU) ---
    gpu_lm_head_transposed: Option<GpuTensor>,
    device: Device, // Store the device it was loaded on
    context: Option<Arc<WgpuContext>>,
}

impl LlamaModel {
    /// A list of the specific model types supported by this implementation.
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::Llama3_2_1B,
        ModelType::Llama3_2_3B,

        
        ModelType::Llama3_2_3B_Instruct,
        ModelType::Llama3_8B_Instruct,
        // Add other Llama variants here as you support them
    ];
    pub fn concrete_config(&self) -> &Arc<LlamaConfig> {
        &self.config
    }
    /// Creates a `LlamaModel` from the HuggingFace model registry.
    ///
    /// This will download the model files to a local cache directory if they
    /// are not already present.
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        decoder_config: Option<DecoderLoadConfig>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!("Unsupported LLaMA model type: {:?}", model_type));
        }
        if model_type.info().architecture != ModelArchitecture::Decoder {
            return Err(anyhow!("Model {:?} is not a decoder model.", model_type));
        }

        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("edgetransformers")
        });
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));
        log::info!("Loading Llama model from {:?}", model_dir);
        download_model_files(&model_dir, &model_type.info().paths).await?;
        Self::from_pretrained(&model_dir, model_type, device, context, decoder_config)
    }

    /// Creates a `LlamaModel` from a local directory containing the model files.
    pub fn from_pretrained(
        model_path: &Path,
        _model_type: ModelType, // Used for registry validation, not needed here
        device: Device,
        context: Option<Arc<WgpuContext>>,
        decoder_config: Option<DecoderLoadConfig>,
    ) -> Result<Self> {
        edgetransformers::utils::configure_threading();

        let weights = ModelWeights::new(model_path)?;
        let mut tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let config = Arc::new(LlamaConfig::from_json(&weights.config_json)?);


        // Set up tokenizer truncation, but no padding for autoregressive generation.
        let truncation_params = tokenizers::TruncationParams {
            max_length: config.max_position_embeddings(),
            ..Default::default()
        };
        tokenizer.with_truncation(Some(truncation_params)).unwrap();
        tokenizer.with_padding(None);


        // let mut gpu_lm_head_transposed = None;

        // Create the RoPE module, passing the scaling config if it exists.
        let cpu_rope = Arc::new(RoPE::new_with_scaling(
            config.head_dim(),
            config.max_position_embeddings(),
            config.rope_theta,
            config.rope_scaling.as_ref(),
        ));
        
        // GpuRoPE::new(context, )

        // The generic TransformerDecoder will be built using the LlamaConfig.
        // let decoder = TransformerDecoder::new(
        //     &weights,
        //     config.clone() as Arc<dyn DecoderArchitecture + Send + Sync>,
        //     device,
        //     context,
        //     Some(rope.clone()),
        // )?;

        // Llama ties the embedding and LM head weights.
        let load_config = decoder_config.unwrap_or_default();
        // let lm_head = weights.get_array2(config.get_lm_head_name())?;
        // Determine Target DType
        let target_dtype = load_config.target_dtype;

        if target_dtype.is_some() {
            log::info!("Loading Llama model in {:?}", target_dtype.unwrap());
        };


        // Load Head using the new LinearLayer loader
        let lm_head = LinearLayer::from_weights(&weights, config.get_lm_head_name(), target_dtype)?;
        match device {
            Device::Cpu => {
                // cpu_decoder = Some(TransformerDecoder::new(
                //     &weights,
                //     config.clone() as Arc<dyn DecoderArchitecture + Send + Sync>,
                //     device,
                //     None, // No context for CPU
                //     Some(cpu_rope),
                //     target_dtype,
                // )?);
                let cpu_decoder = Some(LlamaCpuDecoder::new(
                    &weights,
                    config.clone(),
                    cpu_rope,
                    target_dtype,
                )?);
                Ok(Self {
                    config,
                    decoder: cpu_decoder,
                    tokenizer,
                    lm_head: lm_head,
                    gpu_decoder: None,
                    gpu_lm_head_transposed: None,
                    device,
                    context: None,
                })
            }
            Device::Wgpu => {
                let ctx = context.ok_or_else(|| anyhow!("WGPU device requires a context"))?;

                // Create GPU RoPE
                let gpu_rope = GpuRoPE::new(&ctx, &cpu_rope.cos_cache, &cpu_rope.sin_cache)?;

                let gpu_decoder = Some(LlamaGpuDecoder::new(
                    &ctx,
                    &weights,
                    config.clone(),
                    gpu_rope,
                )?);
                let gpu_lm_head_transposed = if !load_config.offload_lm_head {
                    log::info!("Loading LM Head to VRAM (Unified Layout [Vocab, Hidden])");
                    
                    let head_name = config.get_lm_head_name();
                    let tensor = if let Ok(raw) = weights.get_raw(head_name) {
                         // Load Raw (BF16 or F32) without modification
                         GpuTensor::from_raw(&ctx, &raw, "lm_head")?
                    } else {
                        // Fallback: Load as Array2 (F32), but DO NOT TRANSPOSE.
                        // Standard ndarray load is [Vocab, Hidden]
                        let arr = weights.get_array2(head_name)?;
                        GpuTensor::from_ndarray(&ctx, &arr)?
                    };
                    
                    Some(tensor)
                } else {
                    None
                };
                Ok(Self {
                    config,
                    decoder: None,
                    tokenizer,
                    lm_head: lm_head,
                    gpu_decoder,
                    gpu_lm_head_transposed, // This is now BF16 [Vocab, Hidden] OR F32 [Hidden, Vocab]
                    device,
                    context: Some(ctx),
                })
            }
        }
    }
}

// --- Trait Implementations ---
// These implementations make `LlamaModel` compatible with the generic `Generator`.

impl TransformerModel for LlamaModel {
    fn device(&self) -> Device {
        self.device
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        // self.gpu_decoder().unwrap().context()
        let ctx = self.context.clone();
        ctx
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl LanguageModel for LlamaModel {
    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        _num_beams: usize,
    ) -> Result<Box<dyn Cache>> {
        Ok(match self.device() {
            Device::Cpu => Box::new(CpuKVCache::new(
                self.num_layers(),
                batch_size,
                max_len,
                self.config().kv_dim(),
            )),
            Device::Wgpu => {
                let context = self
                    .context()
                    .ok_or_else(|| anyhow!("GPU model missing context"))?;
                let head_dim = self.hidden_size() / self.num_heads();
                Box::new(GpuKVCache::new(
                    &context,
                    self.num_layers(),
                    batch_size,
                    self.config().num_key_value_heads(), // Llama has a specific field for this
                    head_dim,
                    max_len,
                )?)
            }
        })
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn config(&self) -> &dyn LanguageModelConfig {
        self.config.as_ref()
    }
    fn bos_token_id(&self) -> Option<u32> {
        Some(128000)
    }
    fn eos_token_id(&self) -> Option<u32> {
        Some(self.config.eos_token_id)
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }
    fn num_heads(&self) -> usize {
        self.config.num_attention_heads
    }
}

#[async_trait]
impl DecoderLanguageModel for LlamaModel {
    fn decoder(&self) -> &dyn Decoder<Input=Array2<u32>, Output=DecoderOutput> {
        self.decoder
            .as_ref()
            .expect("CPU decoder not initialized - use Device::Cpu")
    }
    fn lm_head(&self) -> &Array2<f32> {
        // &self.lm_head.to_array2_f32()
        self.lm_head.as_f32().expect("lm_head is in BF16 mode, cannot access as &Array2<f32>. Use project_to_logits instead.")
    }
    fn autoregressive_loop(&self) -> AutoregressiveLoop {
        AutoregressiveLoop::Pipelined
    }
    // --- GPU-centric methods ---
    fn gpu_decoder(&self) -> Result<&(dyn GpuDecoder + Send + Sync)> {
        // Correct way to cast Option<ConcreteType> to a trait object Result
        self.gpu_decoder
            .as_ref()
            .map(|d| d as &(dyn GpuDecoder + Send + Sync))
            .ok_or_else(|| anyhow!("Not a GPU model"))
    }

    fn gpu_lm_head_transposed(&self) -> Result<&GpuTensor> {
        self.gpu_lm_head_transposed
            .as_ref()
            .ok_or_else(|| anyhow!("Not a GPU model"))
    }

    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden) = hidden_states.dim();
        let hidden_2d = hidden_states.view().into_shape_with_order((batch * seq, hidden))?;

        // Dispatch happens here automatically
        log::info!("lm_head dtype: {:?}, shape: {:?}",
        if self.lm_head.as_f32().is_some() { "F32" } else { "BF16" },
        self.lm_head.shape()
    );
        let logits_2d = self.lm_head.matmul(&hidden_2d);

        logits_2d
            .into_shape_with_order((batch, seq, self.vocab_size()))
            .map_err(|e| anyhow!(e))
    }
}
