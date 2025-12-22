//! LLaMA-style decoder-only language model.
//!
//! This module provides the `LlamaModel`, a model container responsible for loading
//! weights and configuration for Llama and its variants.
//!
//! The actual text generation is handled by the generic `Generator` struct.

// --- Standard Library ---
use std::path::{Path, PathBuf};
use std::sync::Arc;

// --- External Crates ---
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use ndarray::{s, Array2, Array3};
use tokenizers::Tokenizer;

// --- Crate-Specific ---
use crate::models::llama::{
    config::LlamaConfig, cpu_decoder::LlamaCpuDecoder, gpu_decoder::LlamaGpuDecoder,
};
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::traits::ModelConfig;
// --- Workspace Crates ---
use kjarni_transformers::{
    decoder::prelude::*,
    gpu_ops::{
        blocks::rope::GpuRoPE, primitives::{linear::GpuLinearLayer, matmul::GpuMatMul}, GpuFrameContext,
        GpuTensor,
        Kernel,
    },
    linear_layer::LinearLayer,
    models::{
        base::AutoregressiveLoop, download_model_files, LanguageModel, ModelArchitecture, ModelType,
    },
    prelude::*,
    rope::RoPE,
    tensor::{DType, RawTensor},
    traits::{DecoderArchitecture, LanguageModelConfig},
    weights::ModelWeights,
    WgpuContext,
};

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
    gpu_lm_head_layer: Option<GpuLinearLayer>,
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
        decoder_config: Option<ModelLoadConfig>,
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
                .join("kjarni")
        });
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));
        log::info!("Loading Llama model from {:?}", model_dir);
        download_model_files(&model_dir, &model_type.info().paths).await?;

        if device.is_gpu() && context.is_none() {
            Self::from_pretrained(
                &model_dir,
                device,
                Some(WgpuContext::new().await?),
                decoder_config,
            )
        } else {
            Self::from_pretrained(&model_dir, device, context, decoder_config)
        }
    }

    /// Creates a `LlamaModel` from a local directory containing the model files.
    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        decoder_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        kjarni_transformers::utils::configure_threading();

        // load weights and config
        let weights = ModelWeights::new(model_path)?;
        let config =
            LlamaConfig::from_loader(&*weights.loader, Some(weights.config_json.as_ref()))?;

        let metadata = config.metadata();
        let layout = config.layout();

        // resolve tokenizer
        let tokenizer_path = if model_path.is_file() {
            model_path.parent().unwrap().join("tokenizer.json")
        } else {
            model_path.join("tokenizer.json")
        };
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;

        // Set up tokenizer truncation, but no padding for autoregressive generation.
        let truncation_params = tokenizers::TruncationParams {
            max_length: metadata.max_seq_len,
            ..Default::default()
        };
        tokenizer.with_truncation(Some(truncation_params)).unwrap();
        tokenizer.with_padding(None);

        // Create the RoPE module, passing the scaling config if it exists.
        let cpu_rope = Arc::new(RoPE::new_with_scaling(
            metadata.head_dim,
            metadata.max_seq_len,
            metadata.rope_theta.unwrap_or(500000.0),
            metadata.rope_scaling.as_ref(),
        ));

        // Llama ties the embedding and LM head weights.
        let load_config = decoder_config.unwrap_or_default();
        let target_dtype = load_config.target_dtype;

        if target_dtype.is_some() {
            log::info!("Loading Llama model in {:?}", target_dtype.unwrap());
        };

        // Load Head using the new LinearLayer loader
        let lm_head =
            LinearLayer::from_weights(&weights, &layout.lm_head, None, target_dtype, None)?;
        match device {
            Device::Cpu => {
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
                    gpu_lm_head_layer: None,
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
                    load_config,
                )?);
                let gpu_lm_head_transposed = if !load_config.offload_lm_head {
                    log::info!("Loading LM Head to VRAM (Unified Layout [Vocab, Hidden])");

                    let tensor = if let Ok(raw) = weights.get_raw(&layout.lm_head) {
                        // Load Raw (BF16 or F32) without modification
                        GpuTensor::from_raw(&ctx, &raw, "lm_head")?
                    } else {
                        let arr = weights.get_array2(&layout.lm_head)?;
                        GpuTensor::from_ndarray(&ctx, &arr)?
                    };

                    Some(tensor)
                } else {
                    None
                };
                let gpu_lm_head_layer = Some(GpuLinearLayer::new(&ctx));
                Ok(Self {
                    config,
                    decoder: None,
                    tokenizer,
                    lm_head: lm_head,
                    gpu_decoder,
                    gpu_lm_head_transposed,
                    gpu_lm_head_layer: gpu_lm_head_layer,
                    device,
                    context: Some(ctx),
                })
            }
        }
    }
}

impl TransformerModel for LlamaModel {
    fn device(&self) -> Device {
        self.device
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
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
        // 1. Get the unified metadata
        let meta = self.config.metadata();

        // 2. Calculate dimensions for the cache
        let head_dim = meta.head_dim;
        let kv_dim = head_dim * meta.num_kv_heads;

        Ok(match self.device() {
            Device::Cpu => {
                // Initialize CPU KV Cache
                Box::new(CpuKVCache::new(
                    meta.num_layers,
                    batch_size,
                    max_len,
                    kv_dim,
                ))
            }
            Device::Wgpu => {
                let context = self
                    .context()
                    .ok_or_else(|| anyhow!("GPU model missing context"))?;

                // Initialize GPU KV Cache
                Box::new(GpuKVCache::new(
                    &context,
                    meta.num_layers,
                    batch_size,
                    meta.num_kv_heads,
                    head_dim,
                    max_len,
                )?)
            }
        })
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn vocab_size(&self) -> usize {
        self.config.metadata().vocab_size
    }

    fn hidden_size(&self) -> usize {
        self.config.metadata().hidden_size
    }

    fn num_layers(&self) -> usize {
        self.config.metadata().num_layers
    }

    fn num_heads(&self) -> usize {
        self.config.metadata().num_attention_heads
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(self.config.bos_token_id)
    }

    fn eos_token_id(&self) -> Option<u32> {
        Some(self.config.eos_token_id)
    }

    fn context_size(&self) -> usize {
        self.config.metadata().max_seq_len
    }

    fn forced_bos_token_id(&self) -> Option<u32> {
        None
    }
    fn pad_token_id(&self) -> Option<u32> {
        self.config.pad_token_id
    }
}

impl CpuDecoderOps for LlamaModel {
    fn decoder(&self) -> &dyn CpuDecoder {
        self.decoder
            .as_ref()
            .expect("CPU decoder not initialized - use Device::Cpu")
    }

    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let t_start = std::time::Instant::now();

        let (batch, seq, hidden) = hidden_states.dim();
        let fast = false; // batch == 1 && seq == 1;

        let logits_2d = if fast {
            // Optimized path for single-token decode
            let hidden_1d = hidden_states.slice(s![0, 0, ..]);
            let logits_1d = self.lm_head.project_logits(&hidden_1d);

            logits_1d.insert_axis(ndarray::Axis(0))
        } else {
            // Batched/prefill path
            let hidden_2d = hidden_states
                .view()
                .into_shape_with_order((batch * seq, hidden))?;
            self.lm_head.matmul(&hidden_2d)
        };
        log::info!("[Logits Projection] Time: {:?}", t_start.elapsed());
        logits_2d
            .into_shape_with_order((batch, seq, self.vocab_size()))
            .map_err(|e| anyhow!(e))
    }

    fn get_attention_mask(&self, seq_len: usize, _past_len: usize) -> Result<Array2<f32>> {
        Ok(kjarni_transformers::utils::create_full_attention_mask(
            1, seq_len,
        ))
    }
}

impl GpuDecoderOps for LlamaModel {
    fn decoder(&self) -> &dyn GpuDecoder {
        self.gpu_decoder
            .as_ref()
            .expect("GPU decoder not initialized - use Device::Wgpu")
    }

    fn get_attention_mask(
        &self,
        ctx: &mut GpuFrameContext,
        seq_len: usize,
        max_len: usize,
    ) -> Result<GpuTensor> {
        let mask_data: Vec<f32> = (0..max_len)
            .map(|i| if i < seq_len { 1.0 } else { 0.0 })
            .collect();

        let tensor = GpuTensor::from_raw(
            ctx.context,
            &RawTensor {
                bytes: std::borrow::Cow::Owned(bytemuck::cast_slice(&mask_data).to_vec()),
                shape: vec![1, max_len],
                dtype: kjarni_transformers::tensor::DType::F32,
                name: "AttentionMask".to_string(),
            },
            "AttentionMask",
        )?;

        Ok(tensor)
    }

    fn project_to_logits(
        &self,
        ctx: &mut GpuFrameContext,
        last_hidden_state: &GpuTensor,
    ) -> Result<GpuTensor> {
        let lm_head = self
            .gpu_lm_head_transposed
            .as_ref()
            .ok_or_else(|| anyhow!("GPU LM head not loaded"))?;

        let linear_layer = self
            .gpu_lm_head_layer
            .as_ref()
            .ok_or_else(|| anyhow!("GPU Linear Layer not initialized"))?;

        let (batch, seq, _hidden) = last_hidden_state.dims3();
        let (vocab, _) = lm_head.dims2();

        let logits = ctx.pool_guard.get(vec![batch, seq, vocab]);

        linear_layer.encode(
            ctx.encoder.as_mut().unwrap(),
            last_hidden_state, // Input
            lm_head,           // Weights
            &logits,           // Output
        );

        Ok(logits)
    }
}

#[async_trait(?Send)]
impl DecoderLanguageModel for LlamaModel {
    fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps> {
        if self.device == Device::Cpu {
            Some(self)
        } else {
            None
        }
    }

    fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps> {
        if self.device == Device::Wgpu {
            Some(self)
        } else {
            None
        }
    }

    fn autoregressive_loop(&self) -> AutoregressiveLoop {
        AutoregressiveLoop::Pipelined
    }
}
