//! LLaMA-style decoder-only language model.
//!
//! This module provides the `LlamaModel`, a model container responsible for loading
//! weights and configuration for Llama and its variants.
//!
//! ## Architecture
//!
//! The model uses a `DecoderPipeline` to manage its components:
//! - `LoadedEmbeddings` - Token embeddings (CPU or GPU)
//! - `LlamaCpuDecoder` / `LlamaGpuDecoder` - Transformer layers
//! - `LoadedLMHead` - Language model head for logit projection
//!
//! The `ExecutionPlan` determines where each stage runs, enabling:
//! - Full GPU execution (default)
//! - Full CPU execution
//! - Hybrid: GPU layers with CPU embeddings/head (VRAM savings)
//!
//! ## Usage
//!
//! ```rust
//! // Simple usage
//! let model = LlamaModel::from_pretrained(path, Device::Wgpu, context, None)?;
//!
//! // With configuration
//! let config = ModelLoadConfig::builder()
//!     .offload_lm_head(true)  // Save ~500MB VRAM
//!     .target_dtype(Some(DType::BF16))
//!     .build();
//! let model = LlamaModel::from_pretrained(path, Device::Wgpu, context, Some(config))?;
//!
//! // Change execution plan at runtime
//! model.pipeline_mut().set_plan(ExecutionPlan::gpu_offload_head())?;
//! ```

// =============================================================================
// Imports
// =============================================================================

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use tokenizers::Tokenizer;

// Crate-specific imports
use crate::models::llama::{
    config::LlamaConfig, cpu_decoder::LlamaCpuDecoder, gpu_decoder::LlamaGpuDecoder,
};

// Workspace imports
use kjarni_transformers::{
    WgpuContext,
    // Cache
    cache::{Cache, CpuKVCache, GpuKVCache},
    // Decoder traits
    decoder::prelude::*,
    // Embeddings
    embeddings::{EmbeddingConfig, LoadedEmbeddings},
    // Execution planning
    execution::ExecutionPlan,
    // GPU operations
    gpu_ops::{GpuFrameContext, GpuTensor, blocks::rope::GpuRoPE},
    // LM Head
    lm_head::{LMHeadConfig, LoadedLMHead},
    // Model infrastructure
    models::base::{AutoregressiveLoop, ModelLoadConfig},
    models::{LanguageModel, ModelArchitecture, ModelType, download_model_files},
    // Pipeline
    pipeline::{DecoderPipeline, DecoderPipelineConfig},
    // Utilities
    prelude::*,
    rope::RoPE,
    tensor::{DType, TensorView},
    traits::{InferenceModel, ModelConfig},
    weights::ModelWeights,
};

// =============================================================================
// Model Definition
// =============================================================================

/// A model container for LLaMA and its variants (Llama 2, Llama 3, Code Llama, etc.).
///
/// This struct holds the model's components via a `DecoderPipeline` and implements
/// the necessary traits for the generation infrastructure to use it.
///
/// ## Supported Models
///
/// - Llama 3.2 1B / 3B
/// - Llama 3 8B Instruct
/// - Other Llama variants with compatible architecture
///
/// ## Example
///
/// ```rust
/// use kjarni_models::LlamaModel;
/// use kjarni_transformers::prelude::Device;
///
/// let model = LlamaModel::from_pretrained(
///     Path::new("/models/llama-3.2-1b"),
///     Device::Wgpu,
///     Some(context),
///     None,
/// )?;
/// ```
pub struct LlamaModel {
    /// The unified pipeline containing all model components.
    /// Access via `pipeline()` / `pipeline_mut()` for runtime configuration.
    pipeline: DecoderPipeline,

    /// The tokenizer for encoding/decoding text.
    tokenizer: Tokenizer,

    /// Model configuration (architecture, sizes, special tokens, etc.)
    config: Arc<LlamaConfig>,

    /// The device this model is primarily loaded on.
    /// Note: With hybrid execution, some components may be on different devices.
    device: Device,

    /// GPU context for WGPU operations. None if CPU-only.
    context: Option<Arc<WgpuContext>>,
}

// =============================================================================
// Model Loading
// =============================================================================

impl LlamaModel {
    /// List of model types this implementation supports.
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::Llama3_2_1B,
        ModelType::Llama3_2_3B,
        ModelType::Llama3_2_3B_Instruct,
        ModelType::Llama3_8B_Instruct,
    ];

    /// Creates a `LlamaModel` from the HuggingFace model registry.
    ///
    /// Downloads model files to a local cache directory if not already present.
    ///
    /// # Arguments
    ///
    /// * `model_type` - The specific Llama variant to load
    /// * `cache_dir` - Optional custom cache directory (defaults to system cache)
    /// * `device` - Target device (CPU or GPU)
    /// * `context` - GPU context (required for GPU, created automatically if None)
    /// * `decoder_config` - Optional loading configuration
    ///
    /// # Example
    ///
    /// ```rust
    /// let model = LlamaModel::from_registry(
    ///     ModelType::Llama3_2_1B,
    ///     None,  // Use default cache
    ///     Device::Wgpu,
    ///     None,  // Auto-create context
    ///     None,  // Default config
    /// ).await?;
    /// ```
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        decoder_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        // Validate model type
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!("Unsupported LLaMA model type: {:?}", model_type));
        }
        if model_type.info().architecture != ModelArchitecture::Decoder {
            return Err(anyhow!("Model {:?} is not a decoder model.", model_type));
        }

        // Resolve cache directory
        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("kjarni")
        });
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        log::info!("Loading Llama model from {:?}", model_dir);

        // Download model files if needed
        download_model_files(&model_dir, &model_type.info().paths).await?;

        // Create GPU context if needed and not provided
        let context = if device.is_gpu() && context.is_none() {
            log::info!("No GPU context provided, creating a new one...");
            Some(WgpuContext::new().await?)
        } else {
            context
        };

        Self::from_pretrained(&model_dir, device, context, decoder_config)
    }

    /// Creates a `LlamaModel` from a local directory containing model files.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to directory containing model files (safetensors, config.json, tokenizer.json)
    /// * `device` - Target device (CPU or GPU)
    /// * `context` - GPU context (required if device is GPU)
    /// * `decoder_config` - Optional loading configuration
    ///
    /// # Loading Configuration
    ///
    /// The `ModelLoadConfig` allows customizing:
    /// - `target_dtype` - Override weight dtype (e.g., force BF16)
    /// - `offload_lm_head` - Keep LM head on CPU to save VRAM
    /// - `offload_embeddings` - Keep embeddings on CPU
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = ModelLoadConfig::builder()
    ///     .target_dtype(Some(DType::BF16))
    ///     .offload_lm_head(true)
    ///     .build();
    ///
    /// let model = LlamaModel::from_pretrained(
    ///     Path::new("/models/llama"),
    ///     Device::Wgpu,
    ///     Some(context),
    ///     Some(config),
    /// )?;
    /// ```
    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        decoder_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        // Configure threading for CPU operations
        kjarni_transformers::utils::configure_threading();

        // =========================================================================
        // 1. Load Configuration
        // =========================================================================

        let weights = ModelWeights::new(model_path)?;
        let config =
            LlamaConfig::from_loader(&*weights.loader, Some(weights.config_json.as_ref()))?;

        let metadata = config.metadata();
        let layout = config.layout();
        let load_config = decoder_config.unwrap_or_default();
        let target_dtype = load_config.target_dtype;

        if let Some(dtype) = target_dtype {
            log::info!("Loading Llama model in {:?}", dtype);
        }

        // =========================================================================
        // 2. Load Tokenizer
        // =========================================================================

        let tokenizer = Self::load_tokenizer(model_path, metadata.max_seq_len)?;

        // =========================================================================
        // 3. Create RoPE (Rotary Position Embeddings)
        // =========================================================================

        // CPU RoPE is always created (needed for cache initialization even on GPU)
        let cpu_rope = Arc::new(RoPE::new_with_scaling(
            metadata.head_dim,
            metadata.max_seq_len,
            metadata.rope_theta.unwrap_or(500000.0),
            metadata.rope_scaling.as_ref(),
        ));

        // =========================================================================
        // 4. Determine Execution Plan
        // =========================================================================

        // The execution plan determines where each component runs.
        // This affects what we need to load.
        let plan = Self::create_execution_plan(device, &load_config);

        // Determine what to load based on plan requirements
        let load_cpu = plan.needs_cpu() || device == Device::Cpu;
        let load_gpu = plan.needs_gpu() && device == Device::Wgpu;

        log::info!(
            "Execution plan: embeddings={:?}, layers={:?}, lm_head={:?}",
            plan.embeddings,
            plan.layers,
            plan.lm_head
        );

        // =========================================================================
        // 5. Load Embeddings
        // =========================================================================
        let target_dtype = load_config.target_dtype;
        let embeddings = LoadedEmbeddings::new(
            context.as_ref(),
            &weights,
            EmbeddingConfig::new(&layout.token_embedding, metadata.hidden_size),
            load_cpu || load_config.offload_embeddings,
            load_gpu && !load_config.offload_embeddings,
            target_dtype,
        )?;

        // =========================================================================
        // 6. Load Decoders
        // =========================================================================

        // CPU Decoder
        let cpu_decoder: Option<Box<dyn CpuDecoder>> = if load_cpu {
            log::info!("Loading CPU decoder...");
            Some(Box::new(LlamaCpuDecoder::new(
                &weights,
                config.clone(),
                cpu_rope.clone(),
                target_dtype,
            )?))
        } else {
            None
        };

        // GPU Decoder
        let gpu_decoder: Option<Box<dyn GpuDecoder>> = if load_gpu {
            let ctx = context
                .as_ref()
                .ok_or_else(|| anyhow!("GPU device requires a WgpuContext"))?;

            log::info!("Loading GPU decoder...");

            // Create GPU RoPE from precomputed CPU caches
            let gpu_rope = GpuRoPE::new(ctx, &cpu_rope.cos_cache, &cpu_rope.sin_cache)?;

            Some(Box::new(LlamaGpuDecoder::new(
                ctx,
                &weights,
                config.clone(),
                gpu_rope,
                load_config.clone(),
            )?))
        } else {
            None
        };

        // =========================================================================
        // 7. Load LM Head
        // =========================================================================

        let lm_head = LoadedLMHead::new(
            context.as_ref(),
            &weights,
            LMHeadConfig::new(&layout.lm_head, metadata.vocab_size, metadata.hidden_size),
            load_cpu || load_config.offload_lm_head,
            load_gpu && !load_config.offload_lm_head,
            target_dtype,
        )?;

        // =========================================================================
        // 8. Create Pipeline
        // =========================================================================

        let pipeline = DecoderPipeline::new(
            embeddings,
            cpu_decoder,
            gpu_decoder,
            lm_head,
            plan,
            context.clone(),
            DecoderPipelineConfig {
                num_layers: metadata.num_layers,
                hidden_size: metadata.hidden_size,
                vocab_size: metadata.vocab_size,
            },
        )?;

        // =========================================================================
        // 9. Return Model
        // =========================================================================

        Ok(Self {
            pipeline,
            tokenizer,
            config,
            device,
            context,
        })
    }

    // =========================================================================
    // Private Loading Helpers
    // =========================================================================

    /// Loads and configures the tokenizer.
    fn load_tokenizer(model_path: &Path, max_seq_len: usize) -> Result<Tokenizer> {
        let tokenizer_path = if model_path.is_file() {
            model_path.parent().unwrap().join("tokenizer.json")
        } else {
            model_path.join("tokenizer.json")
        };

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e))?;

        // Configure truncation for max sequence length
        let truncation_params = tokenizers::TruncationParams {
            max_length: max_seq_len,
            ..Default::default()
        };
        tokenizer
            .with_truncation(Some(truncation_params))
            .map_err(|e| anyhow!("Failed to configure tokenizer truncation: {}", e))?;

        // No padding for autoregressive generation
        tokenizer.with_padding(None);

        Ok(tokenizer)
    }

    /// Creates the execution plan based on device and configuration.
    fn create_execution_plan(device: Device, config: &ModelLoadConfig) -> ExecutionPlan {
        match device {
            Device::Cpu => ExecutionPlan::full_cpu(),
            Device::Wgpu => {
                // Check for offloading options
                let offload_embeddings = config.offload_embeddings;
                let offload_head = config.offload_lm_head;

                match (offload_embeddings, offload_head) {
                    (true, true) => ExecutionPlan::gpu_offload_ends(),
                    (false, true) => ExecutionPlan::gpu_offload_head(),
                    (true, false) => ExecutionPlan::custom(Device::Cpu, Device::Wgpu, Device::Wgpu),
                    (false, false) => ExecutionPlan::full_gpu(),
                }
            }
        }
    }
}

// =============================================================================
// Public Accessors
// =============================================================================

impl LlamaModel {
    /// Returns a reference to the model configuration.
    pub fn config(&self) -> &Arc<LlamaConfig> {
        &self.config
    }

    /// Returns the concrete Llama configuration.
    /// Alias for `config()` for backward compatibility.
    pub fn concrete_config(&self) -> &Arc<LlamaConfig> {
        &self.config
    }

    /// Returns a reference to the decoder pipeline.
    ///
    /// Use this to inspect the current execution plan or access components.
    pub fn pipeline(&self) -> &DecoderPipeline {
        &self.pipeline
    }

    /// Returns a mutable reference to the decoder pipeline.
    ///
    /// Use this to change the execution plan at runtime:
    ///
    /// ```rust
    /// // Switch to CPU-offloaded head to save VRAM
    /// model.pipeline_mut().set_plan(ExecutionPlan::gpu_offload_head())?;
    /// ```
    pub fn pipeline_mut(&mut self) -> &mut DecoderPipeline {
        &mut self.pipeline
    }

    /// Returns the current execution plan.
    pub fn execution_plan(&self) -> &ExecutionPlan {
        self.pipeline.plan()
    }
}

// =============================================================================
// InferenceModel Implementation
// =============================================================================

impl InferenceModel for LlamaModel {
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

// =============================================================================
// LanguageModel Implementation
// =============================================================================

impl LanguageModel for LlamaModel {
    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        _num_beams: usize,
    ) -> Result<Box<dyn Cache>> {
        let meta = self.config.metadata();
        let head_dim = meta.head_dim;

        // Create cache based on where layers run (not the model's primary device)
        match self.pipeline.plan().layers {
            Device::Cpu => {
                let kv_dim = head_dim * meta.num_kv_heads;
                Ok(Box::new(CpuKVCache::new(
                    meta.num_layers,
                    batch_size,
                    max_len,
                    kv_dim,
                )))
            }
            Device::Wgpu => {
                let context = self
                    .context()
                    .ok_or_else(|| anyhow!("GPU cache requires WgpuContext"))?;

                Ok(Box::new(GpuKVCache::new(
                    &context,
                    meta.num_layers,
                    batch_size,
                    meta.num_kv_heads,
                    head_dim,
                    max_len,
                )?))
            }
        }
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

    fn pad_token_id(&self) -> Option<u32> {
        self.config.pad_token_id
    }

    fn context_size(&self) -> usize {
        self.config.metadata().max_seq_len
    }

    fn forced_bos_token_id(&self) -> Option<u32> {
        None // Llama doesn't use forced BOS
    }

    fn forced_eos_token_id(&self) -> Option<u32> {
        None // Llama doesn't use forced EOS
    }
}

// =============================================================================
// CpuDecoderOps Implementation
// =============================================================================

impl CpuDecoderOps for LlamaModel {
    fn decoder(&self) -> &dyn CpuDecoder {
        self.pipeline
            .cpu_decoder()
            .expect("CPU decoder not available - check ExecutionPlan or load with Device::Cpu")
    }

    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        self.pipeline.lm_head().forward_cpu(hidden_states)
    }

    fn get_attention_mask(&self, seq_len: usize, _past_len: usize) -> Result<Array2<f32>> {
        // Llama uses full attention (no sliding window in base model)
        Ok(kjarni_transformers::utils::create_full_attention_mask(
            1, seq_len,
        ))
    }
}

// =============================================================================
// GpuDecoderOps Implementation
// =============================================================================

impl GpuDecoderOps for LlamaModel {
    fn decoder(&self) -> &dyn GpuDecoder {
        self.pipeline
            .gpu_decoder()
            .expect("GPU decoder not available - check ExecutionPlan or load with Device::Wgpu")
    }

    fn get_attention_mask(
        &self,
        ctx: &mut GpuFrameContext,
        seq_len: usize,
        max_len: usize,
    ) -> Result<GpuTensor> {
        // Create a simple padding mask: 1.0 for valid positions, 0.0 for padding
        // The actual causal masking is handled inside the attention kernel
        let mask_data: Vec<f32> = (0..max_len)
            .map(|i| if i < seq_len { 1.0 } else { 0.0 })
            .collect();

        GpuTensor::from_raw(
            ctx.context,
            &TensorView {
                bytes: std::borrow::Cow::Owned(bytemuck::cast_slice(&mask_data).to_vec()),
                shape: vec![1, max_len],
                dtype: DType::F32,
                name: "AttentionMask".to_string(),
            },
            "AttentionMask",
        )
    }

    fn project_to_logits(
        &self,
        ctx: &mut GpuFrameContext,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        let (enc, pool) = ctx.resources();
        self.pipeline
            .lm_head()
            .forward_gpu(enc, pool, hidden_states)
    }
}

// =============================================================================
// DecoderLanguageModel Implementation
// =============================================================================

#[async_trait(?Send)]
impl DecoderLanguageModel for LlamaModel {
    fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps> {
        // Return self if CPU decoder is available
        if self.pipeline.cpu_decoder().is_some() {
            Some(self)
        } else {
            None
        }
    }

    fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps> {
        // Return self if GPU decoder is available
        if self.pipeline.gpu_decoder().is_some() {
            Some(self)
        } else {
            None
        }
    }

    fn autoregressive_loop(&self) -> AutoregressiveLoop {
        // Llama uses the modern pipelined approach
        AutoregressiveLoop::Pipelined
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_models() {
        assert!(LlamaModel::SUPPORTED_MODELS.contains(&ModelType::Llama3_2_1B));
        assert!(LlamaModel::SUPPORTED_MODELS.contains(&ModelType::Llama3_2_3B));
    }

    #[test]
    fn test_execution_plan_creation() {
        // Full CPU
        let plan = LlamaModel::create_execution_plan(Device::Cpu, &ModelLoadConfig::default());
        assert_eq!(plan, ExecutionPlan::full_cpu());

        // Full GPU
        let plan = LlamaModel::create_execution_plan(Device::Wgpu, &ModelLoadConfig::default());
        assert_eq!(plan, ExecutionPlan::full_gpu());

        // GPU with offloaded head
        let config = ModelLoadConfig {
            offload_lm_head: true,
            ..Default::default()
        };
        let plan = LlamaModel::create_execution_plan(Device::Wgpu, &config);
        assert_eq!(plan, ExecutionPlan::gpu_offload_head());

        // GPU with offloaded ends
        let config = ModelLoadConfig {
            offload_lm_head: true,
            offload_embeddings: true,
            ..Default::default()
        };
        let plan = LlamaModel::create_execution_plan(Device::Wgpu, &config);
        assert_eq!(plan, ExecutionPlan::gpu_offload_ends());
    }
}
