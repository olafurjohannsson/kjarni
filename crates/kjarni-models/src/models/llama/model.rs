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
//! ```ignore
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
use kjarni_transformers::ChatTemplate;
use kjarni_transformers::chat::llama3::Llama3ChatTemplate;
use kjarni_transformers::common::{
    DecodingStrategy, GenerationConfig, HFGenerationDefaults, SamplingParams,
};
use kjarni_transformers::pipeline::DecoderModelFactory;
use kjarni_transformers::rope::loader::LoadedRoPE;
use kjarni_transformers::traits::{ModelLayout, ModelMetadata};
use ndarray::{Array2, Array3};
use tokenizers::Tokenizer;

// Crate-specific imports
use crate::models::llama::{
    config::LlamaConfig, cpu_decoder::LlamaCpuDecoder, gpu_decoder::LlamaGpuDecoder,
};

// Workspace imports
use kjarni_transformers::{
    // Utilities
    WgpuContext,
    cache::{Cache, CpuKVCache, GpuKVCache},
    decoder::prelude::*,
    execution::ExecutionPlan,
    gpu_ops::{GpuFrameContext, GpuTensor},
    // Decoder traits
    models::base::{AutoregressiveLoop, ModelLoadConfig},
    // Embeddings
    models::{LanguageModel, ModelType},
    // Execution planning
    pipeline::{DecoderPipeline},
    // GPU operations
    prelude::*,
    traits::{InferenceModel, ModelConfig},
    // Pipeline
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
/// ```ignore
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
    pipeline: DecoderPipeline,
    tokenizer: Tokenizer,
    config: Arc<LlamaConfig>,
    chat_template: Option<Box<dyn ChatTemplate>>,
    generation_defaults: Option<HFGenerationDefaults>,
}

// =============================================================================
// Model Loading
// =============================================================================

impl DecoderModelFactory for LlamaModel {
    type Config = LlamaConfig;

    fn load_config(weights: &ModelWeights) -> Result<Arc<Self::Config>> {
        LlamaConfig::from_loader(&*weights.loader(), Some(&weights.config_json()))
    }

    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        rope: &LoadedRoPE,
        load_config: ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(Option<Box<dyn CpuDecoder>>, Option<Box<dyn GpuDecoder>>)> {
        let mut cpu = None;
        let mut gpu = None;
        if device.is_cpu() || load_config.offload_embeddings {
            cpu = Some(Box::new(LlamaCpuDecoder::new(
                weights,
                meta.clone(),
                layout.clone(),
                rope.cpu.clone(),
                load_config.target_dtype,
            )?) as Box<dyn CpuDecoder>);
        } else if let Some(ctx) = context
            && device.is_gpu()
        {
            gpu = Some(Box::new(LlamaGpuDecoder::new(
                ctx,
                weights,
                meta.clone(),
                layout.clone(),
                rope.gpu.clone(),
                load_config,
            )?) as Box<dyn GpuDecoder>);
        } else {
            log::error!("Invalid device");
        }
        Ok((cpu, gpu))
    }

    fn new_from_pipeline(
        pipeline: DecoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<LlamaConfig>,
        model_type: Option<ModelType>,
        generation_defaults: Option<HFGenerationDefaults>,
        chat_template: Option<Box<dyn ChatTemplate>>,
    ) -> Self {
        Self {
            pipeline,
            tokenizer,
            config,
            chat_template,
            generation_defaults,
        }
    }
}

impl LlamaModel {
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        kjarni_transformers::pipeline::GenericLoader::load_from_registry::<Self>(
            model_type,
            cache_dir,
            device,
            context,
            load_config,
        )
        .await
    }
    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        decoder_config: Option<ModelLoadConfig>,
        model_type: Option<ModelType>,
    ) -> Result<Self> {
        kjarni_transformers::pipeline::GenericLoader::load_from_pretrained::<Self>(
            model_path,
            device,
            context,
            decoder_config,
            model_type,
        )
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
    /// ```ignore
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
        // self.device
        self.pipeline.plan().layers
    }

    fn context(&self) -> Option<Arc<WgpuContext>> {
        // self.context.clone()
        self.pipeline.context().cloned()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// =============================================================================
// LanguageModel Implementation
// =============================================================================

impl LanguageModel for LlamaModel {
    // TODO: extract to builder
    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        _num_beams: usize,
    ) -> Result<Box<dyn Cache>> {
        let meta = self.config.metadata();
        let head_dim = meta.head_dim;

        let effective_max_len = self.pipeline.max_sequence_length().unwrap_or(max_len);
        let effective_batch_size = self.pipeline.max_batch_size().unwrap_or(batch_size);

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
                    effective_batch_size,
                    meta.num_kv_heads,
                    head_dim,
                    effective_max_len,
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

    fn eos_token_ids(&self) -> Option<Vec<u32>> {
        Some(self.config.eos_token_id.clone())
    }

    fn eos_token_id(&self) -> Option<u32> {
        if self.config.eos_token_id.len() == 1 {
            Some(self.config.eos_token_id[0])
        } else if self.config.eos_token_id.len() > 1 {
            log::warn!("Model has multiple EOS token IDs; returning the first one.");
            Some(self.config.eos_token_id[0])
        } else {
            None
        }
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

    fn get_attention_mask(&self, seq: usize, past: usize) -> Result<Array2<f32>> {
        // Llama uses full attention (no sliding window in base model)
        // Ok(kjarni_transformers::utils::create_full_attention_mask(
        //     1, seq_len,
        // ))
        Ok(kjarni_transformers::utils::create_causal_mask(seq, seq + past))
        // Ok(kjarni_transformers::utils::create_causal_mask(seq_len, _past_len))
    }



    // CpuDecoderOps (for Autoregressive Decoders)
    fn embed(&self, tokens: &Array2<u32>, pos: usize) -> Result<Array3<f32>> {
        self.pipeline.embeddings().embed_cpu(&tokens, None, pos)
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
        seq: usize,
        max: usize,
    ) -> Result<GpuTensor> {
        let mask: Vec<f32> = (0..max)
            .map(|i| if i < seq { 1.0 } else { 0.0 })
            .collect();
            
        GpuTensor::create(ctx.context, &mask, vec![1, max], "AttentionMask")
    }

    fn project_to_logits(
        &self,
        ctx: &mut GpuFrameContext,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        let lm_head = self.pipeline.lm_head();

        match (lm_head.has_gpu(), lm_head.has_cpu()) {
            (true, _) => {
                let (enc, pool) = ctx.resources();
                lm_head.forward_gpu(enc, pool, hidden_states)
            }
            (false, true) => {
                // CPU fallback - sync read/write
                log::debug!("Using CPU fallback for LM head projection");
                pollster::block_on(async {
                    let hidden_cpu = hidden_states.to_ndarray_3d().await?;
                    let logits_cpu = lm_head.forward_cpu(&hidden_cpu)?;
                    GpuTensor::from_ndarray(ctx.context, &logits_cpu)
                })
            }
            (false, false) => Err(anyhow!("No LM head available on any device")),
        }
    }
}

// =============================================================================
// DecoderLanguageModel Implementation
// =============================================================================

#[async_trait]
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
        AutoregressiveLoop::Pipelined
    }
    fn chat_template(&self) -> Option<&dyn ChatTemplate> {
        self.chat_template.as_deref()
    }
    fn get_default_generation_config(&self) -> GenerationConfig {
        // Use generation_config.json if loaded
        if let Some(defaults) = &self.generation_defaults {
            return defaults
                .clone()
                .into_generation_config(self.config.max_position_embeddings);
        }
        // Llama fallback: sampling
        GenerationConfig {
            max_new_tokens: Some(256),
            max_length: self.config.max_position_embeddings,
            min_length: 0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
            add_bos_token: true,
            strategy: DecodingStrategy::Sample(SamplingParams {
                temperature: 0.6,
                top_k: None,
                top_p: Some(0.9),
                min_p: Some(0.05),
            }),
        }
    }
}
