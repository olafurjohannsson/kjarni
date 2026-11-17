#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use kjarni_transformers::Device;
use ndarray::{Array2, Array3};
use tokenizers::Tokenizer;

use kjarni_transformers::{
    ChatTemplate, WgpuContext,
    cache::Cache,
    common::{DecodingStrategy, GenerationConfig, HFGenerationDefaults, SamplingParams},
    decoder::prelude::*,
    loaders::LoadedRoPE,
    models::base::{AutoregressiveLoop, ModelLoadConfig},
    models::{LanguageModel, ModelType},
    pipeline::{DecoderModelFactory, DecoderPipeline},
    traits::{InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

use super::cache::HybridCache;
use super::config::Qwen3_5Config;
use super::cpu_decoder::Qwen3_5CpuDecoder;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use super::gpu_decoder::Qwen3_5GpuDecoder;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::gpu::{GpuFrameContext, GpuTensor};

/// Qwen 3.5: Qwen's hybrid decoder, three Gated DeltaNet layers to every
/// attention layer. Text only; a multimodal checkpoint loads its decoder.
pub struct Qwen3_5Model {
    pipeline: DecoderPipeline,
    tokenizer: Tokenizer,
    config: Arc<Qwen3_5Config>,
    chat_template: Option<Box<dyn ChatTemplate>>,
    generation_defaults: Option<HFGenerationDefaults>,
}

impl DecoderModelFactory for Qwen3_5Model {
    type Config = Qwen3_5Config;

    fn load_config(weights: &ModelWeights) -> Result<Arc<Self::Config>> {
        Qwen3_5Config::from_weights(weights)
    }

    fn rope_metadata(config: &Self::Config) -> ModelMetadata {
        ModelMetadata {
            head_dim: config.rotary_dim,
            ..config.metadata()
        }
    }

    fn build_backends(
        weights: &ModelWeights,
        _meta: &ModelMetadata,
        _layout: &ModelLayout,
        rope: &LoadedRoPE,
        load_config: ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(Option<Box<dyn CpuDecoder>>, Option<Box<dyn GpuDecoder>>)> {
        let config = Qwen3_5Config::from_weights(weights)?;
        if device.is_gpu() {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
            {
                let context =
                    context.ok_or_else(|| anyhow!("Qwen 3.5 on the GPU needs a GPU context"))?;
                let rope = rope
                    .gpu
                    .clone()
                    .ok_or_else(|| anyhow!("Qwen 3.5 on the GPU needs the rotary tables there"))?;
                let decoder = Qwen3_5GpuDecoder::new(
                    context,
                    weights,
                    config,
                    rope,
                    load_config.target_dtype,
                )?;
                return Ok((None, Some(Box::new(decoder))));
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
            {
                let _ = context;
                return Err(anyhow!("GPU decoding is not available in WebAssembly"));
            }
        }
        let decoder =
            Qwen3_5CpuDecoder::new(weights, config, rope.cpu.clone(), load_config.target_dtype)?;
        Ok((Some(Box::new(decoder)), None))
    }

    fn new_from_pipeline(
        pipeline: DecoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<Qwen3_5Config>,
        _model_type: Option<ModelType>,
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

impl Qwen3_5Model {
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        kjarni_transformers::pipeline::DecoderLoader::load_from_registry::<Self>(
            model_type,
            cache_dir,
            device,
            context,
            load_config,
        )
        .await
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
        model_type: Option<ModelType>,
    ) -> Result<Self> {
        kjarni_transformers::pipeline::DecoderLoader::load_from_pretrained::<Self>(
            model_path,
            device,
            context,
            load_config,
            model_type,
        )
    }

    pub fn config(&self) -> &Arc<Qwen3_5Config> {
        &self.config
    }
    pub fn pipeline(&self) -> &DecoderPipeline {
        &self.pipeline
    }
}

impl InferenceModel for Qwen3_5Model {
    fn device(&self) -> Device {
        self.pipeline.plan().layers
    }
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    fn context(&self) -> Option<Arc<WgpuContext>> {
        self.pipeline.context().cloned()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl LanguageModel for Qwen3_5Model {
    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        _beams: usize,
    ) -> Result<Box<dyn Cache>> {
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
        if self.pipeline.gpu_decoder().is_some() {
            let context = self
                .context()
                .ok_or_else(|| anyhow!("GPU context required"))?;
            return Ok(Box::new(Qwen3_5GpuDecoder::new_cache(
                &context,
                &self.config,
                batch_size,
                max_len,
            )?));
        }
        Ok(Box::new(HybridCache::new(
            &Qwen3_5CpuDecoder::cache_dims(&self.config),
            batch_size,
            max_len,
        )))
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
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
    fn bos_token_id(&self) -> Option<u32> {
        self.config.bos_token_id
    }
    fn eos_token_ids(&self) -> Option<Vec<u32>> {
        Some(self.config.eos_token_id.clone())
    }
    fn eos_token_id(&self) -> Option<u32> {
        self.config.eos_token_id.first().copied()
    }
    fn pad_token_id(&self) -> Option<u32> {
        self.config.pad_token_id
    }
    fn context_size(&self) -> usize {
        self.config.max_position_embeddings
    }
    fn forced_bos_token_id(&self) -> Option<u32> {
        None
    }
    fn forced_eos_token_id(&self) -> Option<u32> {
        None
    }
    fn stop_token_ids(&self) -> std::collections::HashSet<u32> {
        let mut set: std::collections::HashSet<u32> =
            self.config.eos_token_id.iter().copied().collect();
        if let Some(im_end) = self.tokenizer.token_to_id("<|im_end|>") {
            set.insert(im_end);
        }
        set
    }
}

#[async_trait]
impl DecoderLanguageModel for Qwen3_5Model {
    fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps> {
        self.pipeline
            .cpu_decoder()
            .map(|_| self as &dyn CpuDecoderOps)
    }
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps> {
        self.pipeline
            .gpu_decoder()
            .map(|_| self as &dyn GpuDecoderOps)
    }
    fn autoregressive_loop(&self) -> AutoregressiveLoop {
        AutoregressiveLoop::Pipelined
    }
    fn chat_template(&self) -> Option<&dyn ChatTemplate> {
        self.chat_template.as_deref()
    }
    /// The model card's settings for its default, non-thinking mode.
    fn get_default_generation_config(&self) -> GenerationConfig {
        if let Some(d) = &self.generation_defaults {
            return d
                .clone()
                .into_generation_config(self.config.max_position_embeddings);
        }
        GenerationConfig {
            max_new_tokens: Some(512),
            max_length: self.config.max_position_embeddings,
            min_length: 0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
            add_bos_token: false,
            strategy: DecodingStrategy::Sample(SamplingParams {
                temperature: 0.7,
                top_k: Some(20),
                top_p: Some(0.8),
                min_p: None,
            }),
            speculation: None,
        }
    }
}

impl CpuDecoderOps for Qwen3_5Model {
    fn decoder(&self) -> &dyn CpuDecoder {
        self.pipeline
            .cpu_decoder()
            .expect("decoder_cpu_ops only hands this out with a CPU decoder")
    }
    fn project_to_logits(&self, h: &Array3<f32>) -> Result<Array3<f32>> {
        self.pipeline.lm_head().forward_cpu(h)
    }
    fn get_attention_mask(&self, seq: usize, past: usize) -> Result<Array2<f32>> {
        Ok(kjarni_transformers::utils::create_causal_mask(
            seq,
            seq + past,
        ))
    }
    fn embed(&self, tokens: &Array2<u32>, pos: usize) -> Result<Array3<f32>> {
        self.pipeline.embeddings().embed_cpu(tokens, None, pos)
    }
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
impl GpuDecoderOps for Qwen3_5Model {
    fn decoder(&self) -> &dyn GpuDecoder {
        self.pipeline
            .gpu_decoder()
            .expect("decoder_gpu_ops only hands this out with a GPU decoder")
    }
    /// A padding mask: the attention kernels add causality themselves.
    fn get_attention_mask(
        &self,
        ctx: &mut GpuFrameContext,
        seq: usize,
        max: usize,
    ) -> Result<GpuTensor> {
        let mask: Vec<f32> = (0..max).map(|i| if i < seq { 1.0 } else { 0.0 }).collect();
        GpuTensor::create(ctx.context, &mask, vec![1, max], "AttentionMask")
    }
    fn project_to_logits(&self, ctx: &mut GpuFrameContext, h: &GpuTensor) -> Result<GpuTensor> {
        let lm = self.pipeline.lm_head();
        anyhow::ensure!(
            lm.has_gpu(),
            "Qwen 3.5 on the GPU needs its LM head there too"
        );
        let (encoder, pool) = ctx.resources();
        lm.forward_gpu(encoder, pool, h)
    }
}
