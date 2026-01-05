use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use kjarni_transformers::Device;
use ndarray::{Array2, Array3};
use tokenizers::Tokenizer;

// Reuse Llama Decoders
use crate::models::llama::{cpu_decoder::LlamaCpuDecoder, gpu_decoder::LlamaGpuDecoder};
use crate::models::mistral::config::MistralConfig;


use kjarni_transformers::chat::mistral::MistralChatTemplate; // todo mode tom models?

use kjarni_transformers::{
    ChatTemplate, WgpuContext,
    cache::{Cache, CpuKVCache, GpuKVCache},
    common::{DecodingStrategy, GenerationConfig, HFGenerationDefaults, SamplingParams},
    decoder::prelude::*,
    embeddings::{EmbeddingConfig, LoadedEmbeddings},
    execution::ExecutionPlan,
    gpu_ops::{GpuFrameContext, GpuTensor},
    models::base::{AutoregressiveLoop, ModelLoadConfig},
    models::{LanguageModel, ModelArchitecture, ModelType},
    pipeline::{DecoderModelFactory, DecoderPipeline},
    rope::loader::LoadedRoPE,
    tensor::{DType, TensorView},
    traits::{InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

pub struct MistralModel {
    pipeline: DecoderPipeline,
    tokenizer: Tokenizer,
    config: Arc<MistralConfig>,
    chat_template: Option<Box<dyn ChatTemplate>>,
    generation_defaults: Option<HFGenerationDefaults>,
}

impl DecoderModelFactory for MistralModel {
    type Config = MistralConfig;

    fn load_config(weights: &ModelWeights) -> Result<Arc<Self::Config>> {
        MistralConfig::from_loader(&*weights.loader, Some(&weights.config_json))
    }

    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        rope: &LoadedRoPE,
        load_config: ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
    ) -> Result<(Option<Box<dyn CpuDecoder>>, Option<Box<dyn GpuDecoder>>)> {
        let mut cpu = None;
        let mut gpu = None;

        if load_config.gpu_layers.is_none() || load_config.offload_embeddings {
            cpu = Some(Box::new(LlamaCpuDecoder::new(
                weights,
                meta.clone(),
                layout.clone(),
                rope.cpu.clone(),
                load_config.target_dtype,
            )?) as Box<dyn CpuDecoder>);
        }

        if let Some(ctx) = context {
            gpu = Some(Box::new(LlamaGpuDecoder::new(
                ctx,
                weights,
                meta.clone(),
                layout.clone(),
                rope.gpu.clone(),
                load_config,
            )?) as Box<dyn GpuDecoder>);
        }

        Ok((cpu, gpu))
    }

    fn new_from_pipeline(
        pipeline: DecoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<MistralConfig>,
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

// ... Boilerplate accessors (Standard Pattern) ...

impl MistralModel {
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        kjarni_transformers::pipeline::GenericLoader::load_from_registry::<Self>(
            model_type, cache_dir, device, context, load_config
        ).await
    }
    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        decoder_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        kjarni_transformers::pipeline::GenericLoader::load_from_pretrained::<Self>(
            model_path, device, context, decoder_config, Some(ModelType::Mistral7B_v0_3_Instruct)
        )
    }
    pub fn config(&self) -> &Arc<MistralConfig> { &self.config }
    pub fn pipeline(&self) -> &DecoderPipeline { &self.pipeline }
}

impl InferenceModel for MistralModel {
    fn device(&self) -> Device { self.pipeline.plan().layers }
    fn context(&self) -> Option<Arc<WgpuContext>> { self.pipeline.context().cloned() }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

impl LanguageModel for MistralModel {
    fn new_cache(&self, batch: usize, max_len: usize, _b: usize) -> Result<Box<dyn Cache>> {
        let meta = self.config.metadata();
        match self.pipeline.plan().layers {
            Device::Cpu => {
                let kv = meta.head_dim * meta.num_kv_heads;
                Ok(Box::new(CpuKVCache::new(meta.num_layers, batch, max_len, kv)))
            }
            Device::Wgpu => {
                let ctx = self.context().unwrap();
                Ok(Box::new(GpuKVCache::new(&ctx, meta.num_layers, batch, meta.num_kv_heads, meta.head_dim, max_len)?))
            }
        }
    }
    fn tokenizer(&self) -> &Tokenizer { &self.tokenizer }
    fn vocab_size(&self) -> usize { self.config.vocab_size }
    fn hidden_size(&self) -> usize { self.config.hidden_size }
    fn num_layers(&self) -> usize { self.config.num_hidden_layers }
    fn num_heads(&self) -> usize { self.config.num_attention_heads }
    fn bos_token_id(&self) -> Option<u32> { Some(self.config.bos_token_id) }
    fn eos_token_ids(&self) -> Option<Vec<u32>> { Some(self.config.eos_token_id.clone()) }
    fn eos_token_id(&self) -> Option<u32> { self.config.eos_token_id.first().copied() }
    fn pad_token_id(&self) -> Option<u32> { self.config.pad_token_id }
    fn context_size(&self) -> usize { self.config.max_position_embeddings }
    fn forced_bos_token_id(&self) -> Option<u32> { None }
    fn forced_eos_token_id(&self) -> Option<u32> { None }
}

#[async_trait]
impl DecoderLanguageModel for MistralModel {
    fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps> {
        if self.pipeline.cpu_decoder().is_some() { Some(self) } else { None }
    }
    fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps> {
        if self.pipeline.gpu_decoder().is_some() { Some(self) } else { None }
    }
    fn autoregressive_loop(&self) -> AutoregressiveLoop { AutoregressiveLoop::Pipelined }

    fn chat_template(&self) -> Option<&dyn ChatTemplate> {
        self.chat_template.as_deref()
    }
    fn get_default_generation_config(&self) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: Some(512),
            max_length: self.config.max_position_embeddings,
            min_length: 0,
            repetition_penalty: 1.15, // Mistral likes higher penalty
            no_repeat_ngram_size: 0,
            add_bos_token: true,
            strategy: DecodingStrategy::Sample(SamplingParams {
                temperature: 0.7,
                top_k: Some(40),
                top_p: Some(0.9),
                min_p: Some(0.05),
            }),
        }
    }
}

// Forward Ops
impl CpuDecoderOps for MistralModel {
    fn decoder(&self) -> &dyn CpuDecoder { self.pipeline.cpu_decoder().unwrap() }
    fn project_to_logits(&self, h: &Array3<f32>) -> Result<Array3<f32>> { self.pipeline.lm_head().forward_cpu(h) }
    fn get_attention_mask(&self, seq: usize, _past: usize) -> Result<Array2<f32>> {
        Ok(kjarni_transformers::utils::create_full_attention_mask(1, seq))
    }
}

impl GpuDecoderOps for MistralModel {
    fn decoder(&self) -> &dyn GpuDecoder { self.pipeline.gpu_decoder().unwrap() }
    fn get_attention_mask(&self, ctx: &mut GpuFrameContext, seq: usize, max: usize) -> Result<GpuTensor> {
        let mask: Vec<f32> = (0..max).map(|i| if i < seq { 1.0 } else { 0.0 }).collect();
        GpuTensor::from_raw(ctx.context, &TensorView {
            bytes: std::borrow::Cow::Owned(bytemuck::cast_slice(&mask).to_vec()),
            shape: vec![1, max],
            dtype: DType::F32,
            name: "AttentionMask".to_string(),
        }, "AttentionMask")
    }
    fn project_to_logits(&self, ctx: &mut GpuFrameContext, h: &GpuTensor) -> Result<GpuTensor> {
        let lm = self.pipeline.lm_head();
        if lm.has_gpu() {
            let (enc, pool) = ctx.resources();
            lm.forward_gpu(enc, pool, h)
        } else {
            pollster::block_on(async {
                let h_cpu = h.to_ndarray_3d().await?;
                let logits = lm.forward_cpu(&h_cpu)?;
                GpuTensor::from_ndarray(ctx.context, &logits)
            })
        }
    }
}