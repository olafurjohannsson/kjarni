use crate::{
    WgpuContext,
    cpu::encoder::{CpuEncoder, GpuEncoder},
    encoder_decoder::traits::{CpuCrossDecoder, GpuCrossDecoder},
    execution::ExecutionPlan,
    gpu::GpuTensor,
    loaders::{LMHeadConfig, LoadedLMHead},
    models::base::ModelLoadConfig,
    pipeline::{EncoderDecoderPipeline, EncoderDecoderPipelineConfig},
    traits::{Device, ModelConfig},
    weights::ModelWeights,
    {EmbeddingConfig, EmbeddingData, LoadedEmbeddings},
};

use anyhow::{Result, anyhow};
use std::sync::Arc;

pub struct EncoderDecoderPipelineBuilder<'a> {
    weights: &'a ModelWeights,
    config: Arc<dyn ModelConfig>,
    load_config: ModelLoadConfig,
    context: Option<Arc<WgpuContext>>,
    cpu_encoder_backend: Option<Box<dyn CpuEncoder>>,
    gpu_encoder_backend: Option<Box<dyn GpuEncoder>>,
    cpu_decoder_backend: Option<Box<dyn CpuCrossDecoder>>,
    gpu_decoder_backend: Option<Box<dyn GpuCrossDecoder>>,
    pub is_audio_encoder: bool,
}

impl<'a> EncoderDecoderPipelineBuilder<'a> {
    pub fn new(weights: &'a ModelWeights, config: Arc<dyn ModelConfig>) -> Self {
        Self {
            weights,
            config,
            load_config: ModelLoadConfig::default(),
            context: None,
            cpu_encoder_backend: None,
            gpu_encoder_backend: None,
            cpu_decoder_backend: None,
            gpu_decoder_backend: None,
            is_audio_encoder: false,
        }
    }
    pub fn with_context(mut self, context: Option<Arc<WgpuContext>>) -> Self {
        self.context = context;
        self
    }
    pub fn with_load_config(mut self, cfg: ModelLoadConfig) -> Self {
        self.load_config = cfg;
        self
    }

    pub fn with_context_opt(mut self, ctx: Option<Arc<WgpuContext>>) -> Self {
        self.context = ctx;
        self
    }

    pub fn with_decoder_backends(
        mut self,
        cpu_decoder: Option<Box<dyn CpuCrossDecoder>>,
        gpu_decoder: Option<Box<dyn GpuCrossDecoder>>,
    ) -> Self {
        self.cpu_decoder_backend = cpu_decoder;
        self.gpu_decoder_backend = gpu_decoder;
        self
    }

    pub fn with_encoder_backends(
        mut self,
        cpu_encoder: Option<Box<dyn CpuEncoder>>,
        gpu_encoder: Option<Box<dyn GpuEncoder>>,
    ) -> Self {
        self.cpu_encoder_backend = cpu_encoder;
        self.gpu_encoder_backend = gpu_encoder;
        self
    }

    fn load_shared_word_embeddings(
        &self,
        key: &str,
        load_cpu: bool,
        load_gpu: bool,
    ) -> Result<(Option<EmbeddingData>, Option<GpuTensor>)> {
        let cpu = if load_cpu {
            let arr = self.weights.get_array2(key)?;
            Some(EmbeddingData::F32(Arc::new(arr)))
        } else {
            None
        };

        let gpu = if load_gpu {
            let ctx = self
                .context
                .as_ref()
                .ok_or_else(|| anyhow!("GPU loading requires context"))?;
            Some(GpuTensor::from_model_weights(
                ctx,
                self.weights,
                key,
                self.load_config.target_dtype,
                "shared_word_embeddings",
            )?)
        } else {
            None
        };

        Ok((cpu, gpu))
    }

    pub fn build(self) -> Result<EncoderDecoderPipeline> {
        let meta = self.config.metadata();
        let layout = self.config.layout();
        let enc_layout = layout.encoder.as_ref();
        let dec_layout = layout
            .decoder
            .as_ref()
            .ok_or_else(|| anyhow!("Decoder layout required"))?;

        let ctx = self.context.as_ref();
        let target_dt = self.load_config.target_dtype;

        let primary_device = if ctx.is_some() {
            Device::Wgpu
        } else {
            Device::Cpu
        };
        let plan = ExecutionPlan::from_load_config(primary_device, &self.load_config);

        let mut emb_builder = EmbeddingConfig::builder(&layout.token_embedding, meta.hidden_size);
        if let Some(pos) = &dec_layout.position_embedding {
            emb_builder = emb_builder.position_embedding(pos);
        }
        if let Some(tok) = &dec_layout.token_type_embedding {
            emb_builder = emb_builder.type_embedding(tok);
        }
        let tied_weights = layout.lm_head == layout.token_embedding;
        let emb_load_cpu =
            plan.embeddings == Device::Cpu || (tied_weights && plan.lm_head == Device::Cpu);
        let emb_load_gpu =
            plan.embeddings == Device::Wgpu || (tied_weights && plan.lm_head == Device::Wgpu);

        let (shared_word_cpu, shared_word_gpu) =
            self.load_shared_word_embeddings(&layout.token_embedding, emb_load_cpu, emb_load_gpu)?;

        let encoder_embeddings = if let Some(enc) = enc_layout {
            if !self.is_audio_encoder {
                let mut enc_emb_builder =
                    EmbeddingConfig::builder(&layout.token_embedding, meta.hidden_size)
                        .position_offset(meta.extra_pos_embeddings)
                        .scale_embeddings(meta.scale_embeddings);

                if let Some(pos) = &enc.position_embedding {
                    enc_emb_builder = enc_emb_builder.position_embedding(pos);
                }

                Some(LoadedEmbeddings::with_shared_words(
                    self.context.as_ref(),
                    self.weights,
                    enc_emb_builder.build(),
                    shared_word_cpu.clone(),
                    shared_word_gpu.clone(),
                    emb_load_cpu,
                    emb_load_gpu,
                    self.load_config.target_dtype,
                )?)
            } else {
                None 
            }
        } else {
            None
        };

        let dec_config = EmbeddingConfig::builder(&layout.token_embedding, meta.hidden_size)
            .position_embedding(dec_layout.position_embedding.as_deref().unwrap_or_default())
            .position_offset(meta.extra_pos_embeddings)
            .scale_embeddings(meta.scale_embeddings)
            .build();

        let decoder_embeddings = LoadedEmbeddings::with_shared_words(
            self.context.as_ref(),
            self.weights,
            dec_config,
            shared_word_cpu.clone(),
            shared_word_gpu.clone(),
            emb_load_cpu,
            emb_load_gpu,
            self.load_config.target_dtype,
        )?;

        let final_logits_bias = self
            .weights
            .get_array2("final_logits_bias")
            .ok()
            .map(|t| t.into_dimensionality::<ndarray::Ix2>())
            .transpose()?;

        let lm_head = if tied_weights {
            log::info!("Using tied weights between embeddings and LM head");
            LoadedLMHead::from_shared_weights(
                ctx,
                shared_word_cpu.map(|e| e.to_linear_layer()),
                final_logits_bias,
                shared_word_gpu,
                LMHeadConfig::new(&layout.lm_head, meta.vocab_size, meta.hidden_size),
                None,
            )?
        } else {
            LoadedLMHead::new(
                ctx,
                self.weights,
                final_logits_bias,
                LMHeadConfig::new(&layout.lm_head, meta.vocab_size, meta.hidden_size),
                plan.lm_head == Device::Cpu,
                plan.lm_head == Device::Wgpu,
                target_dt,
            )?
        };

        EncoderDecoderPipeline::new(
            encoder_embeddings,
            decoder_embeddings,
            self.cpu_decoder_backend,
            self.gpu_decoder_backend,
            self.cpu_encoder_backend,
            self.gpu_encoder_backend,
            lm_head,
            None,
            plan,
            self.context.clone(),
            EncoderDecoderPipelineConfig {
                num_layers: meta.num_layers,
                hidden_size: meta.hidden_size,
                vocab_size: meta.vocab_size,
            },
        )
    }
}
