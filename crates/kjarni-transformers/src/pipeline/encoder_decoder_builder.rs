use crate::decoder::prelude::{CpuDecoder, GpuDecoder};
use crate::embeddings::{EmbeddingConfig, LoadedEmbeddings};
use crate::encoder::{CpuEncoder, GpuEncoder};
use crate::encoder_decoder::traits::{CpuCrossDecoder, GpuCrossDecoder};
use crate::execution::ExecutionPlan;
use crate::lm_head::{LMHeadConfig, LoadedLMHead};
use crate::models::base::ModelLoadConfig;
use crate::pipeline::{EncoderDecoderPipeline, EncoderDecoderPipelineConfig};
use crate::traits::{Device, ModelConfig};
use crate::weights::ModelWeights;
use crate::WgpuContext;
use anyhow::{anyhow, Context, Result};
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

    pub fn build(self) -> Result<EncoderDecoderPipeline> {
        let meta = self.config.metadata();
        let layout = self.config.layout();
        let ctx = self.context.as_ref();
        let target_dt = self.load_config.target_dtype;

        // 1. Determine Device Strategy
        let primary_device = if ctx.is_some() {
            Device::Wgpu
        } else {
            Device::Cpu
        };
        let plan = ExecutionPlan::from_load_config(primary_device, &self.load_config);

        // 3. Extract Decoder Layout
        let dec_layout = layout
            .decoder
            .as_ref()
            .ok_or_else(|| anyhow!("Pipeline requires a DecoderLayout in ModelLayout"))?;

        // 4. Load Embeddings
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
        let embeddings = LoadedEmbeddings::new(
            ctx,
            self.weights,
            emb_builder
                .position_offset(meta.extra_pos_embeddings)
                .scale_embeddings(meta.scale_embeddings)
                .build(),
            emb_load_cpu,
            emb_load_gpu,
            target_dt,
        )?;

        // 5. Load LM Head (The "Tied" Check)
        let lm_head = if tied_weights {
            log::info!("Using tied weights between embeddings and LM head");
            LoadedLMHead::from_shared_weights(
                ctx,
                embeddings.word_embeddings_cpu(),
                embeddings.word_embeddings_gpu(),
                LMHeadConfig::new(&layout.lm_head, meta.vocab_size, meta.hidden_size),
                None,
            )?
        } else {
            LoadedLMHead::new(
                ctx,
                self.weights,
                LMHeadConfig::new(&layout.lm_head, meta.vocab_size, meta.hidden_size),
                plan.lm_head == Device::Cpu,
                plan.lm_head == Device::Wgpu,
                target_dt,
            )?
        };

        // 6. Build Model Backends (Handover RoPE and DType)
        // We resolve backends here using the specific model's factory logic
        // This is where you call build_backends from the GenericLoader

        // Return dummy/empty backends for now; the GenericLoader will populate these
        EncoderDecoderPipeline::new(
            embeddings,
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
