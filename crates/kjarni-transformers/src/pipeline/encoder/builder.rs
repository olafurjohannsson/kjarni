//! Builder for EncoderPipeline.

use anyhow::{Result, anyhow};
use std::sync::Arc;

use crate::WgpuContext;
use crate::execution::ExecutionPlan;
use crate::models::base::ModelLoadConfig;
use crate::traits::{Device, ModelConfig};
use crate::weights::ModelWeights;
use crate::{EmbeddingConfig, LoadedEmbeddings};
use crate::{
    cpu::encoder::{
        classifier::CpuSequenceClassificationHead,
        config::PoolingStrategy,
        traits::{CpuEncoder, GpuEncoder},
    },
    pipeline::encoder::pipeline::{EncoderPipeline, EncoderPipelineConfig},
};

/// Builder for constructing an EncoderPipeline.
pub struct EncoderPipelineBuilder<'a> {
    weights: &'a ModelWeights,
    config: Arc<dyn ModelConfig>,
    load_config: ModelLoadConfig,
    context: Option<Arc<WgpuContext>>,

    // Backends (set by the model factory)
    cpu_encoder: Option<Box<dyn CpuEncoder>>,
    gpu_encoder: Option<Box<dyn GpuEncoder>>,

    // Optional head (set by the model factory)
    cpu_head: Option<CpuSequenceClassificationHead>,

    // Pooling strategy (varies by model type)
    pooling_strategy: PoolingStrategy,
}

impl<'a> EncoderPipelineBuilder<'a> {
    pub fn new(weights: &'a ModelWeights, config: Arc<dyn ModelConfig>) -> Self {
        Self {
            weights,
            config,
            load_config: ModelLoadConfig::default(),
            context: None,
            cpu_encoder: None,
            gpu_encoder: None,
            cpu_head: None,
            pooling_strategy: PoolingStrategy::Mean, // Default for embeddings
        }
    }

    pub fn with_load_config(mut self, cfg: ModelLoadConfig) -> Self {
        self.load_config = cfg;
        self
    }

    pub fn with_context(mut self, ctx: Option<Arc<WgpuContext>>) -> Self {
        self.context = ctx;
        self
    }

    pub fn with_backends(
        mut self,
        cpu: Option<Box<dyn CpuEncoder>>,
        gpu: Option<Box<dyn GpuEncoder>>,
    ) -> Self {
        self.cpu_encoder = cpu;
        self.gpu_encoder = gpu;
        self
    }

    pub fn with_head(mut self, head: Option<CpuSequenceClassificationHead>) -> Self {
        self.cpu_head = head;
        self
    }

    pub fn with_pooling_strategy(mut self, strategy: PoolingStrategy) -> Self {
        self.pooling_strategy = strategy;
        self
    }

    pub fn build(self) -> Result<EncoderPipeline> {
        let meta = self.config.metadata();
        let layout = self.config.layout();
        let ctx = self.context.as_ref();
        let target_dtype = self.load_config.target_dtype;

        // 1. Determine execution plan
        let primary_device = if ctx.is_some() {
            Device::Wgpu
        } else {
            Device::Cpu
        };
        let plan = ExecutionPlan::from_load_config(primary_device, &self.load_config);

        // 2. Get encoder layout
        let enc_layout = layout
            .encoder
            .as_ref()
            .ok_or_else(|| anyhow!("Pipeline requires an EncoderLayout in ModelLayout"))?;

        // 3. Load embeddings
        let mut emb_builder = EmbeddingConfig::builder(&layout.token_embedding, meta.hidden_size)
            .position_offset(meta.extra_pos_embeddings);

        if let Some(pos) = &enc_layout.position_embedding {
            emb_builder = emb_builder.position_embedding(pos);
        }
        if let Some(tok) = &enc_layout.token_type_embedding {
            emb_builder = emb_builder.type_embedding(tok);
        }

        let emb_load_cpu = plan.embeddings == Device::Cpu;
        let emb_load_gpu = plan.embeddings == Device::Wgpu;

        let embeddings = LoadedEmbeddings::new(
            ctx,
            self.weights,
            emb_builder.build(),
            emb_load_cpu,
            emb_load_gpu,
            target_dtype,
        )?;

        // 4. Build pipeline config
        let pipeline_config = EncoderPipelineConfig {
            num_layers: meta.num_layers,
            hidden_size: meta.hidden_size,
            vocab_size: meta.vocab_size,
            max_seq_length: meta.max_seq_len,
            pooling_strategy: self.pooling_strategy,
            has_head: self.cpu_head.is_some(),
            num_labels: self.cpu_head.as_ref().map(|h| h.num_classes()),
        };

        // 5. Build pipeline
        EncoderPipeline::new(
            embeddings,
            self.cpu_encoder,
            self.gpu_encoder,
            self.cpu_head,
            plan,
            self.context,
            pipeline_config,
        )
    }
}
