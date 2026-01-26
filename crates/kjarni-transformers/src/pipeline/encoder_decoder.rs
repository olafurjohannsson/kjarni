use crate::WgpuContext;
use crate::cpu::encoder::{CpuEncoder, GpuEncoder};
use crate::LoadedEmbeddings;
use crate::encoder_decoder::traits::{CpuCrossDecoder, GpuCrossDecoder};
use crate::execution::ExecutionPlan;
use crate::gpu_ops::primitives::broadcast::GpuBroadcast;
use crate::lm_head::LoadedLMHead;
use crate::prelude::Device;
use anyhow::{Result, anyhow};
use ndarray::Array2;
use std::sync::Arc;

/// A container that holds all components needed for encoder decoder inference.
///
/// This provides:
/// - Unified access to embeddings, encoder, decoder layers, and LM head
/// - ExecutionPlan for configuring where each stage runs
/// - Validation that components match the plan
///
/// The actual inference is still handled by the existing backends
/// through the `*Ops` traits that the model implements.
pub struct EncoderDecoderPipeline {
    // Core components
    encoder_embeddings: Option<LoadedEmbeddings>,
    decoder_embeddings: LoadedEmbeddings,

    // Encoders
    cpu_encoder: Option<Box<dyn CpuEncoder>>,
    gpu_encoder: Option<Box<dyn GpuEncoder>>,

    cpu_decoder: Option<Box<dyn CpuCrossDecoder>>,
    gpu_decoder: Option<Box<dyn GpuCrossDecoder>>,

    lm_head: LoadedLMHead,
    final_logits_bias: Option<Array2<f32>>, // todo: inside LMHead?

    gpu_broadcast: Option<GpuBroadcast>,

    // Configuration
    plan: ExecutionPlan,

    // Runtime context
    context: Option<Arc<WgpuContext>>,

    // Metadata
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
}

/// Builder configuration for DecoderPipeline
pub struct EncoderDecoderPipelineConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl EncoderDecoderPipeline {
    /// Creates a new decoder pipeline.
    pub fn new(
        encoder_embeddings: Option<LoadedEmbeddings>,
        decoder_embeddings: LoadedEmbeddings,
        cpu_decoder: Option<Box<dyn CpuCrossDecoder>>,
        gpu_decoder: Option<Box<dyn GpuCrossDecoder>>,
        cpu_encoder: Option<Box<dyn CpuEncoder>>,
        gpu_encoder: Option<Box<dyn GpuEncoder>>,
        lm_head: LoadedLMHead,
        final_logits_bias: Option<Array2<f32>>,
        plan: ExecutionPlan,
        context: Option<Arc<WgpuContext>>,
        config: EncoderDecoderPipelineConfig,
    ) -> Result<Self> {
        let gpu_broadcast = if let Some(ctx) = &context {
            Some(GpuBroadcast::new(ctx)?)
        } else {
            None
        };
        let pipeline = Self {
            encoder_embeddings,
            decoder_embeddings,
            cpu_decoder,
            gpu_decoder,
            cpu_encoder,
            gpu_encoder,
            lm_head,
            final_logits_bias,
            gpu_broadcast,
            plan,
            context,
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
        };

        // Validate the plan against available components
        pipeline.validate_plan(&pipeline.plan)?;

        Ok(pipeline)
    }

    // ========================================================================
    // Plan Management
    // ========================================================================

    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Update the execution plan.
    ///
    /// Returns an error if the new plan requires components that aren't loaded.
    pub fn set_plan(&mut self, plan: ExecutionPlan) -> Result<()> {
        self.validate_plan(&plan)?;
        self.plan = plan;
        Ok(())
    }

    fn validate_plan(&self, plan: &ExecutionPlan) -> Result<()> {
        match plan.embeddings {
            Device::Cpu if !self.decoder_embeddings.is_cpu() => {
                return Err(anyhow!(
                    "Plan requires CPU embeddings but decoder embeddings not loaded on CPU"
                ));
            }
            Device::Wgpu if !self.decoder_embeddings.is_gpu() => {
                return Err(anyhow!(
                    "Plan requires GPU embeddings but decoder embeddings not loaded on GPU"
                ));
            }
            _ => {}
        }

        // Validate encoder embeddings (if present)
        if let Some(ref enc_emb) = self.encoder_embeddings {
            match plan.embeddings {
                Device::Cpu if !enc_emb.is_cpu() => {
                    return Err(anyhow!(
                        "Plan requires CPU embeddings but encoder embeddings not loaded on CPU"
                    ));
                }
                Device::Wgpu if !enc_emb.is_gpu() => {
                    return Err(anyhow!(
                        "Plan requires GPU embeddings but encoder embeddings not loaded on GPU"
                    ));
                }
                _ => {}
            }
        }

        // Validate layers
        match plan.layers {
            Device::Cpu if self.cpu_decoder.is_none() => {
                return Err(anyhow!("Plan requires CPU decoder but not loaded"));
            }
            Device::Wgpu if self.gpu_decoder.is_none() => {
                return Err(anyhow!("Plan requires GPU decoder but not loaded"));
            }
            _ => {}
        }

        // Validate LM head
        match plan.lm_head {
            Device::Cpu if !self.lm_head.has_cpu() => {
                return Err(anyhow!("Plan requires CPU LM head but not loaded"));
            }
            Device::Wgpu if !self.lm_head.has_gpu() => {
                return Err(anyhow!("Plan requires GPU LM head but not loaded"));
            }
            _ => {}
        }

        // Validate GPU context if needed
        if plan.needs_gpu() && self.context.is_none() {
            return Err(anyhow!("Plan requires GPU but no WgpuContext available"));
        }

        Ok(())
    }

    // ========================================================================
    // Component Accessors
    // ========================================================================

    pub fn encoder_embeddings(&self) -> Option<&LoadedEmbeddings> {
        self.encoder_embeddings.as_ref()
    }

    pub fn decoder_embeddings(&self) -> &LoadedEmbeddings {
        &self.decoder_embeddings
    }

    pub fn cpu_decoder(&self) -> Option<&dyn CpuCrossDecoder> {
        self.cpu_decoder.as_ref().map(|d| d.as_ref())
    }

    pub fn gpu_decoder(&self) -> Option<&dyn GpuCrossDecoder> {
        self.gpu_decoder.as_ref().map(|d| d.as_ref())
    }

    pub fn lm_head(&self) -> &LoadedLMHead {
        &self.lm_head
    }

    pub fn gpu_broadcast(&self) -> Option<&GpuBroadcast> {
        self.gpu_broadcast.as_ref()
    }

    pub fn final_logits_bias(&self) -> Option<&Array2<f32>> {
        self.final_logits_bias.as_ref()
    }

    pub fn cpu_encoder(&self) -> Option<&dyn CpuEncoder> {
        self.cpu_encoder.as_ref().map(|d| d.as_ref())
    }

    pub fn gpu_encoder(&self) -> Option<&dyn GpuEncoder> {
        self.gpu_encoder.as_ref().map(|d| d.as_ref())
    }

    pub fn context(&self) -> Option<&Arc<WgpuContext>> {
        self.context.as_ref()
    }

    // ========================================================================
    // Metadata
    // ========================================================================

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    // ========================================================================
    // Convenience Methods for Ops Implementation
    // ========================================================================

    /// Get the active decoder based on the current plan.
    pub fn active_cpu_decoder(&self) -> Result<&dyn CpuCrossDecoder> {
        self.cpu_decoder
            .as_ref()
            .map(|d| d.as_ref())
            .ok_or_else(|| anyhow!("CPU decoder not available"))
    }

    /// Get the active GPU decoder based on the current plan.
    pub fn active_gpu_decoder(&self) -> Result<&dyn GpuCrossDecoder> {
        self.gpu_decoder
            .as_ref()
            .map(|d| d.as_ref())
            .ok_or_else(|| anyhow!("GPU decoder not available"))
    }
}
