// kjarni-transformers/src/pipeline/decoder.rs

use crate::decoder::prelude::{CpuDecoder, GpuDecoder};
use crate::embeddings::LoadedEmbeddings;
use crate::execution::ExecutionPlan;
use crate::lm_head::LoadedLMHead;
use crate::prelude::Device;
use crate::WgpuContext;
use anyhow::{anyhow, Result};
use std::sync::Arc;

/// A container that holds all components needed for decoder inference.
///
/// This provides:
/// - Unified access to embeddings, decoder layers, and LM head
/// - ExecutionPlan for configuring where each stage runs
/// - Validation that components match the plan
///
/// The actual inference is still handled by the existing backends
/// through the `*Ops` traits that the model implements.
pub struct DecoderPipeline {
    // Core components
    embeddings: LoadedEmbeddings,
    cpu_decoder: Option<Box<dyn CpuDecoder>>,
    gpu_decoder: Option<Box<dyn GpuDecoder>>,
    lm_head: LoadedLMHead,

    // Configuration
    plan: ExecutionPlan,

    // Runtime context
    context: Option<Arc<WgpuContext>>,

    // Metadata
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    max_sequence_length: Option<usize>,
}

/// Builder configuration for DecoderPipeline
pub struct DecoderPipelineConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_sequence_length: Option<usize>,
    pub max_batch_size: Option<usize>,
}

impl DecoderPipeline {
    /// Creates a new decoder pipeline.
    pub fn new(
        embeddings: LoadedEmbeddings,
        cpu_decoder: Option<Box<dyn CpuDecoder>>,
        gpu_decoder: Option<Box<dyn GpuDecoder>>,
        lm_head: LoadedLMHead,
        plan: ExecutionPlan,
        context: Option<Arc<WgpuContext>>,
        config: DecoderPipelineConfig,
    ) -> Result<Self> {
        let pipeline = Self {
            embeddings,
            cpu_decoder,
            gpu_decoder,
            lm_head,
            plan,
            context,
            num_layers: config.num_layers,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            max_sequence_length: config.max_sequence_length,
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
        // Validate embeddings
        match plan.embeddings {
            Device::Cpu if !self.embeddings.is_cpu() => {
                return Err(anyhow!("Plan requires CPU embeddings but not loaded"));
            }
            Device::Wgpu if !self.embeddings.is_gpu() => {
                return Err(anyhow!("Plan requires GPU embeddings but not loaded"));
            }
            _ => {}
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

    pub fn embeddings(&self) -> &LoadedEmbeddings {
        &self.embeddings
    }

    pub fn cpu_decoder(&self) -> Option<&dyn CpuDecoder> {
        self.cpu_decoder.as_ref().map(|d| d.as_ref())
    }

    pub fn gpu_decoder(&self) -> Option<&dyn GpuDecoder> {
        self.gpu_decoder.as_ref().map(|d| d.as_ref())
    }

    pub fn lm_head(&self) -> &LoadedLMHead {
        &self.lm_head
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
    pub fn max_sequence_length(&self) -> Option<usize> {
        self.max_sequence_length
    }

    pub fn max_batch_size(&self) -> Option<usize> {
        None
    }

    // ========================================================================
    // Convenience Methods for Ops Implementation
    // ========================================================================

    /// Get the active decoder based on the current plan.
    pub fn active_cpu_decoder(&self) -> Result<&dyn CpuDecoder> {
        self.cpu_decoder
            .as_ref()
            .map(|d| d.as_ref())
            .ok_or_else(|| anyhow!("CPU decoder not available"))
    }

    /// Get the active GPU decoder based on the current plan.
    pub fn active_gpu_decoder(&self) -> Result<&dyn GpuDecoder> {
        self.gpu_decoder
            .as_ref()
            .map(|d| d.as_ref())
            .ok_or_else(|| anyhow!("GPU decoder not available"))
    }
}