// kjarni-transformers/src/execution/plan.rs

use crate::prelude::Device;

/// Describes where each stage of inference runs.
///
/// This allows flexible hybrid execution strategies like:
/// - Full GPU (default for VRAM-rich systems)
/// - Full CPU (for systems without GPU)
/// - GPU with CPU offloading (embeddings/head on CPU to save VRAM)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPlan {
    /// Where token embeddings are computed
    pub embeddings: Device,
    /// Where transformer layers run (all layers on same device for now)
    pub layers: Device,
    /// Where the LM head projection runs
    pub lm_head: Device,
}

impl ExecutionPlan {
    /// All stages on GPU. Best performance when VRAM is sufficient.
    pub fn full_gpu() -> Self {
        Self {
            embeddings: Device::Wgpu,
            layers: Device::Wgpu,
            lm_head: Device::Wgpu,
        }
    }

    /// All stages on CPU. For systems without GPU or for debugging.
    pub fn full_cpu() -> Self {
        Self {
            embeddings: Device::Cpu,
            layers: Device::Cpu,
            lm_head: Device::Cpu,
        }
    }

    /// GPU layers with CPU embeddings and LM head.
    /// Saves ~500MB-1GB VRAM for large vocab models.
    pub fn gpu_offload_ends() -> Self {
        Self {
            embeddings: Device::Cpu,
            layers: Device::Wgpu,
            lm_head: Device::Cpu,
        }
    }

    /// GPU layers with only LM head on CPU.
    /// Good balance for models with tied embeddings.
    pub fn gpu_offload_head() -> Self {
        Self {
            embeddings: Device::Wgpu,
            layers: Device::Wgpu,
            lm_head: Device::Cpu,
        }
    }

    /// Custom plan
    pub fn custom(embeddings: Device, layers: Device, lm_head: Device) -> Self {
        Self {
            embeddings,
            layers,
            lm_head,
        }
    }

    /// Check if this plan requires GPU components
    pub fn needs_gpu(&self) -> bool {
        self.embeddings == Device::Wgpu 
            || self.layers == Device::Wgpu 
            || self.lm_head == Device::Wgpu
    }

    /// Check if this plan requires CPU components
    pub fn needs_cpu(&self) -> bool {
        self.embeddings == Device::Cpu 
            || self.layers == Device::Cpu 
            || self.lm_head == Device::Cpu
    }
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self::full_gpu()
    }
}