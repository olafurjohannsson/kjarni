use crate::models::base::ModelLoadConfig;
use crate::prelude::Device;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub embeddings: Device,
    pub layers: Device,
    pub lm_head: Device,
}

impl ExecutionPlan {
    pub fn from_load_config(device: Device, config: &ModelLoadConfig) -> Self {
        match device {
            Device::Cpu => Self::full_cpu(),
            Device::Wgpu => {
                let emb = if config.offload_embeddings {
                    Device::Cpu
                } else {
                    Device::Wgpu
                };
                let head = if config.offload_lm_head {
                    Device::Cpu
                } else {
                    Device::Wgpu
                };
                let layers = Device::Wgpu;

                Self {
                    embeddings: emb,
                    layers,
                    lm_head: head,
                }
            }
        }
    }
    pub fn full_gpu() -> Self {
        Self {
            embeddings: Device::Wgpu,
            layers: Device::Wgpu,
            lm_head: Device::Wgpu,
        }
    }

    pub fn full_cpu() -> Self {
        Self {
            embeddings: Device::Cpu,
            layers: Device::Cpu,
            lm_head: Device::Cpu,
        }
    }
    pub fn gpu_offload_ends() -> Self {
        Self {
            embeddings: Device::Cpu,
            layers: Device::Wgpu,
            lm_head: Device::Cpu,
        }
    }

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

    pub fn needs_gpu(&self) -> bool {
        self.embeddings == Device::Wgpu
            || self.layers == Device::Wgpu
            || self.lm_head == Device::Wgpu
    }

    pub fn needs_cpu(&self) -> bool {
        self.embeddings == Device::Cpu || self.layers == Device::Cpu || self.lm_head == Device::Cpu
    }
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self::full_gpu()
    }
}
