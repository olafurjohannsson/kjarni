pub mod layer_norm;
pub mod rms_norm;

use crate::gpu::{GpuTensor};

pub use layer_norm::{GpuLayerNorm, GpuLayerNormWeights};
pub use rms_norm::{GpuRMSNorm, GpuRMSNormWeights};



pub enum GpuNormalizationWeights {
    LayerNorm(GpuLayerNormWeights),
    RMSNorm(GpuRMSNormWeights),
}

pub enum GpuNormalization {
    LayerNorm(GpuLayerNorm),
    RMSNorm(GpuRMSNorm),
}

impl GpuNormalization {
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        weights: &GpuNormalizationWeights,
        input: &GpuTensor,
        output: &GpuTensor,
    ) {
        match (self, weights) {
            (Self::LayerNorm(ln), GpuNormalizationWeights::LayerNorm(w)) => {
                ln.encode(encoder, w, input, output)
            }
            (Self::RMSNorm(rms), GpuNormalizationWeights::RMSNorm(w)) => {
                rms.encode(encoder, w, input, output)
            }
            _ => panic!("Normalization type mismatch"),
        }
    }
}
