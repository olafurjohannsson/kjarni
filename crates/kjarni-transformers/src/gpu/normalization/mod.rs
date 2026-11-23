pub mod layer_norm;
pub mod rms_norm;

use crate::gpu::GpuTensor;

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
        encoder: &mut crate::gpu::GpuEncoder,
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

    /// `sum = input + addend` and `output = norm(sum)`; see
    /// [`GpuRMSNorm::encode_add`]. RMSNorm only.
    pub fn encode_add(
        &self,
        encoder: &mut crate::gpu::GpuEncoder,
        weights: &GpuNormalizationWeights,
        input: &GpuTensor,
        addend: &GpuTensor,
        sum: &GpuTensor,
        output: &GpuTensor,
    ) -> anyhow::Result<()> {
        match (self, weights) {
            (Self::RMSNorm(rms), GpuNormalizationWeights::RMSNorm(w)) => {
                rms.encode_add(encoder, w, input, addend, sum, output);
                Ok(())
            }
            (Self::LayerNorm(_), _) => {
                anyhow::bail!("a fused residual add is implemented for RMSNorm only")
            }
            _ => panic!("Normalization type mismatch"),
        }
    }
}
