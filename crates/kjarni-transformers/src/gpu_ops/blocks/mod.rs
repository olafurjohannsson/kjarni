pub mod attention;
pub mod cache;
pub mod embeddings;
pub mod encoder;
pub mod ffn;
pub mod ffn_swiglu;
pub mod layer_norm;
pub mod rms_norm;
pub mod rope;
pub mod decoder_cross_attention;

use crate::gpu_ops::{GpuTensor, GpuTensorPool};


pub use layer_norm::{GpuLayerNorm, GpuLayerNormWeights};
pub use rms_norm::{GpuRMSNorm, GpuRMSNormWeights};

pub use ffn::{GpuFeedForward as GpuFeedForwardStd, GpuFeedForwardWeights as GpuFeedForwardWeightsStd};
pub use ffn_swiglu::{GpuSwiGLUFFN, GpuSwiGLUFFNWeights};
// pub use decoder_cross_attention::GpuCrossAttentionDecoder;

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

pub enum GpuFeedForwardWeights {
    Standard(GpuFeedForwardWeightsStd),
    SwiGLU(GpuSwiGLUFFNWeights),
}

pub enum GpuFeedForward {
    Standard(GpuFeedForwardStd),
    SwiGLU(GpuSwiGLUFFN),
}

impl GpuFeedForward {
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        weights: &GpuFeedForwardWeights,
        input: &GpuTensor,
        output: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) {
        match (self, weights) {
            (Self::Standard(ffn), GpuFeedForwardWeights::Standard(w)) => {
                ffn.encode_2(encoder, input, w, pool, output)
            }
            (Self::SwiGLU(swi), GpuFeedForwardWeights::SwiGLU(w)) => {
                swi.encode(encoder, w, input, output, pool)
            }
            _ => panic!("FFN type mismatch"),
        }
    }
}
