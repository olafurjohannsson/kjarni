pub mod layer_norm;
pub mod rms_norm;

pub use crate::normalization::{layer_norm::LayerNorm, rms_norm::RMSNorm};

use ndarray::{Array1, Array3};

/// Normalization type for decoder layers
pub enum Normalization {
    LayerNorm(LayerNorm),
    RMSNorm(RMSNorm),
}

impl Normalization {
    pub fn forward(&self, input: &Array3<f32>) -> Array3<f32> {
        match self {
            Normalization::LayerNorm(ln) => ln.forward_3d(input),
            Normalization::RMSNorm(rms) => rms.forward_3d(input),
        }
    }
    pub fn as_rms_norm(&self) -> Option<&RMSNorm> {
        match self {
            Normalization::RMSNorm(rms) => Some(rms),
            _ => None,
        }
    }
    pub fn as_layer_norm(&self) -> Option<&LayerNorm> {
        match self {
            Normalization::LayerNorm(ln) => Some(ln),
            _ => None,
        }
    }

    pub fn gamma(&self) -> &Array1<f32> {
        match self {
            Normalization::LayerNorm(ln) => &ln.weight,
            Normalization::RMSNorm(rms) => &rms.weight,
        }
    }
    pub fn beta(&self) -> Option<&Array1<f32>> {
        match self {
            Normalization::LayerNorm(ln) => Some(&ln.bias),
            Normalization::RMSNorm(_) => None,
        }
    }
    pub fn eps(&self) -> f32 {
        match self {
            Normalization::LayerNorm(ln) => ln.eps,
            Normalization::RMSNorm(rms) => rms.eps,
        }
    }
}
