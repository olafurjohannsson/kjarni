pub mod layer_norm;
pub mod rms_norm;

pub use crate::cpu::normalization::{layer_norm::LayerNorm, rms_norm::RMSNorm};

use ndarray::{Array1, Array3, ArrayView2, ArrayViewMut2};

/// Normalization type
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
    pub fn forward_2d_noalloc(&self, input: &ArrayView2<f32>, output: &mut ArrayViewMut2<f32>) {
        match self {
            Normalization::LayerNorm(ln) => ln.forward_2d_noalloc(input, output),
            Normalization::RMSNorm(_rms) => panic!("RMSNorm does not support noalloc 2D forward"),
        }
    }
    /// `rows = norm(rows + residual)` in place; see `LayerNorm::add_and_norm_rows`.
    pub fn add_and_norm_rows(&self, rows: &mut [f32], residual: Option<&[f32]>) {
        match self {
            Normalization::LayerNorm(ln) => ln.add_and_norm_rows(rows, residual),
            Normalization::RMSNorm(_rms) => panic!("RMSNorm does not support add_and_norm_rows"),
        }
    }

    /// `norm(hidden)` as a new array, leaving `hidden` for the residual.
    /// Layer norm runs across threads; RMS norm as it always has.
    pub fn forward_rows(&self, hidden: &Array3<f32>) -> Array3<f32> {
        match (self, hidden.as_slice()) {
            (Normalization::LayerNorm(ln), Some(rows)) => {
                let mut out = Array3::<f32>::zeros(hidden.dim());
                ln.norm_rows_into(rows, out.as_slice_mut().expect("fresh array"));
                out
            }
            _ => self.forward(hidden),
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
