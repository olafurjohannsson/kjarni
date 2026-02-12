use anyhow::Result;
use ndarray::{Array3, ArrayView2};

pub mod legacy;
pub mod standard;
pub mod standard_new;
pub mod swiglu;

use crate::cpu::encoder::buffers::EncoderBuffers;
pub use crate::cpu::feedforward::{
    legacy::LegacyFeedForward, standard::StdFeedForward, standard_new::StdFeedForwardNew,
    swiglu::SwiGluFeedForward,
};

pub enum FeedForward {
    Standard(StdFeedForward),
    StandardNew(StdFeedForwardNew),
    Legacy(LegacyFeedForward),
    SwiGLU(SwiGluFeedForward),
}

impl FeedForward {
    pub fn forward(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        match self {
            FeedForward::Standard(ffn) => ffn.forward(input),
            FeedForward::StandardNew(ffn) => ffn.forward(input),
            FeedForward::SwiGLU(swiglu) => swiglu.forward(input),
            FeedForward::Legacy(ffn) => ffn.forward(input),
        }
    }
    pub fn out_features(&self) -> usize {
        match self {
            FeedForward::Standard(ffn) => ffn.dense1_weight.shape()[0],
            FeedForward::StandardNew(ffn) => ffn.fc1.out_features(),
            FeedForward::SwiGLU(swiglu) => swiglu.up.out_features(),
            FeedForward::Legacy(ffn) => ffn.dense1_weight.shape()[0],
        }
    }
    pub fn forward_noalloc(&self, hidden: &ArrayView2<f32>, buffers: &mut EncoderBuffers) {
        match self {
            FeedForward::StandardNew(ffn) => ffn.forward_noalloc(hidden, buffers),
            FeedForward::Standard(ffn) => ffn.forward_noalloc(hidden, buffers),
            _ => panic!("No-alloc forward not implemented for this FeedForward type"),
        }
    }
}

#[cfg(test)]
mod tests;
