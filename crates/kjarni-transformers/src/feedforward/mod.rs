use anyhow::Result;
use ndarray::Array3;

pub mod legacy;
pub mod standard;
pub mod swiglu;
pub mod standard_new;

pub use crate::feedforward::{legacy::LegacyFeedForward, swiglu::SwiGluFeedForward, standard::StdFeedForward, standard_new::StdFeedForwardNew};

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
}


#[cfg(test)]
mod tests;