use anyhow::Result;
use ndarray::Array3;

pub mod standard;
pub mod swiglu;

pub use crate::feedforward::{standard::StdFeedForward, swiglu::SwiGluFeedForward};

pub enum FeedForward {
    Standard(StdFeedForward),
    SwiGLU(SwiGluFeedForward),
}

impl FeedForward {
    pub fn forward(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        match self {
            FeedForward::Standard(ffn) => ffn.forward(input),
            FeedForward::SwiGLU(swiglu) => swiglu.forward(input),
        }
    }
}
