//! CPU-based linear transformation layers with multi-dtype support.

mod builder;
mod linear_layer;

pub use builder::LinearLayerBuilder;
pub use linear_layer::{F32MatmulStrategy, KQuantMatrix, LinearData, LinearLayer};

#[cfg(test)]
mod tests;
