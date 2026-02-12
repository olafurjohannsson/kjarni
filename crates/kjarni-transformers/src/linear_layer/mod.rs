//! CPU-based linear transformation layers with multi-dtype support.

mod linear_layer;
mod builder;

pub use linear_layer::{LinearLayer, LinearData, F32MatmulStrategy};
pub use builder::LinearLayerBuilder;

#[cfg(test)]
mod tests;