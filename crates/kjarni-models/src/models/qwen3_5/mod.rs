//! Qwen 3.5: a hybrid decoder of gated attention and Gated DeltaNet layers.

pub mod cache;
pub mod config;
pub mod cpu_decoder;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
pub mod gpu_decoder;
pub mod layers;
pub mod model;
#[cfg(test)]
mod tests;

pub use config::Qwen3_5Config;
pub use model::Qwen3_5Model;
