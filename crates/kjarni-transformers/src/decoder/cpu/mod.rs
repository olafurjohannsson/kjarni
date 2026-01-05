//! CPU-specific building blocks for constructing decoder compute components.

pub mod decoder_backend;
pub use decoder_backend::CpuDecoderBackend;
pub mod rope_decoder_layer;
pub mod decoder_layer;
pub mod decoder_attention;

pub use rope_decoder_layer::CpuRoPEDecoderLayer;
pub use decoder_layer::DecoderLayer;
pub use decoder_attention::DecoderAttention;

#[cfg(test)]
mod tests;
