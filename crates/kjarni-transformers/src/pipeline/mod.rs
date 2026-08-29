//! High-level inference pipelines for transformer models.

// The encoder path is available everywhere, including wasm32. The decoder and
// encoder-decoder pipelines are not yet: they are wired to the GPU backend, which
// has no wasm build. Splitting the module this way is what lets a browser build
// use the real engine for embeddings, classification and reranking instead of a
// parallel implementation.
mod cpu_factory;
mod encoder;

#[cfg(not(target_arch = "wasm32"))]
mod audio;
mod decoder;
#[cfg(not(target_arch = "wasm32"))]
mod encoder_decoder;
#[cfg(not(target_arch = "wasm32"))]
mod encoder_decoder_builder;
#[cfg(not(target_arch = "wasm32"))]
mod encoder_decoder_loader;
#[cfg(not(target_arch = "wasm32"))]
mod seq2seq_cpu_factory;

pub use cpu_factory::CpuLayerFactory;

pub use decoder::{
    DecoderModelFactory, DecoderPipeline, DecoderPipelineBuilder, DecoderPipelineConfig,
    DecoderLoader,
};
#[cfg(not(target_arch = "wasm32"))]
pub use encoder_decoder::{EncoderDecoderPipeline, EncoderDecoderPipelineConfig};
#[cfg(not(target_arch = "wasm32"))]
pub use encoder_decoder_builder::EncoderDecoderPipelineBuilder;
#[cfg(not(target_arch = "wasm32"))]
pub use encoder_decoder_loader::{EncoderDecoderModelFactory, Seq2SeqLoader};
#[cfg(not(target_arch = "wasm32"))]
pub use seq2seq_cpu_factory::Seq2SeqFactory;

pub use encoder::{
    EncoderLoader, EncoderModelFactory, EncoderPipeline, EncoderPipelineBuilder,
    EncoderPipelineConfig,
};

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
