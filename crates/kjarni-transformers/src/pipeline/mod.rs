//! High-level inference pipelines for transformer models.

mod audio;
mod cpu_factory;
mod decoder;
mod encoder;
mod encoder_decoder;
mod encoder_decoder_builder;
mod encoder_decoder_loader;
mod seq2seq_cpu_factory;

pub use cpu_factory::CpuLayerFactory;
pub use decoder::{
    DecoderModelFactory, DecoderPipeline, DecoderPipelineBuilder, DecoderPipelineConfig,
    DecoderLoader,
};
pub use encoder_decoder::{EncoderDecoderPipeline, EncoderDecoderPipelineConfig};
pub use encoder_decoder_builder::EncoderDecoderPipelineBuilder;
pub use encoder_decoder_loader::{EncoderDecoderModelFactory, Seq2SeqLoader};
pub use seq2seq_cpu_factory::Seq2SeqFactory;

pub use encoder::{
    EncoderLoader, EncoderModelFactory, EncoderPipeline, EncoderPipelineBuilder,
    EncoderPipelineConfig,
};

#[cfg(test)]
mod tests;
