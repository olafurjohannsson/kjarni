//! High-level model traits and pretrained model metadata

pub mod base;
pub mod registry;

// Re-export commonly used items
pub use base::{
    LanguageModel,
    EncoderLanguageModel,
    DecoderLanguageModel,
    Seq2SeqLanguageModel,
    project_to_vocab,
    l2_normalize,
};

pub use registry::{
    ModelType,
    ModelArchitecture,
    ModelInfo,
    ModelPaths,
};