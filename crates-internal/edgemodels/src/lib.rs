//! GPT implementation using edgeTransformers
//! 
//! Provides autoregressive language models for text generation.

pub mod gptconfig;
pub mod bertconfig;
pub mod model;
pub mod gptweights;
pub mod bertweights;
pub mod tokenizer;
// pub mod generation;
pub mod tests;
pub mod sentence_encoder;
pub mod cross_encoder;

pub use sentence_encoder::SentenceEncoder;
pub use cross_encoder::CrossEncoder;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Re-exports
pub use gptconfig::GPTConfig;
// pub use model::{GenerativeModel, GenerativeModelType};
pub use gptweights::GPTModelWeights;
// pub use generation::{GenerationConfig, SamplingStrategy};

#[cfg(not(target_arch = "wasm32"))]
pub use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
pub use tokenizer::wasm::BPETokenizer;