//! GPT implementation using edgeTransformers
//! 
//! Provides autoregressive language models for text generation.

pub mod tokenizer;
pub mod seq2seq;

pub mod text_generation;
pub mod sentence_encoder;
pub mod cross_encoder;
mod generation2;
pub mod generation;

pub use sentence_encoder::SentenceEncoder;
pub use cross_encoder::CrossEncoder;
pub use text_generation::TextGenerator;

/// A callback for streaming generated tokens.
///
/// The callback receives the token ID (`u32`) and its decoded text representation (`&str`).
/// It should return `true` to continue generation or `false` to stop early.
pub type TokenCallback<'a> = Box<dyn FnMut(u32, &str) -> bool + 'a>;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(test)]
pub mod tests;

#[cfg(not(target_arch = "wasm32"))]
pub use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
pub use tokenizer::wasm::BPETokenizer;