//! Sentence encoder for semantic similarity and embeddings.

mod configs;
mod model;
pub use configs::{BertConfig, DistilBertConfig, MpnetConfig};
pub use model::SentenceEncoder;

#[cfg(test)]
mod tests;
