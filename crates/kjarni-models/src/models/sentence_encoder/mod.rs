//! Sentence encoder for semantic similarity and embeddings.

mod model;
mod configs;
pub use model::SentenceEncoder;
pub use configs::{BertConfig, DistilBertConfig, MpnetConfig};

#[cfg(test)]
mod tests;
