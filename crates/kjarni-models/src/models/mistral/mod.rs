//! Mistral

pub mod config;
pub mod model;

pub use config::MistralConfig;
pub use model::MistralModel;

#[cfg(test)]
mod tests;
