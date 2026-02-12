//! Mistral


pub mod config;
pub mod model;

pub use model::MistralModel;
pub use config::MistralConfig;



#[cfg(test)]
mod tests;