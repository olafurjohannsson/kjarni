//! T5 and FLAN-T5 model family

pub mod config;
pub mod model;
pub mod cpu_encoder;
pub mod cpu_decoder;
pub mod gpu_encoder;
pub mod gpu_decoder;

pub use config::T5Config;
pub use model::T5Model;