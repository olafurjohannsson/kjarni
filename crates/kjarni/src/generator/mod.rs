// =============================================================================
// kjarni/src/generator/mod.rs
// =============================================================================

//! Raw text generator for decoder language models.
//!
//! The Generator provides direct access to language model text generation
//! without chat formatting, system prompts, or conversation history.
//!
//! Use `Generator` for:
//! - Text completion (GPT-2 style)
//! - Custom prompt formats
//! - Base models without instruction tuning
//! - Maximum control over generation
//!
//! For conversational AI with chat templates, use `Chat` instead.

mod builder;
mod model;
mod types;
mod validation;

pub use builder::GeneratorBuilder;
pub use model::Generator;
pub use types::*;