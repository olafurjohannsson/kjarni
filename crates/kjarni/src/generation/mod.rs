//! Generation configuration and resolution.
//!
//! This module handles the merging of generation parameters from
//! multiple sources: model defaults, user preferences, and runtime overrides.

pub mod overrides;
pub mod resolved;

pub mod resolution;

pub use overrides::GenerationOverrides;
pub use resolution::resolve_generation_config;
pub use resolved::ResolvedGenerationConfig;


pub type GenConfig = GenerationOverrides;