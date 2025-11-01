//! Foreign Function Interface for EdgeGPT

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "c-bindings")]
pub mod c;

pub mod types;

// Re-export C functions when c-bindings feature is enabled
#[cfg(feature = "c-bindings")]
pub use c::*;