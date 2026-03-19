//! Matrix multiplication dispatch.
//!
//! Platform-specific implementations:
//! - `matmul_native`: Multi-threaded with AVX2/NEON SIMD (Linux, macOS, Windows)
//! - `matmul_wasm`: Single-threaded with WASM SIMD128 (Browser, Obsidian)
//!
//! Both modules export the same public API. Code above this level
//! (LinearLayer, EncoderPipeline, etc.) is platform-agnostic.

#[cfg(not(target_arch = "wasm32"))]
pub mod matmul_native;
#[cfg(not(target_arch = "wasm32"))]
pub use matmul_native::*;

#[cfg(target_arch = "wasm32")]
pub mod matmul_wasm;
#[cfg(target_arch = "wasm32")]
pub use matmul_wasm::*;


#[cfg(test)]
mod tests;