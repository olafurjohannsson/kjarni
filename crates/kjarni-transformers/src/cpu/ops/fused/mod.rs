//! Fused CPU operations for improved memory bandwidth efficiency.
//!
//! This module provides fused kernels that combine multiple operations
//! to reduce memory traffic. Key optimizations:
//!
//! - **Fused Gate+Up+SiLU**: Combines gate projection, up projection,
//!   SiLU activation, and element-wise multiply into a single pass
//!   over the input tensor.
//!
//! # When Fusing Helps
//!
//! Fused kernels are most beneficial when:
//! - **Memory-bound operations**: Small batch sizes (especially decode with batch=1)
//! - **Large weight matrices**: Where weight loading dominates compute time
//! - **Repeated input access**: Same input used for multiple projections
//!
//! # Supported Data Types
//!
//! All fused operations support:
//! - F32 (with AVX2/FMA SIMD on x86, NEON on ARM)
//! - BF16 (mixed precision with F32 accumulation)
//! - Q8_0 (8-bit block quantized)
//! - Q4_K (4-bit K-quant block quantized)
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::cpu::ops::fused;
//!
//! // Fused FFN gate+up+silu (replaces 2 matmuls + activation + multiply)
//! fused::fused_gate_up_silu(
//!     &gate_layer,
//!     &up_layer,
//!     input_slice,
//!     output_slice,
//!     num_tokens,
//! )?;
//! ```

mod gate_up_silu;

pub use gate_up_silu::fused_gate_up_silu;

#[cfg(test)]
mod tests;


mod simd_x86;