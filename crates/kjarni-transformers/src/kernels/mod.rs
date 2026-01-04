//! Ultra-optimized logits projection kernel for Llama lm_head
//!
//! Specifically tuned for:
//! - Single token decode: [1, hidden_size] × [vocab_size, hidden_size]^T
//! - Llama 3.2 1B: hidden_size=2048, vocab_size=128256
//! - BF16 weights, F32 activations
//!
//! Key optimizations:
//! - 8-way output unrolling (8 vocab entries per iteration)
//! - 4-way inner loop unrolling (32 elements per iteration)
//! - Aggressive prefetching
//! - Cache-blocking for L2/L3
//! - Parallel over vocab chunks
//!
#![allow(unsafe_code)]
pub mod q_common;
// pub(crate) mod q_common;
// pub(crate) mod scalar;
pub mod scalar;
pub mod quantize;
pub mod dequantize;


#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;


#[cfg(test)]
mod quantize_tests;