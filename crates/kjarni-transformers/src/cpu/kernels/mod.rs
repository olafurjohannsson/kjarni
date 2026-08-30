#![allow(unsafe_code)]
pub mod dequantize;
pub mod q_common;
pub mod quantize;
pub mod scalar;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;

#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm32;
