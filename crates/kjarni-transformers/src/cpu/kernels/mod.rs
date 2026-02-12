#![allow(unsafe_code)]
pub mod q_common;
pub mod scalar;
pub mod quantize;
pub mod dequantize;


#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;
