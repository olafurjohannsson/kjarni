#![allow(unsafe_code)]
pub(crate) mod scalar;
pub(crate) mod q_common;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;