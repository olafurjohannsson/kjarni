
//! Common data structures and constants for quantized kernels.

use bytemuck::{Pod, Zeroable}; // <-- IMPORT THE TRAITS
use half::f16;

// =======================================================================
//  Constants for K-Quants
// =======================================================================

pub const QK_K: usize = 256;
pub const QS_K: usize = 32;

// =======================================================================
//  Quantization Block Definitions
// =======================================================================

/// An 8-bit quantization block (type 0).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)] // <-- DERIVE THE TRAITS
pub struct BlockQ8_0 {
    /// The block-specific scale factor.
    pub d: f16,
    /// The quantized 8-bit signed integer weights.
    pub qs: [i8; 32],
}

/// A 4-bit "K-Quant" quantization block.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)] // <-- DERIVE THE TRAITS
pub struct BlockQ4_K {
    /// Super-block scale and min, used to derive sub-block scales/mins.
    pub d: f16,
    pub dmin: f16,
    /// Packed scales and mins for the 8 sub-blocks (32 elements each).
    pub scales: [u8; 12],
    /// The quantized 4-bit weights, packed two per byte.
    pub qs: [u8; QK_K / 2],
}

// Compile-time sanity checks for memory layout.
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);
const _: () = assert!(std::mem::size_of::<BlockQ4_K>() == 144);