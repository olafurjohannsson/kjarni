
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

/// Q8_K block structure for quantized activations
/// Used as the "right-hand side" in Q4_K × Q8_K dot products
#[repr(C)]
#[derive(Clone, Debug)]
pub struct BlockQ8_K {
    /// Scale factor for dequantization
    pub d: f32,
    /// Quantized 8-bit values (256 per block)
    pub qs: [i8; 256],
    /// Block sums: sum of each 16-element sub-block (16 sums)
    pub bsums: [i16; 16],
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

/// A 6-bit "K-Quant" quantization block (256 elements).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ6_K {
    pub ql: [u8; 128],      // Lower 4 bits of the 6-bit values
    pub qh: [u8; 64],       // Upper 2 bits of the 6-bit values
    pub scales: [i8; 16],   // 8-bit scales
    pub d: f16,             // Global scale (f16)
}

const _: () = assert!(std::mem::size_of::<BlockQ6_K>() == 210);

// Compile-time sanity checks for memory layout.
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);
const _: () = assert!(std::mem::size_of::<BlockQ4_K>() == 144);