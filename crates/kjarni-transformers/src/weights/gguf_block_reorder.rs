use anyhow::{Result, anyhow};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};

use crate::kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0};
use crate::tensor::{CpuTensor, DType, QuantizedMatrix, TensorView};

/// Safely cast bytes to a typed slice, handling unaligned data.
pub fn cast_or_copy<T: bytemuck::Pod + bytemuck::Zeroable>(bytes: &[u8]) -> Vec<T> {
    if let Ok(slice) = bytemuck::try_cast_slice(bytes) {
        slice.to_vec()
    } else {
        let mut aligned = vec![0u8; bytes.len()];
        aligned.copy_from_slice(bytes);
        bytemuck::cast_slice(&aligned).to_vec()
    }
}

// =============================================================================
// GGUF Block Reordering
// =============================================================================
//
// CRITICAL: GGUF files store quantized blocks in an interleaved "super-block"
// layout, NOT simple row-major order. We discovered this after extensive
// debugging in December 2024.
//
// ## The Problem
//
// For a Q4_K weight matrix [2048, 2048] with 8 blocks per row:
//   - Expected: Block 0→Row 0, Block 8→Row 1, Block 16→Row 2
//   - Reality:  Block 0→Row 0, Block 8→Row 32, Block 16→Row 1, Block 24→Row 33
//
// ## The Pattern
//
// Blocks are organized in "super-tiles" of 64 rows:
//   - Even block groups (0,2,4,...) → first 32 rows of tile (0-31)
//   - Odd block groups (1,3,5,...)  → second 32 rows of tile (32-63)
//
// ## Why?
//
// llama.cpp uses this layout to optimize SIMD memory access patterns.
// Their kernels expect this order, but our row-major kernels don't.
//
// ## The Fix
//
// We reorder blocks to row-major at load time. This adds ~10ms overhead
// but makes all downstream code (matmul, dequantization) work correctly.
//
// For maximum performance, one could port llama.cpp's SIMD kernels that
// expect the interleaved layout directly.
// =============================================================================

const SUPER_TILE_SIZE: usize = 64;
const HALF_TILE: usize = 32;

/// Computes the GGUF block group index for a given logical row.
///
/// This implements the inverse of GGUF's interleaving pattern:
/// - Logical rows 0-31 of each super-tile are stored in even block groups
/// - Logical rows 32-63 of each super-tile are stored in odd block groups
#[inline]
fn gguf_block_group_for_row(logical_row: usize) -> usize {
    let super_tile = logical_row / SUPER_TILE_SIZE;
    let within_tile = logical_row % SUPER_TILE_SIZE;

    if within_tile < HALF_TILE {
        // First half of tile: even block groups (0, 2, 4, ...)
        super_tile * SUPER_TILE_SIZE + within_tile * 2
    } else {
        // Second half of tile: odd block groups (1, 3, 5, ...)
        super_tile * SUPER_TILE_SIZE + 1 + (within_tile - HALF_TILE) * 2
    }
}

/// Reorders Q4_K blocks from GGUF interleaved layout to row-major.
fn reorder_q4k_blocks(blocks: Vec<BlockQ4_K>, rows: usize, cols: usize) -> Vec<BlockQ4_K> {
    let blocks_per_row = cols / 256;

    // Skip for small matrices OR large embedding matrices
    if rows <= HALF_TILE || rows > 8192 {
        return blocks;
    }

    let mut reordered = Vec::with_capacity(blocks.len());

    for logical_row in 0..rows {
        let gguf_block_group = gguf_block_group_for_row(logical_row);

        for b in 0..blocks_per_row {
            let src_idx = gguf_block_group * blocks_per_row + b;
            reordered.push(blocks[src_idx].clone());
        }
    }

    reordered
}

/// Reorders Q6_K blocks from GGUF interleaved layout to row-major.
pub fn reorder_q6k_blocks(blocks: Vec<BlockQ6_K>, rows: usize, cols: usize) -> Vec<BlockQ6_K> {
    let blocks_per_row = cols / 256;

    // Skip reordering for:
    // 1. Small matrices (already row-major)
    // 2. Large embedding matrices (vocab_size x hidden_dim) - NOT interleaved!
    if rows <= HALF_TILE || rows > 8192 {
        return blocks;
    }

    let mut reordered = Vec::with_capacity(blocks.len());

    for logical_row in 0..rows {
        let gguf_block_group = gguf_block_group_for_row(logical_row);

        for b in 0..blocks_per_row {
            let src_idx = gguf_block_group * blocks_per_row + b;
            reordered.push(blocks[src_idx].clone());
        }
    }

    reordered
}

/// Reorders Q8_0 blocks from GGUF interleaved layout to row-major.
fn reorder_q8_0_blocks(blocks: Vec<BlockQ8_0>, rows: usize, cols: usize) -> Vec<BlockQ8_0> {
    let blocks_per_row = cols / 32;

    // Skip for small matrices OR large embedding matrices
    if rows <= HALF_TILE || rows > 8192 {
        return blocks;
    }

    let mut reordered = Vec::with_capacity(blocks.len());

    for logical_row in 0..rows {
        let gguf_block_group = gguf_block_group_for_row(logical_row);

        for b in 0..blocks_per_row {
            let src_idx = gguf_block_group * blocks_per_row + b;
            reordered.push(blocks[src_idx].clone());
        }
    }

    reordered
}
fn needs_gguf_reordering(name: &str, rows: usize) -> bool {
    // Must have enough rows to be interleaved (> 32)
    if rows <= HALF_TILE {
        return false;
    }

    // Only Q and K projections are interleaved
    name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("attn_q")
        || name.contains("attn_k") // GGUF naming
}

/// Converts a `TensorView` into a `CpuTensor`.
///
/// # Arguments
///
/// * `raw` - The raw tensor view from the weight loader
/// * `is_gguf` - If true, applies block reordering for quantized types
///
/// # GGUF Block Reordering
///
/// GGUF stores quantized blocks in an interleaved pattern for SIMD optimization.
/// When `is_gguf` is true, blocks are reordered to row-major layout so that:
/// - `to_array2_f32()` dequantization works correctly
/// - Row-major matmul kernels work correctly
///
/// See module-level documentation for the full story on why this is needed.
pub fn raw_to_typed_gguf(raw: TensorView<'_>, is_gguf: bool) -> Result<CpuTensor> {
    match raw.dtype {
        DType::F32 => {
            let data: Vec<f32> = cast_or_copy(&raw.bytes);
            Ok(CpuTensor::F32(ArrayD::from_shape_vec(
                IxDyn(&raw.shape),
                data,
            )?))
        }
        DType::F16 => {
            let data: Vec<f16> = cast_or_copy(&raw.bytes);
            Ok(CpuTensor::F16(ArrayD::from_shape_vec(
                IxDyn(&raw.shape),
                data,
            )?))
        }
        DType::BF16 => {
            let data: Vec<bf16> = cast_or_copy(&raw.bytes);
            Ok(CpuTensor::BF16(ArrayD::from_shape_vec(
                IxDyn(&raw.shape),
                data,
            )?))
        }
        DType::Q8_0 => {
            let blocks: Vec<BlockQ8_0> = cast_or_copy(&raw.bytes);
            let shape = [raw.shape[0], raw.shape[1]];

            let blocks =
                if is_gguf && raw.shape.len() == 2 && needs_gguf_reordering(&raw.name, shape[0]) {
                    log::debug!(
                        "Reordering Q8_0 blocks for '{}' [{}, {}]",
                        raw.name,
                        shape[0],
                        shape[1]
                    );
                    reorder_q8_0_blocks(blocks, shape[0], shape[1])
                } else {
                    blocks
                };

            Ok(CpuTensor::Q8_0(QuantizedMatrix { blocks, shape }))
        }
        DType::Q4_K => {
            let blocks: Vec<BlockQ4_K> = cast_or_copy(&raw.bytes);
            let shape = [raw.shape[0], raw.shape[1]];

            let blocks =
                if is_gguf && raw.shape.len() == 2 && needs_gguf_reordering(&raw.name, shape[0]) {
                    log::debug!(
                        "Reordering Q4_K blocks for '{}' [{}, {}]",
                        raw.name,
                        shape[0],
                        shape[1]
                    );
                    reorder_q4k_blocks(blocks, shape[0], shape[1])
                } else {
                    blocks
                };

            Ok(CpuTensor::Q4_K(QuantizedMatrix { blocks, shape }))
        }
        DType::Q6_K => {
            let blocks: Vec<BlockQ6_K> = cast_or_copy(&raw.bytes);
            let shape = [raw.shape[0], raw.shape[1]];

            // Q6_K tensors (v_proj, down_proj, embeddings) are NOT interleaved
            // No reordering needed!

            Ok(CpuTensor::Q6_K(QuantizedMatrix { blocks, shape }))
        }
        _ => Err(anyhow!("Unsupported dtype: {:?}", raw.dtype)),
    }
}

/// Legacy function - assumes non-GGUF source (no reordering).
pub fn raw_to_typed_no_reorder(raw: TensorView<'_>) -> Result<CpuTensor> {
    raw_to_typed_gguf(raw, false)
}
