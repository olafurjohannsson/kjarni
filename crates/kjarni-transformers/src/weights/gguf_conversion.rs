//! GGUF tensor conversion and block reordering.
//!
//! This module handles the critical task of converting raw GGUF tensor data into
//! typed CPU tensors, including the essential block reordering step for quantized
//! weights.
//!
//! # ⚠️ CRITICAL: Do Not Modify Reordering Logic ⚠️
//!
//! The block reordering functions in this module are the result of extensive
//! debugging and reverse-engineering of the GGUF format. **Any changes to the
//! reordering logic will break quantized model inference.**
//!
//! # Background: The GGUF Interleaving Problem
//!
//! GGUF files (used by llama.cpp) store quantized weight blocks in an interleaved
//! "super-block" layout optimized for SIMD memory access patterns. This layout
//! differs from the simple row-major order expected by our matmul kernels.
//!
//! ## The Discovery (December 2024)
//!
//! After days of debugging incorrect inference results, we discovered that:
//!
//! - **Expected layout**: Block 0 → Row 0, Block 8 → Row 1, Block 16 → Row 2, ...
//! - **Actual GGUF layout**: Block 0 → Row 0, Block 8 → Row 32, Block 16 → Row 1, ...
//!
//! The pattern organizes blocks in "super-tiles" of 64 rows (or head_dim rows for
//! attention weights):
//!
//! - Even block groups (0, 2, 4, ...) → first half of tile (rows 0-31)
//! - Odd block groups (1, 3, 5, ...) → second half of tile (rows 32-63)
//!
//! ## Why llama.cpp Uses This Layout
//!
//! llama.cpp's SIMD kernels are optimized for this interleaved access pattern,
//! which improves cache utilization during matrix multiplication. However, our
//! row-major kernels expect standard layout.
//!
//! ## Our Solution
//!
//! We reorder blocks to row-major layout at load time. This adds ~10ms overhead
//! per weight matrix but makes all downstream code work correctly:
//!
//! - `to_array2_f32()` dequantization produces correct values
//! - Row-major matmul kernels compute correct results
//! - Attention patterns match PyTorch/HuggingFace reference
//!
//! ## Alternative Approach
//!
//! For maximum performance, one could port llama.cpp's SIMD kernels that expect
//! the interleaved layout directly, avoiding the reordering overhead.
//!
//! # Which Weights Need Reordering?
//!
//! Only Q and K projection weights in attention layers use the interleaved layout:
//!
//! - `*.q_proj.*` / `*.attn_q.*`
//! - `*.k_proj.*` / `*.attn_k.*`
//!
//! V projections, O projections, and MLP weights use standard row-major layout.

use anyhow::{Result, anyhow};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};

use crate::cpu::kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0, QK_K};
use crate::tensor::raw_tensor::TensorView;
use crate::tensor::{CpuTensor, DType, QuantizedMatrix};
use crate::weights::model_weights::AttentionLayout;

// =============================================================================
// Constants
// =============================================================================

/// Size of a GGUF super-tile in rows.
///
/// Blocks are organized in groups of 64 rows, with the first 32 rows stored
/// in even block positions and the second 32 rows in odd positions.
const SUPER_TILE_SIZE: usize = 64;

/// Half of a super-tile (32 rows).
///
/// Rows 0-31 of each tile map to even block groups.
/// Rows 32-63 of each tile map to odd block groups.
const HALF_TILE: usize = 32;

// =============================================================================
// Byte Casting Utilities
// =============================================================================

/// Safely cast bytes to a typed vector, handling memory alignment.
///
/// Attempts zero-copy casting first for performance. If the bytes are not
/// properly aligned for the target type, copies to an aligned buffer first.
///
/// # Type Parameters
///
/// * `T` - Target type, must be plain-old-data (Pod) and zero-initializable
///
/// # Arguments
///
/// * `bytes` - Raw byte slice to cast
///
/// # Returns
///
/// A vector of the target type containing the cast data.
///
/// # Performance
///
/// - Aligned data: Zero-copy cast + single allocation for the vector
/// - Unaligned data: Two allocations (alignment buffer + result vector)
pub fn cast_or_copy<T: bytemuck::Pod + bytemuck::Zeroable>(bytes: &[u8]) -> Vec<T> {
    if let Ok(slice) = bytemuck::try_cast_slice(bytes) {
        slice.to_vec()
    } else {
        // Data is misaligned - copy to aligned buffer first
        let mut aligned = vec![0u8; bytes.len()];
        aligned.copy_from_slice(bytes);
        bytemuck::cast_slice(&aligned).to_vec()
    }
}

// =============================================================================
// Block Group Indexing (Q8_0)
// =============================================================================

/// Computes the GGUF block group index for a given logical row.
///
/// This function implements the inverse of GGUF's 64-row super-tile interleaving:
///
/// - Logical rows 0-31 of each super-tile → even block groups (0, 2, 4, ...)
/// - Logical rows 32-63 of each super-tile → odd block groups (1, 3, 5, ...)
///
/// # Arguments
///
/// * `logical_row` - The row index in standard row-major order (0, 1, 2, ...)
///
/// # Returns
///
/// The block group index where this row's data is stored in the GGUF file.
///
/// # Example
///
/// For a matrix with 64 rows:
/// ```text
/// logical_row  0 → block_group  0 (even, first half)
/// logical_row  1 → block_group  2
/// logical_row 31 → block_group 62
/// logical_row 32 → block_group  1 (odd, second half)
/// logical_row 33 → block_group  3
/// logical_row 63 → block_group 63
/// ```
#[inline]
pub fn gguf_block_group_for_row(logical_row: usize) -> usize {
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

// =============================================================================
// Head-Dimension Based Row Mapping (Q4_K, Q6_K)
// =============================================================================

/// Computes the GGUF source row for attention weight reordering.
///
/// For attention projection weights (Q, K), GGUF uses head-dimension-based
/// interleaving rather than the fixed 64-row super-tiles used for Q8_0.
///
/// The pattern within each attention head:
/// - First half of head (rows 0 to head_dim/2 - 1) → even positions
/// - Second half of head (rows head_dim/2 to head_dim - 1) → odd positions
///
/// # Arguments
///
/// * `logical_row` - The target row in standard row-major order
/// * `head_dim` - The attention head dimension (64 for 1B models, 128 for 3B+)
///
/// # Returns
///
/// The row index in the GGUF file that contains this logical row's data.
///
/// # Panics
///
/// Debug builds will panic if `logical_row` is not aligned to `head_dim`.
#[inline]
fn get_gguf_src_row(logical_row: usize, head_dim: usize) -> usize {
    let half_dim = head_dim / 2;

    // Which attention head does this row belong to?
    let tile = logical_row / head_dim;
    // Position within that head
    let within_tile = logical_row % head_dim;

    if within_tile < half_dim {
        // First half of head maps to even rows in GGUF
        tile * head_dim + within_tile * 2
    } else {
        // Second half of head maps to odd rows in GGUF
        tile * head_dim + (within_tile - half_dim) * 2 + 1
    }
}

// =============================================================================
// Block Reordering Functions
// =============================================================================

/// Reorders Q8_0 blocks from GGUF interleaved layout to row-major.
///
/// Uses the 64-row super-tile pattern defined by [`gguf_block_group_for_row`].
///
/// # Arguments
///
/// * `blocks` - Quantized blocks in GGUF interleaved order
/// * `rows` - Number of rows in the weight matrix
/// * `cols` - Number of columns in the weight matrix
///
/// # Returns
///
/// Blocks reordered to standard row-major layout.
///
/// # Skip Conditions
///
/// Returns blocks unchanged if:
/// - `rows <= 32`: Too small to be interleaved
/// - `rows > 8192`: Large embedding matrices use standard layout
fn reorder_q8_0_blocks(blocks: Vec<BlockQ8_0>, rows: usize, cols: usize) -> Vec<BlockQ8_0> {
    // Q8_0 blocks contain 32 elements each
    let blocks_per_row = cols / 32;

    // Skip reordering for small matrices or large embeddings
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

/// Reorders Q4_K blocks from GGUF interleaved layout to row-major.
///
/// Uses head-dimension-based interleaving via [`get_gguf_src_row`].
/// Works correctly for both 1B models (head_dim=64) and 3B+ models (head_dim=128).
///
/// # Arguments
///
/// * `blocks` - Quantized blocks in GGUF interleaved order
/// * `rows` - Number of rows in the weight matrix
/// * `cols` - Number of columns in the weight matrix
/// * `head_dim` - Attention head dimension (64 or 128)
///
/// # Returns
///
/// Blocks reordered to standard row-major layout.
///
/// # Panics
///
/// Debug builds will panic if:
/// - Block count doesn't match `rows * blocks_per_row`
/// - `rows` is not a multiple of `head_dim`
fn reorder_q4k_blocks(
    blocks: Vec<BlockQ4_K>,
    rows: usize,
    cols: usize,
    head_dim: usize,
) -> Vec<BlockQ4_K> {
    // Q4_K blocks contain 256 elements each
    let blocks_per_row = cols / 256;

    // Sanity checks
    debug_assert_eq!(blocks.len(), rows * blocks_per_row, "Block count mismatch");
    debug_assert_eq!(rows % head_dim, 0, "Rows must align with head dimension");

    let mut reordered = Vec::with_capacity(blocks.len());

    // Iterate through logical rows (0, 1, 2, ...)
    for r in 0..rows {
        // Find which row in the GGUF data holds this logical row's data
        let src_r = get_gguf_src_row(r, head_dim);
        let src_base = src_r * blocks_per_row;

        // Copy the entire row of blocks
        for b in 0..blocks_per_row {
            reordered.push(blocks[src_base + b].clone());
        }
    }

    reordered
}

/// Reorders Q6_K blocks from GGUF interleaved layout to row-major.
///
/// Uses head-dimension-based interleaving via [`get_gguf_src_row`].
/// Works correctly for both 1B models (head_dim=64) and 3B+ models (head_dim=128).
///
/// # Arguments
///
/// * `blocks` - Quantized blocks in GGUF interleaved order
/// * `rows` - Number of rows in the weight matrix
/// * `cols` - Number of columns in the weight matrix
/// * `head_dim` - Attention head dimension (64 or 128)
///
/// # Returns
///
/// Blocks reordered to standard row-major layout.
///
/// # Panics
///
/// Debug builds will panic if:
/// - Block count doesn't match `rows * blocks_per_row`
/// - `rows` is not a multiple of `head_dim`
pub fn reorder_q_k_blocks3(
    blocks: Vec<BlockQ6_K>,
    rows: usize,
    cols: usize,
    head_dim: usize,
) -> Vec<BlockQ6_K> {
    // QK_K is typically 256 for Q6_K
    let blocks_per_row = cols / QK_K;

    // Sanity checks
    debug_assert_eq!(blocks.len(), rows * blocks_per_row);
    debug_assert_eq!(
        rows % head_dim,
        0,
        "Rows must be a multiple of head dimension"
    );

    let mut reordered = vec![blocks[0]; blocks.len()];

    for r in 0..rows {
        // Calculate where this logical row actually lives in the GGUF file
        let src_r = get_gguf_src_row(r, head_dim);

        let dst_base = r * blocks_per_row;
        let src_base = src_r * blocks_per_row;

        reordered[dst_base..dst_base + blocks_per_row]
            .copy_from_slice(&blocks[src_base..src_base + blocks_per_row]);
    }

    reordered
}

// =============================================================================
// Reordering Detection
// =============================================================================

/// Determines if a weight matrix needs GGUF block reordering.
///
/// Only Q and K projection weights use the interleaved layout.
/// V projections, O projections, and MLP weights use standard row-major.
///
/// # Arguments
///
/// * `name` - The tensor name (HuggingFace or GGUF format)
/// * `_rows` - Number of rows (currently unused, reserved for future heuristics)
///
/// # Returns
///
/// `true` if this tensor's blocks need reordering, `false` otherwise.
///
/// # Recognized Patterns
///
/// - HuggingFace: `*.q_proj.*`, `*.k_proj.*`
/// - GGUF: `*.attn_q.*`, `*.attn_k.*`
fn needs_gguf_reordering(name: &str, _rows: usize) -> bool {
    name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("attn_q")
        || name.contains("attn_k")
}

// =============================================================================
// Main Conversion Function
// =============================================================================

/// Converts a raw GGUF tensor view into a typed CPU tensor.
///
/// Handles:
/// - Float types (F32, F16, BF16): Direct conversion
/// - Quantized types (Q8_0, Q4_K, Q6_K): Conversion with optional block reordering
///
/// # Arguments
///
/// * `raw` - The raw tensor view from the GGUF loader
/// * `attn` - Optional attention layout for head-dimension-based reordering
///
/// # Returns
///
/// A typed `CpuTensor` ready for computation.
///
/// # Reordering Behavior
///
/// For quantized tensors (Q8_0, Q4_K, Q6_K):
///
/// 1. Checks if the tensor name matches Q/K projection patterns
/// 2. If yes, applies the appropriate reordering based on quantization type:
///    - Q8_0: Uses 64-row super-tile pattern
///    - Q4_K/Q6_K: Uses head_dim-based pattern (requires `attn` layout)
/// 3. If no, returns blocks in original order
///
/// # Errors
///
/// Returns an error if:
/// - The dtype is not supported
/// - Shape/block count validation fails
///
/// # Example
///
/// ```ignore
/// let attn_layout = AttentionLayout {
///     n_heads: 32,
///     n_kv_heads: 8,
///     head_dim: 128,
/// };
///
/// let cpu_tensor = raw_to_typed_gguf(tensor_view, Some(attn_layout))?;
/// ```
pub fn raw_to_typed_gguf(raw: TensorView<'_>, attn: Option<AttentionLayout>) -> Result<CpuTensor> {
    match raw.dtype {
        // =====================================================================
        // Floating Point Types (no reordering needed)
        // =====================================================================
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

        // =====================================================================
        // Q8_0: 8-bit quantization with 64-row super-tile reordering
        // =====================================================================
        DType::Q8_0 => {
            let blocks: Vec<BlockQ8_0> = cast_or_copy(&raw.bytes);
            let shape = [raw.shape[0], raw.shape[1]];

            let blocks = if raw.shape.len() == 2 && needs_gguf_reordering(&raw.name, raw.shape[0]) {
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

        // =====================================================================
        // Q4_K: 4-bit k-quant with head_dim-based reordering
        // =====================================================================
        DType::Q4_K => {
            let mut blocks: Vec<BlockQ4_K> = cast_or_copy(&raw.bytes);
            let shape = [raw.shape[0], raw.shape[1]];

            if let Some(attn) = attn {
                if needs_gguf_reordering(&raw.name, raw.shape[0]) {
                    let head_dim = attn.head_dim;
                    debug_assert!(
                        head_dim == 64 || head_dim == 128,
                        "Unexpected head_dim {}",
                        head_dim
                    );

                    log::debug!(
                        "Reordering Q4_K blocks for '{}' [{}, {}] with head_dim={}",
                        raw.name,
                        shape[0],
                        shape[1],
                        head_dim
                    );

                    blocks = reorder_q4k_blocks(blocks, raw.shape[0], raw.shape[1], head_dim);
                }
            }

            Ok(CpuTensor::Q4_K(QuantizedMatrix { blocks, shape }))
        }

        // =====================================================================
        // Q6_K: 6-bit k-quant with head_dim-based reordering
        // =====================================================================
        DType::Q6_K => {
            let mut blocks: Vec<BlockQ6_K> = cast_or_copy(&raw.bytes);
            let shape = [raw.shape[0], raw.shape[1]];

            if let Some(attn) = attn {
                if needs_gguf_reordering(&raw.name, shape[0]) {
                    log::debug!(
                        "Reordering Q6_K blocks for '{}' [{}, {}] with head_dim={}",
                        raw.name,
                        shape[0],
                        shape[1],
                        attn.head_dim
                    );

                    blocks = reorder_q_k_blocks3(blocks, shape[0], shape[1], attn.head_dim);
                }
            }

            Ok(CpuTensor::Q6_K(QuantizedMatrix { blocks, shape }))
        }

        // =====================================================================
        // Unsupported types
        // =====================================================================
        _ => Err(anyhow!(
            "Unsupported dtype {:?} for tensor '{}'",
            raw.dtype,
            raw.name
        )),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_block_group_mapping() {
        // First half of first tile → even indices
        assert_eq!(gguf_block_group_for_row(0), 0);
        assert_eq!(gguf_block_group_for_row(1), 2);
        assert_eq!(gguf_block_group_for_row(31), 62);

        // Second half of first tile → odd indices
        assert_eq!(gguf_block_group_for_row(32), 1);
        assert_eq!(gguf_block_group_for_row(33), 3);
        assert_eq!(gguf_block_group_for_row(63), 63);

        // Second tile
        assert_eq!(gguf_block_group_for_row(64), 64);
        assert_eq!(gguf_block_group_for_row(96), 65);
    }

    #[test]
    fn test_get_gguf_src_row_head64() {
        let head_dim = 64;

        // First half of first head → even positions
        assert_eq!(get_gguf_src_row(0, head_dim), 0);
        assert_eq!(get_gguf_src_row(1, head_dim), 2);
        assert_eq!(get_gguf_src_row(31, head_dim), 62);

        // Second half of first head → odd positions
        assert_eq!(get_gguf_src_row(32, head_dim), 1);
        assert_eq!(get_gguf_src_row(33, head_dim), 3);
        assert_eq!(get_gguf_src_row(63, head_dim), 63);
    }

    #[test]
    fn test_get_gguf_src_row_head128() {
        let head_dim = 128;

        // First half of first head → even positions
        assert_eq!(get_gguf_src_row(0, head_dim), 0);
        assert_eq!(get_gguf_src_row(1, head_dim), 2);
        assert_eq!(get_gguf_src_row(63, head_dim), 126);

        // Second half of first head → odd positions
        assert_eq!(get_gguf_src_row(64, head_dim), 1);
        assert_eq!(get_gguf_src_row(65, head_dim), 3);
        assert_eq!(get_gguf_src_row(127, head_dim), 127);
    }

    #[test]
    fn test_needs_gguf_reordering() {
        // Should reorder
        assert!(needs_gguf_reordering(
            "model.layers.0.self_attn.q_proj.weight",
            2048
        ));
        assert!(needs_gguf_reordering(
            "model.layers.0.self_attn.k_proj.weight",
            2048
        ));
        assert!(needs_gguf_reordering("blk.0.attn_q.weight", 2048));
        assert!(needs_gguf_reordering("blk.0.attn_k.weight", 2048));

        // Should NOT reorder
        assert!(!needs_gguf_reordering(
            "model.layers.0.self_attn.v_proj.weight",
            2048
        ));
        assert!(!needs_gguf_reordering(
            "model.layers.0.self_attn.o_proj.weight",
            2048
        ));
        assert!(!needs_gguf_reordering(
            "model.layers.0.mlp.gate_proj.weight",
            2048
        ));
        assert!(!needs_gguf_reordering("model.embed_tokens.weight", 128256));
    }
}
