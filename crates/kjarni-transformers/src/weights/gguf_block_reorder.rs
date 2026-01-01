use anyhow::{Result, anyhow};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};

use crate::kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0, QK_K};
use crate::tensor::{CpuTensor, DType, QuantizedMatrix, TensorView};
use crate::weights::model_weights::AttentionLayout;

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
pub fn reorder_q_k_blocks3(
    blocks: Vec<BlockQ6_K>,
    rows: usize,
    cols: usize,
    head_dim: usize, // <--- Pass 64 for 1B, 128 for 3B
) -> Vec<BlockQ6_K> {
    let blocks_per_row = cols / QK_K; // QK_K usually 256 for Q6_K logic, check your constants

    // Sanity check
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
/// Generalized reorder for Q4_K blocks that works for both 1B and 3B models
pub fn reorder_q4k_blocks3(
    blocks: Vec<BlockQ4_K>,
    rows: usize,
    cols: usize,
    head_dim: usize, // <--- CRITICAL: 64 for 1B, 128 for 3B
) -> Vec<BlockQ4_K> {
    // Q4_K blocks are always 256 elements wide
    let blocks_per_row = cols / 256;

    // Safety checks
    debug_assert_eq!(blocks.len(), rows * blocks_per_row, "Block count mismatch");
    debug_assert_eq!(rows % head_dim, 0, "Rows must align with head dimension");

    // Allocate target vector
    let mut reordered = Vec::with_capacity(blocks.len());
    // (Or use vec![blocks[0]; blocks.len()] if BlockQ4_K implements Clone/Copy)
    // unsafe { reordered.set_len(blocks.len()) }; // Optimization if you prefer

    // Iterate through logical rows (0, 1, 2...)
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
#[inline]
fn get_gguf_src_row(logical_row: usize, head_dim: usize) -> usize {
    let half_dim = head_dim / 2;

    // Which head does this row belong to?
    let tile = logical_row / head_dim;
    // Which row inside that head?
    let within_tile = logical_row % head_dim;

    if within_tile < half_dim {
        // First half of the head maps to even rows in GGUF
        tile * head_dim + within_tile * 2
    } else {
        // Second half of the head maps to odd rows in GGUF
        tile * head_dim + (within_tile - half_dim) * 2 + 1
    }
}
pub fn reorder_q4k_blocks2(blocks: Vec<BlockQ4_K>, rows: usize, cols: usize) -> Vec<BlockQ4_K> {
    let blocks_per_row = cols / QK_K;

    if rows <= 32 {
        return blocks;
    }

    debug_assert_eq!(blocks.len(), rows * blocks_per_row);

    let mut reordered = vec![blocks[0]; blocks.len()];

    for r in 0..rows {
        let src_r = interleave_64(r);

        let dst_base = r * blocks_per_row;
        let src_base = src_r * blocks_per_row;

        reordered[dst_base..dst_base + blocks_per_row]
            .copy_from_slice(&blocks[src_base..src_base + blocks_per_row]);
    }

    reordered
}

#[inline]
fn interleave_64(r: usize) -> usize {
    let tile = r / 64;
    let in_tile = r % 64;

    if in_tile < 32 {
        tile * 64 + in_tile * 2
    } else {
        tile * 64 + (in_tile - 32) * 2 + 1
    }
}

pub fn reorder_q6k_blocks2(blocks: Vec<BlockQ6_K>, rows: usize, cols: usize) -> Vec<BlockQ6_K> {
    let blocks_per_row = cols / QK_K;

    // GGUF Q/K/V interleave applies only for tiles > 32 rows
    if rows <= 32 {
        return blocks;
    }

    debug_assert_eq!(blocks.len(), rows * blocks_per_row);
    debug_assert_eq!(rows % 64, 0);

    let mut reordered = vec![blocks[0]; blocks.len()];

    for r in 0..rows {
        let src_r = interleave_64(r);

        let dst_base = r * blocks_per_row;
        let src_base = src_r * blocks_per_row;

        reordered[dst_base..dst_base + blocks_per_row]
            .copy_from_slice(&blocks[src_base..src_base + blocks_per_row]);
    }

    reordered
}
/// Reorder Q4_K blocks from GGUF GQA layout to row-major.
pub fn reorder_q4k_blocks_gqa(
    blocks: Vec<BlockQ4_K>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cols: usize,
) -> Vec<BlockQ4_K> {
    let blocks_per_row = cols / 256;
    let num_block_rows = blocks.len() / blocks_per_row;

    if num_block_rows <= 32 {
        return blocks;
    }

    let mut reordered = blocks.clone();

    for r in 0..num_block_rows {
        let head = r % n_heads;
        let row_in_head = r / n_heads;
        let src_r = std::cmp::min(
            (head % n_kv_heads) * head_dim + row_in_head * n_kv_heads + (head / n_kv_heads),
            num_block_rows - 1,
        );

        let dst_base = r * blocks_per_row;
        let src_base = src_r * blocks_per_row;

        reordered[dst_base..dst_base + blocks_per_row]
            .copy_from_slice(&blocks[src_base..src_base + blocks_per_row]);
    }

    reordered
}

/// Reorder Q6_K blocks from GGUF GQA layout to row-major.
pub fn reorder_q6k_blocks_gqa(
    blocks: Vec<BlockQ6_K>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cols: usize,
) -> Vec<BlockQ6_K> {
    let blocks_per_row = cols / 256;
    let num_block_rows = blocks.len() / blocks_per_row;

    if num_block_rows <= 32 {
        return blocks;
    }

    let mut reordered = blocks.clone();

    for r in 0..num_block_rows {
        let head = r % n_heads;
        let row_in_head = r / n_heads;
        let src_r = std::cmp::min(
            (head % n_kv_heads) * head_dim + row_in_head * n_kv_heads + (head / n_kv_heads),
            num_block_rows - 1,
        );

        let dst_base = r * blocks_per_row;
        let src_base = src_r * blocks_per_row;

        reordered[dst_base..dst_base + blocks_per_row]
            .copy_from_slice(&blocks[src_base..src_base + blocks_per_row]);
    }

    reordered
}
pub fn reorder_with_map<B: Clone>(
    blocks: Vec<B>,
    rows: usize,
    cols: usize,
    block_size: usize,
    map: &[usize],
) -> Vec<B> {
    let blocks_per_row = cols / block_size;
    if map.len() < rows {
        // Map is incomplete, can't reorder safely
        return blocks;
    }

    let mut reordered = Vec::with_capacity(blocks.len());

    // For each logical row we want to construct...
    for logical_row in 0..rows {
        // ...find which physical row in the GGUF file contains its data.
        let physical_row = map[logical_row];

        // Copy the blocks from that physical row.
        let start_idx = physical_row * blocks_per_row;
        for b in 0..blocks_per_row {
            reordered.push(blocks[start_idx + b].clone());
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

/// Robust check for GGUF interleaving.
///
/// Instead of guessing based on names ("q_proj"), we check the structural properties.
/// If a tensor is Q4_K, 2D, large enough, and NOT the vocab embedding,
/// it is almost certainly interleaved by llama.cpp.
pub fn should_reorder_gguf(
    dtype: DType,
    shape: &[usize],
    vocab_size: usize, // Passed from ModelConfig
) -> bool {
    // 1. Check DType: Only specific K-Quants are interleaved
    let is_interleaved_type = match dtype {
        DType::Q4_K | DType::Q5_K => true, // Q2_K, Q3_K also if you support them
        _ => false,                        // Q6_K, Q8_0, F16, F32 are NEVER interleaved
    };

    if !is_interleaved_type {
        return false;
    }

    // 2. Check Dimensions: Must be 2D weight matrix
    if shape.len() != 2 {
        return false;
    }

    let rows = shape[0];

    // 3. Check Size: Must be at least one super-block (64 rows)
    // The interleaving logic requires groups of 64.
    if rows < 64 {
        return false;
    }

    // 4. Exclusion: Do not reorder Token Embeddings or Output Head
    // These matrices have a dimension equal to vocab_size (e.g., 128256).
    // Note: We use a fuzzy match because sometimes vocab is padded.
    if rows == vocab_size || (rows > 0 && vocab_size > 0 && rows.abs_diff(vocab_size) < 256) {
        log::debug!("Skipping reorder for vocab-sized tensor (rows={})", rows);
        return false;
    }

    // If it passed all checks, it's a Linear Layer (Attention or MLP)
    // that uses Q4_K, so it IS interleaved.
    true
}

fn needs_gguf_reordering2(name: &str, rows: usize) -> bool {
    if rows <= 32 {
        return false;
    }

    let name_lower = name.to_lowercase();

    // Must be in attention context (not random "q" somewhere)
    let is_attention = name_lower.contains("attn")
        || name_lower.contains("self_attn")
        || name_lower.contains("attention");

    // Check for Q or K projection specifically
    let is_q = name_lower.contains("q_proj")
        || name_lower.contains("attn_q")
        || name_lower.contains(".wq.");

    let is_k = name_lower.contains("k_proj")
        || name_lower.contains("attn_k")
        || name_lower.contains(".wk.");

    // Either explicit Q/K pattern, or attention + contains just "q"/"k"
    is_q || is_k || (is_attention && (name_lower.contains(".q.") || name_lower.contains(".k.")))
}

fn needs_gguf_reordering(name: &str, rows: usize) -> bool {
    // Must have enough rows to be interleaved (> 32)
    // if rows <= HALF_TILE {
    //     return false;
    // }
    // name.contains("q_proj")
    // || name.contains("k_proj")
    // || name.contains("v_proj")
    // || name.contains("attn_q")
    // || name.contains("attn_k")
    // || name.contains("attn_v")
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
pub fn raw_to_typed_gguf(
    raw: TensorView<'_>,
    tied_embeddings: bool,
    vocab_size: usize,
    attn: Option<AttentionLayout>,
) -> Result<CpuTensor> {
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

            let blocks = if raw.shape.len() == 2 && needs_gguf_reordering(&raw.name, raw.shape[0]) {
                log::error!(
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
            let mut blocks: Vec<BlockQ4_K> = cast_or_copy(&raw.bytes);
            let shape = [raw.shape[0], raw.shape[1]];

            if let Some(attn) = attn {
                if needs_gguf_reordering(&raw.name, raw.shape[0]) {
                    // blocks = reorder_q4k_blocks2(
                    //     blocks,
                    //     raw.shape[0], // ← ALWAYS THIS
                    //     raw.shape[1],
                    // );
                    let head_dim = attn.head_dim;
                    assert!(head_dim == 64 || head_dim == 128, "Unexpected head_dim {}", head_dim);
                    blocks = reorder_q4k_blocks3(
                        blocks,
                        raw.shape[0],
                        raw.shape[1],
                        head_dim, // Pass the calculated dimension here
                    );
                }
            }

            Ok(CpuTensor::Q4_K(QuantizedMatrix { blocks, shape }))
        }
        // DType::Q4_K => {
        //     let mut blocks: Vec<BlockQ4_K> = cast_or_copy(&raw.bytes);
        //     let shape = [raw.shape[0], raw.shape[1]];

        //     if let Some(attn) = attn {
        //         if needs_gguf_reordering(&raw.name, raw.shape[0]) {
        //             let gguf_rows = attn.n_kv_heads * attn.head_dim;

        //             blocks = reorder_q4k_blocks_gqa(
        //                 blocks,
        //                 attn.n_heads,
        //                 attn.n_kv_heads,
        //                 attn.head_dim,
        //                 raw.shape[1],
        //             );
        //             // blocks = reorder_q4k_blocks(blocks, shape[0], shape[1]);
        //         }
        //     }

        //     Ok(CpuTensor::Q4_K(QuantizedMatrix { blocks, shape }))
        // }
        // DType::Q6_K => {
        //     let mut blocks: Vec<BlockQ6_K> = cast_or_copy(&raw.bytes);
        //     let shape = [raw.shape[0], raw.shape[1]];

        //     if let Some(attn) = attn {
        //         let gguf_rows = attn.n_kv_heads * attn.head_dim;
        //         if needs_gguf_reordering(&raw.name, shape[0]) {
        //             blocks = reorder_q6k_blocks_gqa(
        //                 blocks,
        //                 attn.n_heads,
        //                 attn.n_kv_heads,
        //                 attn.head_dim,
        //                 raw.shape[1],
        //             );
        //         }
        //     }

        //     Ok(CpuTensor::Q6_K(QuantizedMatrix { blocks, shape }))
        // }
        DType::Q6_K => {
            let mut blocks: Vec<BlockQ6_K> = cast_or_copy(&raw.bytes);
            let shape = [raw.shape[0], raw.shape[1]];

            if let Some(attn) = attn {
                if needs_gguf_reordering(&raw.name, shape[0]) {
                    blocks = reorder_q_k_blocks3(blocks, shape[0], shape[1], attn.head_dim);
                }
            }

            Ok(CpuTensor::Q6_K(QuantizedMatrix { blocks, shape }))
        }
        _ => Err(anyhow!("Unsupported dtype: {:?}", raw.dtype)),
    }
}
