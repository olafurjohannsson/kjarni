//! GGUF tensor conversion

use anyhow::{Result, anyhow};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};

use crate::cpu::kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0, QK_K};
use crate::tensor::raw_tensor::TensorView;
use crate::tensor::{CpuTensor, DType, QuantizedMatrix};
use crate::weights::model_weights::AttentionLayout;

const SUPER_TILE_SIZE: usize = 64;
const HALF_TILE: usize = 32;

/// Safely casts bytes to a typed vector, handling alignment.
pub fn cast_or_copy<T: bytemuck::Pod + bytemuck::Zeroable>(bytes: &[u8]) -> Vec<T> {
    if let Ok(slice) = bytemuck::try_cast_slice(bytes) {
        slice.to_vec()
    } else {
        let mut aligned = vec![0u8; bytes.len()];
        aligned.copy_from_slice(bytes);
        bytemuck::cast_slice(&aligned).to_vec()
    }
}

/// Computes the GGUF block group index for a given logical row.
#[inline]
pub fn gguf_block_group_for_row(logical_row: usize) -> usize {
    let super_tile = logical_row / SUPER_TILE_SIZE;
    let within_tile = logical_row % SUPER_TILE_SIZE;

    if within_tile < HALF_TILE {
        super_tile * SUPER_TILE_SIZE + within_tile * 2
    } else {
        super_tile * SUPER_TILE_SIZE + 1 + (within_tile - HALF_TILE) * 2
    }
}

/// Computes the GGUF source row for attention weight reordering.
#[inline]
fn get_gguf_src_row(logical_row: usize, head_dim: usize) -> usize {
    let half_dim = head_dim / 2;
    let tile = logical_row / head_dim;
    let within_tile = logical_row % head_dim;

    if within_tile < half_dim {
        tile * head_dim + within_tile * 2
    } else {
        tile * head_dim + (within_tile - half_dim) * 2 + 1
    }
}

/// Reorders Q8_0 blocks from GGUF interleaved layout to row-major.
fn reorder_q8_0_blocks(blocks: Vec<BlockQ8_0>, rows: usize, cols: usize) -> Vec<BlockQ8_0> {
    let blocks_per_row = cols / 32;

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
fn reorder_q4k_blocks(
    blocks: Vec<BlockQ4_K>,
    rows: usize,
    cols: usize,
    head_dim: usize,
) -> Vec<BlockQ4_K> {
    let blocks_per_row = cols / 256;

    debug_assert_eq!(blocks.len(), rows * blocks_per_row, "block count mismatch");
    debug_assert_eq!(rows % head_dim, 0, "rows must align with head dimension");

    let mut reordered = Vec::with_capacity(blocks.len());

    for r in 0..rows {
        let src_r = get_gguf_src_row(r, head_dim);
        let src_base = src_r * blocks_per_row;

        for b in 0..blocks_per_row {
            reordered.push(blocks[src_base + b].clone());
        }
    }

    reordered
}

/// Reorders Q6_K blocks from GGUF interleaved layout to row-major.
pub fn reorder_q_k_blocks3(
    blocks: Vec<BlockQ6_K>,
    rows: usize,
    cols: usize,
    head_dim: usize,
) -> Vec<BlockQ6_K> {
    let blocks_per_row = cols / QK_K;

    debug_assert_eq!(blocks.len(), rows * blocks_per_row);
    debug_assert_eq!(rows % head_dim, 0, "rows must be a multiple of head dimension");

    let mut reordered = vec![blocks[0]; blocks.len()];

    for r in 0..rows {
        let src_r = get_gguf_src_row(r, head_dim);

        let dst_base = r * blocks_per_row;
        let src_base = src_r * blocks_per_row;

        reordered[dst_base..dst_base + blocks_per_row]
            .copy_from_slice(&blocks[src_base..src_base + blocks_per_row]);
    }

    reordered
}

/// Returns `true` if this tensor needs GGUF block reordering.
fn needs_gguf_reordering(name: &str, _rows: usize) -> bool {
    name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("attn_q")
        || name.contains("attn_k")
}

/// Converts a raw GGUF tensor view into a typed CPU tensor.
pub fn raw_to_typed_gguf(raw: TensorView<'_>, attn: Option<AttentionLayout>) -> Result<CpuTensor> {
    match raw.dtype {
        DType::F32 => {
            let data: Vec<f32> = cast_or_copy(&raw.bytes);
            Ok(CpuTensor::F32(ArrayD::from_shape_vec(IxDyn(&raw.shape), data)?))
        }

        DType::F16 => {
            let data: Vec<f16> = cast_or_copy(&raw.bytes);
            Ok(CpuTensor::F16(ArrayD::from_shape_vec(IxDyn(&raw.shape), data)?))
        }

        DType::BF16 => {
            let data: Vec<bf16> = cast_or_copy(&raw.bytes);
            Ok(CpuTensor::BF16(ArrayD::from_shape_vec(IxDyn(&raw.shape), data)?))
        }

        DType::Q8_0 => {
            let blocks: Vec<BlockQ8_0> = cast_or_copy(&raw.bytes);
            let shape = [raw.shape[0], raw.shape[1]];

            let blocks = if raw.shape.len() == 2 && needs_gguf_reordering(&raw.name, raw.shape[0]) {
                log::debug!(
                    "reordering Q8_0 blocks for '{}' [{}, {}]",
                    raw.name, shape[0], shape[1]
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
                    let head_dim = attn.head_dim;
                    debug_assert!(
                        head_dim == 64 || head_dim == 128,
                        "unexpected head_dim {}",
                        head_dim
                    );

                    log::debug!(
                        "reordering Q4_K blocks for '{}' [{}, {}] with head_dim={}",
                        raw.name, shape[0], shape[1], head_dim
                    );

                    blocks = reorder_q4k_blocks(blocks, raw.shape[0], raw.shape[1], head_dim);
                }
            }

            Ok(CpuTensor::Q4_K(QuantizedMatrix { blocks, shape }))
        }

        DType::Q6_K => {
            let mut blocks: Vec<BlockQ6_K> = cast_or_copy(&raw.bytes);
            let shape = [raw.shape[0], raw.shape[1]];

            if let Some(attn) = attn {
                if needs_gguf_reordering(&raw.name, shape[0]) {
                    log::debug!(
                        "reordering Q6_K blocks for '{}' [{}, {}] with head_dim={}",
                        raw.name, shape[0], shape[1], attn.head_dim
                    );

                    blocks = reorder_q_k_blocks3(blocks, shape[0], shape[1], attn.head_dim);
                }
            }

            Ok(CpuTensor::Q6_K(QuantizedMatrix { blocks, shape }))
        }

        _ => Err(anyhow!(
            "unsupported dtype {:?} for tensor '{}'",
            raw.dtype, raw.name
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_block_group_mapping() {
        assert_eq!(gguf_block_group_for_row(0), 0);
        assert_eq!(gguf_block_group_for_row(1), 2);
        assert_eq!(gguf_block_group_for_row(31), 62);

        assert_eq!(gguf_block_group_for_row(32), 1);
        assert_eq!(gguf_block_group_for_row(33), 3);
        assert_eq!(gguf_block_group_for_row(63), 63);

        assert_eq!(gguf_block_group_for_row(64), 64);
        assert_eq!(gguf_block_group_for_row(96), 65);
    }

    #[test]
    fn test_get_gguf_src_row_head64() {
        let head_dim = 64;

        assert_eq!(get_gguf_src_row(0, head_dim), 0);
        assert_eq!(get_gguf_src_row(1, head_dim), 2);
        assert_eq!(get_gguf_src_row(31, head_dim), 62);

        assert_eq!(get_gguf_src_row(32, head_dim), 1);
        assert_eq!(get_gguf_src_row(33, head_dim), 3);
        assert_eq!(get_gguf_src_row(63, head_dim), 63);
    }

    #[test]
    fn test_get_gguf_src_row_head128() {
        let head_dim = 128;

        assert_eq!(get_gguf_src_row(0, head_dim), 0);
        assert_eq!(get_gguf_src_row(1, head_dim), 2);
        assert_eq!(get_gguf_src_row(63, head_dim), 126);

        assert_eq!(get_gguf_src_row(64, head_dim), 1);
        assert_eq!(get_gguf_src_row(65, head_dim), 3);
        assert_eq!(get_gguf_src_row(127, head_dim), 127);
    }

    #[test]
    fn test_needs_gguf_reordering() {
        assert!(needs_gguf_reordering("model.layers.0.self_attn.q_proj.weight", 2048));
        assert!(needs_gguf_reordering("model.layers.0.self_attn.k_proj.weight", 2048));
        assert!(needs_gguf_reordering("blk.0.attn_q.weight", 2048));
        assert!(needs_gguf_reordering("blk.0.attn_k.weight", 2048));

        assert!(!needs_gguf_reordering("model.layers.0.self_attn.v_proj.weight", 2048));
        assert!(!needs_gguf_reordering("model.layers.0.self_attn.o_proj.weight", 2048));
        assert!(!needs_gguf_reordering("model.layers.0.mlp.gate_proj.weight", 2048));
        assert!(!needs_gguf_reordering("model.embed_tokens.weight", 128256));
    }
}