//! Fused Gate + Up + SiLU kernel for SwiGLU FFN layers.
//!
//! This module provides a fused implementation of the SwiGLU activation:
//!
//! ```text
//! output = SiLU(input @ gate_weight.T) * (input @ up_weight.T)
//! ```
//!
//! By fusing these operations, we:
//! 1. Read the input tensor once instead of twice
//! 2. Stream both weight matrices together, improving cache utilization
//! 3. Apply activation and multiply in registers before writing output
//!
//! # Performance
//!
//! For decode (seq_len=1), this can reduce memory bandwidth by ~50%
//! compared to separate gate and up projections.

use anyhow::{anyhow, Result};
use half::bf16;
use rayon::prelude::*;

// Import from parent crate - adjust these paths for your project structure
use crate::cpu::kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0};
use crate::linear_layer::{LinearData, LinearLayer};

/// Fused gate+up+silu: `output = SiLU(input @ gate.T) * (input @ up.T)`
///
/// Reads input once, computes both projections per output neuron,
/// then applies SiLU and multiply in one pass.
///
/// # Arguments
///
/// * `gate` - Gate projection layer `[intermediate_dim, hidden_dim]`
/// * `up` - Up projection layer `[intermediate_dim, hidden_dim]`
/// * `input` - Input slice, length = `tokens * hidden_dim`
/// * `output` - Output slice, length = `tokens * intermediate_dim`
/// * `tokens` - Number of tokens (1 for decode, >1 for prefill)
///
/// # Returns
///
/// `Ok(())` on success, `Err` if dtypes don't match or dimensions are invalid.
///
/// # Example
///
/// ```ignore
/// let mut output = vec![0.0f32; intermediate_dim];
/// fused_gate_up_silu(&gate, &up, &input, &mut output, 1)?;
/// ```
pub fn fused_gate_up_silu(
    gate: &LinearLayer,
    up: &LinearLayer,
    input: &[f32],
    output: &mut [f32],
    tokens: usize,
) -> Result<()> {
    let hidden_dim = gate.in_features();
    let intermediate_dim = gate.out_features();

    // =========================================================================
    // Validation
    // =========================================================================
    if gate.dtype() != up.dtype() {
        return Err(anyhow!(
            "gate and up must have same dtype: {:?} vs {:?}",
            gate.dtype(),
            up.dtype()
        ));
    }
    if gate.in_features() != up.in_features() || gate.out_features() != up.out_features() {
        return Err(anyhow!(
            "gate/up dimension mismatch: gate [{}, {}] vs up [{}, {}]",
            gate.out_features(),
            gate.in_features(),
            up.out_features(),
            up.in_features()
        ));
    }
    if input.len() != tokens * hidden_dim {
        return Err(anyhow!(
            "input length {} != tokens {} * hidden {}",
            input.len(),
            tokens,
            hidden_dim
        ));
    }
    if output.len() != tokens * intermediate_dim {
        return Err(anyhow!(
            "output length {} != tokens {} * intermediate {}",
            output.len(),
            tokens,
            intermediate_dim
        ));
    }

    // =========================================================================
    // Dispatch based on dtype
    // =========================================================================
    match (&gate.data, &up.data) {
        // ---------------------------------------------------------------------
        // F32
        // ---------------------------------------------------------------------
        (LinearData::F32(g), LinearData::F32(u)) => {
            let g_slice = g
                .as_slice()
                .ok_or_else(|| anyhow!("gate weights not contiguous"))?;
            let u_slice = u
                .as_slice()
                .ok_or_else(|| anyhow!("up weights not contiguous"))?;

            if tokens == 1 {
                fused_gate_up_silu_f32_decode(input, g_slice, u_slice, output, hidden_dim);
            } else {
                fused_gate_up_silu_f32_prefill(
                    input,
                    g_slice,
                    u_slice,
                    output,
                    tokens,
                    hidden_dim,
                    intermediate_dim,
                );
            }
        }

        // ---------------------------------------------------------------------
        // BF16
        // ---------------------------------------------------------------------
        (LinearData::BF16(g), LinearData::BF16(u)) => {
            let g_slice = g
                .as_slice()
                .ok_or_else(|| anyhow!("gate weights not contiguous"))?;
            let u_slice = u
                .as_slice()
                .ok_or_else(|| anyhow!("up weights not contiguous"))?;

            if tokens == 1 {
                fused_gate_up_silu_bf16_decode(input, g_slice, u_slice, output, hidden_dim);
            } else {
                fused_gate_up_silu_bf16_prefill(
                    input,
                    g_slice,
                    u_slice,
                    output,
                    tokens,
                    hidden_dim,
                    intermediate_dim,
                );
            }
        }

        // ---------------------------------------------------------------------
        // Q8_0
        // ---------------------------------------------------------------------
        (LinearData::Q8_0(g), LinearData::Q8_0(u)) => {
            if tokens == 1 {
                fused_gate_up_silu_q8_0_decode(input, &g.blocks, &u.blocks, output, hidden_dim);
            } else {
                fused_gate_up_silu_q8_0_prefill(
                    input,
                    &g.blocks,
                    &u.blocks,
                    output,
                    tokens,
                    hidden_dim,
                    intermediate_dim,
                );
            }
        }

        // ---------------------------------------------------------------------
        // Q4_K
        // ---------------------------------------------------------------------
        (LinearData::Q4_K(g), LinearData::Q4_K(u)) => {
            if tokens == 1 {
                fused_gate_up_silu_q4_k_decode(input, &g.blocks, &u.blocks, output, hidden_dim);
            } else {
                fused_gate_up_silu_q4_k_prefill(
                    input,
                    &g.blocks,
                    &u.blocks,
                    output,
                    tokens,
                    hidden_dim,
                    intermediate_dim,
                );
            }
        }

        // ---------------------------------------------------------------------
        // Q6_K
        // ---------------------------------------------------------------------
        (LinearData::Q6_K(g), LinearData::Q6_K(u)) => {
            if tokens == 1 {
                fused_gate_up_silu_q6_k_decode(input, &g.blocks, &u.blocks, output, hidden_dim);
            } else {
                fused_gate_up_silu_q6_k_prefill(
                    input,
                    &g.blocks,
                    &u.blocks,
                    output,
                    tokens,
                    hidden_dim,
                    intermediate_dim,
                );
            }
        }

        // ---------------------------------------------------------------------
        // Unsupported
        // ---------------------------------------------------------------------
        _ => {
            return Err(anyhow!(
                "Fused gate+up+silu not supported for dtype {:?}",
                gate.dtype()
            ));
        }
    }

    Ok(())
}

// =============================================================================
// SiLU Activation
// =============================================================================

/// SiLU (Swish) activation: x / (1 + exp(-x))
#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// =============================================================================
// SIMD Module Import
// =============================================================================

//#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::cpu::ops::fused::simd_x86;

// =============================================================================
// F32 Kernels
// =============================================================================

/// F32 fused kernel for decode (tokens=1).
/// Uses AVX2/FMA SIMD when available, falls back to scalar.
fn fused_gate_up_silu_f32_decode(
    input: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    output: &mut [f32],
    k: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                simd_x86::fused_gate_up_silu_f32_avx2(
                    output,
                    input.as_ptr(),
                    gate_w.as_ptr(),
                    up_w.as_ptr(),
                    k,
                );
            }
            return;
        }
    }

    // Scalar fallback
    output.par_iter_mut().enumerate().for_each(|(n, out)| {
        let offset = n * k;
        let (gate_sum, up_sum) = dot_pair_f32_scalar(input, &gate_w[offset..], &up_w[offset..], k);
        *out = silu(gate_sum) * up_sum;
    });
}

/// F32 fused kernel for prefill (tokens>1).
/// Parallelizes over tokens.
fn fused_gate_up_silu_f32_prefill(
    input: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    output: &mut [f32],
    tokens: usize,
    k: usize,
    n: usize,
) {
    output
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(t, out_row)| {
            let in_offset = t * k;
            let in_ptr = &input[in_offset..in_offset + k];

            for (j, out) in out_row.iter_mut().enumerate() {
                let w_offset = j * k;

                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        unsafe {
                            *out = fused_dot_silu_f32_avx2(
                                in_ptr.as_ptr(),
                                gate_w.as_ptr().add(w_offset),
                                up_w.as_ptr().add(w_offset),
                                k,
                            );
                        }
                        continue;
                    }
                }

                let (gate_sum, up_sum) =
                    dot_pair_f32_scalar(in_ptr, &gate_w[w_offset..], &up_w[w_offset..], k);
                *out = silu(gate_sum) * up_sum;
            }
        });
}

/// Scalar fallback: compute two dot products simultaneously
#[inline]
fn dot_pair_f32_scalar(input: &[f32], gate_w: &[f32], up_w: &[f32], k: usize) -> (f32, f32) {
    let mut gate_sum = 0.0f32;
    let mut up_sum = 0.0f32;

    for i in 0..k {
        let val = unsafe { *input.get_unchecked(i) };
        gate_sum += val * unsafe { *gate_w.get_unchecked(i) };
        up_sum += val * unsafe { *up_w.get_unchecked(i) };
    }

    (gate_sum, up_sum)
}

// =============================================================================
// F32 AVX2/FMA SIMD Kernel
// =============================================================================


#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fused_dot_silu_f32_avx2(
    input: *const f32,
    gate_w: *const f32,
    up_w: *const f32,
    k: usize,
) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut gate_acc0 = _mm256_setzero_ps();
    let mut gate_acc1 = _mm256_setzero_ps();
    let mut up_acc0 = _mm256_setzero_ps();
    let mut up_acc1 = _mm256_setzero_ps();

    let mut i = 0;
    let mut in_ptr = input;
    let mut g_ptr = gate_w;
    let mut u_ptr = up_w;

    // Main loop: 16 elements at a time
    while i + 16 <= k {
        let in0 = _mm256_loadu_ps(in_ptr);
        let in1 = _mm256_loadu_ps(in_ptr.add(8));

        let g0 = _mm256_loadu_ps(g_ptr);
        let g1 = _mm256_loadu_ps(g_ptr.add(8));
        let u0 = _mm256_loadu_ps(u_ptr);
        let u1 = _mm256_loadu_ps(u_ptr.add(8));

        gate_acc0 = _mm256_fmadd_ps(in0, g0, gate_acc0);
        gate_acc1 = _mm256_fmadd_ps(in1, g1, gate_acc1);
        up_acc0 = _mm256_fmadd_ps(in0, u0, up_acc0);
        up_acc1 = _mm256_fmadd_ps(in1, u1, up_acc1);

        in_ptr = in_ptr.add(16);
        g_ptr = g_ptr.add(16);
        u_ptr = u_ptr.add(16);
        i += 16;
    }

    // Combine accumulators
    gate_acc0 = _mm256_add_ps(gate_acc0, gate_acc1);
    up_acc0 = _mm256_add_ps(up_acc0, up_acc1);

    // Horizontal sum
    let mut gate_sum = hsum_avx(gate_acc0);
    let mut up_sum = hsum_avx(up_acc0);

    // Remainder
    while i < k {
        let val = *input.add(i);
        gate_sum += val * *gate_w.add(i);
        up_sum += val * *up_w.add(i);
        i += 1;
    }

    // SiLU and multiply
    silu(gate_sum) * up_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx(v: std::arch::x86_64::__m256) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // Sum 256-bit vector to scalar
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(high, low);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    _mm_cvtss_f32(sum32)
}

// =============================================================================
// BF16 Kernels
// =============================================================================

/// BF16 fused kernel for decode (tokens=1).
/// Uses AVX2/FMA SIMD when available.
fn fused_gate_up_silu_bf16_decode(
    input: &[f32],
    gate_w: &[bf16],
    up_w: &[bf16],
    output: &mut [f32],
    k: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                simd_x86::fused_gate_up_silu_bf16_avx2(
                    output,
                    input.as_ptr(),
                    gate_w.as_ptr() as *const u16,
                    up_w.as_ptr() as *const u16,
                    k,
                );
            }
            return;
        }
    }

    // Scalar fallback
    output.par_iter_mut().enumerate().for_each(|(n, out)| {
        let offset = n * k;
        let (gate_sum, up_sum) =
            dot_pair_bf16_scalar(input, &gate_w[offset..], &up_w[offset..], k);
        *out = silu(gate_sum) * up_sum;
    });
}

/// BF16 fused kernel for prefill (tokens>1).
fn fused_gate_up_silu_bf16_prefill(
    input: &[f32],
    gate_w: &[bf16],
    up_w: &[bf16],
    output: &mut [f32],
    tokens: usize,
    k: usize,
    n: usize,
) {
    output
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(t, out_row)| {
            let in_offset = t * k;
            let in_ptr = &input[in_offset..in_offset + k];

            for (j, out) in out_row.iter_mut().enumerate() {
                let w_offset = j * k;
                let (gate_sum, up_sum) =
                    dot_pair_bf16_scalar(in_ptr, &gate_w[w_offset..], &up_w[w_offset..], k);
                *out = silu(gate_sum) * up_sum;
            }
        });
}

/// BF16 scalar kernel: accumulate in F32
#[inline]
fn dot_pair_bf16_scalar(input: &[f32], gate_w: &[bf16], up_w: &[bf16], k: usize) -> (f32, f32) {
    let mut gate_sum = 0.0f32;
    let mut up_sum = 0.0f32;

    for i in 0..k {
        let val = unsafe { *input.get_unchecked(i) };
        let g = unsafe { gate_w.get_unchecked(i).to_f32() };
        let u = unsafe { up_w.get_unchecked(i).to_f32() };
        gate_sum += val * g;
        up_sum += val * u;
    }

    (gate_sum, up_sum)
}

// =============================================================================
// Q8_0 Kernels (32 elements per block)
// =============================================================================

const Q8_0_BLOCK_SIZE: usize = 32;

/// Q8_0 fused kernel for decode.
/// Uses AVX2/FMA SIMD when available.
fn fused_gate_up_silu_q8_0_decode(
    input: &[f32],
    gate_blocks: &[BlockQ8_0],
    up_blocks: &[BlockQ8_0],
    output: &mut [f32],
    k: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                simd_x86::fused_gate_up_silu_q8_0_avx2(
                    output,
                    input.as_ptr(),
                    gate_blocks,
                    up_blocks,
                    k,
                );
            }
            return;
        }
    }

    // Scalar fallback
    let blocks_per_row = k / Q8_0_BLOCK_SIZE;

    output.par_iter_mut().enumerate().for_each(|(n, out)| {
        let row_offset = n * blocks_per_row;
        let (gate_sum, up_sum) = dot_pair_q8_0(
            input,
            &gate_blocks[row_offset..row_offset + blocks_per_row],
            &up_blocks[row_offset..row_offset + blocks_per_row],
        );
        *out = silu(gate_sum) * up_sum;
    });
}

/// Q8_0 fused kernel for prefill.
fn fused_gate_up_silu_q8_0_prefill(
    input: &[f32],
    gate_blocks: &[BlockQ8_0],
    up_blocks: &[BlockQ8_0],
    output: &mut [f32],
    tokens: usize,
    k: usize,
    n: usize,
) {
    let blocks_per_row = k / Q8_0_BLOCK_SIZE;

    output
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(t, out_row)| {
            let in_offset = t * k;
            let in_slice = &input[in_offset..in_offset + k];

            for (j, out) in out_row.iter_mut().enumerate() {
                let row_offset = j * blocks_per_row;
                let (gate_sum, up_sum) = dot_pair_q8_0(
                    in_slice,
                    &gate_blocks[row_offset..row_offset + blocks_per_row],
                    &up_blocks[row_offset..row_offset + blocks_per_row],
                );
                *out = silu(gate_sum) * up_sum;
            }
        });
}

/// Fused dot product for Q8_0 blocks
#[inline]
fn dot_pair_q8_0(input: &[f32], gate_blocks: &[BlockQ8_0], up_blocks: &[BlockQ8_0]) -> (f32, f32) {
    let mut gate_sum = 0.0f32;
    let mut up_sum = 0.0f32;

    for (b, (g_block, u_block)) in gate_blocks.iter().zip(up_blocks.iter()).enumerate() {
        let g_scale = g_block.d.to_f32();
        let u_scale = u_block.d.to_f32();
        let in_offset = b * Q8_0_BLOCK_SIZE;

        for i in 0..Q8_0_BLOCK_SIZE {
            let val = input[in_offset + i];
            gate_sum += val * (g_block.qs[i] as f32) * g_scale;
            up_sum += val * (u_block.qs[i] as f32) * u_scale;
        }
    }

    (gate_sum, up_sum)
}

// =============================================================================
// Q4_K Kernels (256 elements per block)
// =============================================================================

const Q4_K_BLOCK_SIZE: usize = 256;

/// Q4_K fused kernel for decode.
/// Uses AVX2/FMA SIMD when available.
fn fused_gate_up_silu_q4_k_decode(
    input: &[f32],
    gate_blocks: &[BlockQ4_K],
    up_blocks: &[BlockQ4_K],
    output: &mut [f32],
    k: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                simd_x86::fused_gate_up_silu_q4_k_avx2(
                    output,
                    input.as_ptr(),
                    gate_blocks,
                    up_blocks,
                    k,
                );
            }
            return;
        }
    }

    // Scalar fallback
    let blocks_per_row = k / Q4_K_BLOCK_SIZE;

    output.par_iter_mut().enumerate().for_each(|(n, out)| {
        let row_offset = n * blocks_per_row;
        let (gate_sum, up_sum) = dot_pair_q4_k(
            input,
            &gate_blocks[row_offset..row_offset + blocks_per_row],
            &up_blocks[row_offset..row_offset + blocks_per_row],
        );
        *out = silu(gate_sum) * up_sum;
    });
}

/// Q4_K fused kernel for prefill.
fn fused_gate_up_silu_q4_k_prefill(
    input: &[f32],
    gate_blocks: &[BlockQ4_K],
    up_blocks: &[BlockQ4_K],
    output: &mut [f32],
    tokens: usize,
    k: usize,
    n: usize,
) {
    let blocks_per_row = k / Q4_K_BLOCK_SIZE;

    output
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(t, out_row)| {
            let in_offset = t * k;
            let in_slice = &input[in_offset..in_offset + k];

            for (j, out) in out_row.iter_mut().enumerate() {
                let row_offset = j * blocks_per_row;
                let (gate_sum, up_sum) = dot_pair_q4_k(
                    in_slice,
                    &gate_blocks[row_offset..row_offset + blocks_per_row],
                    &up_blocks[row_offset..row_offset + blocks_per_row],
                );
                *out = silu(gate_sum) * up_sum;
            }
        });
}

/// Fused dot product for Q4_K blocks.
/// Q4_K structure: 8 sub-blocks of 32 elements each (256 total).
#[inline]
fn dot_pair_q4_k(input: &[f32], gate_blocks: &[BlockQ4_K], up_blocks: &[BlockQ4_K]) -> (f32, f32) {
    let mut gate_sum = 0.0f32;
    let mut up_sum = 0.0f32;

    for (b, (g_block, u_block)) in gate_blocks.iter().zip(up_blocks.iter()).enumerate() {
        let in_offset = b * Q4_K_BLOCK_SIZE;
        let in_chunk = &input[in_offset..in_offset + Q4_K_BLOCK_SIZE];

        let (g, u) = dot_pair_q4_k_block(in_chunk, g_block, u_block);
        gate_sum += g;
        up_sum += u;
    }

    (gate_sum, up_sum)
}

/// Fused dot product for a single Q4_K block
fn dot_pair_q4_k_block(input: &[f32], g: &BlockQ4_K, u: &BlockQ4_K) -> (f32, f32) {
    let g_d = g.d.to_f32();
    let g_dmin = g.dmin.to_f32();
    let u_d = u.d.to_f32();
    let u_dmin = u.dmin.to_f32();

    let mut gate_sum = 0.0f32;
    let mut up_sum = 0.0f32;

    // Q4_K: 8 sub-blocks of 32 elements each
    // scales[0..5] contain 6-bit scales, scales[6..11] extend them
    for sb in 0..8 {
        // Decode 6-bit scale and min from packed format
        let (g_scale, g_min) = decode_q4_k_scale_min(&g.scales, sb, g_d, g_dmin);
        let (u_scale, u_min) = decode_q4_k_scale_min(&u.scales, sb, u_d, u_dmin);

        let base = sb * 32;

        // Each sub-block: 32 elements packed as 16 bytes (4 bits each)
        // Lower nibble: elements 0-15, Upper nibble: elements 16-31
        for i in 0..16 {
            let byte_idx = sb * 16 + i;
            let val0 = input[base + i];
            let val1 = input[base + 16 + i];

            let g_q0 = (g.qs[byte_idx] & 0x0F) as f32;
            let g_q1 = (g.qs[byte_idx] >> 4) as f32;
            let u_q0 = (u.qs[byte_idx] & 0x0F) as f32;
            let u_q1 = (u.qs[byte_idx] >> 4) as f32;

            gate_sum += val0 * (g_q0 * g_scale - g_min);
            gate_sum += val1 * (g_q1 * g_scale - g_min);
            up_sum += val0 * (u_q0 * u_scale - u_min);
            up_sum += val1 * (u_q1 * u_scale - u_min);
        }
    }

    (gate_sum, up_sum)
}

/// Decode Q4_K scale and min for a sub-block
#[inline]
fn decode_q4_k_scale_min(scales: &[u8; 12], sb: usize, d: f32, dmin: f32) -> (f32, f32) {
    // Q4_K scale encoding (from llama.cpp):
    // scales[0..5]: lower 6 bits of scales for sub-blocks 0-7
    // scales[6..11]: lower 6 bits of mins for sub-blocks 0-7
    // Upper 2 bits are packed differently

    let scale_idx = sb;
    let min_idx = sb;

    let sc = if sb < 4 {
        // Sub-blocks 0-3: scale in lower 6 bits
        (scales[sb] & 63) as f32
    } else {
        // Sub-blocks 4-7: scale in lower 6 bits of scales[sb-4+4]
        (scales[sb] & 63) as f32
    };

    let mn = if sb < 4 {
        // Sub-blocks 0-3: min in scales[sb+4]
        (scales[sb + 4] & 63) as f32
    } else {
        // Sub-blocks 4-7: min in scales[sb+4]
        (scales[sb + 4] & 63) as f32
    };

    (sc * d, mn * dmin)
}

// =============================================================================
// Q6_K Kernels (256 elements per block)
// =============================================================================

const Q6_K_BLOCK_SIZE: usize = 256;

/// Q6_K fused kernel for decode.
/// Uses AVX2/FMA SIMD when available.
fn fused_gate_up_silu_q6_k_decode(
    input: &[f32],
    gate_blocks: &[BlockQ6_K],
    up_blocks: &[BlockQ6_K],
    output: &mut [f32],
    k: usize,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                simd_x86::fused_gate_up_silu_q6_k_avx2(
                    output,
                    input.as_ptr(),
                    gate_blocks,
                    up_blocks,
                    k,
                );
            }
            return;
        }
    }

    // Scalar fallback
    let blocks_per_row = k / Q6_K_BLOCK_SIZE;

    output.par_iter_mut().enumerate().for_each(|(n, out)| {
        let row_offset = n * blocks_per_row;
        let (gate_sum, up_sum) = dot_pair_q6_k(
            input,
            &gate_blocks[row_offset..row_offset + blocks_per_row],
            &up_blocks[row_offset..row_offset + blocks_per_row],
        );
        *out = silu(gate_sum) * up_sum;
    });
}

/// Q6_K fused kernel for prefill.
fn fused_gate_up_silu_q6_k_prefill(
    input: &[f32],
    gate_blocks: &[BlockQ6_K],
    up_blocks: &[BlockQ6_K],
    output: &mut [f32],
    tokens: usize,
    k: usize,
    n: usize,
) {
    let blocks_per_row = k / Q6_K_BLOCK_SIZE;

    output
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(t, out_row)| {
            let in_offset = t * k;
            let in_slice = &input[in_offset..in_offset + k];

            for (j, out) in out_row.iter_mut().enumerate() {
                let row_offset = j * blocks_per_row;
                let (gate_sum, up_sum) = dot_pair_q6_k(
                    in_slice,
                    &gate_blocks[row_offset..row_offset + blocks_per_row],
                    &up_blocks[row_offset..row_offset + blocks_per_row],
                );
                *out = silu(gate_sum) * up_sum;
            }
        });
}

/// Fused dot product for Q6_K blocks
#[inline]
fn dot_pair_q6_k(input: &[f32], gate_blocks: &[BlockQ6_K], up_blocks: &[BlockQ6_K]) -> (f32, f32) {
    let mut gate_sum = 0.0f32;
    let mut up_sum = 0.0f32;

    for (b, (g_block, u_block)) in gate_blocks.iter().zip(up_blocks.iter()).enumerate() {
        let in_offset = b * Q6_K_BLOCK_SIZE;
        let in_chunk = &input[in_offset..in_offset + Q6_K_BLOCK_SIZE];

        let (g, u) = dot_pair_q6_k_block(in_chunk, g_block, u_block);
        gate_sum += g;
        up_sum += u;
    }

    (gate_sum, up_sum)
}

/// Fused dot product for a single Q6_K block.
/// Q6_K structure:
/// - ql[128]: lower 4 bits of each 6-bit value
/// - qh[64]: upper 2 bits packed
/// - scales[16]: 8-bit scales for 16 sub-blocks
/// - d: global scale (f16)
fn dot_pair_q6_k_block(input: &[f32], g: &BlockQ6_K, u: &BlockQ6_K) -> (f32, f32) {
    let g_d = g.d.to_f32();
    let u_d = u.d.to_f32();

    let mut gate_sum = 0.0f32;
    let mut up_sum = 0.0f32;

    // Q6_K: 16 sub-blocks of 16 elements each
    for sb in 0..16 {
        let g_scale = (g.scales[sb] as f32) * g_d;
        let u_scale = (u.scales[sb] as f32) * u_d;

        let base = sb * 16;
        let ql_base = sb * 8; // 16 elements = 8 bytes in ql (lower 4 bits)
        let qh_base = sb * 4; // 16 elements = 4 bytes in qh (upper 2 bits)

        for i in 0..16 {
            let val = input[base + i];

            // Reconstruct 6-bit value: lower 4 from ql, upper 2 from qh
            let ql_byte_idx = ql_base + i / 2;
            let qh_byte_idx = qh_base + i / 4;

            let g_ql = if i % 2 == 0 {
                g.ql[ql_byte_idx] & 0x0F
            } else {
                g.ql[ql_byte_idx] >> 4
            };
            let g_qh = (g.qh[qh_byte_idx] >> ((i % 4) * 2)) & 0x03;
            let g_q = ((g_qh << 4) | g_ql) as i8 - 32; // 6-bit signed

            let u_ql = if i % 2 == 0 {
                u.ql[ql_byte_idx] & 0x0F
            } else {
                u.ql[ql_byte_idx] >> 4
            };
            let u_qh = (u.qh[qh_byte_idx] >> ((i % 4) * 2)) & 0x03;
            let u_q = ((u_qh << 4) | u_ql) as i8 - 32;

            gate_sum += val * (g_q as f32) * g_scale;
            up_sum += val * (u_q as f32) * u_scale;
        }
    }

    (gate_sum, up_sum)
}