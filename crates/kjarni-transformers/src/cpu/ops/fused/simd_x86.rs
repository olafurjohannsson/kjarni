//! SIMD-optimized fused gate+up+silu kernels.
//!
//! These kernels compute `output[n] = silu(dot(input, gate[n])) * dot(input, up[n])`
//! by doing both dot products in the same loop, reading input once.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

use crate::cpu::kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0};
use half::bf16;

// =============================================================================
// Helpers
// =============================================================================

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_ps_avx(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(high, low);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    _mm_cvtss_f32(sum32)
}

/// Decode Q4_K scale and min for a sub-block (from llama.cpp)
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}


/// Fused gate+up+silu for F32 weights using AVX2/FMA.
///
/// Computes: `output[n] = silu(dot(input, gate[n])) * dot(input, up[n])`
///
/// Reads input once per output neuron, computing both gate and up dot products
/// in the same loop to maximize cache efficiency.
///
/// # Arguments
///
/// * `output` - Output slice `[n]` where n = number of output neurons
/// * `input` - Input pointer, length = k
/// * `gate_weights` - Gate weight matrix in row-major `[n, k]`
/// * `up_weights` - Up weight matrix in row-major `[n, k]`
/// * `k` - Input dimension (hidden size)
///
/// # Safety
///
/// Requires AVX2 and FMA. Caller must ensure pointers are valid.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fused_gate_up_silu_f32_avx2(
    output: &mut [f32],
    input: *const f32,
    gate_weights: *const f32,
    up_weights: *const f32,
    k: usize,
) {
    let n = output.len();

    for i in 0..n {
        let gate_row = gate_weights.add(i * k);
        let up_row = up_weights.add(i * k);

        // 4 accumulators each to hide FMA latency (4-5 cycle latency, 0.5 cycle throughput)
        let mut gate_sum0 = _mm256_setzero_ps();
        let mut gate_sum1 = _mm256_setzero_ps();
        let mut gate_sum2 = _mm256_setzero_ps();
        let mut gate_sum3 = _mm256_setzero_ps();

        let mut up_sum0 = _mm256_setzero_ps();
        let mut up_sum1 = _mm256_setzero_ps();
        let mut up_sum2 = _mm256_setzero_ps();
        let mut up_sum3 = _mm256_setzero_ps();

        let mut a_ptr = input;
        let mut g_ptr = gate_row;
        let mut u_ptr = up_row;
        let mut remaining = k;

        // Main loop: 32 elements at a time (4 × 8-wide vectors)
        while remaining >= 32 {
            // Load 32 floats from input (shared between gate and up)
            let a0 = _mm256_loadu_ps(a_ptr);
            let a1 = _mm256_loadu_ps(a_ptr.add(8));
            let a2 = _mm256_loadu_ps(a_ptr.add(16));
            let a3 = _mm256_loadu_ps(a_ptr.add(24));

            // Load 32 gate weights
            let g0 = _mm256_loadu_ps(g_ptr);
            let g1 = _mm256_loadu_ps(g_ptr.add(8));
            let g2 = _mm256_loadu_ps(g_ptr.add(16));
            let g3 = _mm256_loadu_ps(g_ptr.add(24));

            // Load 32 up weights
            let u0 = _mm256_loadu_ps(u_ptr);
            let u1 = _mm256_loadu_ps(u_ptr.add(8));
            let u2 = _mm256_loadu_ps(u_ptr.add(16));
            let u3 = _mm256_loadu_ps(u_ptr.add(24));

            // FMA for gate: gate_sum += a * g
            gate_sum0 = _mm256_fmadd_ps(a0, g0, gate_sum0);
            gate_sum1 = _mm256_fmadd_ps(a1, g1, gate_sum1);
            gate_sum2 = _mm256_fmadd_ps(a2, g2, gate_sum2);
            gate_sum3 = _mm256_fmadd_ps(a3, g3, gate_sum3);

            // FMA for up: up_sum += a * u
            up_sum0 = _mm256_fmadd_ps(a0, u0, up_sum0);
            up_sum1 = _mm256_fmadd_ps(a1, u1, up_sum1);
            up_sum2 = _mm256_fmadd_ps(a2, u2, up_sum2);
            up_sum3 = _mm256_fmadd_ps(a3, u3, up_sum3);

            a_ptr = a_ptr.add(32);
            g_ptr = g_ptr.add(32);
            u_ptr = u_ptr.add(32);
            remaining -= 32;
        }

        // Secondary loop: 8 elements at a time
        while remaining >= 8 {
            let a = _mm256_loadu_ps(a_ptr);
            let g = _mm256_loadu_ps(g_ptr);
            let u = _mm256_loadu_ps(u_ptr);

            gate_sum0 = _mm256_fmadd_ps(a, g, gate_sum0);
            up_sum0 = _mm256_fmadd_ps(a, u, up_sum0);

            a_ptr = a_ptr.add(8);
            g_ptr = g_ptr.add(8);
            u_ptr = u_ptr.add(8);
            remaining -= 8;
        }

        // Combine the 4 accumulators
        let gate_combined = _mm256_add_ps(
            _mm256_add_ps(gate_sum0, gate_sum1),
            _mm256_add_ps(gate_sum2, gate_sum3),
        );
        let up_combined = _mm256_add_ps(
            _mm256_add_ps(up_sum0, up_sum1),
            _mm256_add_ps(up_sum2, up_sum3),
        );

        // Horizontal sum
        let mut gate_sum = hsum_ps_avx(gate_combined);
        let mut up_sum = hsum_ps_avx(up_combined);

        // Scalar remainder
        while remaining > 0 {
            let val_a = *a_ptr;
            gate_sum += val_a * *g_ptr;
            up_sum += val_a * *u_ptr;
            a_ptr = a_ptr.add(1);
            g_ptr = g_ptr.add(1);
            u_ptr = u_ptr.add(1);
            remaining -= 1;
        }

        // Apply SiLU to gate and multiply with up
        output[i] = silu(gate_sum) * up_sum;
    }
}

/// Parallel wrapper that distributes output neurons across threads.
///
/// Uses rayon to parallelize over chunks of output neurons.
pub fn fused_gate_up_silu_f32_parallel(
    output: &mut [f32],
    input: &[f32],
    gate_weights: &[f32],
    up_weights: &[f32],
    k: usize,
) {
    use rayon::prelude::*;

    let n = output.len();
    let num_threads = rayon::current_num_threads();
    let chunk_size = (n + num_threads - 1) / num_threads;

    output
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start_n = chunk_idx * chunk_size;

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    unsafe {
                        fused_gate_up_silu_f32_avx2(
                            out_chunk,
                            input.as_ptr(),
                            gate_weights.as_ptr().add(start_n * k),
                            up_weights.as_ptr().add(start_n * k),
                            k,
                        );
                    }
                    return;
                }
            }

            // Scalar fallback
            for (i, out) in out_chunk.iter_mut().enumerate() {
                let n_idx = start_n + i;
                let offset = n_idx * k;
                let mut gate_sum = 0.0f32;
                let mut up_sum = 0.0f32;

                for j in 0..k {
                    let val = input[j];
                    gate_sum += val * gate_weights[offset + j];
                    up_sum += val * up_weights[offset + j];
                }

                *out = silu(gate_sum) * up_sum;
            }
        });
}

// =============================================================================
// BF16 Fused SIMD Kernel
// =============================================================================

/// Fused gate+up+silu for BF16 weights using AVX2/FMA.
/// 
/// Reads input once per output neuron, computing both gate and up dot products
/// in the same loop iteration.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fused_gate_up_silu_bf16_avx2(
    output: &mut [f32],
    input: *const f32,
    gate_weights: *const u16,  // bf16 as u16
    up_weights: *const u16,    // bf16 as u16
    k: usize,
) {
    let n = output.len();
    
    for i in 0..n {
        let gate_row = gate_weights.add(i * k);
        let up_row = up_weights.add(i * k);
        
        // 4 accumulators each to hide FMA latency
        let mut gate_sum0 = _mm256_setzero_ps();
        let mut gate_sum1 = _mm256_setzero_ps();
        let mut gate_sum2 = _mm256_setzero_ps();
        let mut gate_sum3 = _mm256_setzero_ps();
        
        let mut up_sum0 = _mm256_setzero_ps();
        let mut up_sum1 = _mm256_setzero_ps();
        let mut up_sum2 = _mm256_setzero_ps();
        let mut up_sum3 = _mm256_setzero_ps();
        
        let mut a_ptr = input;
        let mut g_ptr = gate_row;
        let mut u_ptr = up_row;
        let mut remaining = k;
        
        // Main loop: 32 elements at a time
        while remaining >= 32 {
            // Prefetch next cache lines
            _mm_prefetch(g_ptr.add(128) as *const i8, _MM_HINT_T0);
            _mm_prefetch(u_ptr.add(128) as *const i8, _MM_HINT_T0);
            
            // Load 32 floats from input (shared between gate and up)
            let a0 = _mm256_loadu_ps(a_ptr);
            let a1 = _mm256_loadu_ps(a_ptr.add(8));
            let a2 = _mm256_loadu_ps(a_ptr.add(16));
            let a3 = _mm256_loadu_ps(a_ptr.add(24));
            
            // Load 32 bf16 gate weights and convert to f32
            let g0_u16 = _mm_loadu_si128(g_ptr as *const __m128i);
            let g1_u16 = _mm_loadu_si128(g_ptr.add(8) as *const __m128i);
            let g2_u16 = _mm_loadu_si128(g_ptr.add(16) as *const __m128i);
            let g3_u16 = _mm_loadu_si128(g_ptr.add(24) as *const __m128i);
            
            let g0_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(g0_u16), 16));
            let g1_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(g1_u16), 16));
            let g2_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(g2_u16), 16));
            let g3_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(g3_u16), 16));
            
            // Load 32 bf16 up weights and convert to f32
            let u0_u16 = _mm_loadu_si128(u_ptr as *const __m128i);
            let u1_u16 = _mm_loadu_si128(u_ptr.add(8) as *const __m128i);
            let u2_u16 = _mm_loadu_si128(u_ptr.add(16) as *const __m128i);
            let u3_u16 = _mm_loadu_si128(u_ptr.add(24) as *const __m128i);
            
            let u0_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(u0_u16), 16));
            let u1_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(u1_u16), 16));
            let u2_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(u2_u16), 16));
            let u3_f = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(u3_u16), 16));
            
            // FMA for gate
            gate_sum0 = _mm256_fmadd_ps(a0, g0_f, gate_sum0);
            gate_sum1 = _mm256_fmadd_ps(a1, g1_f, gate_sum1);
            gate_sum2 = _mm256_fmadd_ps(a2, g2_f, gate_sum2);
            gate_sum3 = _mm256_fmadd_ps(a3, g3_f, gate_sum3);
            
            // FMA for up
            up_sum0 = _mm256_fmadd_ps(a0, u0_f, up_sum0);
            up_sum1 = _mm256_fmadd_ps(a1, u1_f, up_sum1);
            up_sum2 = _mm256_fmadd_ps(a2, u2_f, up_sum2);
            up_sum3 = _mm256_fmadd_ps(a3, u3_f, up_sum3);
            
            a_ptr = a_ptr.add(32);
            g_ptr = g_ptr.add(32);
            u_ptr = u_ptr.add(32);
            remaining -= 32;
        }
        
        // Combine accumulators
        let gate_combined = _mm256_add_ps(
            _mm256_add_ps(gate_sum0, gate_sum1),
            _mm256_add_ps(gate_sum2, gate_sum3),
        );
        let up_combined = _mm256_add_ps(
            _mm256_add_ps(up_sum0, up_sum1),
            _mm256_add_ps(up_sum2, up_sum3),
        );
        
        let mut gate_sum = hsum_ps_avx(gate_combined);
        let mut up_sum = hsum_ps_avx(up_combined);
        
        // Scalar remainder
        while remaining > 0 {
            let val_a = *a_ptr;
            let val_g = f32::from_bits((*g_ptr as u32) << 16);
            let val_u = f32::from_bits((*u_ptr as u32) << 16);
            gate_sum += val_a * val_g;
            up_sum += val_a * val_u;
            a_ptr = a_ptr.add(1);
            g_ptr = g_ptr.add(1);
            u_ptr = u_ptr.add(1);
            remaining -= 1;
        }
        
        // Apply SiLU and multiply
        output[i] = silu(gate_sum) * up_sum;
    }
}

// =============================================================================
// Q8_0 Fused SIMD Kernel
// =============================================================================

/// Fused gate+up+silu for Q8_0 weights using AVX2/FMA.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fused_gate_up_silu_q8_0_avx2(
    output: &mut [f32],
    input: *const f32,
    gate_blocks: &[BlockQ8_0],
    up_blocks: &[BlockQ8_0],
    k: usize,
) {
    let n = output.len();
    let num_blocks_per_row = k / 32;
    
    for i in 0..n {
        let gate_row_blocks = &gate_blocks[i * num_blocks_per_row..(i + 1) * num_blocks_per_row];
        let up_row_blocks = &up_blocks[i * num_blocks_per_row..(i + 1) * num_blocks_per_row];
        
        let mut a_ptr = input;
        
        // 4 accumulators each
        let mut gate_sum0 = _mm256_setzero_ps();
        let mut gate_sum1 = _mm256_setzero_ps();
        let mut gate_sum2 = _mm256_setzero_ps();
        let mut gate_sum3 = _mm256_setzero_ps();
        
        let mut up_sum0 = _mm256_setzero_ps();
        let mut up_sum1 = _mm256_setzero_ps();
        let mut up_sum2 = _mm256_setzero_ps();
        let mut up_sum3 = _mm256_setzero_ps();
        
        // Process blocks in groups of 4 (128 elements)
        let mut block_idx = 0;
        while block_idx + 4 <= num_blocks_per_row {
            // --- Block 0 ---
            {
                let g = &gate_row_blocks[block_idx];
                let u = &up_row_blocks[block_idx];
                let g_d = _mm256_set1_ps(g.d.to_f32());
                let u_d = _mm256_set1_ps(u.d.to_f32());
                
                let g_q_ptr = g.qs.as_ptr() as *const __m128i;
                let u_q_ptr = u.qs.as_ptr() as *const __m128i;
                
                let g_lo = _mm_loadu_si128(g_q_ptr);
                let g_hi = _mm_loadu_si128(g_q_ptr.add(1));
                let u_lo = _mm_loadu_si128(u_q_ptr);
                let u_hi = _mm_loadu_si128(u_q_ptr.add(1));
                
                let a0 = _mm256_loadu_ps(a_ptr);
                let a1 = _mm256_loadu_ps(a_ptr.add(8));
                let a2 = _mm256_loadu_ps(a_ptr.add(16));
                let a3 = _mm256_loadu_ps(a_ptr.add(24));
                
                let g0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo));
                let g1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo, 8)));
                let g2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi));
                let g3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi, 8)));
                
                let u0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo));
                let u1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo, 8)));
                let u2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi));
                let u3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi, 8)));
                
                gate_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(g0_f, g_d), a0, gate_sum0);
                gate_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(g1_f, g_d), a1, gate_sum0);
                gate_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(g2_f, g_d), a2, gate_sum0);
                gate_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(g3_f, g_d), a3, gate_sum0);
                
                up_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(u0_f, u_d), a0, up_sum0);
                up_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(u1_f, u_d), a1, up_sum0);
                up_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(u2_f, u_d), a2, up_sum0);
                up_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(u3_f, u_d), a3, up_sum0);
            }
            
            // --- Block 1 ---
            {
                let g = &gate_row_blocks[block_idx + 1];
                let u = &up_row_blocks[block_idx + 1];
                let g_d = _mm256_set1_ps(g.d.to_f32());
                let u_d = _mm256_set1_ps(u.d.to_f32());
                
                let g_q_ptr = g.qs.as_ptr() as *const __m128i;
                let u_q_ptr = u.qs.as_ptr() as *const __m128i;
                
                let g_lo = _mm_loadu_si128(g_q_ptr);
                let g_hi = _mm_loadu_si128(g_q_ptr.add(1));
                let u_lo = _mm_loadu_si128(u_q_ptr);
                let u_hi = _mm_loadu_si128(u_q_ptr.add(1));
                
                let a0 = _mm256_loadu_ps(a_ptr.add(32));
                let a1 = _mm256_loadu_ps(a_ptr.add(40));
                let a2 = _mm256_loadu_ps(a_ptr.add(48));
                let a3 = _mm256_loadu_ps(a_ptr.add(56));
                
                let g0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo));
                let g1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo, 8)));
                let g2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi));
                let g3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi, 8)));
                
                let u0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo));
                let u1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo, 8)));
                let u2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi));
                let u3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi, 8)));
                
                gate_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(g0_f, g_d), a0, gate_sum1);
                gate_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(g1_f, g_d), a1, gate_sum1);
                gate_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(g2_f, g_d), a2, gate_sum1);
                gate_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(g3_f, g_d), a3, gate_sum1);
                
                up_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(u0_f, u_d), a0, up_sum1);
                up_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(u1_f, u_d), a1, up_sum1);
                up_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(u2_f, u_d), a2, up_sum1);
                up_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(u3_f, u_d), a3, up_sum1);
            }
            
            // --- Block 2 ---
            {
                let g = &gate_row_blocks[block_idx + 2];
                let u = &up_row_blocks[block_idx + 2];
                let g_d = _mm256_set1_ps(g.d.to_f32());
                let u_d = _mm256_set1_ps(u.d.to_f32());
                
                let g_q_ptr = g.qs.as_ptr() as *const __m128i;
                let u_q_ptr = u.qs.as_ptr() as *const __m128i;
                
                let g_lo = _mm_loadu_si128(g_q_ptr);
                let g_hi = _mm_loadu_si128(g_q_ptr.add(1));
                let u_lo = _mm_loadu_si128(u_q_ptr);
                let u_hi = _mm_loadu_si128(u_q_ptr.add(1));
                
                let a0 = _mm256_loadu_ps(a_ptr.add(64));
                let a1 = _mm256_loadu_ps(a_ptr.add(72));
                let a2 = _mm256_loadu_ps(a_ptr.add(80));
                let a3 = _mm256_loadu_ps(a_ptr.add(88));
                
                let g0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo));
                let g1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo, 8)));
                let g2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi));
                let g3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi, 8)));
                
                let u0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo));
                let u1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo, 8)));
                let u2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi));
                let u3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi, 8)));
                
                gate_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(g0_f, g_d), a0, gate_sum2);
                gate_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(g1_f, g_d), a1, gate_sum2);
                gate_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(g2_f, g_d), a2, gate_sum2);
                gate_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(g3_f, g_d), a3, gate_sum2);
                
                up_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(u0_f, u_d), a0, up_sum2);
                up_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(u1_f, u_d), a1, up_sum2);
                up_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(u2_f, u_d), a2, up_sum2);
                up_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(u3_f, u_d), a3, up_sum2);
            }
            
            // --- Block 3 ---
            {
                let g = &gate_row_blocks[block_idx + 3];
                let u = &up_row_blocks[block_idx + 3];
                let g_d = _mm256_set1_ps(g.d.to_f32());
                let u_d = _mm256_set1_ps(u.d.to_f32());
                
                let g_q_ptr = g.qs.as_ptr() as *const __m128i;
                let u_q_ptr = u.qs.as_ptr() as *const __m128i;
                
                let g_lo = _mm_loadu_si128(g_q_ptr);
                let g_hi = _mm_loadu_si128(g_q_ptr.add(1));
                let u_lo = _mm_loadu_si128(u_q_ptr);
                let u_hi = _mm_loadu_si128(u_q_ptr.add(1));
                
                let a0 = _mm256_loadu_ps(a_ptr.add(96));
                let a1 = _mm256_loadu_ps(a_ptr.add(104));
                let a2 = _mm256_loadu_ps(a_ptr.add(112));
                let a3 = _mm256_loadu_ps(a_ptr.add(120));
                
                let g0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo));
                let g1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo, 8)));
                let g2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi));
                let g3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi, 8)));
                
                let u0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo));
                let u1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo, 8)));
                let u2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi));
                let u3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi, 8)));
                
                gate_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(g0_f, g_d), a0, gate_sum3);
                gate_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(g1_f, g_d), a1, gate_sum3);
                gate_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(g2_f, g_d), a2, gate_sum3);
                gate_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(g3_f, g_d), a3, gate_sum3);
                
                up_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(u0_f, u_d), a0, up_sum3);
                up_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(u1_f, u_d), a1, up_sum3);
                up_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(u2_f, u_d), a2, up_sum3);
                up_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(u3_f, u_d), a3, up_sum3);
            }
            
            a_ptr = a_ptr.add(128);
            block_idx += 4;
        }
        
        // Handle remainder blocks
        while block_idx < num_blocks_per_row {
            let g = &gate_row_blocks[block_idx];
            let u = &up_row_blocks[block_idx];
            let g_d = _mm256_set1_ps(g.d.to_f32());
            let u_d = _mm256_set1_ps(u.d.to_f32());
            
            let g_q_ptr = g.qs.as_ptr() as *const __m128i;
            let u_q_ptr = u.qs.as_ptr() as *const __m128i;
            
            let g_lo = _mm_loadu_si128(g_q_ptr);
            let g_hi = _mm_loadu_si128(g_q_ptr.add(1));
            let u_lo = _mm_loadu_si128(u_q_ptr);
            let u_hi = _mm_loadu_si128(u_q_ptr.add(1));
            
            let a0 = _mm256_loadu_ps(a_ptr);
            let a1 = _mm256_loadu_ps(a_ptr.add(8));
            let a2 = _mm256_loadu_ps(a_ptr.add(16));
            let a3 = _mm256_loadu_ps(a_ptr.add(24));
            
            let g0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo));
            let g1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo, 8)));
            let g2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi));
            let g3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi, 8)));
            
            let u0_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo));
            let u1_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo, 8)));
            let u2_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi));
            let u3_f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi, 8)));
            
            gate_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(g0_f, g_d), a0, gate_sum0);
            gate_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(g1_f, g_d), a1, gate_sum0);
            gate_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(g2_f, g_d), a2, gate_sum0);
            gate_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(g3_f, g_d), a3, gate_sum0);
            
            up_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(u0_f, u_d), a0, up_sum0);
            up_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(u1_f, u_d), a1, up_sum0);
            up_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(u2_f, u_d), a2, up_sum0);
            up_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(u3_f, u_d), a3, up_sum0);
            
            a_ptr = a_ptr.add(32);
            block_idx += 1;
        }
        
        // Combine accumulators
        let gate_total = _mm256_add_ps(
            _mm256_add_ps(gate_sum0, gate_sum1),
            _mm256_add_ps(gate_sum2, gate_sum3),
        );
        let up_total = _mm256_add_ps(
            _mm256_add_ps(up_sum0, up_sum1),
            _mm256_add_ps(up_sum2, up_sum3),
        );
        
        let gate_sum = hsum_ps_avx(gate_total);
        let up_sum = hsum_ps_avx(up_total);
        
        output[i] = silu(gate_sum) * up_sum;
    }
}

// =============================================================================
// Q4_K Fused SIMD Kernel
// =============================================================================

/// Fused gate+up+silu for Q4_K weights using AVX2/FMA.
/// Mirrors the structure of matmul_vec_q4_k_avx2 but does two dot products.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fused_gate_up_silu_q4_k_avx2(
    output: &mut [f32],
    input: *const f32,
    gate_blocks: &[BlockQ4_K],
    up_blocks: &[BlockQ4_K],
    k: usize,
) {
    let n = output.len();
    let num_blocks_per_row = k / 256;
    
    for i in 0..n {
        let g_blocks = &gate_blocks[i * num_blocks_per_row..(i + 1) * num_blocks_per_row];
        let u_blocks = &up_blocks[i * num_blocks_per_row..(i + 1) * num_blocks_per_row];
        
        let mut g_acc0 = _mm256_setzero_ps();
        let mut g_acc1 = _mm256_setzero_ps();
        let mut g_acc2 = _mm256_setzero_ps();
        let mut g_acc3 = _mm256_setzero_ps();
        
        let mut u_acc0 = _mm256_setzero_ps();
        let mut u_acc1 = _mm256_setzero_ps();
        let mut u_acc2 = _mm256_setzero_ps();
        let mut u_acc3 = _mm256_setzero_ps();
        
        let mut a_ptr = input;
        
        for (g_block, u_block) in g_blocks.iter().zip(u_blocks.iter()) {
            let g_d = _mm256_set1_ps(g_block.d.to_f32());
            let g_dmin = _mm256_set1_ps(g_block.dmin.to_f32());
            let u_d = _mm256_set1_ps(u_block.d.to_f32());
            let u_dmin = _mm256_set1_ps(u_block.dmin.to_f32());
            
            let g_qs = g_block.qs.as_ptr();
            let u_qs = u_block.qs.as_ptr();
            
            // Process 4 parts of 64 elements each (256 total)
            // PART 0: bytes 0-31, weights 0-63
            {
                let (g_sc1, g_m1) = get_scale_min_k4(0, &g_block.scales);
                let (g_sc2, g_m2) = get_scale_min_k4(1, &g_block.scales);
                let (u_sc1, u_m1) = get_scale_min_k4(0, &u_block.scales);
                let (u_sc2, u_m2) = get_scale_min_k4(1, &u_block.scales);
                
                let g_scale1 = _mm256_mul_ps(_mm256_set1_ps(g_sc1 as f32), g_d);
                let g_min1 = _mm256_mul_ps(_mm256_set1_ps(g_m1 as f32), g_dmin);
                let g_scale2 = _mm256_mul_ps(_mm256_set1_ps(g_sc2 as f32), g_d);
                let g_min2 = _mm256_mul_ps(_mm256_set1_ps(g_m2 as f32), g_dmin);
                
                let u_scale1 = _mm256_mul_ps(_mm256_set1_ps(u_sc1 as f32), u_d);
                let u_min1 = _mm256_mul_ps(_mm256_set1_ps(u_m1 as f32), u_dmin);
                let u_scale2 = _mm256_mul_ps(_mm256_set1_ps(u_sc2 as f32), u_d);
                let u_min2 = _mm256_mul_ps(_mm256_set1_ps(u_m2 as f32), u_dmin);
                
                let g_q1 = _mm_loadu_si128(g_qs as *const __m128i);
                let g_q2 = _mm_loadu_si128(g_qs.add(16) as *const __m128i);
                let u_q1 = _mm_loadu_si128(u_qs as *const __m128i);
                let u_q2 = _mm_loadu_si128(u_qs.add(16) as *const __m128i);
                
                // Load input
                let a0 = _mm256_loadu_ps(a_ptr);
                let a1 = _mm256_loadu_ps(a_ptr.add(8));
                let a2 = _mm256_loadu_ps(a_ptr.add(16));
                let a3 = _mm256_loadu_ps(a_ptr.add(24));
                let a4 = _mm256_loadu_ps(a_ptr.add(32));
                let a5 = _mm256_loadu_ps(a_ptr.add(40));
                let a6 = _mm256_loadu_ps(a_ptr.add(48));
                let a7 = _mm256_loadu_ps(a_ptr.add(56));
                
                let mask_lo = _mm_set1_epi8(0x0F);
                
                // Low nibbles (elements 0-31)
                let g_lo1 = _mm_and_si128(g_q1, mask_lo);
                let g_lo2 = _mm_and_si128(g_q2, mask_lo);
                let u_lo1 = _mm_and_si128(u_q1, mask_lo);
                let u_lo2 = _mm_and_si128(u_q2, mask_lo);
                
                let g_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo1));
                let g_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo1, 8)));
                let g_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo2));
                let g_f4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo2, 8)));
                
                let u_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo1));
                let u_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo1, 8)));
                let u_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo2));
                let u_f4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo2, 8)));
                
                g_acc0 = _mm256_fmadd_ps(a0, _mm256_sub_ps(_mm256_mul_ps(g_f1, g_scale1), g_min1), g_acc0);
                g_acc0 = _mm256_fmadd_ps(a1, _mm256_sub_ps(_mm256_mul_ps(g_f2, g_scale1), g_min1), g_acc0);
                g_acc0 = _mm256_fmadd_ps(a2, _mm256_sub_ps(_mm256_mul_ps(g_f3, g_scale1), g_min1), g_acc0);
                g_acc0 = _mm256_fmadd_ps(a3, _mm256_sub_ps(_mm256_mul_ps(g_f4, g_scale1), g_min1), g_acc0);
                
                u_acc0 = _mm256_fmadd_ps(a0, _mm256_sub_ps(_mm256_mul_ps(u_f1, u_scale1), u_min1), u_acc0);
                u_acc0 = _mm256_fmadd_ps(a1, _mm256_sub_ps(_mm256_mul_ps(u_f2, u_scale1), u_min1), u_acc0);
                u_acc0 = _mm256_fmadd_ps(a2, _mm256_sub_ps(_mm256_mul_ps(u_f3, u_scale1), u_min1), u_acc0);
                u_acc0 = _mm256_fmadd_ps(a3, _mm256_sub_ps(_mm256_mul_ps(u_f4, u_scale1), u_min1), u_acc0);
                
                // High nibbles (elements 32-63)
                let g_hi1 = _mm_and_si128(_mm_srli_epi16(g_q1, 4), mask_lo);
                let g_hi2 = _mm_and_si128(_mm_srli_epi16(g_q2, 4), mask_lo);
                let u_hi1 = _mm_and_si128(_mm_srli_epi16(u_q1, 4), mask_lo);
                let u_hi2 = _mm_and_si128(_mm_srli_epi16(u_q2, 4), mask_lo);
                
                let g_f5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi1));
                let g_f6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi1, 8)));
                let g_f7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi2));
                let g_f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi2, 8)));
                
                let u_f5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi1));
                let u_f6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi1, 8)));
                let u_f7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi2));
                let u_f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi2, 8)));
                
                g_acc0 = _mm256_fmadd_ps(a4, _mm256_sub_ps(_mm256_mul_ps(g_f5, g_scale2), g_min2), g_acc0);
                g_acc0 = _mm256_fmadd_ps(a5, _mm256_sub_ps(_mm256_mul_ps(g_f6, g_scale2), g_min2), g_acc0);
                g_acc0 = _mm256_fmadd_ps(a6, _mm256_sub_ps(_mm256_mul_ps(g_f7, g_scale2), g_min2), g_acc0);
                g_acc0 = _mm256_fmadd_ps(a7, _mm256_sub_ps(_mm256_mul_ps(g_f8, g_scale2), g_min2), g_acc0);
                
                u_acc0 = _mm256_fmadd_ps(a4, _mm256_sub_ps(_mm256_mul_ps(u_f5, u_scale2), u_min2), u_acc0);
                u_acc0 = _mm256_fmadd_ps(a5, _mm256_sub_ps(_mm256_mul_ps(u_f6, u_scale2), u_min2), u_acc0);
                u_acc0 = _mm256_fmadd_ps(a6, _mm256_sub_ps(_mm256_mul_ps(u_f7, u_scale2), u_min2), u_acc0);
                u_acc0 = _mm256_fmadd_ps(a7, _mm256_sub_ps(_mm256_mul_ps(u_f8, u_scale2), u_min2), u_acc0);
            }
            
            // PART 1: bytes 32-63, weights 64-127
            {
                let (g_sc1, g_m1) = get_scale_min_k4(2, &g_block.scales);
                let (g_sc2, g_m2) = get_scale_min_k4(3, &g_block.scales);
                let (u_sc1, u_m1) = get_scale_min_k4(2, &u_block.scales);
                let (u_sc2, u_m2) = get_scale_min_k4(3, &u_block.scales);
                
                let g_scale1 = _mm256_mul_ps(_mm256_set1_ps(g_sc1 as f32), g_d);
                let g_min1 = _mm256_mul_ps(_mm256_set1_ps(g_m1 as f32), g_dmin);
                let g_scale2 = _mm256_mul_ps(_mm256_set1_ps(g_sc2 as f32), g_d);
                let g_min2 = _mm256_mul_ps(_mm256_set1_ps(g_m2 as f32), g_dmin);
                
                let u_scale1 = _mm256_mul_ps(_mm256_set1_ps(u_sc1 as f32), u_d);
                let u_min1 = _mm256_mul_ps(_mm256_set1_ps(u_m1 as f32), u_dmin);
                let u_scale2 = _mm256_mul_ps(_mm256_set1_ps(u_sc2 as f32), u_d);
                let u_min2 = _mm256_mul_ps(_mm256_set1_ps(u_m2 as f32), u_dmin);
                
                let g_q1 = _mm_loadu_si128(g_qs.add(32) as *const __m128i);
                let g_q2 = _mm_loadu_si128(g_qs.add(48) as *const __m128i);
                let u_q1 = _mm_loadu_si128(u_qs.add(32) as *const __m128i);
                let u_q2 = _mm_loadu_si128(u_qs.add(48) as *const __m128i);
                
                let a_base = a_ptr.add(64);
                let a0 = _mm256_loadu_ps(a_base);
                let a1 = _mm256_loadu_ps(a_base.add(8));
                let a2 = _mm256_loadu_ps(a_base.add(16));
                let a3 = _mm256_loadu_ps(a_base.add(24));
                let a4 = _mm256_loadu_ps(a_base.add(32));
                let a5 = _mm256_loadu_ps(a_base.add(40));
                let a6 = _mm256_loadu_ps(a_base.add(48));
                let a7 = _mm256_loadu_ps(a_base.add(56));
                
                let mask_lo = _mm_set1_epi8(0x0F);
                
                // Low nibbles
                let g_lo1 = _mm_and_si128(g_q1, mask_lo);
                let g_lo2 = _mm_and_si128(g_q2, mask_lo);
                let u_lo1 = _mm_and_si128(u_q1, mask_lo);
                let u_lo2 = _mm_and_si128(u_q2, mask_lo);
                
                let g_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo1));
                let g_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo1, 8)));
                let g_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo2));
                let g_f4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo2, 8)));
                
                let u_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo1));
                let u_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo1, 8)));
                let u_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo2));
                let u_f4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo2, 8)));
                
                g_acc1 = _mm256_fmadd_ps(a0, _mm256_sub_ps(_mm256_mul_ps(g_f1, g_scale1), g_min1), g_acc1);
                g_acc1 = _mm256_fmadd_ps(a1, _mm256_sub_ps(_mm256_mul_ps(g_f2, g_scale1), g_min1), g_acc1);
                g_acc1 = _mm256_fmadd_ps(a2, _mm256_sub_ps(_mm256_mul_ps(g_f3, g_scale1), g_min1), g_acc1);
                g_acc1 = _mm256_fmadd_ps(a3, _mm256_sub_ps(_mm256_mul_ps(g_f4, g_scale1), g_min1), g_acc1);
                
                u_acc1 = _mm256_fmadd_ps(a0, _mm256_sub_ps(_mm256_mul_ps(u_f1, u_scale1), u_min1), u_acc1);
                u_acc1 = _mm256_fmadd_ps(a1, _mm256_sub_ps(_mm256_mul_ps(u_f2, u_scale1), u_min1), u_acc1);
                u_acc1 = _mm256_fmadd_ps(a2, _mm256_sub_ps(_mm256_mul_ps(u_f3, u_scale1), u_min1), u_acc1);
                u_acc1 = _mm256_fmadd_ps(a3, _mm256_sub_ps(_mm256_mul_ps(u_f4, u_scale1), u_min1), u_acc1);
                
                // High nibbles
                let g_hi1 = _mm_and_si128(_mm_srli_epi16(g_q1, 4), mask_lo);
                let g_hi2 = _mm_and_si128(_mm_srli_epi16(g_q2, 4), mask_lo);
                let u_hi1 = _mm_and_si128(_mm_srli_epi16(u_q1, 4), mask_lo);
                let u_hi2 = _mm_and_si128(_mm_srli_epi16(u_q2, 4), mask_lo);
                
                let g_f5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi1));
                let g_f6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi1, 8)));
                let g_f7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi2));
                let g_f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi2, 8)));
                
                let u_f5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi1));
                let u_f6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi1, 8)));
                let u_f7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi2));
                let u_f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi2, 8)));
                
                g_acc1 = _mm256_fmadd_ps(a4, _mm256_sub_ps(_mm256_mul_ps(g_f5, g_scale2), g_min2), g_acc1);
                g_acc1 = _mm256_fmadd_ps(a5, _mm256_sub_ps(_mm256_mul_ps(g_f6, g_scale2), g_min2), g_acc1);
                g_acc1 = _mm256_fmadd_ps(a6, _mm256_sub_ps(_mm256_mul_ps(g_f7, g_scale2), g_min2), g_acc1);
                g_acc1 = _mm256_fmadd_ps(a7, _mm256_sub_ps(_mm256_mul_ps(g_f8, g_scale2), g_min2), g_acc1);
                
                u_acc1 = _mm256_fmadd_ps(a4, _mm256_sub_ps(_mm256_mul_ps(u_f5, u_scale2), u_min2), u_acc1);
                u_acc1 = _mm256_fmadd_ps(a5, _mm256_sub_ps(_mm256_mul_ps(u_f6, u_scale2), u_min2), u_acc1);
                u_acc1 = _mm256_fmadd_ps(a6, _mm256_sub_ps(_mm256_mul_ps(u_f7, u_scale2), u_min2), u_acc1);
                u_acc1 = _mm256_fmadd_ps(a7, _mm256_sub_ps(_mm256_mul_ps(u_f8, u_scale2), u_min2), u_acc1);
            }
            
            // PART 2: bytes 64-95, weights 128-191
            {
                let (g_sc1, g_m1) = get_scale_min_k4(4, &g_block.scales);
                let (g_sc2, g_m2) = get_scale_min_k4(5, &g_block.scales);
                let (u_sc1, u_m1) = get_scale_min_k4(4, &u_block.scales);
                let (u_sc2, u_m2) = get_scale_min_k4(5, &u_block.scales);
                
                let g_scale1 = _mm256_mul_ps(_mm256_set1_ps(g_sc1 as f32), g_d);
                let g_min1 = _mm256_mul_ps(_mm256_set1_ps(g_m1 as f32), g_dmin);
                let g_scale2 = _mm256_mul_ps(_mm256_set1_ps(g_sc2 as f32), g_d);
                let g_min2 = _mm256_mul_ps(_mm256_set1_ps(g_m2 as f32), g_dmin);
                
                let u_scale1 = _mm256_mul_ps(_mm256_set1_ps(u_sc1 as f32), u_d);
                let u_min1 = _mm256_mul_ps(_mm256_set1_ps(u_m1 as f32), u_dmin);
                let u_scale2 = _mm256_mul_ps(_mm256_set1_ps(u_sc2 as f32), u_d);
                let u_min2 = _mm256_mul_ps(_mm256_set1_ps(u_m2 as f32), u_dmin);
                
                let g_q1 = _mm_loadu_si128(g_qs.add(64) as *const __m128i);
                let g_q2 = _mm_loadu_si128(g_qs.add(80) as *const __m128i);
                let u_q1 = _mm_loadu_si128(u_qs.add(64) as *const __m128i);
                let u_q2 = _mm_loadu_si128(u_qs.add(80) as *const __m128i);
                
                let a_base = a_ptr.add(128);
                let a0 = _mm256_loadu_ps(a_base);
                let a1 = _mm256_loadu_ps(a_base.add(8));
                let a2 = _mm256_loadu_ps(a_base.add(16));
                let a3 = _mm256_loadu_ps(a_base.add(24));
                let a4 = _mm256_loadu_ps(a_base.add(32));
                let a5 = _mm256_loadu_ps(a_base.add(40));
                let a6 = _mm256_loadu_ps(a_base.add(48));
                let a7 = _mm256_loadu_ps(a_base.add(56));
                
                let mask_lo = _mm_set1_epi8(0x0F);
                
                let g_lo1 = _mm_and_si128(g_q1, mask_lo);
                let g_lo2 = _mm_and_si128(g_q2, mask_lo);
                let u_lo1 = _mm_and_si128(u_q1, mask_lo);
                let u_lo2 = _mm_and_si128(u_q2, mask_lo);
                
                let g_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo1));
                let g_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo1, 8)));
                let g_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo2));
                let g_f4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo2, 8)));
                
                let u_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo1));
                let u_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo1, 8)));
                let u_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo2));
                let u_f4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo2, 8)));
                
                g_acc2 = _mm256_fmadd_ps(a0, _mm256_sub_ps(_mm256_mul_ps(g_f1, g_scale1), g_min1), g_acc2);
                g_acc2 = _mm256_fmadd_ps(a1, _mm256_sub_ps(_mm256_mul_ps(g_f2, g_scale1), g_min1), g_acc2);
                g_acc2 = _mm256_fmadd_ps(a2, _mm256_sub_ps(_mm256_mul_ps(g_f3, g_scale1), g_min1), g_acc2);
                g_acc2 = _mm256_fmadd_ps(a3, _mm256_sub_ps(_mm256_mul_ps(g_f4, g_scale1), g_min1), g_acc2);
                
                u_acc2 = _mm256_fmadd_ps(a0, _mm256_sub_ps(_mm256_mul_ps(u_f1, u_scale1), u_min1), u_acc2);
                u_acc2 = _mm256_fmadd_ps(a1, _mm256_sub_ps(_mm256_mul_ps(u_f2, u_scale1), u_min1), u_acc2);
                u_acc2 = _mm256_fmadd_ps(a2, _mm256_sub_ps(_mm256_mul_ps(u_f3, u_scale1), u_min1), u_acc2);
                u_acc2 = _mm256_fmadd_ps(a3, _mm256_sub_ps(_mm256_mul_ps(u_f4, u_scale1), u_min1), u_acc2);
                
                let g_hi1 = _mm_and_si128(_mm_srli_epi16(g_q1, 4), mask_lo);
                let g_hi2 = _mm_and_si128(_mm_srli_epi16(g_q2, 4), mask_lo);
                let u_hi1 = _mm_and_si128(_mm_srli_epi16(u_q1, 4), mask_lo);
                let u_hi2 = _mm_and_si128(_mm_srli_epi16(u_q2, 4), mask_lo);
                
                let g_f5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi1));
                let g_f6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi1, 8)));
                let g_f7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi2));
                let g_f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi2, 8)));
                
                let u_f5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi1));
                let u_f6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi1, 8)));
                let u_f7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi2));
                let u_f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi2, 8)));
                
                g_acc2 = _mm256_fmadd_ps(a4, _mm256_sub_ps(_mm256_mul_ps(g_f5, g_scale2), g_min2), g_acc2);
                g_acc2 = _mm256_fmadd_ps(a5, _mm256_sub_ps(_mm256_mul_ps(g_f6, g_scale2), g_min2), g_acc2);
                g_acc2 = _mm256_fmadd_ps(a6, _mm256_sub_ps(_mm256_mul_ps(g_f7, g_scale2), g_min2), g_acc2);
                g_acc2 = _mm256_fmadd_ps(a7, _mm256_sub_ps(_mm256_mul_ps(g_f8, g_scale2), g_min2), g_acc2);
                
                u_acc2 = _mm256_fmadd_ps(a4, _mm256_sub_ps(_mm256_mul_ps(u_f5, u_scale2), u_min2), u_acc2);
                u_acc2 = _mm256_fmadd_ps(a5, _mm256_sub_ps(_mm256_mul_ps(u_f6, u_scale2), u_min2), u_acc2);
                u_acc2 = _mm256_fmadd_ps(a6, _mm256_sub_ps(_mm256_mul_ps(u_f7, u_scale2), u_min2), u_acc2);
                u_acc2 = _mm256_fmadd_ps(a7, _mm256_sub_ps(_mm256_mul_ps(u_f8, u_scale2), u_min2), u_acc2);
            }
            
            // PART 3: bytes 96-127, weights 192-255
            {
                let (g_sc1, g_m1) = get_scale_min_k4(6, &g_block.scales);
                let (g_sc2, g_m2) = get_scale_min_k4(7, &g_block.scales);
                let (u_sc1, u_m1) = get_scale_min_k4(6, &u_block.scales);
                let (u_sc2, u_m2) = get_scale_min_k4(7, &u_block.scales);
                
                let g_scale1 = _mm256_mul_ps(_mm256_set1_ps(g_sc1 as f32), g_d);
                let g_min1 = _mm256_mul_ps(_mm256_set1_ps(g_m1 as f32), g_dmin);
                let g_scale2 = _mm256_mul_ps(_mm256_set1_ps(g_sc2 as f32), g_d);
                let g_min2 = _mm256_mul_ps(_mm256_set1_ps(g_m2 as f32), g_dmin);
                
                let u_scale1 = _mm256_mul_ps(_mm256_set1_ps(u_sc1 as f32), u_d);
                let u_min1 = _mm256_mul_ps(_mm256_set1_ps(u_m1 as f32), u_dmin);
                let u_scale2 = _mm256_mul_ps(_mm256_set1_ps(u_sc2 as f32), u_d);
                let u_min2 = _mm256_mul_ps(_mm256_set1_ps(u_m2 as f32), u_dmin);
                
                let g_q1 = _mm_loadu_si128(g_qs.add(96) as *const __m128i);
                let g_q2 = _mm_loadu_si128(g_qs.add(112) as *const __m128i);
                let u_q1 = _mm_loadu_si128(u_qs.add(96) as *const __m128i);
                let u_q2 = _mm_loadu_si128(u_qs.add(112) as *const __m128i);
                
                let a_base = a_ptr.add(192);
                let a0 = _mm256_loadu_ps(a_base);
                let a1 = _mm256_loadu_ps(a_base.add(8));
                let a2 = _mm256_loadu_ps(a_base.add(16));
                let a3 = _mm256_loadu_ps(a_base.add(24));
                let a4 = _mm256_loadu_ps(a_base.add(32));
                let a5 = _mm256_loadu_ps(a_base.add(40));
                let a6 = _mm256_loadu_ps(a_base.add(48));
                let a7 = _mm256_loadu_ps(a_base.add(56));
                
                let mask_lo = _mm_set1_epi8(0x0F);
                
                let g_lo1 = _mm_and_si128(g_q1, mask_lo);
                let g_lo2 = _mm_and_si128(g_q2, mask_lo);
                let u_lo1 = _mm_and_si128(u_q1, mask_lo);
                let u_lo2 = _mm_and_si128(u_q2, mask_lo);
                
                let g_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo1));
                let g_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo1, 8)));
                let g_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_lo2));
                let g_f4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_lo2, 8)));
                
                let u_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo1));
                let u_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo1, 8)));
                let u_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_lo2));
                let u_f4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_lo2, 8)));
                
                g_acc3 = _mm256_fmadd_ps(a0, _mm256_sub_ps(_mm256_mul_ps(g_f1, g_scale1), g_min1), g_acc3);
                g_acc3 = _mm256_fmadd_ps(a1, _mm256_sub_ps(_mm256_mul_ps(g_f2, g_scale1), g_min1), g_acc3);
                g_acc3 = _mm256_fmadd_ps(a2, _mm256_sub_ps(_mm256_mul_ps(g_f3, g_scale1), g_min1), g_acc3);
                g_acc3 = _mm256_fmadd_ps(a3, _mm256_sub_ps(_mm256_mul_ps(g_f4, g_scale1), g_min1), g_acc3);
                
                u_acc3 = _mm256_fmadd_ps(a0, _mm256_sub_ps(_mm256_mul_ps(u_f1, u_scale1), u_min1), u_acc3);
                u_acc3 = _mm256_fmadd_ps(a1, _mm256_sub_ps(_mm256_mul_ps(u_f2, u_scale1), u_min1), u_acc3);
                u_acc3 = _mm256_fmadd_ps(a2, _mm256_sub_ps(_mm256_mul_ps(u_f3, u_scale1), u_min1), u_acc3);
                u_acc3 = _mm256_fmadd_ps(a3, _mm256_sub_ps(_mm256_mul_ps(u_f4, u_scale1), u_min1), u_acc3);
                
                let g_hi1 = _mm_and_si128(_mm_srli_epi16(g_q1, 4), mask_lo);
                let g_hi2 = _mm_and_si128(_mm_srli_epi16(g_q2, 4), mask_lo);
                let u_hi1 = _mm_and_si128(_mm_srli_epi16(u_q1, 4), mask_lo);
                let u_hi2 = _mm_and_si128(_mm_srli_epi16(u_q2, 4), mask_lo);
                
                let g_f5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi1));
                let g_f6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi1, 8)));
                let g_f7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_hi2));
                let g_f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_hi2, 8)));
                
                let u_f5 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi1));
                let u_f6 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi1, 8)));
                let u_f7 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_hi2));
                let u_f8 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_hi2, 8)));
                
                g_acc3 = _mm256_fmadd_ps(a4, _mm256_sub_ps(_mm256_mul_ps(g_f5, g_scale2), g_min2), g_acc3);
                g_acc3 = _mm256_fmadd_ps(a5, _mm256_sub_ps(_mm256_mul_ps(g_f6, g_scale2), g_min2), g_acc3);
                g_acc3 = _mm256_fmadd_ps(a6, _mm256_sub_ps(_mm256_mul_ps(g_f7, g_scale2), g_min2), g_acc3);
                g_acc3 = _mm256_fmadd_ps(a7, _mm256_sub_ps(_mm256_mul_ps(g_f8, g_scale2), g_min2), g_acc3);
                
                u_acc3 = _mm256_fmadd_ps(a4, _mm256_sub_ps(_mm256_mul_ps(u_f5, u_scale2), u_min2), u_acc3);
                u_acc3 = _mm256_fmadd_ps(a5, _mm256_sub_ps(_mm256_mul_ps(u_f6, u_scale2), u_min2), u_acc3);
                u_acc3 = _mm256_fmadd_ps(a6, _mm256_sub_ps(_mm256_mul_ps(u_f7, u_scale2), u_min2), u_acc3);
                u_acc3 = _mm256_fmadd_ps(a7, _mm256_sub_ps(_mm256_mul_ps(u_f8, u_scale2), u_min2), u_acc3);
            }
            
            a_ptr = a_ptr.add(256);
        }
        
        // Combine accumulators
        let g_total = _mm256_add_ps(
            _mm256_add_ps(g_acc0, g_acc1),
            _mm256_add_ps(g_acc2, g_acc3),
        );
        let u_total = _mm256_add_ps(
            _mm256_add_ps(u_acc0, u_acc1),
            _mm256_add_ps(u_acc2, u_acc3),
        );
        
        let gate_sum = hsum_ps_avx(g_total);
        let up_sum = hsum_ps_avx(u_total);
        
        output[i] = silu(gate_sum) * up_sum;
    }
}

// =============================================================================
// Parallel Wrappers (for decode - parallelize over output neurons)
// =============================================================================

/// Parallel BF16 fused kernel - distributes output neurons across threads.
pub fn fused_gate_up_silu_bf16_parallel(
    output: &mut [f32],
    input: &[f32],
    gate_weights: &[bf16],
    up_weights: &[bf16],
    k: usize,
) {
    use rayon::prelude::*;
    
    let n = output.len();
    let num_threads = rayon::current_num_threads();
    let chunk_size = (n + num_threads - 1) / num_threads;
    
    output
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start_n = chunk_idx * chunk_size;
            
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    unsafe {
                        fused_gate_up_silu_bf16_avx2(
                            out_chunk,
                            input.as_ptr(),
                            gate_weights.as_ptr().add(start_n * k) as *const u16,
                            up_weights.as_ptr().add(start_n * k) as *const u16,
                            k,
                        );
                    }
                    return;
                }
            }
            
            // Scalar fallback
            for (i, out) in out_chunk.iter_mut().enumerate() {
                let n_idx = start_n + i;
                let offset = n_idx * k;
                let mut gate_sum = 0.0f32;
                let mut up_sum = 0.0f32;
                
                for j in 0..k {
                    let val = input[j];
                    gate_sum += val * gate_weights[offset + j].to_f32();
                    up_sum += val * up_weights[offset + j].to_f32();
                }
                
                *out = silu(gate_sum) * up_sum;
            }
        });
}

/// Parallel Q8_0 fused kernel.
pub fn fused_gate_up_silu_q8_0_parallel(
    output: &mut [f32],
    input: &[f32],
    gate_blocks: &[BlockQ8_0],
    up_blocks: &[BlockQ8_0],
    k: usize,
) {
    use rayon::prelude::*;
    
    let n = output.len();
    let num_threads = rayon::current_num_threads();
    let chunk_size = (n + num_threads - 1) / num_threads;
    let blocks_per_row = k / 32;
    
    output
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start_n = chunk_idx * chunk_size;
            let g_blocks_start = start_n * blocks_per_row;
            let u_blocks_start = start_n * blocks_per_row;
            let num_rows = out_chunk.len();
            
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    unsafe {
                        fused_gate_up_silu_q8_0_avx2(
                            out_chunk,
                            input.as_ptr(),
                            &gate_blocks[g_blocks_start..g_blocks_start + num_rows * blocks_per_row],
                            &up_blocks[u_blocks_start..u_blocks_start + num_rows * blocks_per_row],
                            k,
                        );
                    }
                    return;
                }
            }
            
            // Scalar fallback
            for (i, out) in out_chunk.iter_mut().enumerate() {
                let row_offset = i * blocks_per_row;
                let mut gate_sum = 0.0f32;
                let mut up_sum = 0.0f32;
                
                for b in 0..blocks_per_row {
                    let g = &gate_blocks[g_blocks_start + row_offset + b];
                    let u = &up_blocks[u_blocks_start + row_offset + b];
                    let g_scale = g.d.to_f32();
                    let u_scale = u.d.to_f32();
                    let in_offset = b * 32;
                    
                    for j in 0..32 {
                        let val = input[in_offset + j];
                        gate_sum += val * (g.qs[j] as f32) * g_scale;
                        up_sum += val * (u.qs[j] as f32) * u_scale;
                    }
                }
                
                *out = silu(gate_sum) * up_sum;
            }
        });
}

// =============================================================================
// Q6_K Fused SIMD Kernel
// =============================================================================

/// Fused gate+up+silu for Q6_K weights using AVX2/FMA.
/// Mirrors the structure of matmul_vec_q6_k_avx2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn fused_gate_up_silu_q6_k_avx2(
    output: &mut [f32],
    input: *const f32,
    gate_blocks: &[BlockQ6_K],
    up_blocks: &[BlockQ6_K],
    k: usize,
) {
    let n = output.len();
    let num_blocks_per_row = k / 256;
    
    let mask_low4 = _mm256_set1_epi8(0x0F);
    let mask_low2 = _mm256_set1_epi8(0x03);
    let m32 = _mm256_set1_epi8(32);
    
    for i in 0..n {
        let g_blocks = &gate_blocks[i * num_blocks_per_row..(i + 1) * num_blocks_per_row];
        let u_blocks = &up_blocks[i * num_blocks_per_row..(i + 1) * num_blocks_per_row];
        
        let mut gate_total = 0.0f32;
        let mut up_total = 0.0f32;
        let mut a_ptr = input;
        
        for (g_block, u_block) in g_blocks.iter().zip(u_blocks.iter()) {
            let g_d_val = g_block.d.to_f32();
            let u_d_val = u_block.d.to_f32();
            let g_d_vec = _mm256_set1_ps(g_d_val);
            let u_d_vec = _mm256_set1_ps(u_d_val);
            
            let mut g_sum0 = _mm256_setzero_ps();
            let mut g_sum1 = _mm256_setzero_ps();
            let mut g_sum2 = _mm256_setzero_ps();
            let mut g_sum3 = _mm256_setzero_ps();
            
            let mut u_sum0 = _mm256_setzero_ps();
            let mut u_sum1 = _mm256_setzero_ps();
            let mut u_sum2 = _mm256_setzero_ps();
            let mut u_sum3 = _mm256_setzero_ps();
            
            let g_ql = g_block.ql.as_ptr();
            let g_qh = g_block.qh.as_ptr();
            let g_scales = g_block.scales.as_ptr();
            let u_ql = u_block.ql.as_ptr();
            let u_qh = u_block.qh.as_ptr();
            let u_scales = u_block.scales.as_ptr();
            
            // Process 2 halves (0..128 and 128..256)
            for is in 0..2 {
                let qh_base = is * 32;
                let g_qh_vec = _mm256_loadu_si256(g_qh.add(qh_base) as *const __m256i);
                let u_qh_vec = _mm256_loadu_si256(u_qh.add(qh_base) as *const __m256i);
                
                let ql_base = is * 64;
                let g_ql_vec_0 = _mm256_loadu_si256(g_ql.add(ql_base) as *const __m256i);
                let g_ql_vec_1 = _mm256_loadu_si256(g_ql.add(ql_base + 32) as *const __m256i);
                let u_ql_vec_0 = _mm256_loadu_si256(u_ql.add(ql_base) as *const __m256i);
                let u_ql_vec_1 = _mm256_loadu_si256(u_ql.add(ql_base + 32) as *const __m256i);
                
                let s_base = is * 8;
                
                // Reconstruct weights for gate (q0, q1, q2, q3)
                let g_q0_lo = _mm256_and_si256(g_ql_vec_0, mask_low4);
                let g_q0_hi = _mm256_and_si256(g_qh_vec, mask_low2);
                let g_q0 = _mm256_or_si256(g_q0_lo, _mm256_slli_epi16(g_q0_hi, 4));
                let g_q0_i8 = _mm256_sub_epi8(g_q0, m32);
                
                let g_q2_lo = _mm256_and_si256(_mm256_srli_epi16(g_ql_vec_0, 4), mask_low4);
                let g_q2_hi = _mm256_and_si256(_mm256_srli_epi16(g_qh_vec, 4), mask_low2);
                let g_q2 = _mm256_or_si256(g_q2_lo, _mm256_slli_epi16(g_q2_hi, 4));
                let g_q2_i8 = _mm256_sub_epi8(g_q2, m32);
                
                let g_q1_lo = _mm256_and_si256(g_ql_vec_1, mask_low4);
                let g_q1_hi = _mm256_and_si256(_mm256_srli_epi16(g_qh_vec, 2), mask_low2);
                let g_q1 = _mm256_or_si256(g_q1_lo, _mm256_slli_epi16(g_q1_hi, 4));
                let g_q1_i8 = _mm256_sub_epi8(g_q1, m32);
                
                let g_q3_lo = _mm256_and_si256(_mm256_srli_epi16(g_ql_vec_1, 4), mask_low4);
                let g_q3_hi = _mm256_and_si256(_mm256_srli_epi16(g_qh_vec, 6), mask_low2);
                let g_q3 = _mm256_or_si256(g_q3_lo, _mm256_slli_epi16(g_q3_hi, 4));
                let g_q3_i8 = _mm256_sub_epi8(g_q3, m32);
                
                // Reconstruct weights for up
                let u_q0_lo = _mm256_and_si256(u_ql_vec_0, mask_low4);
                let u_q0_hi = _mm256_and_si256(u_qh_vec, mask_low2);
                let u_q0 = _mm256_or_si256(u_q0_lo, _mm256_slli_epi16(u_q0_hi, 4));
                let u_q0_i8 = _mm256_sub_epi8(u_q0, m32);
                
                let u_q2_lo = _mm256_and_si256(_mm256_srli_epi16(u_ql_vec_0, 4), mask_low4);
                let u_q2_hi = _mm256_and_si256(_mm256_srli_epi16(u_qh_vec, 4), mask_low2);
                let u_q2 = _mm256_or_si256(u_q2_lo, _mm256_slli_epi16(u_q2_hi, 4));
                let u_q2_i8 = _mm256_sub_epi8(u_q2, m32);
                
                let u_q1_lo = _mm256_and_si256(u_ql_vec_1, mask_low4);
                let u_q1_hi = _mm256_and_si256(_mm256_srli_epi16(u_qh_vec, 2), mask_low2);
                let u_q1 = _mm256_or_si256(u_q1_lo, _mm256_slli_epi16(u_q1_hi, 4));
                let u_q1_i8 = _mm256_sub_epi8(u_q1, m32);
                
                let u_q3_lo = _mm256_and_si256(_mm256_srli_epi16(u_ql_vec_1, 4), mask_low4);
                let u_q3_hi = _mm256_and_si256(_mm256_srli_epi16(u_qh_vec, 6), mask_low2);
                let u_q3 = _mm256_or_si256(u_q3_lo, _mm256_slli_epi16(u_q3_hi, 4));
                let u_q3_i8 = _mm256_sub_epi8(u_q3, m32);
                
                let input_base = is * 128;
                
                // Process q0 (indices 0..32)
                {
                    let g_sc0 = _mm256_mul_ps(_mm256_set1_ps(*g_scales.add(s_base) as f32), g_d_vec);
                    let g_sc1 = _mm256_mul_ps(_mm256_set1_ps(*g_scales.add(s_base + 1) as f32), g_d_vec);
                    let u_sc0 = _mm256_mul_ps(_mm256_set1_ps(*u_scales.add(s_base) as f32), u_d_vec);
                    let u_sc1 = _mm256_mul_ps(_mm256_set1_ps(*u_scales.add(s_base + 1) as f32), u_d_vec);
                    
                    let g_q_low128 = _mm256_castsi256_si128(g_q0_i8);
                    let g_q_high128 = _mm256_extracti128_si256(g_q0_i8, 1);
                    let u_q_low128 = _mm256_castsi256_si128(u_q0_i8);
                    let u_q_high128 = _mm256_extracti128_si256(u_q0_i8, 1);
                    
                    let a0 = _mm256_loadu_ps(a_ptr.add(input_base));
                    let a1 = _mm256_loadu_ps(a_ptr.add(input_base + 8));
                    let a2 = _mm256_loadu_ps(a_ptr.add(input_base + 16));
                    let a3 = _mm256_loadu_ps(a_ptr.add(input_base + 24));
                    
                    let g_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_q_low128));
                    let g_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_q_low128, 8)));
                    let g_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_q_high128));
                    let g_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_q_high128, 8)));
                    
                    let u_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_q_low128));
                    let u_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_q_low128, 8)));
                    let u_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_q_high128));
                    let u_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_q_high128, 8)));
                    
                    g_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(g_f0, g_sc0), a0, g_sum0);
                    g_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(g_f1, g_sc0), a1, g_sum0);
                    g_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(g_f2, g_sc1), a2, g_sum0);
                    g_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(g_f3, g_sc1), a3, g_sum0);
                    
                    u_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(u_f0, u_sc0), a0, u_sum0);
                    u_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(u_f1, u_sc0), a1, u_sum0);
                    u_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(u_f2, u_sc1), a2, u_sum0);
                    u_sum0 = _mm256_fmadd_ps(_mm256_mul_ps(u_f3, u_sc1), a3, u_sum0);
                }
                
                // Process q1 (indices 32..64)
                {
                    let g_sc2 = _mm256_mul_ps(_mm256_set1_ps(*g_scales.add(s_base + 2) as f32), g_d_vec);
                    let g_sc3 = _mm256_mul_ps(_mm256_set1_ps(*g_scales.add(s_base + 3) as f32), g_d_vec);
                    let u_sc2 = _mm256_mul_ps(_mm256_set1_ps(*u_scales.add(s_base + 2) as f32), u_d_vec);
                    let u_sc3 = _mm256_mul_ps(_mm256_set1_ps(*u_scales.add(s_base + 3) as f32), u_d_vec);
                    
                    let g_q_low128 = _mm256_castsi256_si128(g_q1_i8);
                    let g_q_high128 = _mm256_extracti128_si256(g_q1_i8, 1);
                    let u_q_low128 = _mm256_castsi256_si128(u_q1_i8);
                    let u_q_high128 = _mm256_extracti128_si256(u_q1_i8, 1);
                    
                    let a0 = _mm256_loadu_ps(a_ptr.add(input_base + 32));
                    let a1 = _mm256_loadu_ps(a_ptr.add(input_base + 40));
                    let a2 = _mm256_loadu_ps(a_ptr.add(input_base + 48));
                    let a3 = _mm256_loadu_ps(a_ptr.add(input_base + 56));
                    
                    let g_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_q_low128));
                    let g_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_q_low128, 8)));
                    let g_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_q_high128));
                    let g_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_q_high128, 8)));
                    
                    let u_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_q_low128));
                    let u_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_q_low128, 8)));
                    let u_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_q_high128));
                    let u_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_q_high128, 8)));
                    
                    g_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(g_f0, g_sc2), a0, g_sum1);
                    g_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(g_f1, g_sc2), a1, g_sum1);
                    g_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(g_f2, g_sc3), a2, g_sum1);
                    g_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(g_f3, g_sc3), a3, g_sum1);
                    
                    u_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(u_f0, u_sc2), a0, u_sum1);
                    u_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(u_f1, u_sc2), a1, u_sum1);
                    u_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(u_f2, u_sc3), a2, u_sum1);
                    u_sum1 = _mm256_fmadd_ps(_mm256_mul_ps(u_f3, u_sc3), a3, u_sum1);
                }
                
                // Process q2 (indices 64..96)
                {
                    let g_sc4 = _mm256_mul_ps(_mm256_set1_ps(*g_scales.add(s_base + 4) as f32), g_d_vec);
                    let g_sc5 = _mm256_mul_ps(_mm256_set1_ps(*g_scales.add(s_base + 5) as f32), g_d_vec);
                    let u_sc4 = _mm256_mul_ps(_mm256_set1_ps(*u_scales.add(s_base + 4) as f32), u_d_vec);
                    let u_sc5 = _mm256_mul_ps(_mm256_set1_ps(*u_scales.add(s_base + 5) as f32), u_d_vec);
                    
                    let g_q_low128 = _mm256_castsi256_si128(g_q2_i8);
                    let g_q_high128 = _mm256_extracti128_si256(g_q2_i8, 1);
                    let u_q_low128 = _mm256_castsi256_si128(u_q2_i8);
                    let u_q_high128 = _mm256_extracti128_si256(u_q2_i8, 1);
                    
                    let a0 = _mm256_loadu_ps(a_ptr.add(input_base + 64));
                    let a1 = _mm256_loadu_ps(a_ptr.add(input_base + 72));
                    let a2 = _mm256_loadu_ps(a_ptr.add(input_base + 80));
                    let a3 = _mm256_loadu_ps(a_ptr.add(input_base + 88));
                    
                    let g_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_q_low128));
                    let g_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_q_low128, 8)));
                    let g_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_q_high128));
                    let g_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_q_high128, 8)));
                    
                    let u_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_q_low128));
                    let u_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_q_low128, 8)));
                    let u_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_q_high128));
                    let u_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_q_high128, 8)));
                    
                    g_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(g_f0, g_sc4), a0, g_sum2);
                    g_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(g_f1, g_sc4), a1, g_sum2);
                    g_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(g_f2, g_sc5), a2, g_sum2);
                    g_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(g_f3, g_sc5), a3, g_sum2);
                    
                    u_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(u_f0, u_sc4), a0, u_sum2);
                    u_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(u_f1, u_sc4), a1, u_sum2);
                    u_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(u_f2, u_sc5), a2, u_sum2);
                    u_sum2 = _mm256_fmadd_ps(_mm256_mul_ps(u_f3, u_sc5), a3, u_sum2);
                }
                
                // Process q3 (indices 96..128)
                {
                    let g_sc6 = _mm256_mul_ps(_mm256_set1_ps(*g_scales.add(s_base + 6) as f32), g_d_vec);
                    let g_sc7 = _mm256_mul_ps(_mm256_set1_ps(*g_scales.add(s_base + 7) as f32), g_d_vec);
                    let u_sc6 = _mm256_mul_ps(_mm256_set1_ps(*u_scales.add(s_base + 6) as f32), u_d_vec);
                    let u_sc7 = _mm256_mul_ps(_mm256_set1_ps(*u_scales.add(s_base + 7) as f32), u_d_vec);
                    
                    let g_q_low128 = _mm256_castsi256_si128(g_q3_i8);
                    let g_q_high128 = _mm256_extracti128_si256(g_q3_i8, 1);
                    let u_q_low128 = _mm256_castsi256_si128(u_q3_i8);
                    let u_q_high128 = _mm256_extracti128_si256(u_q3_i8, 1);
                    
                    let a0 = _mm256_loadu_ps(a_ptr.add(input_base + 96));
                    let a1 = _mm256_loadu_ps(a_ptr.add(input_base + 104));
                    let a2 = _mm256_loadu_ps(a_ptr.add(input_base + 112));
                    let a3 = _mm256_loadu_ps(a_ptr.add(input_base + 120));
                    
                    let g_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_q_low128));
                    let g_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_q_low128, 8)));
                    let g_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g_q_high128));
                    let g_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(g_q_high128, 8)));
                    
                    let u_f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_q_low128));
                    let u_f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_q_low128, 8)));
                    let u_f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(u_q_high128));
                    let u_f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(u_q_high128, 8)));
                    
                    g_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(g_f0, g_sc6), a0, g_sum3);
                    g_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(g_f1, g_sc6), a1, g_sum3);
                    g_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(g_f2, g_sc7), a2, g_sum3);
                    g_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(g_f3, g_sc7), a3, g_sum3);
                    
                    u_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(u_f0, u_sc6), a0, u_sum3);
                    u_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(u_f1, u_sc6), a1, u_sum3);
                    u_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(u_f2, u_sc7), a2, u_sum3);
                    u_sum3 = _mm256_fmadd_ps(_mm256_mul_ps(u_f3, u_sc7), a3, u_sum3);
                }
            }
            
            a_ptr = a_ptr.add(256);
            
            gate_total += hsum_ps_avx(_mm256_add_ps(
                _mm256_add_ps(g_sum0, g_sum1),
                _mm256_add_ps(g_sum2, g_sum3),
            ));
            up_total += hsum_ps_avx(_mm256_add_ps(
                _mm256_add_ps(u_sum0, u_sum1),
                _mm256_add_ps(u_sum2, u_sum3),
            ));
        }
        
        output[i] = silu(gate_total) * up_total;
    }
}

/// Parallel Q6_K fused kernel.
pub fn fused_gate_up_silu_q6_k_parallel(
    output: &mut [f32],
    input: &[f32],
    gate_blocks: &[BlockQ6_K],
    up_blocks: &[BlockQ6_K],
    k: usize,
) {
    use rayon::prelude::*;
    
    let n = output.len();
    let num_threads = rayon::current_num_threads();
    let chunk_size = (n + num_threads - 1) / num_threads;
    let blocks_per_row = k / 256;
    
    output
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start_n = chunk_idx * chunk_size;
            let g_blocks_start = start_n * blocks_per_row;
            let u_blocks_start = start_n * blocks_per_row;
            let num_rows = out_chunk.len();
            
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    unsafe {
                        fused_gate_up_silu_q6_k_avx2(
                            out_chunk,
                            input.as_ptr(),
                            &gate_blocks[g_blocks_start..g_blocks_start + num_rows * blocks_per_row],
                            &up_blocks[u_blocks_start..u_blocks_start + num_rows * blocks_per_row],
                            k,
                        );
                    }
                    return;
                }
            }
            
            // Scalar fallback would go here
            unimplemented!("Q6_K scalar fallback");
        });
}

/// Parallel Q4_K fused kernel.
pub fn fused_gate_up_silu_q4_k_parallel(
    output: &mut [f32],
    input: &[f32],
    gate_blocks: &[BlockQ4_K],
    up_blocks: &[BlockQ4_K],
    k: usize,
) {
    use rayon::prelude::*;
    
    let n = output.len();
    let num_threads = rayon::current_num_threads();
    let chunk_size = (n + num_threads - 1) / num_threads;
    let blocks_per_row = k / 256;
    
    output
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start_n = chunk_idx * chunk_size;
            let g_blocks_start = start_n * blocks_per_row;
            let u_blocks_start = start_n * blocks_per_row;
            let num_rows = out_chunk.len();
            
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    unsafe {
                        fused_gate_up_silu_q4_k_avx2(
                            out_chunk,
                            input.as_ptr(),
                            &gate_blocks[g_blocks_start..g_blocks_start + num_rows * blocks_per_row],
                            &up_blocks[u_blocks_start..u_blocks_start + num_rows * blocks_per_row],
                            k,
                        );
                    }
                    return;
                }
            }
            
            // Scalar fallback would go here
            unimplemented!("Q4_K scalar fallback");
        });
}