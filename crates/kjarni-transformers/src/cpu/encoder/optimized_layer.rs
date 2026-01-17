// kjarni-transformers/src/cpu/encoder/optimized_layer.rs

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, Axis, Zip, s};
use rayon::prelude::*;

use crate::{activations::Activation, linear_layer::LinearLayer, rope::RoPE};

/// Scratch buffers for encoder layer - reused across forward passes
pub struct EncoderScratch {
    // QKV projection output (fused)
    qkv: Array2<f32>, // [batch*seq, 3*hidden]

    // Attention intermediates
    q_heads: Array4<f32>, // [batch, heads, seq, head_dim]
    k_heads: Array4<f32>, // [batch, heads, seq, head_dim]
    v_heads: Array4<f32>, // [batch, heads, seq, head_dim]
    scores: Array4<f32>,  // [batch, heads, seq, seq]
    context: Array4<f32>, // [batch, heads, seq, head_dim]

    // FFN intermediates
    pub intermediate: Array2<f32>, // [batch*seq, intermediate_dim]

    // Output buffer
    pub output_2d: Array2<f32>, // [batch*seq, hidden]
}

impl EncoderScratch {
    pub fn new(batch: usize, seq: usize, hidden: usize, heads: usize, intermediate: usize) -> Self {
        let head_dim = hidden / heads;
        let tokens = batch * seq;

        Self {
            qkv: Array2::zeros((tokens, 3 * hidden)),
            q_heads: Array4::zeros((batch, heads, seq, head_dim)),
            k_heads: Array4::zeros((batch, heads, seq, head_dim)),
            v_heads: Array4::zeros((batch, heads, seq, head_dim)),
            scores: Array4::zeros((batch, heads, seq, seq)),
            context: Array4::zeros((batch, heads, seq, head_dim)),
            intermediate: Array2::zeros((tokens, intermediate)),
            output_2d: Array2::zeros((tokens, hidden)),
        }
    }

    /// Resize if needed (for variable batch/seq)
    pub fn ensure_size(
        &mut self,
        batch: usize,
        seq: usize,
        hidden: usize,
        heads: usize,
        intermediate: usize,
    ) {
        let head_dim = hidden / heads;
        let tokens = batch * seq;

        if self.qkv.dim() != (tokens, 3 * hidden) {
            *self = Self::new(batch, seq, hidden, heads, intermediate);
        }
    }
}

/// Optimized self-attention with buffer reuse
pub struct OptimizedSelfAttention {
    // Fused QKV projection: [hidden, 3*hidden] (transposed for our matmul)
    qkv_proj: LinearLayer,
    out_proj: LinearLayer,

    num_heads: usize,
    head_dim: usize,
    scale_factor: f32,
    scale_qk: bool,
}

impl OptimizedSelfAttention {
    pub fn new(
        q_proj: LinearLayer,
        k_proj: LinearLayer,
        v_proj: LinearLayer,
        out_proj: LinearLayer,
        num_heads: usize,
        scale_qk: bool,
    ) -> Self {
        let head_dim = q_proj.out_features() / num_heads;

        // Fuse Q, K, V weights into single projection
        // This reduces 3 matmuls to 1 (major win for memory bandwidth)
        let qkv_weights = fuse_qkv_weights(&q_proj, &k_proj, &v_proj);
        let qkv_bias = fuse_qkv_bias(&q_proj, &k_proj, &v_proj);

        Self {
            qkv_proj: LinearLayer::new_f32(qkv_weights, qkv_bias),
            out_proj,
            num_heads,
            head_dim,
            scale_factor: 1.0 / (head_dim as f32).sqrt(),
            scale_qk,
        }
    }

    /// Forward pass with buffer reuse
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_bias: Option<&Array4<f32>>,
        rope: Option<&RoPE>,
        scratch: &mut EncoderScratch,
    ) -> Result<()> {
        let (batch, seq_len, hidden_dim) = hidden_states.dim();
        let tokens = batch * seq_len;

        // 1. Fused QKV Projection (1 matmul instead of 3!)
        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((tokens, hidden_dim))?;
        // self.qkv_proj.matmul_into(&hidden_2d, &mut scratch.qkv);
        self.qkv_proj
            .matmul_into_blocked(&hidden_2d, &mut scratch.qkv);

        // 2. Split QKV and reshape to heads (in-place where possible)
        split_qkv_to_heads(
            &scratch.qkv,
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
            &mut scratch.q_heads,
            &mut scratch.k_heads,
            &mut scratch.v_heads,
        );

        // 3. Apply RoPE if needed (in-place)
        // if let Some(r) = rope {
        //     r.apply_inplace(&mut scratch.q_heads, &mut scratch.k_heads, 0)?;
        // }

        // 4. Attention scores: Q @ K^T (batched, writes to scratch.scores)
        batched_qk_matmul(
            &scratch.q_heads,
            &scratch.k_heads,
            &mut scratch.scores,
            self.scale_factor,
            self.scale_qk,
        );

        // 5. Add position bias if provided
        if let Some(bias) = position_bias {
            add_bias_inplace(&mut scratch.scores, bias);
        }

        // 6. Apply mask and softmax (fused, in-place)
        masked_softmax_inplace(&mut scratch.scores, attention_mask);

        // 7. Context: Scores @ V (batched, writes to scratch.context)
        batched_sv_matmul(&scratch.scores, &scratch.v_heads, &mut scratch.context);

        // 8. Merge heads and output projection (writes to scratch.output_2d)
        merge_heads_and_project(
            &scratch.context,
            &self.out_proj,
            batch,
            seq_len,
            &mut scratch.output_2d,
        );

        Ok(())
    }
}

/// Fuse Q, K, V weight matrices into single [hidden, 3*hidden] matrix
fn fuse_qkv_weights(q: &LinearLayer, k: &LinearLayer, v: &LinearLayer) -> Array2<f32> {
    let hidden = q.in_features();
    let out = q.out_features();

    let mut fused = Array2::zeros((3 * out, hidden));

    // Copy weights (assuming they're stored as [out, in])
    fused.slice_mut(s![0..out, ..]).assign(&q.weights_view());
    fused
        .slice_mut(s![out..2 * out, ..])
        .assign(&k.weights_view());
    fused
        .slice_mut(s![2 * out..3 * out, ..])
        .assign(&v.weights_view());

    fused
}

fn fuse_qkv_bias(q: &LinearLayer, k: &LinearLayer, v: &LinearLayer) -> Option<Array1<f32>> {
    match (q.bias(), k.bias(), v.bias()) {
        (Some(qb), Some(kb), Some(vb)) => {
            let out = qb.len();
            let mut fused = Array1::zeros(3 * out);
            fused.slice_mut(s![0..out]).assign(&qb);
            fused.slice_mut(s![out..2 * out]).assign(&kb);
            fused.slice_mut(s![2 * out..3 * out]).assign(&vb);
            Some(fused)
        }
        _ => None,
    }
}

/// Split fused QKV output into separate head tensors (no allocation!)
fn split_qkv_to_heads(
    qkv: &Array2<f32>, // [tokens, 3*hidden]
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    q_out: &mut Array4<f32>, // [batch, heads, seq, head_dim]
    k_out: &mut Array4<f32>,
    v_out: &mut Array4<f32>,
) {
    let hidden = heads * head_dim;

    // Parallel over batch
    q_out
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(k_out.axis_iter_mut(Axis(0)))
        .zip(v_out.axis_iter_mut(Axis(0)))
        .enumerate()
        .for_each(|(b, ((mut q_b, mut k_b), mut v_b))| {
            for s in 0..seq {
                let token_idx = b * seq + s;
                let qkv_row = qkv.row(token_idx);

                for h in 0..heads {
                    let head_start = h * head_dim;
                    for d in 0..head_dim {
                        q_b[[h, s, d]] = qkv_row[head_start + d];
                        k_b[[h, s, d]] = qkv_row[hidden + head_start + d];
                        v_b[[h, s, d]] = qkv_row[2 * hidden + head_start + d];
                    }
                }
            }
        });
}

/// Batched Q @ K^T with scaling (writes directly to output buffer)
fn batched_qk_matmul(
    q: &Array4<f32>,       // [batch, heads, seq, head_dim]
    k: &Array4<f32>,       // [batch, heads, seq, head_dim]
    out: &mut Array4<f32>, // [batch, heads, seq, seq]
    scale: f32,
    apply_scale: bool,
) {
    let (batch, heads, seq, head_dim) = q.dim();
    let scale = if apply_scale { scale } else { 1.0 };

    // Parallel over batch * heads
    Zip::from(out.outer_iter_mut())
        .and(q.outer_iter())
        .and(k.outer_iter())
        .par_for_each(|mut out_b, q_b, k_b| {
            Zip::from(out_b.outer_iter_mut())
                .and(q_b.outer_iter())
                .and(k_b.outer_iter())
                .for_each(|mut out_h, q_h, k_h| {
                    // out_h[i, j] = sum_d(q_h[i, d] * k_h[j, d]) * scale
                    let q_slice = q_h.as_slice().unwrap();
                    let k_slice = k_h.as_slice().unwrap();
                    let out_slice = out_h.as_slice_mut().unwrap();

                    for i in 0..seq {
                        for j in 0..seq {
                            let mut sum = 0.0f32;
                            let q_row = &q_slice[i * head_dim..(i + 1) * head_dim];
                            let k_row = &k_slice[j * head_dim..(j + 1) * head_dim];

                            // SIMD-friendly dot product
                            for d in 0..head_dim {
                                sum += q_row[d] * k_row[d];
                            }
                            out_slice[i * seq + j] = sum * scale;
                        }
                    }
                });
        });
}

/// Apply mask and softmax in one pass (avoids 2 memory passes)
fn masked_softmax_inplace(scores: &mut Array4<f32>, mask: &Array2<f32>) {
    let (batch, heads, seq_q, seq_k) = scores.dim();

    scores
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(b, mut scores_b)| {
            let mask_row = mask.row(b);

            for mut scores_h in scores_b.outer_iter_mut() {
                let data = scores_h.as_slice_mut().unwrap();

                for i in 0..seq_q {
                    let row_start = i * seq_k;
                    let row = &mut data[row_start..row_start + seq_k];

                    // 1. Apply mask (set masked positions to -inf)
                    let mut max_val = f32::NEG_INFINITY;
                    for j in 0..seq_k {
                        if mask_row[j] == 0.0 {
                            row[j] = f32::NEG_INFINITY;
                        }
                        max_val = max_val.max(row[j]);
                    }

                    // 2. Softmax: exp(x - max) / sum(exp(x - max))
                    let mut sum = 0.0f32;
                    for j in 0..seq_k {
                        if row[j] != f32::NEG_INFINITY {
                            row[j] = (row[j] - max_val).exp();
                            sum += row[j];
                        } else {
                            row[j] = 0.0;
                        }
                    }

                    if sum > 0.0 {
                        let inv_sum = 1.0 / sum;
                        for j in 0..seq_k {
                            row[j] *= inv_sum;
                        }
                    }
                }
            }
        });
}

/// Batched Scores @ V
fn batched_sv_matmul(
    scores: &Array4<f32>,  // [batch, heads, seq, seq]
    v: &Array4<f32>,       // [batch, heads, seq, head_dim]
    out: &mut Array4<f32>, // [batch, heads, seq, head_dim]
) {
    let (_, _, seq, head_dim) = v.dim();

    Zip::from(out.outer_iter_mut())
        .and(scores.outer_iter())
        .and(v.outer_iter())
        .par_for_each(|mut out_b, scores_b, v_b| {
            Zip::from(out_b.outer_iter_mut())
                .and(scores_b.outer_iter())
                .and(v_b.outer_iter())
                .for_each(|mut out_h, scores_h, v_h| {
                    // out_h[i, d] = sum_j(scores_h[i, j] * v_h[j, d])
                    let scores_slice = scores_h.as_slice().unwrap();
                    let v_slice = v_h.as_slice().unwrap();
                    let out_slice = out_h.as_slice_mut().unwrap();

                    for i in 0..seq {
                        for d in 0..head_dim {
                            let mut sum = 0.0f32;
                            for j in 0..seq {
                                sum += scores_slice[i * seq + j] * v_slice[j * head_dim + d];
                            }
                            out_slice[i * head_dim + d] = sum;
                        }
                    }
                });
        });
}

/// Merge heads and apply output projection
fn merge_heads_and_project(
    context: &Array4<f32>, // [batch, heads, seq, head_dim]
    out_proj: &LinearLayer,
    batch: usize,
    seq: usize,
    output: &mut Array2<f32>, // [batch*seq, hidden]
) {
    let (_, heads, _, head_dim) = context.dim();
    let hidden = heads * head_dim;

    // First, merge heads into temporary buffer (reuse output as scratch)
    // output will be overwritten by the projection anyway
    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(token_idx, mut out_row)| {
            let b = token_idx / seq;
            let s = token_idx % seq;

            for h in 0..heads {
                for d in 0..head_dim {
                    out_row[h * head_dim + d] = context[[b, h, s, d]];
                }
            }
        });

    // Now project (in-place not possible, need temp)
    // Actually we need a different approach - use separate buffer
    // For now, just do the matmul
    let merged = output.view();
    let projected = out_proj.matmul(&merged);
    output.assign(&projected);
}

/// Optimized FFN with buffer reuse
pub struct OptimizedFeedForward {
    fc1: LinearLayer,
    fc2: LinearLayer,
    activation: Activation,
}

impl OptimizedFeedForward {
    pub fn new(fc1: LinearLayer, fc2: LinearLayer, activation: Activation) -> Self {
        Self {
            fc1,
            fc2,
            activation,
        }
    }

    /// Forward with buffer reuse
    pub fn forward(
        &self,
        hidden: &ArrayView2<f32>,  // [tokens, hidden]
        scratch: &mut Array2<f32>, // [tokens, intermediate]
        output: &mut Array2<f32>,  // [tokens, hidden]
    ) {
        // FC1 -> scratch
        self.fc1.matmul_into_blocked(hidden, scratch);

        // Activation (in-place)
        apply_activation_inplace(scratch.as_slice_mut().unwrap(), self.activation);

        // FC2 -> output
        self.fc2.matmul_into_blocked(&scratch.view(), output);
    }
}

/// In-place activation
fn apply_activation_inplace(data: &mut [f32], activation: Activation) {
    use crate::activations::{gelu_new_scalar, gelu_scalar, relu_scalar, silu_scalar, tanh_scalar};

    match activation {
        Activation::Gelu => data.par_iter_mut().for_each(|x| *x = gelu_scalar(*x)),
        Activation::GeluNew => data.par_iter_mut().for_each(|x| *x = gelu_new_scalar(*x)),
        Activation::Relu => data.par_iter_mut().for_each(|x| *x = relu_scalar(*x)),
        Activation::SilU => data.par_iter_mut().for_each(|x| *x = silu_scalar(*x)),
        Activation::Tanh => data.par_iter_mut().for_each(|x| *x = tanh_scalar(*x)),
    }
}

/// Add bias tensor in-place
fn add_bias_inplace(scores: &mut Array4<f32>, bias: &Array4<f32>) {
    let dim = scores.dim();
    Zip::from(scores)
        .and(bias.broadcast(dim).unwrap())
        .par_for_each(|s, &b| *s += b);
}

// =============================================================================
// Optimized Layer Norm + Residual (FUSED - single memory pass)
// =============================================================================

/// Fused: output = LayerNorm(input + residual)
pub fn fused_residual_layernorm(
    output: &mut Array3<f32>,   // Output buffer (can be same as input)
    input: &ArrayView3<f32>,    // Current tensor
    residual: &ArrayView3<f32>, // Residual connection
    gamma: &Array1<f32>,
    beta: &Array1<f32>,
    eps: f32,
) {
    let (batch, seq, hidden) = input.dim();

    output
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(input.axis_iter(Axis(0)))
        .zip(residual.axis_iter(Axis(0)))
        .for_each(|((mut out_b, inp_b), res_b)| {
            for s in 0..seq {
                let inp_row = inp_b.row(s);
                let res_row = res_b.row(s);
                let mut out_row = out_b.row_mut(s);

                // 1. Add residual and compute mean
                let mut sum = 0.0f32;
                for d in 0..hidden {
                    let val = inp_row[d] + res_row[d];
                    out_row[d] = val;
                    sum += val;
                }
                let mean = sum / hidden as f32;

                // 2. Compute variance
                let mut var_sum = 0.0f32;
                for d in 0..hidden {
                    let diff = out_row[d] - mean;
                    var_sum += diff * diff;
                }
                let std = (var_sum / hidden as f32 + eps).sqrt();
                let inv_std = 1.0 / std;

                // 3. Normalize and apply affine
                for d in 0..hidden {
                    out_row[d] = (out_row[d] - mean) * inv_std * gamma[d] + beta[d];
                }
            }
        });
}

// =============================================================================
// LinearLayer extension for in-place matmul
// =============================================================================

impl LinearLayer {
    /// Get a view of the weights [out_features, in_features]
    pub fn weights_view(&self) -> ArrayView2<f32> {
        match self.data {
            crate::linear_layer::LinearData::F32(ref w) => w.view(),
            _ => panic!("Only f32 LinearLayer supported in optimized path"),
        }
    }

    /// Get weights as a contiguous slice (ensures standard layout)
    pub fn weights_slice(&self) -> &[f32] {
        // If already contiguous, this is free
        match self.data {
            crate::linear_layer::LinearData::F32(ref w) => {
                if w.is_standard_layout() {
                    return w.as_slice().expect("Weights must be contiguous row-major");
                }
                unimplemented!("Non-contiguous weights are not supported");
            }
            _ => panic!("Only f32 LinearLayer supported in optimized path"),
        }
        // self.weights
        //     .as_slice()
        //     .expect("Weights must be contiguous row-major")
    }

    /// Get reference to bias if present
    pub fn bias(&self) -> Option<&Array1<f32>> {
        self.bias.as_ref()
    }

    /// Get bias as slice if present
    pub fn bias_slice(&self) -> Option<&[f32]> {
        self.bias.as_ref().map(|b| b.as_slice().unwrap())
    }

    /// Matmul that writes directly to pre-allocated output buffer
    pub fn matmul_into(&self, input: &ArrayView2<f32>, output: &mut Array2<f32>) {
        let (m, k) = input.dim();
        let n = self.out_features();

        debug_assert_eq!(output.dim(), (m, n));
        debug_assert_eq!(k, self.in_features());

        // Get contiguous slices
        let input_s = input.as_standard_layout();
        let input_slice = input_s.as_slice().unwrap();
        let weights_slice = self.weights_slice();
        let output_slice = output.as_slice_mut().unwrap();

        // Parallel matmul into output buffer
        output_slice
            .par_chunks_mut(n)
            .zip(input_slice.par_chunks(k))
            .for_each(|(out_row, in_row)| {
                unsafe {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        crate::cpu::kernels::x86::f32::matmul_vec_f32(
                            out_row,
                            in_row.as_ptr(),
                            weights_slice.as_ptr(),
                            k,
                        );
                        return;
                    }
                    // Fallback
                    matmul_vec_scalar(out_row, in_row, weights_slice, k);
                }
            });

        // Add bias if present
        if let Some(bias) = self.bias() {
            Zip::from(output.rows_mut()).par_for_each(|mut row| {
                Zip::from(&mut row).and(bias).for_each(|o, &b| *o += b);
            });
        }
    }
    pub fn matmul_into_blocked(&self, input: &ArrayView2<f32>, output: &mut Array2<f32>) {
        let (m, k) = input.dim();
        let n = self.out_features();

        // 1. Get Slices (These are Sync/Send and safe to pass to Rayon)
        let input_slice = input.as_slice().expect("Input contiguous");
        let output_slice = output.as_slice_mut().expect("Output contiguous");
        let weights_slice = self.weights_slice(); // &[f32]
        let bias_slice_opt = self.bias_slice(); // Option<&[f32]>

        const BLOCK_SIZE: usize = 64;

        // 2. Parallelize
        output_slice
            .par_chunks_mut(BLOCK_SIZE * n)
            .zip(input_slice.par_chunks(BLOCK_SIZE * k))
            .for_each(|(out_block, in_block)| {
                // 3. CONVERT TO POINTERS INSIDE THE THREAD
                // This satisfies the compiler because we sent the slice, not the pointer.
                let weights_ptr = weights_slice.as_ptr();
                let bias_ptr = match bias_slice_opt {
                    Some(s) => s.as_ptr(),
                    None => std::ptr::null(),
                };

                let num_tokens = in_block.len() / k;
                let mut t = 0;

                // --- MAIN LOOP: Process 4 Tokens at a time ---
                while t + 4 <= num_tokens {
                    let in_ptr = unsafe { in_block.as_ptr().add(t * k) };
                    let mut j = 0;

                    // Kernel: Process 3 Output Features at a time
                    while j + 3 <= n {
                        let w_ptr = unsafe { weights_ptr.add(j * k) };
                        let out_ptr = unsafe { out_block.as_mut_ptr().add(t * n + j) };

                        // Calculate bias offset for these 3 features
                        let b_ptr_offset = if !bias_ptr.is_null() {
                            unsafe { bias_ptr.add(j) }
                        } else {
                            std::ptr::null()
                        };

                        unsafe {
                            crate::cpu::kernels::x86::f32::matmul_block_4x3_f32(
                                out_ptr,
                                n, // Stride
                                in_ptr,
                                w_ptr,
                                k,
                                b_ptr_offset,
                            );
                        }
                        j += 3;
                    }

                    // Fallback: Remaining Output Features (cols) for these 4 tokens
                    while j < n {
                        let w_ptr = unsafe { weights_ptr.add(j * k) };
                        let b_val = if !bias_ptr.is_null() {
                            unsafe { *bias_ptr.add(j) }
                        } else {
                            0.0
                        };

                        for row_offset in 0..4 {
                            let row_ptr = unsafe { in_ptr.add(row_offset * k) };
                            let dst =
                                unsafe { out_block.as_mut_ptr().add((t + row_offset) * n + j) };

                            let mut sum = 0.0;
                            // Simple scalar loop (compiler auto-vectorizes this well enough)
                            for i in 0..k {
                                sum += unsafe { *row_ptr.add(i) * *w_ptr.add(i) };
                            }
                            unsafe {
                                *dst = sum + b_val;
                            }
                        }
                        j += 1;
                    }
                    t += 4;
                }

                // --- CLEANUP LOOP: Remaining Tokens (< 4) ---
                while t < num_tokens {
                    let in_ptr = unsafe { in_block.as_ptr().add(t * k) };

                    for j in 0..n {
                        let w_ptr = unsafe { weights_ptr.add(j * k) };
                        let b_val = if !bias_ptr.is_null() {
                            unsafe { *bias_ptr.add(j) }
                        } else {
                            0.0
                        };

                        let mut sum = 0.0;
                        for x in 0..k {
                            sum += unsafe { *in_ptr.add(x) * *w_ptr.add(x) };
                        }
                        out_block[t * n + j] = sum + b_val;
                    }
                    t += 1;
                }
            });

        // REMOVED: The second bias loop. Bias is now handled inside the kernel.
    }
}

fn matmul_vec_scalar(out: &mut [f32], a: &[f32], b: &[f32], k: usize) {
    let n = out.len();
    for i in 0..n {
        let mut sum = 0.0f32;
        let b_row = &b[i * k..(i + 1) * k];
        for j in 0..k {
            sum += a[j] * b_row[j];
        }
        out[i] = sum;
    }
}

// Place this in a new file or as a mod block in encoder_layer.rs
// e.g., crates/kjarni-transformers/src/cpu/encoder/optimized_tests.rs

#[cfg(test)]
mod optimized_parity_tests {
    use anyhow::Result;
    use ndarray::{Array1, Array2, Array3, Array4};
    use rayon::iter::IntoParallelIterator;

    use crate::activations::Activation;
    use crate::cpu::encoder::encoder_layer::EncoderLayer;
    use crate::cpu::encoder::encoder_self_attention::EncoderSelfAttention;
    use crate::cpu::encoder::optimized_layer::{
        EncoderScratch, OptimizedFeedForward, OptimizedSelfAttention, fused_residual_layernorm,
    };
    use crate::feedforward::StdFeedForward;
    use crate::linear_layer::LinearLayer;
    use crate::normalization::{LayerNorm, Normalization};

    // =========================================================================
    // Test Utilities
    // =========================================================================

    struct ComparisonResult {
        max_diff: f32,
        mean_diff: f32,
        num_elements: usize,
    }

    fn compare_arrays_3d(a: &Array3<f32>, b: &Array3<f32>) -> ComparisonResult {
        assert_eq!(a.dim(), b.dim(), "Array dimensions must match");
        let diff = a - b;
        let abs_diff = diff.mapv(|x| x.abs());
        let max_diff = abs_diff.iter().fold(0.0f32, |acc, &x| acc.max(x));
        let mean_diff = abs_diff.mean().unwrap_or(0.0);
        ComparisonResult {
            max_diff,
            mean_diff,
            num_elements: a.len(),
        }
    }

    fn compare_arrays_2d(a: &Array2<f32>, b: &Array2<f32>) -> ComparisonResult {
        assert_eq!(a.dim(), b.dim(), "Array dimensions must match");
        let diff = a - b;
        let abs_diff = diff.mapv(|x| x.abs());
        let max_diff = abs_diff.iter().fold(0.0f32, |acc, &x| acc.max(x));
        let mean_diff = abs_diff.mean().unwrap_or(0.0);
        ComparisonResult {
            max_diff,
            mean_diff,
            num_elements: a.len(),
        }
    }

    /// Create deterministic linear layer weights
    fn make_linear(rows: usize, cols: usize, start: &mut usize, bias_val: f32) -> LinearLayer {
        let size = rows * cols;
        let data: Vec<f32> = (*start..*start + size).map(|x| x as f32 * 0.001).collect();
        *start += size;
        let weights = Array2::from_shape_vec((rows, cols), data).unwrap();
        let bias = Array1::from_elem(rows, bias_val);
        LinearLayer::new_f32(weights, Some(bias))
    }

    /// Create deterministic input tensor
    fn make_input_3d(batch: usize, seq: usize, hidden: usize) -> Array3<f32> {
        let data: Vec<f32> = (0..batch * seq * hidden)
            .map(|i| (i as f32) * 0.1)
            .collect();
        Array3::from_shape_vec((batch, seq, hidden), data).unwrap()
    }

    /// Create attention mask (all valid except last position in batch 1)
    fn make_mask(batch: usize, seq: usize) -> Array2<f32> {
        let mut mask = Array2::ones((batch, seq));
        if batch > 1 && seq > 1 {
            mask[[1, seq - 1]] = 0.0; // Mask last position of second batch
        }
        mask
    }

    /// Create position bias
    fn make_position_bias(heads: usize, seq: usize) -> Array4<f32> {
        let data: Vec<f32> = (0..heads * seq * seq).map(|i| (i as f32) * 0.01).collect();
        Array4::from_shape_vec((1, heads, seq, seq), data).unwrap()
    }

    // =========================================================================
    // ATTENTION PARITY TESTS
    // =========================================================================
    #[test]
    fn test_split_qkv_to_heads_parity() {
        // Test the split operation in isolation
        let (batch, seq, hidden, heads) = (4, 16, 64, 4);
        let head_dim = hidden / heads;
        let tokens = batch * seq;

        // Create deterministic QKV data
        let qkv_data: Vec<f32> = (0..tokens * 3 * hidden).map(|i| i as f32 * 0.001).collect();
        let qkv = Array2::from_shape_vec((tokens, 3 * hidden), qkv_data).unwrap();

        // Optimized split
        let mut q_opt = Array4::zeros((batch, heads, seq, head_dim));
        let mut k_opt = Array4::zeros((batch, heads, seq, head_dim));
        let mut v_opt = Array4::zeros((batch, heads, seq, head_dim));

        crate::cpu::encoder::optimized_layer::split_qkv_to_heads(
            &qkv, batch, seq, heads, head_dim, &mut q_opt, &mut k_opt, &mut v_opt,
        );

        // Reference split (simple, obviously correct)
        let mut q_ref = Array4::zeros((batch, heads, seq, head_dim));
        let mut k_ref = Array4::zeros((batch, heads, seq, head_dim));
        let mut v_ref = Array4::zeros((batch, heads, seq, head_dim));

        for b in 0..batch {
            for s in 0..seq {
                let token_idx = b * seq + s;
                for h in 0..heads {
                    for d in 0..head_dim {
                        let q_idx = h * head_dim + d;
                        let k_idx = hidden + h * head_dim + d;
                        let v_idx = 2 * hidden + h * head_dim + d;

                        q_ref[[b, h, s, d]] = qkv[[token_idx, q_idx]];
                        k_ref[[b, h, s, d]] = qkv[[token_idx, k_idx]];
                        v_ref[[b, h, s, d]] = qkv[[token_idx, v_idx]];
                    }
                }
            }
        }

        let q_diff = (&q_opt - &q_ref)
            .mapv(f32::abs)
            .fold(0.0f32, |a, &b| a.max(b));
        let k_diff = (&k_opt - &k_ref)
            .mapv(f32::abs)
            .fold(0.0f32, |a, &b| a.max(b));
        let v_diff = (&v_opt - &v_ref)
            .mapv(f32::abs)
            .fold(0.0f32, |a, &b| a.max(b));

        println!("\n=== SPLIT QKV PARITY ===");
        println!("Q diff: {:.2e}", q_diff);
        println!("K diff: {:.2e}", k_diff);
        println!("V diff: {:.2e}", v_diff);

        assert!(q_diff < 1e-6, "Q split mismatch: {}", q_diff);
        assert!(k_diff < 1e-6, "K split mismatch: {}", k_diff);
        assert!(v_diff < 1e-6, "V split mismatch: {}", v_diff);
    }

    #[test]
    fn test_merge_heads_parity() {
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::ParallelIterator;
        // Test merge operation in isolation
        let (batch, seq, hidden, heads) = (4, 16, 64, 4);
        let head_dim = hidden / heads;
        let tokens = batch * seq;

        // Create deterministic context data
        let context_data: Vec<f32> = (0..batch * heads * seq * head_dim)
            .map(|i| i as f32 * 0.001)
            .collect();
        let context = Array4::from_shape_vec((batch, heads, seq, head_dim), context_data).unwrap();

        // Optimized merge (without projection)
        let mut merged_opt = Array2::zeros((tokens, hidden));
        merged_opt
            .axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(token_idx, mut out_row)| {
                let b = token_idx / seq;
                let s = token_idx % seq;
                for h in 0..heads {
                    for d in 0..head_dim {
                        out_row[h * head_dim + d] = context[[b, h, s, d]];
                    }
                }
            });

        // Reference merge
        let mut merged_ref = Array2::zeros((tokens, hidden));
        for b in 0..batch {
            for s in 0..seq {
                let token_idx = b * seq + s;
                for h in 0..heads {
                    for d in 0..head_dim {
                        merged_ref[[token_idx, h * head_dim + d]] = context[[b, h, s, d]];
                    }
                }
            }
        }

        let diff = (&merged_opt - &merged_ref)
            .mapv(f32::abs)
            .fold(0.0f32, |a, &b| a.max(b));

        println!("\n=== MERGE HEADS PARITY ===");
        println!("Max diff: {:.2e}", diff);

        assert!(diff < 1e-6, "Merge mismatch: {}", diff);
    }
    #[test]
    fn test_attention_parity_small() -> Result<()> {
        let (batch, seq, hidden, heads) = (2, 3, 4, 2);
        let head_dim = hidden / heads;
        let intermediate = 8; // Not used for attention, but needed for scratch
        let bias_val = 0.01;
        let mut count = 1;

        // Create identical linear layers for both paths
        let q_proj = make_linear(hidden, hidden, &mut count, bias_val);
        let k_proj = make_linear(hidden, hidden, &mut count, bias_val);
        let v_proj = make_linear(hidden, hidden, &mut count, bias_val);
        let out_proj = make_linear(hidden, hidden, &mut count, bias_val);

        // Standard attention
        let std_attn = EncoderSelfAttention::new(
            hidden,
            heads,
            q_proj.clone(),
            k_proj.clone(),
            v_proj.clone(),
            out_proj.clone(),
        );

        // Optimized attention
        let opt_attn = OptimizedSelfAttention::new(
            q_proj, k_proj, v_proj, out_proj, heads, true, // scale_qk
        );

        // Create inputs
        let input = make_input_3d(batch, seq, hidden);
        let mask = make_mask(batch, seq);
        let pos_bias = make_position_bias(heads, seq);

        // Run standard path
        let std_output = std_attn.forward(&input, &mask, Some(&pos_bias), None)?;

        // Run optimized path
        let mut scratch = EncoderScratch::new(batch, seq, hidden, heads, intermediate);
        opt_attn.forward(&input, &mask, Some(&pos_bias), None, &mut scratch)?;

        // Reshape optimized output from [tokens, hidden] to [batch, seq, hidden]
        let opt_output = scratch
            .output_2d
            .view()
            .into_shape((batch, seq, hidden))?
            .to_owned();

        // Compare
        let result = compare_arrays_3d(&std_output, &opt_output);

        println!("\n=== ATTENTION PARITY (small) ===");
        println!("Shape: [{}, {}, {}]", batch, seq, hidden);
        println!("Max diff:  {:.6e}", result.max_diff);
        println!("Mean diff: {:.6e}", result.mean_diff);
        println!("Elements:  {}", result.num_elements);

        // Print first few values for debugging
        println!(
            "\nStandard output (first 8): {:?}",
            &std_output.as_slice().unwrap()[..8.min(std_output.len())]
        );
        println!(
            "Optimized output (first 8): {:?}",
            &opt_output.as_slice().unwrap()[..8.min(opt_output.len())]
        );

        assert!(
            result.max_diff < 1e-4,
            "Attention parity failed. Max diff: {:.6e}",
            result.max_diff
        );

        Ok(())
    }

    #[test]
    fn test_attention_parity_medium() -> Result<()> {
        let (batch, seq, hidden, heads) = (4, 16, 64, 4);
        let head_dim = hidden / heads;
        let intermediate = 256;
        let bias_val = 0.01;
        let mut count = 1;

        let q_proj = make_linear(hidden, hidden, &mut count, bias_val);
        let k_proj = make_linear(hidden, hidden, &mut count, bias_val);
        let v_proj = make_linear(hidden, hidden, &mut count, bias_val);
        let out_proj = make_linear(hidden, hidden, &mut count, bias_val);

        let std_attn = EncoderSelfAttention::new(
            hidden,
            heads,
            q_proj.clone(),
            k_proj.clone(),
            v_proj.clone(),
            out_proj.clone(),
        );

        let opt_attn = OptimizedSelfAttention::new(q_proj, k_proj, v_proj, out_proj, heads, true);

        let input = make_input_3d(batch, seq, hidden);
        let mask = make_mask(batch, seq);
        let pos_bias = make_position_bias(heads, seq);

        let std_output = std_attn.forward(&input, &mask, Some(&pos_bias), None)?;

        let mut scratch = EncoderScratch::new(batch, seq, hidden, heads, intermediate);
        opt_attn.forward(&input, &mask, Some(&pos_bias), None, &mut scratch)?;
        let opt_output = scratch
            .output_2d
            .view()
            .into_shape((batch, seq, hidden))?
            .to_owned();

        let result = compare_arrays_3d(&std_output, &opt_output);

        println!("\n=== ATTENTION PARITY (medium) ===");
        println!("Shape: [{}, {}, {}]", batch, seq, hidden);
        println!("Max diff:  {:.6e}", result.max_diff);
        println!("Mean diff: {:.6e}", result.mean_diff);

        assert!(
            result.max_diff < 1e-4,
            "Attention parity failed. Max diff: {:.6e}",
            result.max_diff
        );

        Ok(())
    }

    // =========================================================================
    // FFN PARITY TESTS
    // =========================================================================

    #[test]
    fn test_ffn_parity_small() -> Result<()> {
        let (batch, seq, hidden, intermediate) = (2, 3, 4, 8);
        let bias_val = 0.01;
        let mut count = 1;

        let fc1 = make_linear(intermediate, hidden, &mut count, bias_val);
        let fc2 = make_linear(hidden, intermediate, &mut count, bias_val);

        // Standard FFN (using StdFeedForward)
        let std_ffn = StdFeedForward::new(
            fc1.weights_view().to_owned(),
            fc1.bias().unwrap().clone(),
            fc2.weights_view().to_owned(),
            fc2.bias().unwrap().clone(),
            Activation::Gelu,
        );

        // Optimized FFN
        let opt_ffn = OptimizedFeedForward::new(fc1, fc2, Activation::Gelu);

        // Create input
        let input_3d = make_input_3d(batch, seq, hidden);

        // Run standard path
        let std_output = std_ffn.forward(&input_3d)?;

        // Run optimized path (needs 2D input)
        let tokens = batch * seq;
        let input_2d = input_3d.view().into_shape((tokens, hidden))?.to_owned();
        let mut scratch_intermediate = Array2::zeros((tokens, intermediate));
        let mut opt_output_2d = Array2::zeros((tokens, hidden));

        opt_ffn.forward(
            &input_2d.view(),
            &mut scratch_intermediate,
            &mut opt_output_2d,
        );

        // Reshape to 3D for comparison
        let opt_output = opt_output_2d.into_shape((batch, seq, hidden))?.to_owned();

        let result = compare_arrays_3d(&std_output, &opt_output);

        println!("\n=== FFN PARITY (small) ===");
        println!(
            "Shape: [{}, {}, {}] -> intermediate {}",
            batch, seq, hidden, intermediate
        );
        println!("Max diff:  {:.6e}", result.max_diff);
        println!("Mean diff: {:.6e}", result.mean_diff);

        println!(
            "\nStandard output (first 8): {:?}",
            &std_output.as_slice().unwrap()[..8.min(std_output.len())]
        );
        println!(
            "Optimized output (first 8): {:?}",
            &opt_output.as_slice().unwrap()[..8.min(opt_output.len())]
        );

        assert!(
            result.max_diff < 1e-5,
            "FFN parity failed. Max diff: {:.6e}",
            result.max_diff
        );

        Ok(())
    }

    #[test]
    fn test_ffn_parity_medium() -> Result<()> {
        let (batch, seq, hidden, intermediate) = (4, 16, 64, 256);
        let bias_val = 0.01;
        let mut count = 1;

        let fc1 = make_linear(intermediate, hidden, &mut count, bias_val);
        let fc2 = make_linear(hidden, intermediate, &mut count, bias_val);

        let std_ffn = StdFeedForward::new(
            fc1.weights_view().to_owned(),
            fc1.bias().unwrap().clone(),
            fc2.weights_view().to_owned(),
            fc2.bias().unwrap().clone(),
            Activation::Gelu,
        );

        let opt_ffn = OptimizedFeedForward::new(fc1, fc2, Activation::Gelu);

        let input_3d = make_input_3d(batch, seq, hidden);
        let std_output = std_ffn.forward(&input_3d)?;

        let tokens = batch * seq;
        let input_2d = input_3d.view().into_shape((tokens, hidden))?.to_owned();
        let mut scratch_intermediate = Array2::zeros((tokens, intermediate));
        let mut opt_output_2d = Array2::zeros((tokens, hidden));

        opt_ffn.forward(
            &input_2d.view(),
            &mut scratch_intermediate,
            &mut opt_output_2d,
        );
        let opt_output = opt_output_2d.into_shape((batch, seq, hidden))?.to_owned();

        let result = compare_arrays_3d(&std_output, &opt_output);

        println!("\n=== FFN PARITY (medium) ===");
        println!(
            "Shape: [{}, {}, {}] -> intermediate {}",
            batch, seq, hidden, intermediate
        );
        println!("Max diff:  {:.6e}", result.max_diff);
        println!("Mean diff: {:.6e}", result.mean_diff);

        assert!(
            result.max_diff < 1e-4,
            "FFN parity failed. Max diff: {:.6e}",
            result.max_diff
        );

        Ok(())
    }

    // =========================================================================
    // FUSED RESIDUAL + LAYERNORM PARITY TESTS
    // =========================================================================

    #[test]
    fn test_fused_residual_layernorm_parity_small() -> Result<()> {
        let (batch, seq, hidden) = (2, 3, 4);
        let eps = 1e-5;

        // Create LayerNorm parameters
        let gamma = Array1::from_vec((1..=hidden).map(|i| 1.0 + i as f32 * 0.01).collect());
        let beta = Array1::from_vec((1..=hidden).map(|i| i as f32 * 0.001).collect());

        // Create inputs
        let input = make_input_3d(batch, seq, hidden);
        let residual: Array3<f32> = Array3::from_shape_fn((batch, seq, hidden), |(b, s, h)| {
            ((b * seq * hidden + s * hidden + h) as f32) * 0.05 + 0.1
        });

        // Standard path: separate add + layernorm
        let layer_norm = LayerNorm::new(gamma.clone(), beta.clone(), eps);
        let added = &input + &residual;
        let std_output = layer_norm.forward(&added.view());

        // Optimized path: fused
        let mut opt_output = Array3::zeros((batch, seq, hidden));
        fused_residual_layernorm(
            &mut opt_output,
            &input.view(),
            &residual.view(),
            &gamma,
            &beta,
            eps,
        );

        let result = compare_arrays_3d(&std_output, &opt_output);

        println!("\n=== FUSED RESIDUAL+LAYERNORM PARITY (small) ===");
        println!("Shape: [{}, {}, {}]", batch, seq, hidden);
        println!("Max diff:  {:.6e}", result.max_diff);
        println!("Mean diff: {:.6e}", result.mean_diff);

        println!(
            "\nStandard output (first 8): {:?}",
            &std_output.as_slice().unwrap()[..8.min(std_output.len())]
        );
        println!(
            "Optimized output (first 8): {:?}",
            &opt_output.as_slice().unwrap()[..8.min(opt_output.len())]
        );

        assert!(
            result.max_diff < 1e-5,
            "Fused residual+layernorm parity failed. Max diff: {:.6e}",
            result.max_diff
        );

        Ok(())
    }

    #[test]
    fn test_fused_residual_layernorm_parity_medium() -> Result<()> {
        let (batch, seq, hidden) = (4, 16, 64);
        let eps = 1e-5;

        let gamma = Array1::from_vec((0..hidden).map(|i| 1.0 + (i as f32) * 0.001).collect());
        let beta = Array1::from_vec((0..hidden).map(|i| (i as f32) * 0.0001).collect());

        let input = make_input_3d(batch, seq, hidden);
        let residual: Array3<f32> = Array3::from_shape_fn((batch, seq, hidden), |(b, s, h)| {
            ((b * seq * hidden + s * hidden + h) as f32) * 0.02
        });

        let layer_norm = LayerNorm::new(gamma.clone(), beta.clone(), eps);
        let added = &input + &residual;
        let std_output = layer_norm.forward(&added.view());

        let mut opt_output = Array3::zeros((batch, seq, hidden));
        fused_residual_layernorm(
            &mut opt_output,
            &input.view(),
            &residual.view(),
            &gamma,
            &beta,
            eps,
        );

        let result = compare_arrays_3d(&std_output, &opt_output);

        println!("\n=== FUSED RESIDUAL+LAYERNORM PARITY (medium) ===");
        println!("Shape: [{}, {}, {}]", batch, seq, hidden);
        println!("Max diff:  {:.6e}", result.max_diff);
        println!("Mean diff: {:.6e}", result.mean_diff);

        assert!(
            result.max_diff < 1e-5,
            "Fused residual+layernorm parity failed. Max diff: {:.6e}",
            result.max_diff
        );

        Ok(())
    }

    // =========================================================================
    // MATMUL PARITY TESTS (LinearLayer methods)
    // =========================================================================

    #[test]
    fn test_matmul_into_blocked_parity() -> Result<()> {
        let (tokens, in_features, out_features) = (64, 128, 256);
        let mut count = 1;
        let bias_val = 0.01;

        let linear = make_linear(out_features, in_features, &mut count, bias_val);

        // Create input
        let input_data: Vec<f32> = (0..tokens * in_features)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let input = Array2::from_shape_vec((tokens, in_features), input_data)?;

        // Standard matmul
        let std_output = linear.matmul(&input.view());

        // Blocked matmul
        let mut blocked_output = Array2::zeros((tokens, out_features));
        linear.matmul_into_blocked(&input.view(), &mut blocked_output);

        let result = compare_arrays_2d(&std_output, &blocked_output);

        println!("\n=== MATMUL_INTO_BLOCKED PARITY ===");
        println!(
            "Shape: [{}, {}] @ [{}, {}] -> [{}, {}]",
            tokens, in_features, out_features, in_features, tokens, out_features
        );
        println!("Max diff:  {:.6e}", result.max_diff);
        println!("Mean diff: {:.6e}", result.mean_diff);

        assert!(
            result.max_diff < 1e-4,
            "matmul_into_blocked parity failed. Max diff: {:.6e}",
            result.max_diff
        );

        Ok(())
    }

    // =========================================================================
    // FULL LAYER PARITY TEST (for final verification)
    // =========================================================================

    #[test]
    fn test_full_layer_postnorm_vs_postnorm2_small() -> Result<()> {
        let (batch, seq, hidden, intermediate, heads) = (2, 3, 4, 8, 2);
        let bias_val = 0.01;

        // Use the same create_deterministic_layer logic
        let mut count = 1;

        let q_proj = make_linear(hidden, hidden, &mut count, bias_val);
        let k_proj = make_linear(hidden, hidden, &mut count, bias_val);
        let v_proj = make_linear(hidden, hidden, &mut count, bias_val);
        let out_proj = make_linear(hidden, hidden, &mut count, bias_val);

        let self_attn = EncoderSelfAttention::new(
            hidden,
            heads,
            q_proj.clone(),
            k_proj.clone(),
            v_proj.clone(),
            out_proj.clone(),
        );

        let ln1 = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden),
            Array1::from_elem(hidden, bias_val),
            1e-5,
        ));

        let fc1 = make_linear(intermediate, hidden, &mut count, bias_val);
        let fc2 = make_linear(hidden, intermediate, &mut count, bias_val);

        let feedforward = crate::feedforward::FeedForward::Standard(StdFeedForward::new(
            fc1.weights_view().to_owned(),
            fc1.bias().unwrap().clone(),
            fc2.weights_view().to_owned(),
            fc2.bias().unwrap().clone(),
            Activation::Gelu,
        ));

        let ln2 = Normalization::LayerNorm(LayerNorm::new(
            Array1::ones(hidden),
            Array1::from_elem(hidden, bias_val),
            1e-5,
        ));

        let optimized_attention =
            OptimizedSelfAttention::new(q_proj, k_proj, v_proj, out_proj, heads, true);

        let optimized_feedforward = OptimizedFeedForward::new(fc1, fc2, Activation::Gelu);

        let layer = EncoderLayer::new(self_attn, ln1, feedforward, ln2)
            .with_optimized(optimized_attention, optimized_feedforward);

        // Create inputs matching your golden test
        let input_data: Vec<f32> = (0..batch * seq * hidden)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let input = Array3::from_shape_vec((batch, seq, hidden), input_data)?;

        let mut mask = Array2::ones((batch, seq));
        mask[[1, 2]] = 0.0; // Match your test: last position of batch 1 masked

        let pos_bias_data: Vec<f32> = (0..heads * seq * seq).map(|i| (i as f32) * 0.01).collect();
        let pos_bias = Array4::from_shape_vec((1, heads, seq, seq), pos_bias_data)?;

        // Run postnorm (standard)
        let output_postnorm =
            layer.forward_postnorm(input.clone(), &mask, Some(&pos_bias), None)?;

        // Run postnorm2 (optimized)
        let output_postnorm2 =
            layer.forward_postnorm2(input.clone(), &mask, Some(&pos_bias), None)?;

        let result = compare_arrays_3d(&output_postnorm, &output_postnorm2);

        println!("\n=== FULL LAYER: POSTNORM vs POSTNORM2 (small) ===");
        println!("Shape: [{}, {}, {}]", batch, seq, hidden);
        println!("Max diff:  {:.6e}", result.max_diff);
        println!("Mean diff: {:.6e}", result.mean_diff);

        println!(
            "\nPostnorm output (first 8):  {:?}",
            &output_postnorm.as_slice().unwrap()[..8]
        );
        println!(
            "Postnorm2 output (first 8): {:?}",
            &output_postnorm2.as_slice().unwrap()[..8]
        );

        assert!(
            result.max_diff < 1e-4,
            "Full layer parity failed. Max diff: {:.6e}",
            result.max_diff
        );

        Ok(())
    }

    // =========================================================================
    // PERFORMANCE COMPARISON (run with --nocapture)
    // =========================================================================

    #[test]
    fn test_attention_performance() -> Result<()> {
        use std::time::Instant;

        let (batch, seq, hidden, heads) = (32, 128, 384, 6); // MiniLM-like
        let intermediate = 1536;
        let bias_val = 0.01;
        let mut count = 1;

        let q_proj = make_linear(hidden, hidden, &mut count, bias_val);
        let k_proj = make_linear(hidden, hidden, &mut count, bias_val);
        let v_proj = make_linear(hidden, hidden, &mut count, bias_val);
        let out_proj = make_linear(hidden, hidden, &mut count, bias_val);

        let std_attn = EncoderSelfAttention::new(
            hidden,
            heads,
            q_proj.clone(),
            k_proj.clone(),
            v_proj.clone(),
            out_proj.clone(),
        );

        let opt_attn = OptimizedSelfAttention::new(q_proj, k_proj, v_proj, out_proj, heads, true);

        let input = make_input_3d(batch, seq, hidden);
        let mask = Array2::ones((batch, seq));

        // Warmup
        let _ = std_attn.forward(&input, &mask, None, None)?;
        let mut scratch = EncoderScratch::new(batch, seq, hidden, heads, intermediate);
        opt_attn.forward(&input, &mask, None, None, &mut scratch)?;

        // Benchmark standard
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = std_attn.forward(&input, &mask, None, None)?;
        }
        let std_time = start.elapsed() / iterations;

        // Benchmark optimized
        let start = Instant::now();
        for _ in 0..iterations {
            opt_attn.forward(&input, &mask, None, None, &mut scratch)?;
        }
        let opt_time = start.elapsed() / iterations;

        println!("\n=== ATTENTION PERFORMANCE ===");
        println!("Shape: [{}, {}, {}], {} heads", batch, seq, hidden, heads);
        println!("Standard:  {:?}", std_time);
        println!("Optimized: {:?}", opt_time);
        println!(
            "Speedup:   {:.2}x",
            std_time.as_secs_f64() / opt_time.as_secs_f64()
        );

        Ok(())
    }

    #[test]
    fn test_ffn_performance() -> Result<()> {
        use std::time::Instant;

        let (batch, seq, hidden, intermediate) = (32, 128, 384, 1536);
        let bias_val = 0.01;
        let mut count = 1;

        let fc1 = make_linear(intermediate, hidden, &mut count, bias_val);
        let fc2 = make_linear(hidden, intermediate, &mut count, bias_val);

        let std_ffn = StdFeedForward::new(
            fc1.weights_view().to_owned(),
            fc1.bias().unwrap().clone(),
            fc2.weights_view().to_owned(),
            fc2.bias().unwrap().clone(),
            Activation::Gelu,
        );

        let opt_ffn = OptimizedFeedForward::new(fc1, fc2, Activation::Gelu);

        let input_3d = make_input_3d(batch, seq, hidden);
        let tokens = batch * seq;
        let input_2d = input_3d.view().into_shape((tokens, hidden))?.to_owned();

        // Warmup
        let _ = std_ffn.forward(&input_3d)?;
        let mut scratch_intermediate = Array2::zeros((tokens, intermediate));
        let mut opt_output = Array2::zeros((tokens, hidden));
        opt_ffn.forward(&input_2d.view(), &mut scratch_intermediate, &mut opt_output);

        // Benchmark
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = std_ffn.forward(&input_3d)?;
        }
        let std_time = start.elapsed() / iterations;

        let start = Instant::now();
        for _ in 0..iterations {
            opt_ffn.forward(&input_2d.view(), &mut scratch_intermediate, &mut opt_output);
        }
        let opt_time = start.elapsed() / iterations;

        println!("\n=== FFN PERFORMANCE ===");
        println!(
            "Shape: [{}, {}, {}] -> intermediate {}",
            batch, seq, hidden, intermediate
        );
        println!("Standard:  {:?}", std_time);
        println!("Optimized: {:?}", opt_time);
        println!(
            "Speedup:   {:.2}x",
            std_time.as_secs_f64() / opt_time.as_secs_f64()
        );

        Ok(())
    }
}
