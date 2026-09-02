use crate::activations::softmax_4d_inplace;
use crate::cpu::encoder::{buffers::EncoderBuffers, qkv_projection::QKVProjection};
use crate::linear_layer::LinearLayer;
use crate::rope::RoPE;
use crate::utils::linear_algebra::matmul_4d;
use anyhow::Result;
use ndarray::{Array2, Array3, ArrayView4, Zip, s};

pub struct EncoderSelfAttention {
    pub qkv_proj: QKVProjection,
    pub out_proj: LinearLayer,

    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Scaling factor: 1 / sqrt(head_dim).
    pub scale_factor: f32,
    /// Whether to scale Q@K by sqrt(head_dim). False for T5.
    pub scale_qk: bool,
}

impl EncoderSelfAttention {
    /// Creates a new encoder self-attention module.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        q: LinearLayer,
        k: LinearLayer,
        v: LinearLayer,
        o: LinearLayer,
    ) -> Self {
        let head_dim = hidden_size / num_heads;

        // Construct QKVProjection with automatic strategy selection
        let qkv_proj = QKVProjection::new(q, k, v);

        Self {
            qkv_proj,
            out_proj: o,
            num_heads,
            head_dim,
            scale_factor: 1.0 / (head_dim as f32).sqrt(),
            scale_qk: true,
        }
    }

    /// Disables Q@K scaling (for T5-style attention).
    pub fn with_no_qk_scaling(mut self) -> Self {
        self.scale_qk = false;
        self
    }

    /// Returns the hidden size.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Performs the forward pass of encoder self-attention.
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_bias: Option<&ndarray::Array4<f32>>,
        rope: Option<&RoPE>,
    ) -> Result<Array3<f32>> {
        let (batch, seq_len, _) = hidden_states.dim();
        let hidden_dim = self.num_heads * self.head_dim;

        // Flatten & Project
        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((batch * seq_len, hidden_dim))?;

        let (q, k, v) = self.qkv_proj.forward(&hidden_2d);

        let mut q_3d = q.into_shape_with_order((batch, seq_len, hidden_dim))?;
        let mut k_3d = k.into_shape_with_order((batch, seq_len, hidden_dim))?;

        if let Some(r) = rope {
            let (q_rot, k_rot) = r.apply_3d(&q_3d, &k_3d, self.num_heads, self.num_heads, 0)?;
            q_3d = q_rot;
            k_3d = k_rot;
        }

        // Reshape & Permute to [B, H, S, D]
        let q_heads = q_3d
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .to_owned();

        let k_heads_t = k_3d
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 3, 1])
            .to_owned();

        let v_heads = v
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .to_owned();

        // Compute attention scores: Q @ K^T
        let mut scores = matmul_4d(&q_heads, &k_heads_t);

        // Scale
        if self.scale_qk {
            scores.mapv_inplace(|x| x * self.scale_factor);
        }

        // Add position bias if provided
        if let Some(bias) = position_bias {
            scores += bias;
        }

        // Apply padding mask
        scores = crate::utils::apply_padding_mask(scores, attention_mask)?;

        // Softmax
        softmax_4d_inplace(&mut scores);

        // Compute context: Scores @ V
        let context = matmul_4d(&scores, &v_heads);

        // Merge heads and output projection
        let context_contig = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .to_owned();

        let context_flat = context_contig.into_shape_with_order((batch * seq_len, hidden_dim))?;
        let output = self.out_proj.matmul(&context_flat.view());

        let output_3d = output
            .as_standard_layout()
            .into_owned()
            .into_shape_with_order((batch, seq_len, hidden_dim))?;

        Ok(output_3d)
    }

    /// Forward pass with pre-allocated buffers
    pub fn forward_noalloc(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_bias: Option<&ndarray::Array4<f32>>,
        rope: Option<&RoPE>,
        buffers: &mut EncoderBuffers,
    ) -> Result<()> {
        let (batch, seq_len, hidden_dim) = hidden_states.dim();
        let tokens = batch * seq_len;

        #[cfg(debug_assertions)]
        buffers.validate_dimensions(
            batch,
            seq_len,
            hidden_dim,
            self.num_heads,
            buffers.intermediate_dim(),
        );

        //  Flatten hidden states to 2D for QKV projection
        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((tokens, hidden_dim))?;

        // writes to buffers.q, buffers.k, buffers.v
        self.qkv_proj.forward_noalloc(&hidden_2d, buffers);

        // Get slices
        let q_slice = buffers.q.slice(s![..tokens, ..]);
        let k_slice = buffers.k.slice(s![..tokens, ..]);
        let v_slice = buffers.v.slice(s![..tokens, ..]);

        //  Reshape to heads, into buffers that persist across layers.
        //
        // This used to build three arrays per layer with
        // `permuted_axes(..).as_standard_layout().to_owned()`. On MiniLM at batch
        // 64 that is 1.4MB each, so a six-layer forward pass allocated and dropped
        // about 25MB against a 30MB L3 and evicted the weights every layer. The
        // transposes themselves were only part of the cost: the FFN's GEMMs, which
        // touch none of this, were running at half speed because their weights had
        // been pushed out of cache.
        if let Some(r) = rope {
            let q_3d = q_slice
                .to_owned()
                .into_shape_with_order((batch, seq_len, hidden_dim))?;
            let k_3d = k_slice
                .to_owned()
                .into_shape_with_order((batch, seq_len, hidden_dim))?;
            let (q_rot, k_rot) = r.apply_3d(&q_3d, &k_3d, self.num_heads, self.num_heads, 0)?;

            let q_4d =
                q_rot.into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?;
            let k_4d =
                k_rot.into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?;

            buffers
                .q_heads
                .slice_mut(s![..batch, .., ..seq_len, ..self.head_dim])
                .assign(&q_4d.view().permuted_axes([0, 2, 1, 3]));
            buffers
                .k_heads_t
                .slice_mut(s![..batch, .., ..self.head_dim, ..seq_len])
                .assign(&k_4d.view().permuted_axes([0, 2, 3, 1]));
        } else {
            let q_4d = q_slice.to_shape((batch, seq_len, self.num_heads, self.head_dim))?;
            let k_4d = k_slice.to_shape((batch, seq_len, self.num_heads, self.head_dim))?;

            buffers
                .q_heads
                .slice_mut(s![..batch, .., ..seq_len, ..self.head_dim])
                .assign(&q_4d.view().permuted_axes([0, 2, 1, 3]));
            buffers
                .k_heads_t
                .slice_mut(s![..batch, .., ..self.head_dim, ..seq_len])
                .assign(&k_4d.view().permuted_axes([0, 2, 3, 1]));
        }

        // V is untouched by RoPE, so it is the same either way.
        {
            let v_4d = v_slice.to_shape((batch, seq_len, self.num_heads, self.head_dim))?;
            buffers
                .v_heads
                .slice_mut(s![..batch, .., ..seq_len, ..self.head_dim])
                .assign(&v_4d.view().permuted_axes([0, 2, 1, 3]));
        }

        // Two paths. The T5 style relative position bias is a [1, heads, seq, seq]
        // array added to the scores, so that case still needs the whole tensor and
        // keeps the materialised form. Everything else tiles by query block and
        // never builds the tensor at all.
        if position_bias.is_some() || seq_len < TILE_MIN_SEQ {
            matmul_4d_into(
                &buffers
                    .q_heads
                    .slice(s![..batch, .., ..seq_len, ..self.head_dim]),
                &buffers
                    .k_heads_t
                    .slice(s![..batch, .., ..self.head_dim, ..seq_len]),
                &mut buffers
                    .attn_scores
                    .slice_mut(s![..batch, .., ..seq_len, ..seq_len]),
            );

            if position_bias.is_some() && self.scale_qk {
                buffers
                    .attn_scores
                    .slice_mut(s![..batch, .., ..seq_len, ..seq_len])
                    .mapv_inplace(|x| x * self.scale_factor);
            }
            if let Some(bias) = position_bias {
                let bias_slice = bias.slice(s![.., .., ..seq_len, ..seq_len]);
                buffers
                    .attn_scores
                    .slice_mut(s![..batch, .., ..seq_len, ..seq_len])
                    .zip_mut_with(&bias_slice, |s, &b| *s += b);
            }
            if position_bias.is_some() {
                Self::apply_padding_mask_inplace(
                    &mut buffers
                        .attn_scores
                        .slice_mut(s![..batch, .., ..seq_len, ..seq_len]),
                    attention_mask,
                );
                crate::activations::softmax_4d_view_inplace(
                    &mut buffers
                        .attn_scores
                        .slice_mut(s![..batch, .., ..seq_len, ..seq_len]),
                );
            } else {
                fused_scale_mask_softmax(
                    &mut buffers
                        .attn_scores
                        .slice_mut(s![..batch, .., ..seq_len, ..seq_len]),
                    attention_mask,
                    self.scale_qk.then_some(self.scale_factor),
                );
            }

            matmul_4d_into(
                &buffers
                    .attn_scores
                    .slice(s![..batch, .., ..seq_len, ..seq_len]),
                &buffers
                    .v_heads
                    .slice(s![..batch, .., ..seq_len, ..self.head_dim]),
                &mut buffers
                    .attn_context
                    .slice_mut(s![..batch, .., ..seq_len, ..self.head_dim]),
            );
        } else {
            // Disjoint fields, so the immutable views of q, k and v coexist with
            // the mutable view of the context.
            let EncoderBuffers {
                q_heads,
                k_heads_t,
                v_heads,
                attn_context,
                ..
            } = &mut *buffers;

            tiled_attention(
                &q_heads.slice(s![..batch, .., ..seq_len, ..self.head_dim]),
                &k_heads_t.slice(s![..batch, .., ..self.head_dim, ..seq_len]),
                &v_heads.slice(s![..batch, .., ..seq_len, ..self.head_dim]),
                attention_mask,
                self.scale_qk.then_some(self.scale_factor),
                &mut attn_context.slice_mut(s![..batch, .., ..seq_len, ..self.head_dim]),
            );
        }

        // Merge heads: [batch, heads, seq, head_dim] -> [tokens, hidden] (no-alloc)
        let context_slice = buffers
            .attn_context
            .slice(s![..batch, .., ..seq_len, ..self.head_dim]);
        permute_merge_heads_into(
            &context_slice,
            &mut buffers.merge_scratch.slice_mut(s![..tokens, ..]),
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        );

        // Output projection (no-alloc) -> writes to buffers.attn_output
        self.out_proj.matmul_noalloc(
            &buffers.merge_scratch.slice(s![..tokens, ..]),
            &mut buffers.attn_output,
        );

        Ok(())
    }

    /// Applies padding mask in-place.
    #[inline]
    fn apply_padding_mask_inplace(scores: &mut ndarray::ArrayViewMut4<f32>, mask: &Array2<f32>) {
        let (batch, num_heads, seq_q, seq_k) = scores.dim();

        for b in 0..batch {
            for k in 0..seq_k {
                if mask[[b, k]] == 0.0 {
                    for h in 0..num_heads {
                        for q in 0..seq_q {
                            scores[[b, h, q, k]] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }
    }
}

/// Scale, mask and softmax over the whole score tensor, for sequences short
/// enough that it stays in cache anyway.
fn fused_scale_mask_softmax(
    scores: &mut ndarray::ArrayViewMut4<f32>,
    mask: &Array2<f32>,
    scale: Option<f32>,
) {
    use rayon::prelude::*;

    scores
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(b, mut heads)| {
            heads.outer_iter_mut().for_each(|mut queries| {
                queries.outer_iter_mut().for_each(|mut row| {
                    let mut max = f32::NEG_INFINITY;
                    for (k, v) in row.iter_mut().enumerate() {
                        let x = match scale {
                            Some(s) => *v * s,
                            None => *v,
                        };
                        let x = if mask[[b, k]] == 0.0 {
                            f32::NEG_INFINITY
                        } else {
                            x
                        };
                        *v = x;
                        if x > max {
                            max = x;
                        }
                    }

                    if !max.is_finite() {
                        row.fill(0.0);
                        return;
                    }

                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        let e = (*v - max).exp();
                        *v = e;
                        sum += e;
                    }
                    if sum > 0.0 {
                        let inv = 1.0 / sum;
                        for v in row.iter_mut() {
                            *v *= inv;
                        }
                    }
                });
            });
        });
}

/// Rows of queries handled per tile. At 64 rows a tile is `64 * seq_k * 4` bytes,
/// which is 56KB at 220 tokens and stays in L2 while it is softmaxed and
/// multiplied by V.
const QUERY_BLOCK: usize = 64;

/// Below this sequence length the score tensor is small enough to stay in cache,
/// so materialising it costs nothing and tiling only adds per head setup. Every
/// regression from tiling was at 18 token inputs, and every gain was at 220
/// tokens and above.
const TILE_MIN_SEQ: usize = 64;

/// Attention over blocks of queries, so the score tile never reaches memory.
///
/// The materialised path writes a `[batch, heads, seq, seq]` score array, reads it
/// back for the softmax and reads it a third time for the context matmul. On
/// MiniLM at batch 64 and 220 tokens that array is 148MB per layer, which is why
/// the softmax measured about a fifth of a batched forward pass while accounting
/// for well under one percent of its arithmetic, and why cost grows with the
/// square of the sequence faster than torch's.
///
/// Each tile here spans the whole key dimension, so a row is scaled, masked,
/// exponentiated and normalised over exactly the same values in the same order as
/// before. No online rescaling is needed and the results are bit identical, which
/// `encoder_path_agreement` checks against the allocating path.
#[allow(clippy::too_many_arguments)]
fn tiled_attention(
    q_heads: &ArrayView4<f32>,
    k_heads_t: &ArrayView4<f32>,
    v_heads: &ArrayView4<f32>,
    mask: &Array2<f32>,
    scale: Option<f32>,
    context: &mut ndarray::ArrayViewMut4<f32>,
) {
    let (_, _, seq_q, head_dim) = q_heads.dim();
    let seq_k = k_heads_t.dim().3;

    Zip::from(context.outer_iter_mut())
        .and(q_heads.outer_iter())
        .and(k_heads_t.outer_iter())
        .and(v_heads.outer_iter())
        .and(mask.outer_iter())
        .par_for_each(|mut ctx_b, q_b, kt_b, v_b, mask_b| {
            Zip::from(ctx_b.outer_iter_mut())
                .and(q_b.outer_iter())
                .and(kt_b.outer_iter())
                .and(v_b.outer_iter())
                .par_for_each(|mut ctx_h, q_h, kt_h, v_h| {
                    let q_s = q_h.as_standard_layout();
                    let kt_s = kt_h.as_standard_layout();
                    let v_s = v_h.as_standard_layout();
                    let q_sl = q_s.as_slice().expect("q block contiguous");
                    let kt_sl = kt_s.as_slice().expect("k^T block contiguous");
                    let v_sl = v_s.as_slice().expect("v block contiguous");

                    let rows_max = QUERY_BLOCK.min(seq_q);
                    let mut tile = vec![0.0f32; rows_max * seq_k];
                    let mut out_block = vec![0.0f32; rows_max * head_dim];

                    let mut q0 = 0;
                    while q0 < seq_q {
                        let rows = QUERY_BLOCK.min(seq_q - q0);

                        faer::linalg::matmul::matmul(
                            faer::mat::from_row_major_slice_mut(
                                &mut tile[..rows * seq_k],
                                rows,
                                seq_k,
                            ),
                            faer::mat::from_row_major_slice(
                                &q_sl[q0 * head_dim..(q0 + rows) * head_dim],
                                rows,
                                head_dim,
                            ),
                            faer::mat::from_row_major_slice(kt_sl, head_dim, seq_k),
                            None,
                            1.0,
                            faer::Parallelism::None,
                        );

                        for r in 0..rows {
                            let row = &mut tile[r * seq_k..(r + 1) * seq_k];
                            let mut max = f32::NEG_INFINITY;
                            for (k, v) in row.iter_mut().enumerate() {
                                let x = match scale {
                                    Some(s) => *v * s,
                                    None => *v,
                                };
                                let x = if mask_b[k] == 0.0 {
                                    f32::NEG_INFINITY
                                } else {
                                    x
                                };
                                *v = x;
                                if x > max {
                                    max = x;
                                }
                            }

                            if !max.is_finite() {
                                row.fill(0.0);
                                continue;
                            }

                            let mut sum = 0.0f32;
                            for v in row.iter_mut() {
                                let e = (*v - max).exp();
                                *v = e;
                                sum += e;
                            }
                            if sum > 0.0 {
                                let inv = 1.0 / sum;
                                for v in row.iter_mut() {
                                    *v *= inv;
                                }
                            }
                        }

                        faer::linalg::matmul::matmul(
                            faer::mat::from_row_major_slice_mut(
                                &mut out_block[..rows * head_dim],
                                rows,
                                head_dim,
                            ),
                            faer::mat::from_row_major_slice(&tile[..rows * seq_k], rows, seq_k),
                            faer::mat::from_row_major_slice(v_sl, seq_k, head_dim),
                            None,
                            1.0,
                            faer::Parallelism::None,
                        );

                        let mut dst = ctx_h.slice_mut(s![q0..q0 + rows, ..]);
                        if let Some(d) = dst.as_slice_mut() {
                            d.copy_from_slice(&out_block[..rows * head_dim]);
                        } else {
                            dst.assign(
                                &ndarray::ArrayView2::from_shape(
                                    (rows, head_dim),
                                    &out_block[..rows * head_dim],
                                )
                                .expect("shape"),
                            );
                        }

                        q0 += rows;
                    }
                });
        });
}

/// Batched matmul
#[inline]
pub fn matmul_4d_into(
    a: &ArrayView4<f32>,
    b: &ArrayView4<f32>,
    out: &mut ndarray::ArrayViewMut4<f32>,
) {
    let (_, _, m, k) = a.dim();
    let n = b.dim().3;

    Zip::from(out.outer_iter_mut())
        .and(a.outer_iter())
        .and(b.outer_iter())
        .par_for_each(|mut out_b, a_b, b_b| {
            // Per-batch scratch for strided output
            let mut scratch = ndarray::Array2::<f32>::zeros((m, n));

            Zip::from(out_b.outer_iter_mut())
                .and(a_b.outer_iter())
                .and(b_b.outer_iter())
                .for_each(|mut out_h, a_h, b_h| {
                    let a_s = a_h.as_standard_layout();
                    let b_s = b_h.as_standard_layout();

                    // Check if output head is contiguous
                    if let Some(o_s) = out_h.as_slice_mut() {
                        // Fast path: write directly
                        faer::linalg::matmul::matmul(
                            faer::mat::from_row_major_slice_mut(o_s, m, n),
                            faer::mat::from_row_major_slice(a_s.as_slice().unwrap(), m, k),
                            faer::mat::from_row_major_slice(b_s.as_slice().unwrap(), k, n),
                            None,
                            1.0,
                            faer::Parallelism::None,
                        );
                    } else {
                        // Strided output: use scratch and copy
                        faer::linalg::matmul::matmul(
                            faer::mat::from_row_major_slice_mut(
                                scratch.as_slice_mut().unwrap(),
                                m,
                                n,
                            ),
                            faer::mat::from_row_major_slice(a_s.as_slice().unwrap(), m, k),
                            faer::mat::from_row_major_slice(b_s.as_slice().unwrap(), k, n),
                            None,
                            1.0,
                            faer::Parallelism::None,
                        );
                        out_h.assign(&scratch);
                    }
                });
        });
}

#[inline]
fn permute_merge_heads_into(
    context: &ndarray::ArrayView4<f32>,
    output: &mut ndarray::ArrayViewMut2<f32>,
    _batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) {
    use rayon::prelude::*;

    let _hidden = num_heads * head_dim;

    // Parallel over tokens
    output
        .axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(token_idx, mut out_row)| {
            let b = token_idx / seq_len;
            let s = token_idx % seq_len;
            let out_ptr = out_row.as_mut_ptr();

            for h in 0..num_heads {
                let head_slice = context.slice(ndarray::s![b, h, s, ..]);
                if let Some(src) = head_slice.as_slice() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr(),
                            out_ptr.add(h * head_dim),
                            head_dim,
                        );
                    }
                } else {
                    for d in 0..head_dim {
                        unsafe {
                            *out_ptr.add(h * head_dim + d) = context[[b, h, s, d]];
                        }
                    }
                }
            }
        });
}
