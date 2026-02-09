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
            scores = scores + bias;
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

        //  Reshape to heads
        let (q_heads, k_heads_t, v_heads) = if let Some(r) = rope {
            // RoPe need 3D intermediate for RoPE application
            let mut q_3d = q_slice
                .to_owned()
                .into_shape_with_order((batch, seq_len, hidden_dim))?;
            let mut k_3d = k_slice
                .to_owned()
                .into_shape_with_order((batch, seq_len, hidden_dim))?;

            let (q_rot, k_rot) = r.apply_3d(&q_3d, &k_3d, self.num_heads, self.num_heads, 0)?;
            q_3d = q_rot;
            k_3d = k_rot;

            let q_heads = q_3d
                .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned();

            let k_heads_t = k_3d
                .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned();

            let v_3d = v_slice
                .to_owned()
                .into_shape_with_order((batch, seq_len, hidden_dim))?;
            let v_heads = v_3d
                .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned();

            (q_heads, k_heads_t, v_heads)
        } else {
            let q_4d = q_slice.to_shape((batch, seq_len, self.num_heads, self.head_dim))?;
            let q_heads = q_4d
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned();

            let k_4d = k_slice.to_shape((batch, seq_len, self.num_heads, self.head_dim))?;
            let k_heads_t = k_4d
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned();

            let v_4d = v_slice.to_shape((batch, seq_len, self.num_heads, self.head_dim))?;
            let v_heads = v_4d
                .permuted_axes([0, 2, 1, 3])
                .as_standard_layout()
                .to_owned();

            (q_heads, k_heads_t, v_heads)
        };

        // Compute attention scores: Q @ K^T 
        matmul_4d_into(
            &q_heads.view(),
            &k_heads_t.view(),
            &mut buffers
                .attn_scores
                .slice_mut(s![..batch, .., ..seq_len, ..seq_len]),
        );

        // Scale in-place
        if self.scale_qk {
            buffers
                .attn_scores
                .slice_mut(s![..batch, .., ..seq_len, ..seq_len])
                .mapv_inplace(|x| x * self.scale_factor);
        }

        //Add position bias if provided
        if let Some(bias) = position_bias {
            let bias_slice = bias.slice(s![.., .., ..seq_len, ..seq_len]);
            buffers
                .attn_scores
                .slice_mut(s![..batch, .., ..seq_len, ..seq_len])
                .zip_mut_with(&bias_slice, |s, &b| *s += b);
        }

        // Apply padding mask in-place
        Self::apply_padding_mask_inplace(
            &mut buffers
                .attn_scores
                .slice_mut(s![..batch, .., ..seq_len, ..seq_len]),
            attention_mask,
        );

        // Softmax in-place
        crate::activations::softmax_4d_view_inplace(&mut buffers.attn_scores.slice_mut(s![
            ..batch,
            ..,
            ..seq_len,
            ..seq_len
        ]));

        // Compute context: Scores @ V 
        matmul_4d_into(
            &buffers
                .attn_scores
                .slice(s![..batch, .., ..seq_len, ..seq_len]),
            &v_heads.view(),
            &mut buffers
                .attn_context
                .slice_mut(s![..batch, .., ..seq_len, ..self.head_dim]),
        );

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
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) {
    use rayon::prelude::*;

    let hidden = num_heads * head_dim;

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
