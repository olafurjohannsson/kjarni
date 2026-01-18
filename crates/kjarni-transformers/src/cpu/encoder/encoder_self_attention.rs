use crate::activations::softmax_4d_inplace;
use crate::cpu::encoder::{buffers::EncoderBuffers, qkv_projection::QKVProjection};
use crate::linear_layer::LinearLayer;
use crate::rope::RoPE;
use crate::utils::linear_algebra::matmul_4d;
use anyhow::Result;
use ndarray::{Array2, Array3, ArrayView2, s};

pub struct EncoderSelfAttention {
    /// Unified QKV projection (handles fused vs separate internally)
    pub qkv_proj: QKVProjection,
    /// Output projection layer.
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
    ///
    /// Internally constructs a QKVProjection with automatic strategy selection
    /// (fused for hidden <= 512, separate for larger).
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - The model's hidden dimension.
    /// * `num_heads` - Number of attention heads.
    /// * `q` - Query projection weights.
    /// * `k` - Key projection weights.
    /// * `v` - Value projection weights.
    /// * `o` - Output projection weights.
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
    ///
    /// This is the allocating version for backward compatibility.
    /// For hot paths, use `forward_noalloc`.
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_bias: Option<&ndarray::Array4<f32>>,
        rope: Option<&RoPE>,
    ) -> Result<Array3<f32>> {
        let (batch, seq_len, _) = hidden_states.dim();
        let hidden_dim = self.num_heads * self.head_dim;

        // 1. Flatten & Project
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

        // 2. Reshape & Permute to [B, H, S, D]
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

        // 3. Compute attention scores: Q @ K^T
        let mut scores = matmul_4d(&q_heads, &k_heads_t);

        // 4. Scale
        if self.scale_qk {
            scores.mapv_inplace(|x| x * self.scale_factor);
        }

        // 5. Add position bias if provided
        if let Some(bias) = position_bias {
            scores = scores + bias;
        }

        // 6. Apply padding mask
        scores = crate::utils::apply_padding_mask(scores, attention_mask)?;

        // 7. Softmax
        softmax_4d_inplace(&mut scores);

        // 8. Compute context: Scores @ V
        let context = matmul_4d(&scores, &v_heads);

        // 9. Merge heads and output projection
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

    /// Forward pass with pre-allocated buffers (no allocation in hot path).
    ///
    /// # Note
    ///
    /// Still allocates: RoPE, 4D attention matmuls, reshape/permute operations.
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

        // 1. Flatten hidden states to 2D for QKV projection
        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((tokens, hidden_dim))?;

        // 2. QKV Projection (no-alloc) -> writes to buffers.q, buffers.k, buffers.v
        self.qkv_proj.forward_noalloc(&hidden_2d, buffers);

        // 3. Get slices and reshape to 3D [batch, seq, hidden]
        let q_slice = buffers.q.slice(s![..tokens, ..]);
        let k_slice = buffers.k.slice(s![..tokens, ..]);
        let v_slice = buffers.v.slice(s![..tokens, ..]);

        let mut q_3d = q_slice
            .to_owned()
            .into_shape_with_order((batch, seq_len, hidden_dim))?;
        let mut k_3d = k_slice
            .to_owned()
            .into_shape_with_order((batch, seq_len, hidden_dim))?;

        // 4. Apply RoPE if present
        if let Some(r) = rope {
            let (q_rot, k_rot) = r.apply_3d(&q_3d, &k_3d, self.num_heads, self.num_heads, 0)?;
            q_3d = q_rot;
            k_3d = k_rot;
        }

        // 5. Reshape to heads: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q_heads = q_3d
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .to_owned();

        let k_heads_t = k_3d
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 3, 1]) // Transposed
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

        // 6. Compute attention scores: Q @ K^T
        let scores_computed = matmul_4d(&q_heads, &k_heads_t);

        // 7. Copy into buffer and get mutable slice
        buffers
            .attn_scores
            .slice_mut(s![..batch, .., ..seq_len, ..seq_len])
            .assign(&scores_computed);

        // 8. Scale in-place
        if self.scale_qk {
            buffers
                .attn_scores
                .slice_mut(s![..batch, .., ..seq_len, ..seq_len])
                .mapv_inplace(|x| x * self.scale_factor);
        }

        // 9. Add position bias if provided
        if let Some(bias) = position_bias {
            let bias_slice = bias.slice(s![.., .., ..seq_len, ..seq_len]);
            buffers
                .attn_scores
                .slice_mut(s![..batch, .., ..seq_len, ..seq_len])
                .zip_mut_with(&bias_slice, |s, &b| *s += b);
        }

        // 10. Apply padding mask in-place
        Self::apply_padding_mask_inplace(
            &mut buffers
                .attn_scores
                .slice_mut(s![..batch, .., ..seq_len, ..seq_len]),
            attention_mask,
        );

        // 11. Softmax in-place
        Self::softmax_4d_inplace(&mut buffers.attn_scores.slice_mut(s![
            ..batch,
            ..,
            ..seq_len,
            ..seq_len
        ]));

        // 12. Compute context: Scores @ V
        let scores_view = buffers
            .attn_scores
            .slice(s![..batch, .., ..seq_len, ..seq_len]);
        let context_computed = matmul_4d(&scores_view.to_owned(), &v_heads);

        // 13. Copy into buffer
        buffers
            .attn_context
            .slice_mut(s![..batch, .., ..seq_len, ..])
            .assign(&context_computed);

        // 14. Merge heads: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let context_slice = buffers.attn_context.slice(s![..batch, .., ..seq_len, ..]);
        let context_perm = context_slice
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .to_owned();
        let context_flat = context_perm.into_shape_with_order((tokens, hidden_dim))?;

        // 15. Output projection (no-alloc) -> writes to buffers.attn_output
        self.out_proj
            .matmul_noalloc(&context_flat.view(), &mut buffers.attn_output);

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

    /// Softmax along last dimension, in-place.
    #[inline]
    fn softmax_4d_inplace(scores: &mut ndarray::ArrayViewMut4<f32>) {
        let (batch, heads, seq_q, seq_k) = scores.dim();

        for b in 0..batch {
            for h in 0..heads {
                for q in 0..seq_q {
                    // Max for numerical stability
                    let mut max_val = f32::NEG_INFINITY;
                    for k in 0..seq_k {
                        max_val = max_val.max(scores[[b, h, q, k]]);
                    }

                    // Exp and sum
                    let mut sum = 0.0f32;
                    for k in 0..seq_k {
                        let exp_val = (scores[[b, h, q, k]] - max_val).exp();
                        scores[[b, h, q, k]] = exp_val;
                        sum += exp_val;
                    }

                    // Normalize
                    if sum > 0.0 {
                        for k in 0..seq_k {
                            scores[[b, h, q, k]] /= sum;
                        }
                    }
                }
            }
        }
    }
    /// Q @ K^T matmul for attention scores.
    ///
    /// NOTE: This is a placeholder - true noalloc would need custom implementation.
    #[inline]
    fn matmul_qk_noalloc(
        q: &ndarray::Array4<f32>,       // [batch, heads, seq, head_dim]
        k_t: &ndarray::Array4<f32>,     // [batch, heads, head_dim, seq]
        out: &mut ndarray::Array4<f32>, // [batch, heads, seq, seq]
    ) {
        let result = matmul_4d(q, k_t);
        out.assign(&result);
    }

    /// Scores @ V matmul for context.
    ///
    /// NOTE: This is a placeholder - true noalloc would need custom implementation.
    #[inline]
    fn matmul_sv_noalloc(
        scores: &ndarray::Array4<f32>,  // [batch, heads, seq, seq]
        v: &ndarray::Array4<f32>,       // [batch, heads, seq, head_dim]
        out: &mut ndarray::Array4<f32>, // [batch, heads, seq, head_dim]
    ) {
        let result = matmul_4d(scores, v);
        out.assign(&result);
    }
}
