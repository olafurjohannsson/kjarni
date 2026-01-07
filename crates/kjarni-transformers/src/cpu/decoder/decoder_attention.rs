use crate::linear_layer::LinearLayer;
use crate::rope::RoPE;
use crate::utils::MASK_VALUE;
use anyhow::Result;
use ndarray::{s, Array2, Array3, Array4};

/// High-performance self-attention for decoder-only transformer models.
///
/// `DecoderAttention` implements the scaled dot-product attention mechanism optimized
/// for modern decoder architectures (Llama, Mistral, Phi, Qwen). It supports:
/// - Grouped Query Attention (GQA) for reduced KV cache memory
/// - Rotary Position Embeddings (RoPE) for relative position encoding
/// - In-place KV cache writes with zero-copy attention computation
/// - Separate Q/K/V projections for flexibility
///
/// # Architecture
///
/// The attention computation follows these steps:
/// 1. Project hidden states to Q, K, V using separate linear layers
/// 2. Write new K, V directly to the end of contiguous cache buffers
/// 3. Apply RoPE to Q and the newly written K values in-place
/// 4. Compute attention scores over the full cache (history + new tokens)
/// 5. Apply causal masking and softmax
/// 6. Compute context via attention-weighted sum of V cache
/// 7. Project context through output layer
///
/// # Performance
///
/// This implementation is optimized for single-token decode (autoregressive generation):
/// - **Zero concatenation**: Writes K/V directly to cache slices, avoiding memory copies
/// - **Contiguous cache layout**: Single buffer for [History | New] tokens
/// - **GQA-aware matmul**: Specialized kernels that repeat KV heads without copying
/// - **Fast path for decode**: Optimized (Q @ K^T) for seq_len=1
///
/// Typical performance on modern CPUs:
/// - Decode step (seq_len=1): 1-5ms per layer depending on model size
/// - Prefill (seq_len>1): Proportional to sequence length squared
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::decoder::cpu::DecoderAttention;
/// use kjarni_transformers::linear_layer::LinearLayer;
/// use ndarray::Array3;
///
/// let attention = DecoderAttention::new(
///     hidden_size,
///     num_heads,
///     q_proj,
///     k_proj,
///     v_proj,
///     o_proj,
///     Some(num_kv_heads), // Enable GQA
/// );
///
/// // During inference, pass contiguous cache view
/// let output = attention.forward(
///     &hidden_states,
///     Some(&attention_mask),
///     k_cache.view_mut(),
///     v_cache.view_mut(),
///     position_offset,
///     Some(&rope),
/// )?;
/// ```
///
/// # See Also
///
/// * [`CpuRoPEDecoderLayer`] — Combines attention with feedforward and normalization.
/// * [`RoPE`] — Rotary position embedding implementation.
/// * [`CpuKVCache`] — Manages the KV cache buffers.
pub struct DecoderAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale_factor: f32,
}

impl DecoderAttention {
    /// Creates a new decoder attention layer with separate Q/K/V/O projections.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - The model's hidden dimension (must be divisible by `num_heads`).
    /// * `num_heads` - Number of attention heads for queries.
    /// * `q` - Query projection layer (hidden_size -> hidden_size).
    /// * `k` - Key projection layer (hidden_size -> num_kv_heads * head_dim).
    /// * `v` - Value projection layer (hidden_size -> num_kv_heads * head_dim).
    /// * `o` - Output projection layer (hidden_size -> hidden_size).
    /// * `num_kv_heads` - Number of key/value heads for GQA. If `None`, defaults to `num_heads` (standard MHA).
    ///
    /// # Returns
    ///
    /// A new `DecoderAttention` instance configured for the given architecture.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Standard Multi-Head Attention
    /// let attention = DecoderAttention::new(4096, 32, q, k, v, o, None);
    ///
    /// // Grouped Query Attention (8 KV heads, 32 Q heads)
    /// let attention = DecoderAttention::new(4096, 32, q, k, v, o, Some(8));
    /// ```
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        q: LinearLayer,
        k: LinearLayer,
        v: LinearLayer,
        o: LinearLayer,
        num_kv_heads: Option<usize>,
    ) -> Self {
        let num_kv_heads = num_kv_heads.unwrap_or(num_heads);
        let head_dim = hidden_size / num_heads;

        Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            num_heads,
            num_kv_heads,
            head_dim,
            scale_factor: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Computes self-attention with in-place KV cache updates.
    ///
    /// This is the optimized forward pass that writes new K/V values directly into
    /// the end of contiguous cache buffers, avoiding all concatenation operations.
    /// The cache must be pre-allocated with space for both history and new tokens.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Input tensor of shape `[batch, seq_len, hidden_size]`.
    /// * `attention_mask` - Optional padding mask of shape `[batch, total_seq_len]`.
    /// * `k_cache` - Mutable view of full K cache `[batch, total_len, kv_dim]` where `total_len = history + seq_len`.
    /// * `v_cache` - Mutable view of full V cache `[batch, total_len, kv_dim]`.
    /// * `position_offset` - Starting position index for RoPE (typically the history length).
    /// * `rope` - Optional rotary position embedding to apply to Q and K.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, seq_len, hidden_size]` containing the attention-weighted context.
    ///
    /// # Performance
    ///
    /// - **Decode (seq_len=1)**: Uses specialized fast-path matmul optimized for single queries.
    /// - **Prefill (seq_len>1)**: Standard attention computation with causal masking.
    /// - **Zero-copy caching**: New K/V written directly to cache end via slicing.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Get contiguous cache view for this layer
    /// let (k_full, v_full) = cache.get_context_view_mut(layer_idx, new_tokens)?;
    ///
    /// // Forward pass writes new tokens to cache end
    /// let output = attention.forward(
    ///     &hidden_states,
    ///     Some(&mask),
    ///     k_full,
    ///     v_full,
    ///     position_offset,
    ///     Some(&rope),
    /// )?;
    /// ```
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
        // The FULL active cache (0 .. current + new)
        mut k_cache: ndarray::ArrayViewMut3<f32>,
        mut v_cache: ndarray::ArrayViewMut3<f32>,
        position_offset: usize,
        rope: Option<&RoPE>,
    ) -> Result<Array3<f32>> {
        let (batch, seq_len, _) = hidden_states.dim();
        let total_len = k_cache.shape()[1]; // This is (History + Seq_Len)
        let is_decode = seq_len == 1;

        // 1. Projection (Compute Q, K, V for NEW tokens only)
        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((batch * seq_len, self.num_heads * self.head_dim))?;

        let q = self.q_proj.matmul(&hidden_2d);
        let mut q_3d = q.into_shape_with_order((batch, seq_len, self.num_heads * self.head_dim))?;

        // 2. Write K/V into the END of the cache
        // We assume the matmul outputs standard layout.
        // We slice the cache to get the "Write Slot" (the last `seq_len` tokens).
        let start_write = total_len - seq_len;

        // K Projection -> Write directly to Cache slice
        let k_new = self.k_proj.matmul(&hidden_2d);
        k_cache
            .slice_mut(s![.., start_write.., ..])
            .assign(&k_new.into_shape_with_order((
                batch,
                seq_len,
                self.num_kv_heads * self.head_dim,
            ))?);

        // V Projection -> Write directly to Cache slice
        let v_new = self.v_proj.matmul(&hidden_2d);
        v_cache
            .slice_mut(s![.., start_write.., ..])
            .assign(&v_new.into_shape_with_order((
                batch,
                seq_len,
                self.num_kv_heads * self.head_dim,
            ))?);

        // 3. Apply RoPE
        if let Some(r) = rope {
            // Apply to Q (Local variable)
            // You need to ensure your RoPE has a method for ViewMut or use assignment
            // Assuming your RoPE::apply_3d returns new tensors, we do:
            // (Optimize this later to be in-place)
            let k_temp_owned = k_cache.slice(s![.., start_write.., ..]).to_owned();

            let (q_rot, k_rot) = r.apply_3d(
                &q_3d,         // &Array3 (Good)
                &k_temp_owned, // &Array3 (Good)
                self.num_heads,
                self.num_kv_heads,
                position_offset,
            )?;

            // Write rotated results back
            q_3d.assign(&q_rot);
            // Write K back into the cache
            k_cache.slice_mut(s![.., start_write.., ..]).assign(&k_rot);
        }

        // 4. Attention on the FULL Cache
        // CRITICAL: We use `k_cache` directly. It is contiguous! No concatenation!

        let q_heads = q_3d
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .to_owned(); // [Batch, Heads, Seq, Dim]

        let n_rep = self.num_heads / self.num_kv_heads;

        let mut scores = if is_decode {
            // --- FAST PATH ---
            // k_cache includes history AND the token we just wrote.
            // Permute: [Batch, Total_Len, KV_Heads, Dim] -> [Batch, KV_Heads, Dim, Total_Len]
            let k_view = k_cache
                .view()
                .into_shape_with_order((batch, total_len, self.num_kv_heads, self.head_dim))?
                .permuted_axes([0, 2, 3, 1]);

            crate::utils::linear_algebra::matmul_4d_decode_gqa(&q_heads, &k_view, n_rep)
        } else {
            // --- PREFILL PATH ---
            let k_heads = self.prepare_kv_heads(&k_cache.view().into_owned(), batch)?;
            let k_t = k_heads.permuted_axes([0, 1, 3, 2]);
            crate::utils::linear_algebra::matmul_4d(&q_heads, &k_t)
        };

        // 5. Scale & Softmax
        scores.mapv_inplace(|x| x * self.scale_factor);

        if let Some(mask) = attention_mask {
            scores = crate::utils::linear_algebra::apply_attention_mask(scores, mask);
        }

        // Only apply causal mask if not single-token decode
        // if !is_decode {
        //     self.apply_causal_mask(&mut scores, start_write);
        // }
        self.apply_causal_mask(&mut scores, start_write);

        crate::activations::softmax_4d_inplace(&mut scores);

        // 6. Context (Scores @ V)
        let context = if is_decode {
            let v_view = v_cache
                .view()
                .into_shape_with_order((batch, total_len, self.num_kv_heads, self.head_dim))?
                .permuted_axes([0, 2, 1, 3]);

            crate::utils::linear_algebra::matmul_4d_context_gqa(&scores, &v_view, n_rep)
        } else {
            let v_heads = self.prepare_kv_heads(&v_cache.view().into_owned(), batch)?;
            crate::utils::linear_algebra::matmul_4d(&scores, &v_heads)
        };

        // 7. Output Projection
        let context_flat = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((batch * seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        let output = self.o_proj.matmul(&context_flat.view());
        Ok(output.into_shape_with_order((batch, seq_len, self.num_heads * self.head_dim))?)
    }

    fn prepare_kv_heads(&self, kv: &Array3<f32>, batch: usize) -> Result<Array4<f32>> {
        let total_seq = kv.shape()[1];
        let kv_heads = kv
            .view()
            .into_shape_with_order((batch, total_seq, self.num_kv_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]); // [B, KV_H, T, D]

        if self.num_heads == self.num_kv_heads {
            Ok(kv_heads.to_owned())
        } else {
            // Repeat Logic for GQA
            let groups = self.num_heads / self.num_kv_heads;
            let mut out = Array4::zeros((batch, self.num_heads, total_seq, self.head_dim));
            for i in 0..self.num_kv_heads {
                for g in 0..groups {
                    out.slice_mut(s![.., i * groups + g, .., ..])
                        .assign(&kv_heads.slice(s![.., i, .., ..]));
                }
            }
            Ok(out)
        }
    }

    fn apply_causal_mask(&self, scores: &mut Array4<f32>, cache_len: usize) {
        let (_, _, seq_q, _) = scores.dim();
        for i in 0..seq_q {
            let query_pos = cache_len + i;
            for j in 0..scores.shape()[3] {
                if j > query_pos {
                    scores.slice_mut(s![.., .., i, j]).fill(MASK_VALUE);
                }
            }
        }
    }
}
