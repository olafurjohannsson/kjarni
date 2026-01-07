//! Causal self-attention for decoder models.
//!
//! This module provides `DecoderSelfAttention`, the CPU implementation of
//! causal self-attention with KV caching for autoregressive decoding.
//!
//! # Characteristics
//!
//! - **Causal**: Each token can only attend to previous tokens.
//! - **KV Cache**: Caches key/value states for efficient generation.
//! - **No RoPE**: For models using learned positions (GPT-2, BART decoder).
//!
//! # Used By
//!
//! - GPT-2 (decoder-only)
//! - BART decoder, T5 decoder (self-attention part)
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::cpu_ops::attention::DecoderSelfAttention;
//!
//! let attn = DecoderSelfAttention::new(1024, 16, q_proj, k_proj, v_proj, o_proj);
//!
//! // Prefill (no cache)
//! let (output, new_k, new_v) = attn.forward(&hidden, Some(&mask), None)?;
//!
//! // Decode with cache
//! let (output, new_k, new_v) = attn.forward(
//!     &token_hidden,
//!     Some(&mask),
//!     Some((cached_k.view(), cached_v.view())),
//! )?;
//! ```

use crate::linear_layer::LinearLayer;
use crate::utils::linear_algebra::{
    apply_attention_mask, matmul_4d, matmul_4d_context, matmul_4d_decode
};
use crate::activations::softmax_4d_inplace;
use crate::utils::masks::apply_causal_mask;
use anyhow::Result;
use ndarray::{Array2, Array3, Array4, ArrayView3, Axis};

/// Causal self-attention for decoder models with KV caching.
///
/// Computes masked multi-head self-attention where each token can only
/// attend to itself and previous tokens. Supports incremental decoding
/// via KV caching.
///
/// # Architecture
///
/// ```text
/// Input [B, S_new, H]
///     │
///     ├──► Q = input @ W_q + b_q  ──► Split heads ──┐
///     │                                              │
///     ├──► K = input @ W_k + b_k  ──► Concat cache ──┼──► Causal Attention
///     │                                              │
///     └──► V = input @ W_v + b_v  ──► Concat cache ──┘
///                                                    │
///                                                    ▼
///                                     Merge heads ──► Output projection
///                                                    │
/// Output [B, S_new, H] ◄─────────────────────────────┘
///
/// Also returns: (new_K, new_V) for cache update
/// ```
///
/// # KV Cache Strategy
///
/// The cache stores K and V in 3D format `[B, S_cached, H]` for efficient
/// concatenation along the sequence dimension.
pub struct DecoderSelfAttention {
    /// Query projection layer.
    pub q_proj: LinearLayer,
    /// Key projection layer.
    pub k_proj: LinearLayer,
    /// Value projection layer.
    pub v_proj: LinearLayer,
    /// Output projection layer.
    pub o_proj: LinearLayer,

    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Scaling factor: 1 / sqrt(head_dim).
    pub scale_factor: f32,
    /// If scale_qk
    pub scale_qk: bool,
}

impl DecoderSelfAttention {
    /// Creates a new decoder self-attention module.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - The model's hidden dimension.
    /// * `num_heads` - Number of attention heads.
    /// * `q` - Query projection weights.
    /// * `k` - Key projection weights.
    /// * `v` - Value projection weights.
    /// * `o` - Output projection weights.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Self attention with Out features and In features
    /// use kjarni_transformers::encoder_decoder::DecoderSelfAttention;
    /// use kjarni_transformers::linear_layer::LinearLayer;
    /// use kjarni_transformers::tensor::Dtype;
    /// let q_proj = LinearLayer::new(1024, 1024, Dtype::F32);
    /// let k_proj = LinearLayer::new(1024, 1024, Dtype::F32);
    /// let v_proj = LinearLayer::new(1024, 1024, Dtype::F32);
    /// let o_proj = LinearLayer::new(1024, 1024, Dtype::F32);
    /// let attn = DecoderSelfAttention::new(1024, 16, q_proj, k_proj, v_proj, o_proj);
    /// ```
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        q: LinearLayer,
        k: LinearLayer,
        v: LinearLayer,
        o: LinearLayer,
    ) -> Self {
        let head_dim = hidden_size / num_heads;

        Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            num_heads,
            head_dim,
            scale_factor: 1.0 / (head_dim as f32).sqrt(),
            scale_qk: true,
        }
    }

    pub fn with_no_qk_scaling(mut self) -> Self {
        self.scale_qk = false;
        self
    }
    /// Performs the forward pass of decoder self-attention.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Input tensor of shape `[batch, seq_new, hidden]`.
    /// * `attention_mask` - Optional padding mask `[batch, seq_total]`.
    /// * `past_kv` - Optional cached (K, V) views from previous steps.
    ///
    /// # Returns
    ///
    /// Tuple of:
    /// - Output tensor `[batch, seq_new, hidden]`
    /// - New K tensor `[batch, seq_new, hidden]` for cache update
    /// - New V tensor `[batch, seq_new, hidden]` for cache update
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Prefill (process prompt, no cache)
    /// let (output, new_k, new_v) = attn.forward(&prompt_hidden, Some(&mask), None)?;
    ///
    /// // Decode (generate one token at a time)
    /// let (output, new_k, new_v) = attn.forward(
    ///     &token_hidden,  // [B, 1, H]
    ///     Some(&mask),
    ///     Some((cached_k.view(), cached_v.view())),
    /// )?;
    /// ```
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
        past_kv: Option<(ArrayView3<f32>, ArrayView3<f32>)>,
        position_bias: Option<&Array4<f32>>,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        let (batch, seq_len, _) = hidden_states.dim();
        let hidden_size = self.num_heads * self.head_dim;

        // 1. Flatten & Project
        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((batch * seq_len, hidden_size))?;

        let q = self.q_proj.matmul(&hidden_2d);
        let k = self.k_proj.matmul(&hidden_2d);
        let v = self.v_proj.matmul(&hidden_2d);

        // Q: Split heads immediately [B, H, S_new, D]
        let q_heads = q
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .to_owned();

        // K/V: Keep in 3D [B, S_new, hidden] for caching
        let k_3d = k.into_shape_with_order((batch, seq_len, hidden_size))?;
        let v_3d = v.into_shape_with_order((batch, seq_len, hidden_size))?;

        // 2. Concatenate with cache
        let (full_k, full_v) = if let Some((cached_k, cached_v)) = past_kv {
            let full_k = ndarray::concatenate![Axis(1), cached_k, k_3d.view()]
                .as_standard_layout()
                .to_owned();
            let full_v = ndarray::concatenate![Axis(1), cached_v, v_3d.view()]
                .as_standard_layout()
                .to_owned();
            (full_k, full_v)
        } else {
            (k_3d.clone(), v_3d.clone())
        };

        // 3. Prepare K/V for attention
        let total_seq = full_k.shape()[1];

        let k_heads = full_k
            .view()
            .into_shape_with_order((batch, total_seq, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        // Transpose K for Q @ K^T: [B, H, D, S_total]
        let k_t = k_heads.permuted_axes([0, 1, 3, 2]).to_owned();

        // 4. Compute attention scores
        let mut scores = if seq_len == 1 {
            matmul_4d_decode(&q_heads, &k_t)
        } else {
            matmul_4d(&q_heads, &k_t)
        };

        // Scale
        if self.scale_qk {
            scores.mapv_inplace(|x| x * self.scale_factor);
        }

        // apply relative position bias
        if let Some(bias) = position_bias {
            scores += bias;
        }

        // Apply padding mask
        if let Some(mask) = attention_mask {
            scores = apply_attention_mask(scores, mask);
        }

        // Apply causal mask (only for prefill, decode is inherently causal)
        if seq_len > 1 {
            let cache_len = past_kv.map_or(0, |(k, _)| k.shape()[1]);
            apply_causal_mask(&mut scores, cache_len);
        }

        // Softmax
        softmax_4d_inplace(&mut scores);

        // 5. Compute context: Scores @ V
        let v_heads = full_v
            .view()
            .into_shape_with_order((batch, total_seq, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .to_owned();

        let context = if seq_len == 1 {
            matmul_4d_context(&scores, &v_heads)
        } else {
            matmul_4d(&scores, &v_heads)
        };

        // 6. Merge heads and output projection
        let context_flat = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((batch * seq_len, hidden_size))?
            .to_owned();

        let output = self.o_proj.matmul(&context_flat.view());
        let output_3d = output.into_shape_with_order((batch, seq_len, hidden_size))?;

        // Return output and NEW K/V for cache update
        Ok((output_3d, k_3d, v_3d))
    }
}
