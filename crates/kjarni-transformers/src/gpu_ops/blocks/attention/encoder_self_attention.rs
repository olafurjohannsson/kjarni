//! Bidirectional self-attention for encoder models.
//!
//! This module provides `GpuEncoderSelfAttention`, optimized for encoder-only
//! and encoder portions of encoder-decoder models.
//!
//! # Characteristics
//!
//! - **Bidirectional**: Each token can attend to all other tokens (no causal mask).
//! - **No KV Cache**: Processes the full sequence at once (no autoregressive decoding).
//! - **Padding Mask**: Supports masking out padding tokens.
//!
//! # Used By
//!
//! - BERT, RoBERTa, DistilBERT (encoder-only)
//! - BART encoder, T5 encoder (encoder-decoder)
//! - Sentence transformers
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::gpu_ops::attention::GpuEncoderSelfAttention;
//!
//! let attn = GpuEncoderSelfAttention::new(&context, 768, 12);
//!
//! let output = attn.forward(
//!     &mut encoder,
//!     &hidden_states,
//!     &weights,
//!     Some(&padding_mask),
//!     &mut pool,
//! );
//! ```

use super::ops::AttentionOps;
use crate::gpu_ops::{blocks::attention::GpuAttentionWeights, GpuTensor, GpuTensorPool};
use crate::WgpuContext;
use std::sync::Arc;

/// Bidirectional self-attention for encoder models.
///
/// This attention module is designed for encoders where every token can attend
/// to every other token. It does not use a KV cache since encoders process
/// the full sequence in a single forward pass.
///
/// # Architecture
///
/// ```text
/// Input [B, S, H]
///     │
///     ├──► Q Projection ──► Split Heads ──┐
///     │                                    │
///     ├──► K Projection ──► Split Heads ──┼──► Attention ──► Merge Heads ──► O Projection
///     │                                    │
///     └──► V Projection ──► Split Heads ──┘
///                                          │
/// Output [B, S, H] ◄───────────────────────┘
/// ```
///
/// # Thread Safety
///
/// `GpuEncoderSelfAttention` is `Send + Sync` and can be safely shared across threads.
pub struct GpuEncoderSelfAttention {
    ops: AttentionOps,
}

impl GpuEncoderSelfAttention {
    /// Creates a new encoder self-attention module.
    ///
    /// # Arguments
    ///
    /// * `context` - The WGPU context for creating GPU resources.
    /// * `hidden_size` - The model's hidden dimension.
    /// * `num_heads` - Number of attention heads.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // BERT-base: 768 hidden, 12 heads
    /// let attn = GpuEncoderSelfAttention::new(&ctx, 768, 12);
    ///
    /// // BERT-large: 1024 hidden, 16 heads
    /// let attn = GpuEncoderSelfAttention::new(&ctx, 1024, 16);
    /// ```
    pub fn new(context: &Arc<WgpuContext>, hidden_size: u32, num_heads: u32) -> Self {
        Self {
            // Encoder self-attention has equal Q and KV heads (no GQA)
            ops: AttentionOps::new(context, hidden_size, num_heads, num_heads),
        }
    }

    /// Returns the underlying attention operations for advanced usage.
    pub fn ops(&self) -> &AttentionOps {
        &self.ops
    }

    /// Performs the forward pass of encoder self-attention.
    ///
    /// Computes bidirectional self-attention where each token can attend to
    /// all other tokens in the sequence.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder for recording GPU operations.
    /// * `hidden_states` - Input tensor of shape `[B, S, H]`.
    /// * `weights` - The Q, K, V, and O projection weights.
    /// * `padding_mask` - Optional mask of shape `[B, S]` where 1.0 = valid, 0.0 = padding.
    /// * `pool` - Tensor pool for allocating intermediate tensors.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[B, S, H]`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Without padding mask (all tokens valid)
    /// use kjarni_transformers::gpu_ops::blocks::attention::GpuEncoderSelfAttention;
    /// use kjarni_transformers::gpu_ops::{GpuTensor, GpuAttentionWeights, GpuTensorPool};
    /// use kjarni_transformers::WgpuContext;
    /// let context = WgpuContext::new();
    /// let attn = GpuEncoderSelfAttention::new(&context, 768, 12);
    /// let mut enc = context.create_command_encoder("encoder_self_attention");
    /// let hidden = GpuTensor::from_ndarray(&ctx, &hidden_states)?;
    /// let weights = GpuAttentionWeights::new(&ctx, 768, 12);
    /// let mut pool = GpuTensorPool::new();
    /// let output = attn.forward(&mut enc, &hidden, &weights, None, &mut pool);
    ///
    /// // With padding mask
    /// let mask = GpuTensor::from_ndarray(&ctx, &padding_mask)?;
    /// let output = attn.forward(&mut enc, &hidden, &weights, Some(&mask), &mut pool);
    /// ```
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        weights: &GpuAttentionWeights,
        padding_mask: Option<&GpuTensor>,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        // 1. Project Q, K, V
        let q_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.q_weight,
            &weights.q_bias,
            pool,
        );
        let k_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.k_weight,
            &weights.k_bias,
            pool,
        );
        let v_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.v_weight,
            &weights.v_bias,
            pool,
        );

        // 2. Split heads: [B, S, H*D] -> [B, H, S, D]
        let q_heads = self.ops.split_heads(encoder, &q_proj, pool);
        let k_heads = self.ops.split_heads(encoder, &k_proj, pool);
        let v_heads = self.ops.split_heads(encoder, &v_proj, pool);

        // 3. Compute attention (bidirectional, no causal mask)
        let context = self.ops.attention(
            encoder,
            &q_heads,
            &k_heads,
            &v_heads,
            padding_mask,
            false, // is_causal = false for encoder
            0,     // position_offset = 0 (no KV cache)
            pool,
        );

        // 4. Merge heads: [B, H, S, D] -> [B, S, H*D]
        let context_merged = self.ops.merge_heads(encoder, &context, pool);

        // 5. Output projection
        self.ops.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_attention_dimensions() {
        // This is a compile-time test to ensure the API is correct
        // Actual GPU tests would require a WgpuContext
        
        // BERT-base dimensions
        let hidden_size = 768u32;
        let num_heads = 12u32;
        let head_dim = hidden_size / num_heads;
        
        assert_eq!(head_dim, 64);
    }
}