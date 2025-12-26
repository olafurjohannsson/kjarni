//! Cross-attention for encoder-decoder models.
//!
//! This module provides `GpuCrossAttention`, optimized for attending from
//! decoder states to encoder outputs.

use super::ops::AttentionOps;
use crate::WgpuContext;
use crate::gpu_ops::{GpuTensor, GpuTensorPool, blocks::attention::GpuAttentionWeights};
use std::sync::Arc;

/// Cross-attention for encoder-decoder models.
///
/// Attends from decoder hidden states (Q) to encoder outputs (K, V).
/// The encoder K/V can be precomputed once and reused for all decoding steps.
pub struct GpuCrossAttention {
    ops: AttentionOps,
}

impl GpuCrossAttention {
    /// Creates a new cross-attention module.
    pub fn new(context: &Arc<WgpuContext>, hidden_size: u32, num_heads: u32) -> Self {
        Self {
            ops: AttentionOps::new(context, hidden_size, num_heads, num_heads),
        }
    }

    /// Returns the underlying attention operations.
    pub fn ops(&self) -> &AttentionOps {
        &self.ops
    }

    /// Precomputes K and V from encoder hidden states.
    ///
    /// Call once after encoding, reuse for all decode steps.
    /// K is pre-transposed to `[B, H, D, S]` for efficient attention.
    ///
    /// # Returns
    ///
    /// `(K, V)` tuple where:
    /// - K: `[B, H, D, S_enc]` (transposed for efficient Q @ K^T)
    /// - V: `[B, H, S_enc, D]`
    pub fn precompute_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        encoder_hidden_states: &GpuTensor,
        weights: &GpuAttentionWeights,
        pool: &mut GpuTensorPool,
    ) -> (GpuTensor, GpuTensor) {
        // 1. Project K and V
        let k_proj = self.ops.project(
            encoder,
            encoder_hidden_states,
            &weights.k_weight,
            &weights.k_bias,
            pool,
        );
        let v_proj = self.ops.project(
            encoder,
            encoder_hidden_states,
            &weights.v_weight,
            &weights.v_bias,
            pool,
        );

        // 2. Split heads: [B, S, H*D] -> [B, H, S, D]
        let k_heads = self.ops.split_heads(encoder, &k_proj, pool);
        let v_heads = self.ops.split_heads(encoder, &v_proj, pool);

        // 3. Pre-transpose K: [B, H, S, D] -> [B, H, D, S]
        let k_transposed = k_heads.permute(encoder, self.ops.permute_kernel(), &[0, 1, 3, 2]);

        (k_transposed, v_heads)
    }

    /// Performs cross-attention using precomputed K/V.
    ///
    /// # Arguments
    ///
    /// * `decoder_hidden_states` - Decoder hidden states `[B, S_dec, H]`.
    /// * `precomputed_kv` - Tuple of (K, V) from `precompute_kv`:
    ///   - K: `[B, H, D, S_enc]` (pre-transposed)
    ///   - V: `[B, H, S_enc, D]`
    /// * `encoder_mask` - Optional encoder padding mask `[B, S_enc]`.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[B, S_dec, H]`.
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        decoder_hidden_states: &GpuTensor,
        precomputed_kv: &(GpuTensor, GpuTensor),
        weights: &GpuAttentionWeights,
        encoder_mask: Option<&GpuTensor>,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (precomputed_k, precomputed_v) = precomputed_kv;

        // 1. Project Q from decoder states
        let q_proj = self.ops.project(
            encoder,
            decoder_hidden_states,
            &weights.q_weight,
            &weights.q_bias,
            pool,
        );

        // 2. Split heads: [B, S_dec, H*D] -> [B, H, S_dec, D]
        let q_heads = self.ops.split_heads(encoder, &q_proj, pool);

        // 3. Attention scores: Q @ K^T (K is already transposed)
        // q_heads: [B, H, S_dec, D]
        // precomputed_k: [B, H, D, S_enc]
        // scores: [B, H, S_dec, S_enc]
        let scores = self.ops.bmm_4d(encoder, &q_heads, precomputed_k, pool);

        // 4. Apply mask and softmax (NOT causal)
        self.ops.apply_mask_and_softmax(
            encoder,
            &scores,
            encoder_mask,
            false, // NOT causal
            0,     // No position offset
        );

        // 5. Context: scores @ V
        let context = self.ops.bmm_4d(encoder, &scores, precomputed_v, pool);

        // 6. Merge heads and output projection
        let context_merged = self.ops.merge_heads(encoder, &context, pool);

        self.ops.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        )
    }
}
