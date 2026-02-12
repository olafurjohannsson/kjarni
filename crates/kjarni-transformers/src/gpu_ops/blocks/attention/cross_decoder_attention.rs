//! Cross-attention for encoder-decoder models

use super::ops::AttentionOps;
use crate::WgpuContext;
use crate::gpu::{GpuTensor, GpuTensorPool};
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use std::sync::Arc;

/// Cross-attention for encoder-decoder models
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

    /// Precomputes K and V from encoder hidden states
    pub fn precompute_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        encoder_hidden_states: &GpuTensor,
        weights: &GpuAttentionWeights,
        pool: &mut GpuTensorPool,
    ) -> (GpuTensor, GpuTensor) {
        // Project K and V
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

        // Split heads: [B, S, H*D] -> [B, H, S, D]
        let k_heads = self.ops.split_heads(encoder, &k_proj, pool);
        let v_heads = self.ops.split_heads(encoder, &v_proj, pool);

        // Pre-transpose K: [B, H, S, D] -> [B, H, D, S]
        let k_transposed = k_heads.permute(encoder, self.ops.permute_kernel(), &[0, 1, 3, 2]);

        (k_transposed, v_heads)
    }

    /// Performs cross-attention using precomputed K/V
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

        // Project Q from decoder states
        let q_proj = self.ops.project(
            encoder,
            decoder_hidden_states,
            &weights.q_weight,
            &weights.q_bias,
            pool,
        );

        // Split heads: [B, S_dec, H*D] -> [B, H, S_dec, D]
        let q_heads = self.ops.split_heads(encoder, &q_proj, pool);
        // q_heads: [B, H, S_dec, D]
        // precomputed_k: [B, H, D, S_enc]
        // scores: [B, H, S_dec, S_enc]
        let scores = self.ops.bmm_4d(encoder, &q_heads, precomputed_k, pool);

        // Apply mask and softmax (NOT causal)
        self.ops.apply_mask_and_softmax(
            encoder,
            &scores,
            encoder_mask,
            false, // NOT causal
            0,     // No position offset
        );

        // Context: scores @ V
        let context = self.ops.bmm_4d(encoder, &scores, precomputed_v, pool);

        // Merge heads and output projection
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
