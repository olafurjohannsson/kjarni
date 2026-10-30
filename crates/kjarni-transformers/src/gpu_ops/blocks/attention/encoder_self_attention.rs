//! Bidirectional self-attention for encoder models

use super::ops::AttentionOps;
use crate::WgpuContext;
use crate::gpu::{GpuTensor, GpuTensorPool};
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::rope::GpuRoPE;
use std::sync::Arc;

/// Bidirectional self-attention for encoder models
pub struct GpuEncoderSelfAttention {
    ops: AttentionOps,
}

impl GpuEncoderSelfAttention {
    /// Creates a new encoder self-attention module
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

    /// Performs the forward pass of encoder self-attention
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        weights: &GpuAttentionWeights,
        padding_mask: Option<&GpuTensor>,
        position_bias: Option<&GpuTensor>,
        rope: Option<&GpuRoPE>,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        // Project Q, K, V
        let q_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.q_weight,
            weights.q_bias.as_ref(),
            pool,
        );
        let k_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.k_weight,
            weights.k_bias.as_ref(),
            pool,
        );
        let v_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.v_weight,
            weights.v_bias.as_ref(),
            pool,
        );

        // Split heads: [B, S, H*D] -> [B, H, S, D]
        let mut q_heads = self.ops.split_heads(encoder, &q_proj, pool);
        let mut k_heads = self.ops.split_heads(encoder, &k_proj, pool);

        // Rotary positions (nomic-bert), from position 0: an encoder has no cache.
        if let Some(rope) = rope {
            let q_rot = pool.get(q_heads.shape().to_vec());
            let k_rot = pool.get(k_heads.shape().to_vec());
            rope.encode(encoder, &q_heads, &q_rot, 0);
            rope.encode(encoder, &k_heads, &k_rot, 0);
            q_heads = q_rot;
            k_heads = k_rot;
        }
        let v_heads = self.ops.split_heads(encoder, &v_proj, pool);

        // Compute attention (bidirectional, no causal mask)
        let context = self.ops.attention(
            encoder,
            &q_heads,
            &k_heads,
            &v_heads,
            padding_mask,
            false, // is_causal = false for encoder
            0,     // position_offset = 0 (no KV cache)
            position_bias,
            pool,
        );

        // Merge heads: [B, H, S, D] -> [B, S, H*D]
        let context_merged = self.ops.merge_heads(encoder, &context, pool);

        // Output projection
        self.ops.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            weights.output_bias.as_ref(),
            pool,
        )
    }
}
