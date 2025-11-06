use std::sync::Arc;
use wgpu::CommandEncoder;
use anyhow::Result;

use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::blocks::attention::attention::GpuAttention;
use crate::gpu_ops::blocks::ffn::{GpuFeedForward, GpuFeedForwardWeights};

use crate::gpu_ops::blocks::attention::attention::GpuAttentionWeights;


use crate::gpu_ops::blocks::layer_norm::GpuLayerNorm;
use crate::gpu_ops::blocks::layer_norm::GpuLayerNormWeights;

mod tests;

/// Represents a single layer for a decoder-only transformer model (e.g., GPT-2) on the GPU.
///
/// This struct holds the GPU-accelerated components (attention, layer norms, feed-forward)
/// for one layer. It is designed to be a simple container of these components.
///
/// The actual forward pass orchestration is handled by the top-level `GpuTransformerDecoder`
/// to allow for efficient batching of GPU commands. For example, the decoder can perform
/// all `project_kv` operations for all layers, then all cache updates, then all `attend`
/// operations, minimizing kernel dispatch overhead.
pub struct GpuDecoderLayer {
    // The self-attention block, containing kernels for matmul, softmax, etc.
    pub self_attn: GpuAttention,
    // The weights associated with the self-attention block.
    pub self_attn_weights: GpuAttentionWeights,
    // The layer normalization applied before the self-attention block.
    pub self_attn_layer_norm: GpuLayerNorm,
    pub self_attn_ln_weights: GpuLayerNormWeights,

    // The feed-forward block.
    pub feedforward: GpuFeedForward,
    pub ff_weights: GpuFeedForwardWeights,
    // The layer normalization applied before the feed-forward block.
    pub ffn_layer_norm: GpuLayerNorm,
    pub ffn_ln_weights: GpuLayerNormWeights,
}

impl GpuDecoderLayer {
    /// Creates a new GPU-based decoder layer.
    ///
    /// This function initializes the GPU kernels for each component and moves the
    /// weights to the GPU.
    ///
    /// # Arguments
    /// * `context`: The shared WGPU context.
    /// * `self_attn_weights`: The weights for the self-attention mechanism.
    /// * `self_attn_ln_weights`: The weights for the pre-attention layer norm.
    /// * `ff_weights`: The weights for the feed-forward network.
    /// * `ffn_ln_weights`: The weights for the pre-feed-forward layer norm.
    /// * `hidden_size`: The dimensionality of the model.
    /// * `num_heads`: The number of attention heads.
    /// * `intermediate_size`: The dimensionality of the feed-forward layer's intermediate state.
    pub fn new(
        context: &Arc<WgpuContext>,
        self_attn_weights: GpuAttentionWeights,
        self_attn_ln_weights: GpuLayerNormWeights,
        ff_weights: GpuFeedForwardWeights,
        ffn_ln_weights: GpuLayerNormWeights,
        hidden_size: u32,
        num_heads: u32,
        intermediate_size: u32,
    ) -> Result<Self> {
        let self_attn = GpuAttention::new(context, hidden_size, num_heads);
        let self_attn_layer_norm = GpuLayerNorm::new(context);
        
        let feedforward = GpuFeedForward::new(context, hidden_size, intermediate_size);
        let ffn_layer_norm = GpuLayerNorm::new(context);

        Ok(Self {
            self_attn,
            self_attn_weights,
            self_attn_layer_norm,
            self_attn_ln_weights,
            feedforward,
            ff_weights,
            ffn_layer_norm,
            ffn_ln_weights,
        })
    }

    // NOTE: A monolithic `forward` method is intentionally omitted here.
    // The `GpuTransformerDecoder` will call the components' `encode` methods directly.
    // This allows the orchestrator to control the command encoder and manage the
    // execution flow for maximum efficiency (e.g., batching all projections first).
    //
    // For example, the orchestrator's logic for this layer will look like:
    //
    // 1. residual = hidden_states
    // 2. ln1_out = self.self_attn_layer_norm.encode(encoder, hidden_states, ln1_weights)
    // 3. (new_k, new_v) = self.self_attn.project_kv(encoder, &ln1_out, attn_weights)
    //    -> `new_k`, `new_v` are returned to orchestrator to update cache.
    // 4. attn_out = self.self_attn.attend(encoder, &ln1_out, full_cache_view, ...)
    // 5. attn_block_output = residual + attn_out  (using an "add" kernel)
    // 6. residual = attn_block_output
    // 7. ln2_out = self.ffn_layer_norm.encode(encoder, attn_block_output, ffn_ln_weights)
    // 8. ffn_out = self.feedforward.encode(encoder, &ln2_out, ff_weights)
    // 9. final_output = residual + ffn_out (using an "add" kernel)
}