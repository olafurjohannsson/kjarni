use crate::activations;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::Kernel;
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::attention::{GpuAttention, TempStorage};
use crate::gpu_ops::blocks::ffn::{GpuFeedForward, GpuFeedForwardWeights};
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::primitives::layout::concatenate::GpuConcatenate;
use crate::traits::CrossAttentionDecoder;
use crate::traits::{
    Cache, Decoder, DecoderArchitecture, DecoderOutput, Device, TransformerConfig, TransformerModel,
};
use anyhow::Result;
use std::sync::Arc;
use wgpu::CommandEncoder;

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
pub struct GpuPreNormDecoderLayer {
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

    add: GpuAdd,
    concat: GpuConcatenate,
    config: Arc<dyn DecoderArchitecture + Send + Sync>,
}

impl GpuPreNormDecoderLayer {
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
        config: Arc<dyn DecoderArchitecture + Send + Sync>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size() as u32;
        let num_heads = config.num_attention_heads() as u32;

        let self_attn = GpuAttention::new(context, hidden_size, num_heads);
        let self_attn_layer_norm = GpuLayerNorm::new(context, config.layer_norm_eps());

        let feedforward = GpuFeedForward::new(context, activations::Activation::Gelu)?;
        let ffn_layer_norm = GpuLayerNorm::new(context, config.layer_norm_eps());

        let add = GpuAdd::new(&context);
        let concat = GpuConcatenate::new(&context);

        Ok(Self {
            self_attn,
            self_attn_weights,
            self_attn_layer_norm,
            self_attn_ln_weights,
            feedforward,
            ff_weights,
            ffn_layer_norm,
            ffn_ln_weights,
            config,
            add,
            concat,
        })
    }
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        position_offset: usize,
        // The signature now mirrors the CPU version exactly
        // past_kv: Option<(&GpuTensor, &GpuTensor)>,
        physical_past_kv: Option<(&GpuTensor, &GpuTensor)>,
        temp: &mut TempStorage,
    ) -> Result<(GpuTensor, (GpuTensor, GpuTensor))> {
        // --- 1. First Sub-layer: Self-Attention (Pre-Norm) ---
        let residual = hidden_states;

        let ln1_out = temp.get(hidden_states.shape().to_vec());
        self.self_attn_layer_norm.encode(
            encoder,
            &self.self_attn_ln_weights,
            hidden_states,
            &ln1_out,
        );

        // Project the K/V for this step. These are the raw 3D tensors.
        let (new_k, new_v) =
            self.self_attn
                .project_kv(encoder, &ln1_out, &self.self_attn_weights, temp);

        let logical_len = if let Some((physical_k, _)) = physical_past_kv {
            // Generation pass: logical length = cache + new tokens
            position_offset + new_k.shape()[1]
        } else {
            // Priming pass: logical length = just new tokens
            new_k.shape()[1]
        };

        let attn_out = if let Some((physical_k, physical_v)) = physical_past_kv {
            // ✅ Use strided attention with physical cache
            self.self_attn.attend_with_physical_cache(
                encoder,
                &ln1_out,
                physical_k,
                physical_v,
                attention_mask,
                &self.self_attn_weights,
                position_offset,
                logical_len,
                temp,
            )
        } else {
            // Priming pass: no cache, use standard attend
            // Split heads for new K/V
            let new_k_split = self.self_attn.split_heads(encoder, &new_k, temp);
            let new_v_split = self.self_attn.split_heads(encoder, &new_v, temp);

            self.self_attn.attend(
                encoder,
                &ln1_out,
                &self.self_attn_weights,
                attention_mask,
                true, // is_causal
                (&new_k_split, &new_v_split),
                0, // position_offset = 0 for priming
                temp,
            )
        };

        // --- Cache Preparation (Now identical to CPU logic) ---
        // We must split the heads of the new K/V tensors before concatenation.
        // let new_k_split = self.self_attn.split_heads(encoder, &new_k, temp);
        // let new_v_split = self.self_attn.split_heads(encoder, &new_v, temp);

        // let (full_k, full_v) = if let Some((past_k, past_v)) = past_kv {
        //     // Concatenate the past (already head-split) with the new (now head-split)
        //     let full_k_shape = vec![
        //         past_k.shape()[0],
        //         past_k.shape()[1],
        //         past_k.shape()[2] + new_k_split.shape()[2],
        //         past_k.shape()[3],
        //     ];
        //     let full_v_shape = vec![
        //         past_v.shape()[0],
        //         past_v.shape()[1],
        //         past_v.shape()[2] + new_v_split.shape()[2],
        //         past_v.shape()[3],
        //     ];

        //     let fk = temp.get(full_k_shape);
        //     let fv = temp.get(full_v_shape);

        //     // Use the GpuConcatenate kernel along axis 2 (the sequence dimension)
        //     // TODO: remove and use strides for cache in GpuBatchedMatMul and GpuApplyMask
        //     self.concat.encode(encoder, &[past_k, &new_k_split], &fk, 2);
        //     self.concat.encode(encoder, &[past_v, &new_v_split], &fv, 2);

        //     (fk, fv)
        // } else {
        //     // No cache (priming pass), so the "full" cache is just the new head-split state.
        //     (new_k_split, new_v_split)
        // };

        // // Perform the attention calculation. `full_k` and `full_v` are now correctly shaped.
        // let attn_out = self.self_attn.attend(
        //     encoder,
        //     &ln1_out,
        //     &self.self_attn_weights,
        //     attention_mask,
        //     true, // is_causal
        //     (&full_k, &full_v),
        //     position_offset,
        //     temp,
        // );

        let attn_block_output = temp.get(hidden_states.shape().to_vec());
        self.add
            .encode(encoder, &[residual, &attn_out], &attn_block_output);

        // --- 2. Second Sub-layer: Feed-Forward Network (Pre-Norm) ---
        let residual_2 = &attn_block_output;

        let ln2_out = temp.get(residual_2.shape().to_vec());
        self.ffn_layer_norm
            .encode(encoder, &self.ffn_ln_weights, residual_2, &ln2_out);

        let ffn_out = self
            .feedforward
            .encode(encoder, &ln2_out, &self.ff_weights, temp);

        let final_output = temp.get(residual_2.shape().to_vec());
        self.add
            .encode(encoder, &[residual_2, &ffn_out], &final_output);

        // Return the raw 3D K/V tensors for the cache manager.
        Ok((final_output, (new_k, new_v)))
    }
}
