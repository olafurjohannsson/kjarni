use std::sync::Arc;
use wgpu::CommandEncoder;
use anyhow::Result;

use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::attention::{GpuAttentionWeights, GpuAttention, TempStorage};
use crate::gpu_ops::blocks::ffn::{GpuFeedForward, GpuFeedForwardWeights};
use crate::gpu_ops::blocks::layer_norm::GpuLayerNorm;
use crate::gpu_ops::blocks::layer_norm::GpuLayerNormWeights;
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::{Kernel, GpuTensor};
use crate::activations;
use crate::traits::{EncoderArchitecture, TransformerConfig};


/// Represents a single layer for a standard transformer encoder model (e.g., BERT, BART-encoder).
///
/// This struct holds the GPU-accelerated components and orchestrates a single, stateless
/// forward pass. It supports both Pre-Normalization and Post-Normalization architectures.
pub struct GpuEncoderLayer {
    // --- Components for Self-Attention ---
    self_attn: GpuAttention,
    self_attn_weights: GpuAttentionWeights,
    self_attn_layer_norm: GpuLayerNorm,
    self_attn_ln_weights: GpuLayerNormWeights,

    // --- Components for Feed-Forward Network ---
    feedforward: GpuFeedForward,
    ff_weights: GpuFeedForwardWeights,
    ffn_layer_norm: GpuLayerNorm,
    ffn_ln_weights: GpuLayerNormWeights,

    // --- Utility Kernels ---
    add: GpuAdd,
}

impl GpuEncoderLayer {
    /// Creates a new GPU-based encoder layer.
    pub fn new(
        context: &Arc<WgpuContext>,
        self_attn_weights: GpuAttentionWeights,
        self_attn_ln_weights: GpuLayerNormWeights,
        ff_weights: GpuFeedForwardWeights,
        ffn_ln_weights: GpuLayerNormWeights,
        config: &dyn TransformerConfig, // Use the generic config trait
    ) -> Result<Self> {
        let hidden_size = config.hidden_size() as u32;
        let num_heads = config.num_attention_heads() as u32;

        let self_attn = GpuAttention::new(context, hidden_size, num_heads);
        let self_attn_layer_norm = GpuLayerNorm::new(context, config.layer_norm_eps());
        
        // Assuming GELU is standard for encoders you're targeting
        let feedforward = GpuFeedForward::new(context, activations::Activation::Gelu)?;
        let ffn_layer_norm = GpuLayerNorm::new(context, config.layer_norm_eps());
        
        let add = GpuAdd::new(context);

        Ok(Self {
            self_attn,
            self_attn_weights,
            self_attn_layer_norm,
            self_attn_ln_weights,
            feedforward,
            ff_weights,
            ffn_layer_norm,
            ffn_ln_weights,
            add,
        })
    }

    /// Performs a forward pass through the encoder layer.
    ///
    /// This is a stateless operation that does not use a KV cache.
    /// It can handle both Pre-Norm and Post-Norm architectures based on the config.
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        config: &dyn TransformerConfig,
        temp: &mut TempStorage,
    ) -> Result<GpuTensor> {

        if config.is_prenorm() {
            self.forward_prenorm(encoder, hidden_states, attention_mask, temp)
        } else {
            self.forward_postnorm(encoder, hidden_states, attention_mask, temp)
        }
    }

    /// The forward pass logic for a Pre-Normalization architecture (e.g., GPT-2 style).
    fn forward_prenorm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        temp: &mut TempStorage,
    ) -> Result<GpuTensor> {
        // --- 1. Self-Attention Block (Norm -> Attention -> Add) ---
        let residual = hidden_states;

        let ln1_out = temp.get(hidden_states.shape().to_vec());
        self.self_attn_layer_norm.encode(encoder, &self.self_attn_ln_weights, hidden_states, &ln1_out);

        // Encoder attention is non-causal and has no cache.
        // The `attend` method can be used for this by providing the input's own projections.
        let (new_k, new_v) = self.self_attn.project_kv(encoder, &ln1_out, &self.self_attn_weights, temp);
        let new_k_split = self.self_attn.split_heads(encoder, &new_k, temp);
        let new_v_split = self.self_attn.split_heads(encoder, &new_v, temp);

        let attn_out = self.self_attn.attend(
            encoder,
            &ln1_out, // Query
            &self.self_attn_weights,
            attention_mask,
            false, // is_causal is false for encoders
            (&new_k_split, &new_v_split), // K and V are from the input itself
            0, // No position offset
            temp,
        );

        let attn_block_output = temp.get(hidden_states.shape().to_vec());
        self.add.encode(encoder, &[residual, &attn_out], &attn_block_output);

        // --- 2. Feed-Forward Block (Norm -> FFN -> Add) ---
        let residual_2 = &attn_block_output;
        
        let ln2_out = temp.get(residual_2.shape().to_vec());
        self.ffn_layer_norm.encode(encoder, &self.ffn_ln_weights, residual_2, &ln2_out);

        let ffn_out = self.feedforward.encode(encoder, &ln2_out, &self.ff_weights, temp);

        let final_output = temp.get(residual_2.shape().to_vec());
        self.add.encode(encoder, &[residual_2, &ffn_out], &final_output);

        Ok(final_output)
    }

    /// The forward pass logic for a Post-Normalization architecture (e.g., BERT style).
    fn forward_postnorm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        temp: &mut TempStorage,
    ) -> Result<GpuTensor> {
        // --- 1. Self-Attention Block (Attention -> Add -> Norm) ---
        let residual = hidden_states;

        let (new_k, new_v) = self.self_attn.project_kv(encoder, hidden_states, &self.self_attn_weights, temp);
        let new_k_split = self.self_attn.split_heads(encoder, &new_k, temp);
        let new_v_split = self.self_attn.split_heads(encoder, &new_v, temp);

        let attn_out = self.self_attn.attend(
            encoder,
            hidden_states, // Query
            &self.self_attn_weights,
            attention_mask,
            false, // is_causal
            (&new_k_split, &new_v_split),
            0,
            temp,
        );

        let add_1_out = temp.get(hidden_states.shape().to_vec());
        self.add.encode(encoder, &[residual, &attn_out], &add_1_out);
        
        let attn_block_output = temp.get(hidden_states.shape().to_vec());
        self.self_attn_layer_norm.encode(encoder, &self.self_attn_ln_weights, &add_1_out, &attn_block_output);

        // --- 2. Feed-Forward Block (FFN -> Add -> Norm) ---
        let residual_2 = &attn_block_output;

        let ffn_out = self.feedforward.encode(encoder, residual_2, &self.ff_weights, temp);

        let add_2_out = temp.get(residual_2.shape().to_vec());
        self.add.encode(encoder, &[residual_2, &ffn_out], &add_2_out);

        let final_output = temp.get(residual_2.shape().to_vec());
        self.ffn_layer_norm.encode(encoder, &self.ffn_ln_weights, &add_2_out, &final_output);

        Ok(final_output)
    }
}