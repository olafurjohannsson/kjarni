use anyhow::Result;
use std::sync::Arc;
use wgpu::CommandEncoder;

use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::attention::{GpuAttention, GpuAttentionWeights};
use crate::gpu_ops::blocks::ffn::{GpuFeedForward, GpuFeedForwardWeights};
use crate::gpu_ops::blocks::layer_norm::GpuLayerNorm;
use crate::gpu_ops::blocks::layer_norm::GpuLayerNormWeights;
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::{GpuTensor, Kernel, GpuTensorPool};
use crate::traits::{EncoderArchitecture, LanguageModelConfig, TransformerConfig};
use crate::{EncoderLanguageModel, activations};

pub struct GpuEncoderLayer {
    self_attn: GpuAttention,
    self_attn_weights: GpuAttentionWeights,
    self_attn_layer_norm: GpuLayerNorm,
    self_attn_ln_weights: GpuLayerNormWeights,

    feedforward: GpuFeedForward,
    ff_weights: GpuFeedForwardWeights,
    ffn_layer_norm: GpuLayerNorm,
    ffn_ln_weights: GpuLayerNormWeights,

    add: GpuAdd,
}

impl GpuEncoderLayer {
    pub fn new(
        context: &Arc<WgpuContext>,
        self_attn_weights: GpuAttentionWeights,
        self_attn_ln_weights: GpuLayerNormWeights,
        ff_weights: GpuFeedForwardWeights,
        ffn_ln_weights: GpuLayerNormWeights,
        config: &dyn TransformerConfig,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size() as u32;
        let num_heads = config.num_attention_heads() as u32;
        let num_kv_heads = 0; //config.num_key_value_heads() as u32;

        let self_attn = GpuAttention::new(context, hidden_size, num_heads, num_kv_heads);
        let self_attn_layer_norm = GpuLayerNorm::new(context, config.layer_norm_eps());

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

    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        config: &dyn TransformerConfig,
        pool: &mut GpuTensorPool,
    ) -> Result<GpuTensor> {
        if config.is_prenorm() {
            self.forward_prenorm(encoder, hidden_states, attention_mask, pool)
        } else {
            self.forward_postnorm(encoder, hidden_states, attention_mask, pool)
        }
    }

    fn forward_prenorm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> Result<GpuTensor> {
        let residual = hidden_states;

        let ln1_out = pool.get(hidden_states.shape().to_vec());
        self.self_attn_layer_norm.encode(
            encoder,
            &self.self_attn_ln_weights,
            hidden_states,
            &ln1_out,
        );

        let (new_k, new_v) =
            self.self_attn
                .project_kv(encoder, &ln1_out, &self.self_attn_weights, 0, pool, None);
        let new_k_split = self.self_attn.split_heads(encoder, &new_k, pool);
        let new_v_split = self.self_attn.split_heads(encoder, &new_v, pool);

        let attn_out = self.self_attn.attend(
            encoder,
            &ln1_out, // Query
            &self.self_attn_weights,
            attention_mask,
            false,                        // is_causal is false for encoders
            (&new_k_split, &new_v_split), // K and V are from the input itself
            0,                            // No position offset
            pool,
        );

        let attn_block_output = pool.get(hidden_states.shape().to_vec());
        self.add
            .encode(encoder, &[residual, &attn_out], &attn_block_output);

        let residual_2 = &attn_block_output;

        let ln2_out = pool.get(residual_2.shape().to_vec());
        self.ffn_layer_norm
            .encode(encoder, &self.ffn_ln_weights, residual_2, &ln2_out);

        let ffn_out = self
            .feedforward
            .encode(encoder, &ln2_out, &self.ff_weights, pool);

        let final_output = pool.get(residual_2.shape().to_vec());
        self.add
            .encode(encoder, &[residual_2, &ffn_out], &final_output);

        Ok(final_output)
    }

    /// The forward pass logic for a Post-Normalization architecture (e.g., BERT style).
    fn forward_postnorm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> Result<GpuTensor> {
        let residual = hidden_states;

        let (new_k, new_v) =
            self.self_attn
                .project_kv(encoder, hidden_states, &self.self_attn_weights, 0, pool, None);
        let new_k_split = self.self_attn.split_heads(encoder, &new_k, pool);
        let new_v_split = self.self_attn.split_heads(encoder, &new_v, pool);

        let attn_out = self.self_attn.attend(
            encoder,
            hidden_states, // Query
            &self.self_attn_weights,
            attention_mask,
            false, // is_causal
            (&new_k_split, &new_v_split),
            0,
            pool,
        );

        let add_1_out = pool.get(hidden_states.shape().to_vec());
        self.add.encode(encoder, &[residual, &attn_out], &add_1_out);

        let attn_block_output = pool.get(hidden_states.shape().to_vec());
        self.self_attn_layer_norm.encode(
            encoder,
            &self.self_attn_ln_weights,
            &add_1_out,
            &attn_block_output,
        );

        let residual_2 = &attn_block_output;

        let ffn_out = self
            .feedforward
            .encode(encoder, residual_2, &self.ff_weights, pool);

        let add_2_out = pool.get(residual_2.shape().to_vec());
        self.add
            .encode(encoder, &[residual_2, &ffn_out], &add_2_out);

        let final_output = pool.get(residual_2.shape().to_vec());
        self.ffn_layer_norm
            .encode(encoder, &self.ffn_ln_weights, &add_2_out, &final_output);

        Ok(final_output)
    }
}
