use anyhow::{Context, Result};
use crate::{
    WgpuContext,
    cache::GpuKVCache,
    decoder::prelude::*,
    gpu_ops::{
        GpuTensor, GpuTensorPool, Kernel,
        blocks::{
            GpuFeedForward, GpuFeedForwardWeights, GpuNormalization, GpuNormalizationWeights,
            GpuSwiGLUFFN, GpuSwiGLUFFNWeights,
            attention::{
                GpuAttention, GpuAttentionWeights, GpuDecoderSelfAttention, GpuRoPEAttention,
            },
            embeddings::{GpuEmbeddingWeights, GpuEmbeddings},
            rms_norm::{GpuRMSNorm, GpuRMSNormWeights},
            rope::GpuRoPE,
        },
        primitives::add::GpuAdd,
    },
    models::base::{ModelInput, ModelLoadConfig},
    tensor::DType,
    traits::{ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};
use std::sync::Arc;

pub struct GpuRoPEDecoderLayer {
    pub self_attn: GpuRoPEAttention,
    pub self_attn_weights: GpuAttentionWeights,
    pub self_attn_norm: GpuNormalization,
    pub self_attn_norm_weights: GpuNormalizationWeights,
    pub feedforward: GpuFeedForward,
    pub ff_weights: GpuFeedForwardWeights,
    pub ffn_norm: GpuNormalization,
    pub ffn_norm_weights: GpuNormalizationWeights,
    pub add: GpuAdd,
}

impl GpuRoPEDecoderLayer {
    pub fn new(
        context: &Arc<WgpuContext>,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        self_attn_weights: GpuAttentionWeights,
        self_attn_norm_weights: GpuNormalizationWeights,
        ff_weights: GpuFeedForwardWeights,
        ffn_norm_weights: GpuNormalizationWeights,
        norm_eps: f32,
    ) -> Result<Self> {
        let self_attn = GpuRoPEAttention::new(
            context,
            hidden_size as u32,
            num_heads as u32,
            num_kv_heads as u32,
        );
        let add = GpuAdd::new(context);

        // Llama specific blocks
        let self_attn_norm = GpuNormalization::RMSNorm(GpuRMSNorm::new(context, norm_eps));
        let ffn_norm = GpuNormalization::RMSNorm(GpuRMSNorm::new(context, norm_eps));
        let feedforward = GpuFeedForward::SwiGLU(GpuSwiGLUFFN::new(context)?);

        Ok(Self {
            self_attn,
            self_attn_weights,
            self_attn_norm,
            self_attn_norm_weights,
            feedforward,
            ff_weights,
            ffn_norm,
            ffn_norm_weights,
            add,
        })
    }
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        layer_idx: usize,
        position_offset: usize,
        gpu_cache: Option<&mut GpuKVCache>,
        pool: &mut GpuTensorPool,
        rope: &GpuRoPE,
    ) -> Result<GpuTensor> {
        // --- 1. Self-Attention Block (Pre-Norm) ---
        let residual = hidden_states;
        let ln1_out = pool.get(hidden_states.shape().to_vec());
        self.self_attn_norm.encode(
            encoder,
            &self.self_attn_norm_weights,
            hidden_states,
            &ln1_out,
        );

        // Get cached KV if available (returns Option<(&GpuTensor, &GpuTensor)>)
        let cached_tensors = gpu_cache.as_ref().and_then(|c| c.get(layer_idx));

        let cached_kv: Option<(&GpuTensor, &GpuTensor)> =
            cached_tensors.as_ref().map(|(k, v)| (k, v));

        // Single forward call handles: Q/K/V projection, RoPE, GQA, attention, output projection
        let attn_output = self.self_attn.forward(
            encoder,
            &ln1_out,
            &self.self_attn_weights,
            rope,
            attention_mask,
            cached_kv,
            position_offset,
            pool,
        )?;

        // Update cache with new K/V (need mutable borrow now)
        if let Some(cache) = gpu_cache {
            cache.update(
                encoder,
                layer_idx,
                &attn_output.new_k,
                &attn_output.new_v,
                position_offset,
            )?;
        }

        // Residual add
        let attn_block_output = pool.get(hidden_states.shape().to_vec());
        self.add.encode(
            encoder,
            &[residual, &attn_output.hidden_states],
            &attn_block_output,
        );

        // --- 2. Feed-Forward Block (Pre-Norm) ---
        let residual_2 = &attn_block_output;
        let ln2_out = pool.get(residual_2.shape().to_vec());
        self.ffn_norm
            .encode(encoder, &self.ffn_norm_weights, residual_2, &ln2_out);

        // FFN (needs 2D input)
        let (b, s, h) = ln2_out.dims3();
        let ln2_out_2d = ln2_out.view(vec![b * s, h]);
        let ffn_out_2d = pool.get(vec![b * s, h]);

        self.feedforward
            .encode(encoder, &self.ff_weights, &ln2_out_2d, &ffn_out_2d, pool);
        let ffn_out = ffn_out_2d.view(vec![b, s, h]);

        // Residual add
        let final_output = pool.get(residual_2.shape().to_vec());
        self.add
            .encode(encoder, &[residual_2, &ffn_out], &final_output);

        Ok(final_output)
    }
}
