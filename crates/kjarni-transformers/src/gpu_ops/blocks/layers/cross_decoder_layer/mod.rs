use crate::gpu_ops::blocks::attention::{GpuAttentionWeights, GpuCrossAttention, GpuDecoderSelfAttention};
use crate::gpu_ops::blocks::{
    GpuFeedForward, GpuFeedForwardWeights,
};
use crate::gpu::normalization::{
    GpuNormalization, GpuNormalizationWeights, 
};
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::{GpuTensor, GpuTensorPool, Kernel};
use crate::traits::ModelMetadata;
use crate::WgpuContext;
use anyhow::Result;
use std::sync::Arc;

pub struct GpuCrossDecoderLayer {
    pub self_attn: GpuDecoderSelfAttention,
    pub self_attn_weights: GpuAttentionWeights,
    pub self_attn_norm: GpuNormalization,
    pub self_attn_norm_weights: GpuNormalizationWeights,
    pub cross_attn: GpuCrossAttention,
    pub cross_attn_weights: GpuAttentionWeights,
    pub cross_attn_norm: GpuNormalization,
    pub cross_attn_norm_weights: GpuNormalizationWeights,
    pub feedforward: GpuFeedForward,
    pub ff_weights: GpuFeedForwardWeights,
    pub ffn_norm: GpuNormalization,
    pub ffn_norm_weights: GpuNormalizationWeights,
    pub add: GpuAdd,
}

impl GpuCrossDecoderLayer {
    pub fn new(
        context: &Arc<WgpuContext>,
        self_attn_weights: GpuAttentionWeights,
        self_attn_norm: GpuNormalization,
        self_attn_norm_weights: GpuNormalizationWeights,
        cross_attn_weights: GpuAttentionWeights,
        cross_attn_norm: GpuNormalization,
        cross_attn_norm_weights: GpuNormalizationWeights,
        feedforward: GpuFeedForward,
        ff_weights: GpuFeedForwardWeights,
        ffn_norm: GpuNormalization,
        ffn_norm_weights: GpuNormalizationWeights,
        metadata: &ModelMetadata,
    ) -> Result<Self> {
        let hidden_size = metadata.hidden_size as u32;
        let num_heads = metadata.num_attention_heads as u32;

        Ok(Self {
            self_attn: GpuDecoderSelfAttention::new(context, hidden_size, num_heads),
            self_attn_weights,
            self_attn_norm,
            self_attn_norm_weights,
            cross_attn: GpuCrossAttention::new(context, hidden_size, num_heads),
            cross_attn_weights,
            cross_attn_norm,
            cross_attn_norm_weights,
            feedforward,
            ff_weights,
            ffn_norm,
            ffn_norm_weights,
            add: GpuAdd::new(context),
        })
    }
   
    pub fn precompute_cross_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        encoder_hidden_states: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> (GpuTensor, GpuTensor) {
        self.cross_attn.precompute_kv(
            encoder,
            encoder_hidden_states,
            &self.cross_attn_weights,
            pool,
        )
    }

    /// Forward pass (post-norm for BART).
    ///
    /// # Arguments
    ///
    /// * `cross_kv` - Precomputed (K, V) tuple from `precompute_cross_kv`.
    /// * `encoder_mask` - Encoder padding mask (optional for BART, usually None).
    ///
    /// # Returns
    ///
    /// (output, new_k, new_v) for self-attention cache update.
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        cross_kv: &(GpuTensor, GpuTensor),
        decoder_mask: &GpuTensor,
        encoder_mask: Option<&GpuTensor>,
        cached_kv: Option<(&GpuTensor, &GpuTensor)>,
        cache_len: usize,
        pool: &mut GpuTensorPool,
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor)> {
        // ========================================
        // 1. Self-Attention (Post-Norm)
        // ========================================
        let residual = hidden_states;

        let self_attn_out = self.self_attn.forward(
            encoder,
            residual,
            &self.self_attn_weights,
            decoder_mask,
            cached_kv,
            cache_len,
            pool,
        )?;

        let after_add1 = pool.get(residual.shape().to_vec());
        self.add.encode(encoder, &[residual, &self_attn_out.hidden_states], &after_add1);

        let after_norm1 = pool.get(after_add1.shape().to_vec());
        self.self_attn_norm.encode(
            encoder,
            &self.self_attn_norm_weights,
            &after_add1,
            &after_norm1,
        );

        // ========================================
        // 2. Cross-Attention (Post-Norm)
        // ========================================
        let residual = &after_norm1;

        let cross_attn_out = self.cross_attn.forward(
            encoder,
            residual,
            cross_kv,
            &self.cross_attn_weights,
            encoder_mask,
            pool,
        );

        let after_add2 = pool.get(residual.shape().to_vec());
        self.add.encode(encoder, &[residual, &cross_attn_out], &after_add2);

        let after_norm2 = pool.get(after_add2.shape().to_vec());
        self.cross_attn_norm.encode(
            encoder,
            &self.cross_attn_norm_weights,
            &after_add2,
            &after_norm2,
        );

        // ========================================
        // 3. FFN (Post-Norm)
        // ========================================
        let residual = &after_norm2;

        let ffn_out = pool.get(residual.shape().to_vec());
        self.feedforward.encode(
            encoder,
            &self.ff_weights,
            residual,
            &ffn_out,
            pool,
        );

        let after_add3 = pool.get(residual.shape().to_vec());
        self.add.encode(encoder, &[residual, &ffn_out], &after_add3);

        let final_output = pool.get(after_add3.shape().to_vec());
        self.ffn_norm.encode(
            encoder,
            &self.ffn_norm_weights,
            &after_add3,
            &final_output,
        );

        Ok((final_output, self_attn_out.new_k, self_attn_out.new_v))
    }

}

#[cfg(test)]
mod tests;
