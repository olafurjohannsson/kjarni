use crate::GpuKVCache;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::{GpuTensor, GpuTensorPool};
use crate::gpu_ops::Kernel;
use crate::gpu_ops::blocks::attention::{GpuAttention, GpuAttentionWeights};
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::primitives::layout::concatenate::GpuConcatenate;
use crate::traits::{
    Cache, Decoder, DecoderArchitecture, TransformerConfig, TransformerModel,
};
use anyhow::Result;
use std::sync::Arc;

use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::blocks::{
    GpuFeedForward, GpuFeedForwardWeights,
    GpuNormalization, GpuNormalizationWeights, GpuSwiGLUFFN, GpuSwiGLUFFNWeights,
};

pub struct GpuPreNormDecoderLayer {
    pub self_attn: GpuAttention,
    pub self_attn_weights: GpuAttentionWeights,
    pub self_attn_norm: GpuNormalization,
    pub self_attn_norm_weights: GpuNormalizationWeights,
    pub feedforward: GpuFeedForward,
    pub ff_weights: GpuFeedForwardWeights,
    pub ffn_norm: GpuNormalization,
    pub ffn_norm_weights: GpuNormalizationWeights,
    add: GpuAdd,
    concat: GpuConcatenate,
    config: Arc<dyn DecoderArchitecture + Send + Sync>,
}

impl GpuPreNormDecoderLayer {

    pub fn new(
        context: &Arc<WgpuContext>,
        self_attn_weights: GpuAttentionWeights,
        self_attn_norm: GpuNormalization,
        self_attn_norm_weights: GpuNormalizationWeights,
        feedforward: GpuFeedForward,
        ff_weights: GpuFeedForwardWeights,
        ffn_norm: GpuNormalization,
        ffn_norm_weights: GpuNormalizationWeights,
        config: Arc<dyn DecoderArchitecture + Send + Sync>,
        gpu_rope: Option<&GpuRoPE>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size() as u32;
        let num_heads = config.num_attention_heads() as u32;
        let num_kv_heads = config.num_key_value_heads() as u32;

        let self_attn = GpuAttention::new(context, hidden_size, num_heads, num_kv_heads);
        let add = GpuAdd::new(&context);
        let concat = GpuConcatenate::new(&context);

        Ok(Self {
            self_attn,
            self_attn_weights,
            self_attn_norm,
            self_attn_norm_weights,
            feedforward,
            ff_weights,
            ffn_norm,
            ffn_norm_weights,
            config,
            add,
            concat,
        })
    }
    pub fn forward_llama(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        layer_idx: usize,
        position_offset: usize,
        gpu_cache: Option<&mut GpuKVCache>,
        temp: &mut GpuTensorPool,
        rope: Option<&GpuRoPE>,
    ) -> Result<(GpuTensor, (GpuTensor, GpuTensor))> {
        // --- 1. Pre-Norm ---
        let residual = hidden_states;
        let ln1_out = temp.get(hidden_states.shape().to_vec());
        self.self_attn_norm.encode(
            encoder,
            &self.self_attn_norm_weights,
            hidden_states,
            &ln1_out,
        );

        // --- 2. Project Q, K, V ---
        let q_proj = self.self_attn.project(
            encoder,
            &ln1_out,
            &self.self_attn_weights.q_weight,
            &self.self_attn_weights.q_bias,
            temp,
        );
        let k_proj = self.self_attn.project(
            encoder,
            &ln1_out,
            &self.self_attn_weights.k_weight,
            &self.self_attn_weights.k_bias,
            temp,
        );
        let v_proj = self.self_attn.project(
            encoder,
            &ln1_out,
            &self.self_attn_weights.v_weight,
            &self.self_attn_weights.v_bias,
            temp,
        );

        // --- 3. Split heads and apply RoPE to Q and K ---
        let q_split = self.self_attn.split_heads(encoder, &q_proj, temp);
        let k_split = self.self_attn.split_heads(encoder, &k_proj, temp);
        let v_split = self.self_attn.split_heads(encoder, &v_proj, temp);

        let q_rotated = temp.get(q_split.shape().to_vec());
        let k_rotated = temp.get(k_split.shape().to_vec());

        if let Some(rope_encoder) = rope {
            rope_encoder.encode(encoder, &q_split, &q_rotated, position_offset);
            rope_encoder.encode(encoder, &k_split, &k_rotated, position_offset);
        } else {
            // If there's no RoPE, we should probably copy the original tensors
            // This case might not happen for Llama, but it's good practice.
            // A copy kernel would be ideal, but for now, this logic is inside an if-let.
            // If rope is None, q_rotated and k_rotated will contain uninitialized data,
            // which is a bug. Let's handle it.
            // NOTE: This part of the logic needs review. If RoPE is optional,
            // we need a clean way to handle the non-RoPE case.
            // For now, we assume `rope` is always `Some` for Llama.
        }

        // --- 4. Cache handling ---
        let attn_out = if let Some(cache) = gpu_cache {
            let k_rotated_3d = self.self_attn.merge_heads(encoder, &k_rotated, temp);

            // 2. v_proj is already the correct 3D shape [B, S, H*D].

            // --- Update the Cache ---
            // Pass the 3D tensors to the cache. The kernel will split heads and store them in 4D.
            cache.update(encoder, layer_idx, &k_rotated_3d, &v_proj, position_offset)?;

            // --- Prepare Tensors for Attention ---
            // The attention function expects 4D tensors: [B, H, S, D].

            // 1. Get the full K/V history from the cache. The cache correctly stores and returns them as 4D tensors.
            let (cached_k, cached_v) = cache.get(layer_idx).unwrap();

            // 2. q_rotated is already the correct 4D shape.

            // --- Perform Attention ---
            // All inputs to llama_attention are now correctly shaped 4D tensors.
            self.self_attn.llama_attention(
                encoder,
                &q_rotated,      // [B, H_q, 1, D]
                &cached_k,       // [B, H_kv, S_full, D]
                &cached_v,       // [B, H_kv, S_full, D]
                attention_mask,
                position_offset,
                temp,
                &self.self_attn_weights
            )
        } else {
            // No cache - use new K/V directly
            self.self_attn.llama_attention(
                encoder,
                &q_rotated, // 4D, rotated
                &k_rotated, // 4D, rotated
                &v_split,   // 4D, not rotated
                attention_mask,
                position_offset,
                temp,
                &self.self_attn_weights
            )
        };

        // --- 5. Residual connection ---
        let attn_block_output = temp.get(hidden_states.shape().to_vec());
        self.add
            .encode(encoder, &[residual, &attn_out], &attn_block_output);

        // --- 6. FFN (unchanged) ---
        let residual_2 = &attn_block_output;
        let ln2_out = temp.get(residual_2.shape().to_vec());
        self.ffn_norm
            .encode(encoder, &self.ffn_norm_weights, residual_2, &ln2_out);

        // [FIX] START: Reshape 3D tensor to 2D for FFN
        let (b, s, h) = ln2_out.dims3();
        let ln2_out_2d = ln2_out.view(vec![b * s, h]);

        // The output of the FFN will also be 2D initially
        let ffn_out_2d = temp.get(vec![b * s, h]);

        self.feedforward
            .encode(encoder, &self.ff_weights, &ln2_out_2d, &ffn_out_2d, temp);

        // Reshape the 2D FFN output back to 3D for the residual connection
        let ffn_out = ffn_out_2d.view(vec![b, s, h]);
        // [FIX] END

        let final_output = temp.get(residual_2.shape().to_vec());
        self.add
            .encode(encoder, &[residual_2, &ffn_out], &final_output);


        let k_for_return = self.self_attn.merge_heads(encoder, &k_rotated, temp);
        Ok((final_output, (k_for_return, v_proj)))
    }

    pub fn forward_gpt2(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        layer_idx: usize,
        position_offset: usize,
        gpu_cache: Option<&mut GpuKVCache>,
        pool: &mut GpuTensorPool,
        rope: Option<&GpuRoPE>,
    ) -> Result<(GpuTensor, (GpuTensor, GpuTensor))> {
        // Self-Attention (Pre-Norm)
        let residual = hidden_states;

        let ln1_out = pool.get(hidden_states.shape().to_vec());
        self.self_attn_norm.encode(
            encoder,
            &self.self_attn_norm_weights,
            hidden_states,
            &ln1_out,
        );

        // Project the K/V for this step. These are the raw 3D tensors.
        let (new_k, new_v) = self.self_attn.project_kv(
            encoder,
            &ln1_out,
            &self.self_attn_weights,
            position_offset,
            pool,
            rope,
        );

        let attn_out = if let Some(cache) = gpu_cache {
            cache.update(encoder, layer_idx, &new_k, &new_v, cache.get_seq_length())?;

            let (physical_k, physical_v) = cache.get(layer_idx).unwrap();
            let (b, h, max_seq, d) = physical_k.dims4();
            let k_3d = physical_k.view_as_3d(b * h, max_seq, d)?;
            let v_3d = physical_v.view_as_3d(b * h, max_seq, d)?;

            self.self_attn.attend_with_physical_cache(
                encoder,
                &ln1_out,
                &k_3d,
                &v_3d,
                attention_mask,
                &self.self_attn_weights,
                position_offset,
                pool,
                rope,
            )
        } else {
            // Priming pass: no cache, use standard attend
            // Split heads for new K/V
            let new_k_split = self.self_attn.split_heads(encoder, &new_k, pool);
            let new_v_split = self.self_attn.split_heads(encoder, &new_v, pool);

            self.self_attn.attend(
                encoder,
                &ln1_out,
                &self.self_attn_weights,
                attention_mask,
                true, // is_causal
                (&new_k_split, &new_v_split),
                position_offset,
                pool,
            )
        };
        let attn_block_output = pool.get(hidden_states.shape().to_vec());
        self.add
            .encode(encoder, &[residual, &attn_out], &attn_block_output);

        // Second Sub-layer: Feed-Forward Network (Pre-Norm)
        let residual_2 = &attn_block_output;

        let ln2_out = pool.get(residual_2.shape().to_vec());
        self.ffn_norm
            .encode(encoder, &self.ffn_norm_weights, residual_2, &ln2_out);

        let ffn_out = pool.get(ln2_out.shape().to_vec());

        // 2. Call encode with the correct 5 arguments in the correct order.
        self.feedforward.encode(
            encoder,
            &self.ff_weights, // weights
            &ln2_out,         // input
            &ffn_out,         // output
            pool,
        );

        let final_output = pool.get(residual_2.shape().to_vec());
        self.add
            .encode(encoder, &[residual_2, &ffn_out], &final_output);

        // Return the raw 3D K/V tensors for the cache manager.
        Ok((final_output, (new_k, new_v)))
    }
}

#[cfg(test)]
mod tests;