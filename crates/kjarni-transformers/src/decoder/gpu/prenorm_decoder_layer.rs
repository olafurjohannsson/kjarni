use std::sync::Arc;

use anyhow::Result;

use crate::gpu::normalization::{GpuNormalization, GpuNormalizationWeights};
use crate::gpu_ops::blocks::attention::{GpuAttention, GpuAttentionWeights};
use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::blocks::{GpuFeedForward, GpuFeedForwardWeights};
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu::{GpuTensor, GpuTensorPool, Kernel};
use crate::gpu::cache::GpuKVCache;
use crate::WgpuContext;

pub struct GpuPreNormDecoderLayer {
    pub self_attn: GpuAttention,
    pub self_attn_weights: GpuAttentionWeights,
    pub self_attn_norm: GpuNormalization,
    pub self_attn_norm_weights: GpuNormalizationWeights,
    pub feedforward: GpuFeedForward,
    pub ff_weights: GpuFeedForwardWeights,
    pub ffn_norm: GpuNormalization,
    pub ffn_norm_weights: GpuNormalizationWeights,
    pub add: GpuAdd,
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
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> Result<Self> {
        let self_attn = GpuAttention::new(
            context,
            hidden_size as u32,
            num_heads as u32,
            num_kv_heads as u32,
        );
        let add = GpuAdd::new(context);

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
        let residual = hidden_states;
        let ln1_out = temp.get(hidden_states.shape().to_vec());
        self.self_attn_norm.encode(
            encoder,
            &self.self_attn_norm_weights,
            hidden_states,
            &ln1_out,
        );

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

        let q_split = self.self_attn.split_heads(encoder, &q_proj, temp);
        let k_split = self.self_attn.split_heads(encoder, &k_proj, temp);
        let v_split = self.self_attn.split_heads(encoder, &v_proj, temp);

        let q_rotated = temp.get(q_split.shape().to_vec());
        let k_rotated = temp.get(k_split.shape().to_vec());

        if let Some(rope_encoder) = rope {
            rope_encoder.encode(encoder, &q_split, &q_rotated, position_offset);
            rope_encoder.encode(encoder, &k_split, &k_rotated, position_offset);
        }

        let attn_out = if let Some(cache) = gpu_cache {
            let k_rotated_3d = self.self_attn.merge_heads(encoder, &k_rotated, temp);

            cache.update(encoder, layer_idx, &k_rotated_3d, &v_proj, position_offset)?;

            let (cached_k, cached_v) = cache.get(layer_idx).unwrap();

            self.self_attn.llama_attention(
                encoder,
                &q_rotated,
                &cached_k,
                &cached_v,
                attention_mask,
                position_offset,
                temp,
                &self.self_attn_weights,
            )
        } else {
            self.self_attn.llama_attention(
                encoder,
                &q_rotated,
                &k_rotated,
                &v_split,
                attention_mask,
                position_offset,
                temp,
                &self.self_attn_weights,
            )
        };

        let attn_block_output = temp.get(hidden_states.shape().to_vec());
        self.add
            .encode(encoder, &[residual, &attn_out], &attn_block_output);

        let residual_2 = &attn_block_output;
        let ln2_out = temp.get(residual_2.shape().to_vec());
        self.ffn_norm
            .encode(encoder, &self.ffn_norm_weights, residual_2, &ln2_out);

        let (b, s, h) = ln2_out.dims3();
        let ln2_out_2d = ln2_out.view(vec![b * s, h]);

        let ffn_out_2d = temp.get(vec![b * s, h]);

        self.feedforward
            .encode(encoder, &self.ff_weights, &ln2_out_2d, &ffn_out_2d, temp);

        let ffn_out = ffn_out_2d.view(vec![b, s, h]);

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
        let residual = hidden_states;

        let ln1_out = pool.get(hidden_states.shape().to_vec());
        self.self_attn_norm.encode(
            encoder,
            &self.self_attn_norm_weights,
            hidden_states,
            &ln1_out,
        );

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
            let new_k_split = self.self_attn.split_heads(encoder, &new_k, pool);
            let new_v_split = self.self_attn.split_heads(encoder, &new_v, pool);

            self.self_attn.attend(
                encoder,
                &ln1_out,
                &self.self_attn_weights,
                attention_mask,
                true,
                (&new_k_split, &new_v_split),
                position_offset,
                pool,
            )
        };

        let attn_block_output = pool.get(hidden_states.shape().to_vec());
        self.add
            .encode(encoder, &[residual, &attn_out], &attn_block_output);

        let residual_2 = &attn_block_output;

        let ln2_out = pool.get(residual_2.shape().to_vec());
        self.ffn_norm
            .encode(encoder, &self.ffn_norm_weights, residual_2, &ln2_out);

        let ffn_out = pool.get(ln2_out.shape().to_vec());

        self.feedforward.encode(
            encoder,
            &self.ff_weights,
            &ln2_out,
            &ffn_out,
            pool,
        );

        let final_output = pool.get(residual_2.shape().to_vec());
        self.add
            .encode(encoder, &[residual_2, &ffn_out], &final_output);

        Ok((final_output, (new_k, new_v)))
    }
}