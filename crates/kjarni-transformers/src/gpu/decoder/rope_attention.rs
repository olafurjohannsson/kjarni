//! RoPE-based causal attention for modern LLMs.

use std::sync::Arc;

use anyhow::Result;

use crate::gpu_ops::blocks::attention::ops::AttentionOps;
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::primitives::layout::concatenate::GpuConcatenate;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::gpu_ops::primitives::repeat_kv::GpuRepeatKV;
use crate::gpu_ops::{GpuTensor, GpuTensorPool};
use crate::WgpuContext;

pub struct GpuRoPEAttention {
    ops: AttentionOps,
    concatenate: GpuConcatenate,
    slice: GpuSlice,
    repeat_kv: GpuRepeatKV,
}

pub struct RoPEAttentionOutput {
    pub hidden_states: GpuTensor,
    pub new_k: GpuTensor,
    pub new_v: GpuTensor,
}

impl GpuRoPEAttention {
    pub fn new(
        context: &Arc<WgpuContext>,
        hidden_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
    ) -> Self {
        Self {
            ops: AttentionOps::new(context, hidden_size, num_heads, num_kv_heads),
            concatenate: GpuConcatenate::new(context),
            slice: GpuSlice::new(context),
            repeat_kv: GpuRepeatKV::new(context),
        }
    }

    pub fn ops(&self) -> &AttentionOps {
        &self.ops
    }

    pub fn num_heads(&self) -> u32 {
        self.ops.num_heads()
    }

    pub fn num_kv_heads(&self) -> u32 {
        self.ops.num_kv_heads()
    }

    pub fn gqa_ratio(&self) -> u32 {
        self.ops.num_heads() / self.ops.num_kv_heads()
    }

    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        weights: &GpuAttentionWeights,
        rope: &GpuRoPE,
        attention_mask: &GpuTensor,
        cached_kv: Option<(&GpuTensor, &GpuTensor)>,
        position_offset: usize,
        pool: &mut GpuTensorPool,
    ) -> Result<RoPEAttentionOutput> {
        let num_heads = self.ops.num_heads() as usize;
        let num_kv_heads = self.ops.num_kv_heads() as usize;

        let q_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.q_weight,
            &weights.q_bias,
            pool,
        );
        let k_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.k_weight,
            &weights.k_bias,
            pool,
        );
        let v_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.v_weight,
            &weights.v_bias,
            pool,
        );

        let q_heads = self.ops.split_heads(encoder, &q_proj, pool);
        let k_heads = self.ops.split_heads(encoder, &k_proj, pool);
        let v_heads = self.ops.split_heads(encoder, &v_proj, pool);

        let q_rotated = pool.get(q_heads.shape().to_vec());
        let k_rotated = pool.get(k_heads.shape().to_vec());

        rope.encode(encoder, &q_heads, &q_rotated, position_offset);
        rope.encode(encoder, &k_heads, &k_rotated, position_offset);

        let new_k_3d = self.ops.merge_heads(encoder, &k_rotated, pool);
        let new_v_3d = self.ops.merge_heads(encoder, &v_heads, pool);

        let (full_k, full_v) = if let Some((cached_k, cached_v)) = cached_kv {
            let cache_len = position_offset;
            if cache_len > 0 {
                self.concat_with_cache(
                    encoder, &k_rotated, &v_heads, cached_k, cached_v, cache_len, pool,
                )?
            } else {
                (k_rotated.clone(), v_heads.clone())
            }
        } else {
            (k_rotated.clone(), v_heads.clone())
        };

        let (k_for_attn, v_for_attn) = if num_heads != num_kv_heads {
            self.expand_kv_for_gqa(encoder, &full_k, &full_v, pool)
        } else {
            (full_k, full_v)
        };

        let context = self.compute_attention(
            encoder,
            &q_rotated,
            &k_for_attn,
            &v_for_attn,
            attention_mask,
            position_offset,
            pool,
        );

        let context_merged = self.ops.merge_heads(encoder, &context, pool);
        let output = self.ops.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        );

        Ok(RoPEAttentionOutput {
            hidden_states: output,
            new_k: new_k_3d,
            new_v: new_v_3d,
        })
    }

    fn concat_with_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        k_new: &GpuTensor,
        v_new: &GpuTensor,
        cached_k: &GpuTensor,
        cached_v: &GpuTensor,
        cache_len: usize,
        pool: &mut GpuTensorPool,
    ) -> Result<(GpuTensor, GpuTensor)> {
        let (b, h, s_new, d) = k_new.dims4();
        let full_len = cache_len + s_new;

        let slice_shape = &[b, h, cache_len, d];
        let slice_offset = &[0, 0, 0, 0];

        let valid_cache_k = cached_k.slice(encoder, &self.slice, slice_offset, slice_shape)?;
        let valid_cache_v = cached_v.slice(encoder, &self.slice, slice_offset, slice_shape)?;

        let full_k = pool.get(vec![b, h, full_len, d]);
        let full_v = pool.get(vec![b, h, full_len, d]);

        self.concatenate
            .encode(encoder, &[&valid_cache_k, k_new], &full_k, 2);
        self.concatenate
            .encode(encoder, &[&valid_cache_v, v_new], &full_v, 2);

        Ok((full_k, full_v))
    }

    fn expand_kv_for_gqa(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        k: &GpuTensor,
        v: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> (GpuTensor, GpuTensor) {
        let (batch, _kv_heads, seq_len, head_dim) = k.dims4();
        let num_heads = self.ops.num_heads() as usize;

        let expanded_shape = vec![batch, num_heads, seq_len, head_dim];

        let k_expanded = pool.get(expanded_shape.clone());
        let v_expanded = pool.get(expanded_shape);

        self.repeat_kv.encode(encoder, k, &k_expanded);
        self.repeat_kv.encode(encoder, v, &v_expanded);

        (k_expanded, v_expanded)
    }

    fn compute_attention(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        mask: &GpuTensor,
        position_offset: usize,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let k_transposed = k.permute(encoder, self.ops.permute_kernel(), &[0, 1, 3, 2]);
        let scores = self.ops.bmm_4d(encoder, q, &k_transposed, pool);

        self.ops.apply_mask_and_softmax(
            encoder,
            &scores,
            Some(mask),
            true,
            position_offset,
        );

        self.ops.bmm_4d(encoder, &scores, v, pool)
    }

    pub fn project_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        weights: &GpuAttentionWeights,
        rope: &GpuRoPE,
        position_offset: usize,
        pool: &mut GpuTensorPool,
    ) -> (GpuTensor, GpuTensor) {
        let k_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.k_weight,
            &weights.k_bias,
            pool,
        );
        let v_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.v_weight,
            &weights.v_bias,
            pool,
        );

        let k_heads = self.ops.split_heads(encoder, &k_proj, pool);

        let k_rotated = pool.get(k_heads.shape().to_vec());
        rope.encode(encoder, &k_heads, &k_rotated, position_offset);

        let k_3d = self.ops.merge_heads(encoder, &k_rotated, pool);

        (k_3d, v_proj)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::{Array1, Array2, Array3, Array4};

    use super::*;
    use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
    use crate::gpu_ops::blocks::rope::GpuRoPE;
    use crate::gpu_ops::GpuTensorPool;
    use crate::rope::RoPE;
    use crate::WgpuContext;

    async fn setup() -> Arc<WgpuContext> {
        WgpuContext::new().await.unwrap()
    }

    fn create_dummy_weights(
        ctx: &Arc<WgpuContext>,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> GpuAttentionWeights {
        let head_dim = hidden_size / num_heads;
        let q_dim = hidden_size;
        let kv_dim = num_kv_heads * head_dim;

        let q_weight =
            GpuTensor::from_ndarray(ctx, &Array2::<f32>::zeros((q_dim, hidden_size))).unwrap();
        let q_bias = GpuTensor::from_ndarray(ctx, &Array1::<f32>::zeros(q_dim)).unwrap();

        let k_weight =
            GpuTensor::from_ndarray(ctx, &Array2::<f32>::zeros((kv_dim, hidden_size))).unwrap();
        let k_bias = GpuTensor::from_ndarray(ctx, &Array1::<f32>::zeros(kv_dim)).unwrap();
        let v_weight =
            GpuTensor::from_ndarray(ctx, &Array2::<f32>::zeros((kv_dim, hidden_size))).unwrap();
        let v_bias = GpuTensor::from_ndarray(ctx, &Array1::<f32>::zeros(kv_dim)).unwrap();

        let o_weight =
            GpuTensor::from_ndarray(ctx, &Array2::<f32>::zeros((hidden_size, hidden_size)))
                .unwrap();
        let o_bias = GpuTensor::from_ndarray(ctx, &Array1::<f32>::zeros(hidden_size)).unwrap();

        GpuAttentionWeights::new(
            q_weight,
            Some(q_bias),
            k_weight,
            Some(k_bias),
            v_weight,
            Some(v_bias),
            o_weight,
            Some(o_bias),
        )
        .unwrap()
    }

    fn create_rope(ctx: &Arc<WgpuContext>, head_dim: usize, max_seq_len: usize) -> GpuRoPE {
        let cpu_rope = RoPE::new(head_dim, max_seq_len, 10000.0);
        GpuRoPE::new(ctx, &cpu_rope.cos_cache, &cpu_rope.sin_cache).unwrap()
    }

    #[tokio::test]
    async fn test_rope_attention_prefill_no_cache() {
        let ctx = setup().await;

        let hidden_size = 256;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = hidden_size / num_heads;
        let batch_size = 1;
        let seq_len = 10;

        let attn = GpuRoPEAttention::new(
            &ctx,
            hidden_size as u32,
            num_heads as u32,
            num_kv_heads as u32,
        );
        let weights = create_dummy_weights(&ctx, hidden_size, num_heads, num_kv_heads);
        let rope = create_rope(&ctx, head_dim, 128);

        let hidden = Array3::<f32>::ones((batch_size, seq_len, hidden_size));
        let hidden_gpu = GpuTensor::from_ndarray(&ctx, &hidden).unwrap();

        let mask = Array2::<f32>::ones((1, seq_len));
        let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask).unwrap();

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(ctx.clone());

        let result = attn.forward(
            &mut encoder,
            &hidden_gpu,
            &weights,
            &rope,
            &mask_gpu,
            None,
            0,
            &mut pool,
        );

        assert!(result.is_ok(), "prefill should succeed: {:?}", result.err());

        let output = result.unwrap();
        assert_eq!(
            output.hidden_states.shape(),
            &[batch_size, seq_len, hidden_size]
        );
    }

    #[tokio::test]
    async fn test_rope_attention_decode_with_cache() {
        let ctx = setup().await;

        let hidden_size = 256;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = hidden_size / num_heads;
        let batch_size = 1;
        let cache_len = 10;
        let query_len = 1;
        let total_len = cache_len + query_len;

        let attn = GpuRoPEAttention::new(
            &ctx,
            hidden_size as u32,
            num_heads as u32,
            num_kv_heads as u32,
        );
        let weights = create_dummy_weights(&ctx, hidden_size, num_heads, num_kv_heads);
        let rope = create_rope(&ctx, head_dim, 128);

        let hidden = Array3::<f32>::ones((batch_size, query_len, hidden_size));
        let hidden_gpu = GpuTensor::from_ndarray(&ctx, &hidden).unwrap();

        let mask = Array2::<f32>::ones((1, total_len));
        let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask).unwrap();

        let cached_k = Array4::<f32>::zeros((batch_size, num_kv_heads, cache_len, head_dim));
        let cached_v = Array4::<f32>::zeros((batch_size, num_kv_heads, cache_len, head_dim));
        let cached_k_gpu = GpuTensor::from_ndarray(&ctx, &cached_k).unwrap();
        let cached_v_gpu = GpuTensor::from_ndarray(&ctx, &cached_v).unwrap();

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(ctx.clone());

        let result = attn.forward(
            &mut encoder,
            &hidden_gpu,
            &weights,
            &rope,
            &mask_gpu,
            Some((&cached_k_gpu, &cached_v_gpu)),
            cache_len,
            &mut pool,
        );

        assert!(
            result.is_ok(),
            "decode with cache should succeed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert_eq!(
            output.hidden_states.shape(),
            &[batch_size, query_len, hidden_size]
        );
    }

    #[test]
    fn test_rope_attention_dimensions() {
        let hidden_size = 2048u32;
        let num_heads = 32u32;
        let num_kv_heads = 8u32;
        let head_dim = hidden_size / num_heads;
        let gqa_ratio = num_heads / num_kv_heads;

        assert_eq!(head_dim, 64);
        assert_eq!(gqa_ratio, 4);
    }

    #[test]
    fn test_no_gqa() {
        let num_heads = 32u32;
        let num_kv_heads = 32u32;
        let gqa_ratio = num_heads / num_kv_heads;

        assert_eq!(gqa_ratio, 1);
    }
}