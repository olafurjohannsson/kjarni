//! GPU attention modules for transformer architectures.

use std::sync::Arc;

use anyhow::{Result, anyhow};

use crate::WgpuContext;
use crate::gpu::kernel::Kernel;
use crate::gpu::{GpuTensor, GpuTensorPool};
use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::primitives::add_bias::GpuAddBias;
use crate::gpu_ops::primitives::apply_mask::GpuApplyMask;
use crate::gpu_ops::primitives::bmm::GpuBatchedMatMul;
use crate::gpu_ops::primitives::layout::concatenate::GpuConcatenate;
use crate::gpu_ops::primitives::layout::permute::GpuPermute;
use crate::gpu_ops::primitives::layout::reshape::GpuReshape;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::gpu_ops::primitives::layout::unreshape::GpuUnreshape;
use crate::gpu_ops::primitives::linear::GpuLinearLayer;
use crate::gpu_ops::primitives::matmul::GpuMatMul;
use crate::gpu_ops::primitives::repeat_kv::GpuRepeatKV;
use crate::gpu_ops::primitives::softmax::GpuSoftmax;
use crate::tensor::DType;
use crate::traits::{AttentionLayout, ModelLayout};
use crate::weights::ModelWeights;

mod cross_decoder_attention;
mod decoder_self_attention;
mod encoder_self_attention;
pub mod ops;

pub use cross_decoder_attention::GpuCrossAttention;
pub use decoder_self_attention::{DecoderSelfAttentionOutput, GpuDecoderSelfAttention};
pub use encoder_self_attention::GpuEncoderSelfAttention;
pub use ops::AttentionOps;

/// GPU tensors for attention weights.
pub struct GpuAttentionWeights {
    pub q_weight: GpuTensor,
    pub q_bias: GpuTensor,
    pub k_weight: GpuTensor,
    pub k_bias: GpuTensor,
    pub v_weight: GpuTensor,
    pub v_bias: GpuTensor,
    pub output_weight: GpuTensor,
    pub output_bias: GpuTensor,
}

impl GpuAttentionWeights {
    pub fn from_layout(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        attn_layout: &AttentionLayout,
        layer_idx: usize,
        target_dtype: Option<DType>,
        label_prefix: &str,
        hidden_size: usize, // Add this parameter
    ) -> Result<Self> {
        let i_str = &layer_idx.to_string();

        let load = |template: &str, label: &str| {
            GpuTensor::from_model_weights(
                context,
                weights,
                &template.replace("{}", i_str),
                target_dtype,
                &format!("{}.{}", label_prefix, label),
            )
        };

        // Load optional bias - returns zero tensor if not present
        let load_bias_opt = |template: &Option<String>, label: &str| -> Result<GpuTensor> {
            match template {
                Some(name) => load(name, label),
                None => GpuTensor::zeros(
                    context,
                    vec![hidden_size],
                    target_dtype.unwrap_or(DType::F32),
                    &format!("{}.{}", label_prefix, label),
                ),
            }
        };

        Self::new(
            load(&attn_layout.q_weight, "q_w")?,
            Some(load_bias_opt(&attn_layout.q_bias, "q_b")?),
            load(&attn_layout.k_weight, "k_w")?,
            Some(load_bias_opt(&attn_layout.k_bias, "k_b")?),
            load(&attn_layout.v_weight, "v_w")?,
            Some(load_bias_opt(&attn_layout.v_bias, "v_b")?),
            load(&attn_layout.o_weight, "o_w")?,
            Some(load_bias_opt(&attn_layout.o_bias, "o_b")?),
        )
    }

    pub fn from_decoder_self_attn_layout(
        ctx: &Arc<WgpuContext>,
        weights: &ModelWeights,
        layout: &ModelLayout,
        layer_index: usize,
        target_dt: Option<DType>,
        hidden_size: usize,
    ) -> Result<Self> {
        let decoder_layout = layout
            .decoder
            .as_ref()
            .ok_or_else(|| anyhow!("ModelLayout is missing a decoder layout"))?;

        Self::from_layout(
            ctx,
            weights,
            &decoder_layout.layer.self_attn,
            layer_index,
            target_dt,
            &format!("decoder.layer{}.self_attn", layer_index),
            hidden_size,
        )
    }

    pub fn from_encoder_self_attn_layout(
        ctx: &Arc<WgpuContext>,
        weights: &ModelWeights,
        layout: &ModelLayout,
        layer_index: usize,
        target_dt: Option<DType>,
        hidden_size: usize,
    ) -> Result<Self> {
        let encoder_layout = layout
            .encoder
            .as_ref()
            .ok_or_else(|| anyhow!("ModelLayout is missing an encoder layout"))?;

        Self::from_layout(
            ctx,
            weights,
            &encoder_layout.layer.self_attn,
            layer_index,
            target_dt,
            &format!("encoder.layer{}.self_attn", layer_index),
            hidden_size,
        )
    }

    pub fn from_cross_attn_layout(
        ctx: &Arc<WgpuContext>,
        weights: &ModelWeights,
        layout: &ModelLayout,
        layer_index: usize,
        target_dt: Option<DType>,
        hidden_size: usize,
    ) -> Result<Self> {
        let decoder_layout = layout
            .decoder
            .as_ref()
            .ok_or_else(|| anyhow!("ModelLayout is missing a decoder layout"))?;
        let cross_attn_layout = decoder_layout
            .layer
            .cross_attn
            .as_ref()
            .ok_or_else(|| anyhow!("decoder layout is missing a cross-attention layout"))?;

        Self::from_layout(
            ctx,
            weights,
            cross_attn_layout,
            layer_index,
            target_dt,
            &format!("layer{}.cross_attn", layer_index),
            hidden_size,
        )
    }

    pub fn new(
        q_weight: GpuTensor,
        q_bias: Option<GpuTensor>,
        k_weight: GpuTensor,
        k_bias: Option<GpuTensor>,
        v_weight: GpuTensor,
        v_bias: Option<GpuTensor>,
        output_weight: GpuTensor,
        output_bias: Option<GpuTensor>,
    ) -> Result<Self> {
        let resolve_bias =
            |w: &GpuTensor, b: Option<GpuTensor>, label: &str| -> Result<GpuTensor> {
                match b {
                    Some(tensor) => Ok(tensor),
                    None => {
                        let out_features = w.shape()[0];
                        GpuTensor::zeros(w.context(), vec![out_features], w.dtype(), label)
                    }
                }
            };

        let q_b = resolve_bias(&q_weight, q_bias, "q_bias_zero")?;
        let k_b = resolve_bias(&k_weight, k_bias, "k_bias_zero")?;
        let v_b = resolve_bias(&v_weight, v_bias, "v_bias_zero")?;
        let o_b = resolve_bias(&output_weight, output_bias, "o_bias_zero")?;

        assert_eq!(q_weight.rank(), 2, "Q weight must be 2D");
        assert_eq!(q_b.rank(), 1, "Q bias must be 1D");
        let (q_in, q_out) = q_weight.linear_layer_dims();
        assert_eq!(
            q_out,
            q_b.shape()[0],
            "Q weight output dim must match bias size"
        );

        assert_eq!(k_weight.rank(), 2, "K weight must be 2D");
        assert_eq!(k_b.rank(), 1, "K bias must be 1D");
        let (k_in, k_out) = k_weight.linear_layer_dims();
        assert_eq!(
            k_out,
            k_b.shape()[0],
            "K weight output dim must match bias size"
        );

        assert_eq!(v_weight.rank(), 2, "V weight must be 2D");
        assert_eq!(v_b.rank(), 1, "V bias must be 1D");
        let (v_in, v_out) = v_weight.linear_layer_dims();
        assert_eq!(
            v_out,
            v_b.shape()[0],
            "V weight output dim must match bias size"
        );

        assert_eq!(output_weight.rank(), 2, "output weight must be 2D");
        assert_eq!(o_b.rank(), 1, "output bias must be 1D");
        let (o_in, o_out) = output_weight.linear_layer_dims();
        assert_eq!(
            o_out,
            o_b.shape()[0],
            "output weight output dim must match bias size"
        );

        let hidden_size = q_in;

        assert_eq!(
            k_in, hidden_size,
            "K weight input dim must match Q weight input dim"
        );
        assert_eq!(
            v_in, hidden_size,
            "V weight input dim must match Q weight input dim"
        );
        assert_eq!(
            o_in, q_out,
            "output projection input dim must match Q projection output dim"
        );
        assert_eq!(
            o_out, hidden_size,
            "output projection output dim must match hidden size"
        );

        Ok(Self {
            q_weight,
            q_bias: q_b,
            k_weight,
            k_bias: k_b,
            v_weight,
            v_bias: v_b,
            output_weight,
            output_bias: o_b,
        })
    }
}

pub struct GpuAttention {
    pub matmul: GpuMatMul,
    pub bmm: GpuBatchedMatMul,
    pub linear: GpuLinearLayer,
    pub add_bias: GpuAddBias,
    pub reshape: GpuReshape,
    pub unreshape: GpuUnreshape,
    pub apply_mask: GpuApplyMask,
    pub softmax: GpuSoftmax,
    pub permute: GpuPermute,
    pub repeat_kv: GpuRepeatKV,
    pub gpu_concatenate: GpuConcatenate,
    pub slice_kernel: GpuSlice,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub scale_factor: f32,
    pub head_dim: u32,
}

impl GpuAttention {
    pub fn new(
        context: &Arc<WgpuContext>,
        hidden_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        let scale_factor = 1.0 / (head_dim as f32).sqrt();

        Self {
            matmul: GpuMatMul::new(context),
            bmm: GpuBatchedMatMul::new(context),
            linear: GpuLinearLayer::new(context),
            add_bias: GpuAddBias::new(context),
            reshape: GpuReshape::new(context),
            unreshape: GpuUnreshape::new(context),
            apply_mask: GpuApplyMask::new(context),
            softmax: GpuSoftmax::new(context),
            permute: GpuPermute::new(context),
            repeat_kv: GpuRepeatKV::new(context),
            gpu_concatenate: GpuConcatenate::new(context),
            slice_kernel: GpuSlice::new(context),
            num_heads,
            num_kv_heads,
            scale_factor,
            head_dim,
        }
    }

    pub fn llama_attention(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_heads: &GpuTensor,
        k_heads: &GpuTensor,
        v_heads: &GpuTensor,
        attention_mask: &GpuTensor,
        position_offset: usize,
        pool: &mut GpuTensorPool,
        weights: &GpuAttentionWeights,
    ) -> GpuTensor {
        let (batch, _, _, _) = q_heads.dims4();
        let seq_k = k_heads.shape()[2];

        let (k_expanded, v_expanded) = if self.num_kv_heads != self.num_heads {
            let expanded_shape = vec![
                batch,
                self.num_heads as usize,
                seq_k,
                self.head_dim as usize,
            ];

            let k_repeated = pool.get(expanded_shape.clone());
            let v_repeated = pool.get(expanded_shape);

            self.repeat_kv.encode(encoder, k_heads, &k_repeated);
            self.repeat_kv.encode(encoder, v_heads, &v_repeated);

            (k_repeated, v_repeated)
        } else {
            (k_heads.clone(), v_heads.clone())
        };

        let k_transposed = k_expanded.permute(encoder, &self.permute, &[0, 1, 3, 2]);
        let scores = self.bmm_4d(encoder, q_heads, &k_transposed, pool);

        let logical_key_len = scores.shape()[2] + position_offset;
        self.apply_mask.encode(
            encoder,
            &scores,
            attention_mask,
            true,
            position_offset as u32,
            logical_key_len as u32,
        );

        self.softmax.encode(encoder, &scores, self.scale_factor);

        let context = self.bmm_4d(encoder, &scores, &v_expanded, pool);

        let context_merged = self.merge_heads(encoder, &context, pool);
        self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        )
    }

    pub fn forward_with_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor,
        key_value: Option<&GpuTensor>,
        weights: &GpuAttentionWeights,
        attention_mask: Option<&GpuTensor>,
        is_causal: bool,
        cached_kv: Option<(&GpuTensor, &GpuTensor)>,
        rope: Option<&GpuRoPE>,
        pool: &mut GpuTensorPool,
    ) -> (GpuTensor, GpuTensor, GpuTensor) {
        let kv_source = key_value.unwrap_or(query);
        let (batch_size, query_len, _) = query.dims3();

        let cache_len = cached_kv.map_or(0, |(k, _)| k.shape()[1]);

        let q_proj = self.project(encoder, query, &weights.q_weight, &weights.q_bias, pool);
        let k_proj = self.project(encoder, kv_source, &weights.k_weight, &weights.k_bias, pool);
        let v_proj = self.project(encoder, kv_source, &weights.v_weight, &weights.v_bias, pool);

        let (q_rotated, k_rotated) = if let Some(rope_encoder) = rope {
            let q_split = self.split_heads(encoder, &q_proj, pool);
            let k_split = self.split_heads(encoder, &k_proj, pool);

            let q_rope = pool.get(q_split.shape().to_vec());
            let k_rope = pool.get(k_split.shape().to_vec());

            rope_encoder.encode(encoder, &q_split, &q_rope, cache_len);
            rope_encoder.encode(encoder, &k_split, &k_rope, cache_len);

            let q_3d = self.merge_heads(encoder, &q_rope, pool);
            let k_3d = self.merge_heads(encoder, &k_rope, pool);
            (q_3d, k_3d)
        } else {
            (q_proj, k_proj)
        };

        let (full_k, full_v) = if let Some((cached_k, cached_v)) = cached_kv {
            let full_len = cache_len + query_len;
            let kv_hidden = k_rotated.shape()[2];

            let full_k_tensor = pool.get(vec![batch_size, full_len, kv_hidden]);
            let full_v_tensor = pool.get(vec![batch_size, full_len, kv_hidden]);

            self.gpu_concatenate
                .encode(encoder, &[cached_k, &k_rotated], &full_k_tensor, 1);
            self.gpu_concatenate
                .encode(encoder, &[cached_v, &v_proj], &full_v_tensor, 1);

            (full_k_tensor, full_v_tensor)
        } else {
            (k_rotated.clone(), v_proj.clone())
        };

        let q_heads = self.split_heads(encoder, &q_rotated, pool);
        let k_heads = self.split_heads(encoder, &full_k, pool);
        let v_heads = self.split_heads(encoder, &full_v, pool);

        let (k_expanded, v_expanded) = if self.num_kv_heads != self.num_heads {
            let expanded_shape = vec![
                batch_size,
                self.num_heads as usize,
                full_k.shape()[1],
                self.head_dim as usize,
            ];

            let k_repeated = pool.get(expanded_shape.clone());
            let v_repeated = pool.get(expanded_shape);

            self.repeat_kv.encode(encoder, &k_heads, &k_repeated);
            self.repeat_kv.encode(encoder, &v_heads, &v_repeated);

            (k_repeated, v_repeated)
        } else {
            (k_heads, v_heads)
        };

        let k_transposed = k_expanded.permute(encoder, &self.permute, &[0, 1, 3, 2]);
        let scores = self.bmm_4d(encoder, &q_heads, &k_transposed, pool);

        if is_causal || attention_mask.is_some() {
            let padding_mask = attention_mask.unwrap_or(&scores);
            let logical_key_len = scores.shape()[2] + cache_len;
            self.apply_mask.encode(
                encoder,
                &scores,
                padding_mask,
                is_causal,
                cache_len as u32,
                logical_key_len as u32,
            );
        }

        self.softmax.encode(encoder, &scores, self.scale_factor);

        let context = self.bmm_4d(encoder, &scores, &v_expanded, pool);

        let context_merged = self.merge_heads(encoder, &context, pool);
        let output = self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        );

        (output, k_rotated, v_proj)
    }

    pub fn attend_cached(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_heads: &GpuTensor,
        cache_k: &GpuTensor,
        cache_v: &GpuTensor,
        attention_mask: &GpuTensor,
        weights: &GpuAttentionWeights,
        position_offset: usize,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (batch, _, _, head_dim) = q_heads.dims4();
        let key_len = cache_k.shape()[2];

        let (k_expanded, v_expanded) = if self.num_kv_heads != self.num_heads {
            let expanded_shape = vec![batch, self.num_heads as usize, key_len, head_dim];
            let k_repeated = pool.get(expanded_shape.clone());
            let v_repeated = pool.get(expanded_shape);

            self.repeat_kv.encode(encoder, cache_k, &k_repeated);
            self.repeat_kv.encode(encoder, cache_v, &v_repeated);

            (k_repeated, v_repeated)
        } else {
            (cache_k.clone(), cache_v.clone())
        };

        let k_transposed = k_expanded.permute(encoder, &self.permute, &[0, 1, 3, 2]);
        let scores = self.bmm_4d(encoder, q_heads, &k_transposed, pool);

        let logical_key_len = scores.shape()[2] + position_offset;
        self.apply_mask.encode(
            encoder,
            &scores,
            attention_mask,
            true,
            position_offset as u32,
            logical_key_len as u32,
        );
        self.softmax.encode(encoder, &scores, self.scale_factor);

        let context = self.bmm_4d(encoder, &scores, &v_expanded, pool);
        let context_merged = self.merge_heads(encoder, &context, pool);

        self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        )
    }

    pub fn forward_cross(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor,
        key_value: &GpuTensor,
        weights: &GpuAttentionWeights,
        attention_mask: Option<&GpuTensor>,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let q_proj = self.project(encoder, query, &weights.q_weight, &weights.q_bias, pool);
        let k_proj = self.project(encoder, key_value, &weights.k_weight, &weights.k_bias, pool);
        let v_proj = self.project(encoder, key_value, &weights.v_weight, &weights.v_bias, pool);

        let q_heads = self.split_heads(encoder, &q_proj, pool);
        let k_heads = self.split_heads(encoder, &k_proj, pool);
        let v_heads = self.split_heads(encoder, &v_proj, pool);

        let k_transposed = k_heads.permute(encoder, &self.permute, &[0, 1, 3, 2]);
        let scores = self.bmm_4d(encoder, &q_heads, &k_transposed, pool);

        if let Some(mask) = attention_mask {
            let position_offset = 0u32;
            let logical_key_len = key_value.shape()[1] as u32;
            self.apply_padding_mask(
                encoder,
                &scores,
                mask,
                pool,
                position_offset,
                logical_key_len,
            );
        }

        self.softmax.encode(encoder, &scores, self.scale_factor);

        let context = self.bmm_4d(encoder, &scores, &v_heads, pool);

        let context_merged = self.merge_heads(encoder, &context, pool);
        self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        )
    }

    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor,
        key_value: Option<&GpuTensor>,
        weights: &GpuAttentionWeights,
        attention_mask: Option<&GpuTensor>,
        is_causal: bool,
        rope: Option<&GpuRoPE>,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (output, _, _) = self.forward_with_cache(
            encoder,
            query,
            key_value,
            weights,
            attention_mask,
            is_causal,
            None,
            rope,
            pool,
        );
        output
    }

    pub fn forward_seq2seq(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor,
        weights: &GpuAttentionWeights,
        attention_mask: &GpuTensor,
        cached_kv: Option<(&GpuTensor, &GpuTensor)>,
        cache_len: usize,
        pool: &mut GpuTensorPool,
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor)> {
        let (_, query_len, _hidden_size) = query.dims3();

        let q_proj = self.project(encoder, query, &weights.q_weight, &weights.q_bias, pool);
        let k_proj = self.project(encoder, query, &weights.k_weight, &weights.k_bias, pool);
        let v_proj = self.project(encoder, query, &weights.v_weight, &weights.v_bias, pool);

        let q_heads = self.split_heads(encoder, &q_proj, pool);
        let k_heads_new = self.split_heads(encoder, &k_proj, pool);
        let v_heads_new = self.split_heads(encoder, &v_proj, pool);

        let (full_k, full_v) = if let Some((cached_k, cached_v)) = cached_kv {
            if cache_len > 0 {
                let (b, h, _, d) = k_heads_new.dims4();
                let full_len = cache_len + query_len;

                let slice_shape = &[b, h, cache_len, d];
                let slice_offset = &[0, 0, 0, 0];
                let valid_cache_k =
                    cached_k.slice(encoder, &self.slice_kernel, slice_offset, slice_shape)?;
                let valid_cache_v =
                    cached_v.slice(encoder, &self.slice_kernel, slice_offset, slice_shape)?;

                let full_k_tensor = pool.get(vec![b, h, full_len, d]);
                let full_v_tensor = pool.get(vec![b, h, full_len, d]);

                self.gpu_concatenate.encode(
                    encoder,
                    &[&valid_cache_k, &k_heads_new],
                    &full_k_tensor,
                    2,
                );
                self.gpu_concatenate.encode(
                    encoder,
                    &[&valid_cache_v, &v_heads_new],
                    &full_v_tensor,
                    2,
                );

                (full_k_tensor, full_v_tensor)
            } else {
                (k_heads_new, v_heads_new)
            }
        } else {
            (k_heads_new, v_heads_new)
        };

        let k_transposed = full_k.permute(encoder, &self.permute, &[0, 1, 3, 2]);
        let scores = self.bmm_4d(encoder, &q_heads, &k_transposed, pool);

        let expected_scores_shape = vec![
            q_heads.shape()[0],
            q_heads.shape()[1],
            q_heads.shape()[2],
            k_transposed.shape()[3],
        ];
        assert_eq!(
            scores.shape(),
            &expected_scores_shape,
            "scores tensor has unexpected shape"
        );

        let logical_key_len = scores.shape()[2] + cache_len;
        self.apply_mask.encode(
            encoder,
            &scores,
            attention_mask,
            true,
            cache_len as u32,
            logical_key_len as u32,
        );

        self.softmax.encode(encoder, &scores, self.scale_factor);
        let context = self.bmm_4d(encoder, &scores, &full_v, pool);

        let context_merged = self.merge_heads(encoder, &context, pool);
        let output = self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        );

        Ok((output, k_proj, v_proj))
    }

    pub fn project_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        key_value: &GpuTensor,
        weights: &GpuAttentionWeights,
        position_offset: usize,
        pool: &mut GpuTensorPool,
        rope: Option<&GpuRoPE>,
    ) -> (GpuTensor, GpuTensor) {
        let new_k = self.project(encoder, key_value, &weights.k_weight, &weights.k_bias, pool);
        let new_v = self.project(encoder, key_value, &weights.v_weight, &weights.v_bias, pool);

        let new_k_rotated = if let Some(rope_encoder) = rope {
            let k_split = self.split_heads(encoder, &new_k, pool);
            let k_rotated_4d = pool.get(k_split.shape().to_vec());
            rope_encoder.encode(encoder, &k_split, &k_rotated_4d, position_offset);
            self.merge_heads(encoder, &k_rotated_4d, pool)
        } else {
            new_k
        };

        (new_k_rotated, new_v)
    }

    pub fn attend_with_physical_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor,
        physical_k: &GpuTensor,
        physical_v: &GpuTensor,
        attention_mask: &GpuTensor,
        weights: &GpuAttentionWeights,
        position_offset: usize,
        pool: &mut GpuTensorPool,
        rope: Option<&GpuRoPE>,
    ) -> GpuTensor {
        let (batch, _, _) = query.dims3();
        let (_, max_seq_len, head_dim) = physical_k.dims3();

        let q_proj = self.project(encoder, query, &weights.q_weight, &weights.q_bias, pool);
        let q_split = self.split_heads(encoder, &q_proj, pool);
        let q_for_attn = if let Some(r) = rope {
            let out = pool.get(q_split.shape().to_vec());
            // let _ = pool.get(q_split.shape().to_vec());
            r.encode(encoder, &q_split, &out, position_offset);
            out
        } else {
            q_split
        };

        let num_q_heads = self.num_heads as usize;
        let num_kv_heads = self.num_kv_heads as usize;

        let (k_for_attn, v_for_attn) = if num_q_heads != num_kv_heads {
            let physical_k_4d = physical_k.view(vec![batch, num_kv_heads, max_seq_len, head_dim]);
            let physical_v_4d = physical_v.view(vec![batch, num_kv_heads, max_seq_len, head_dim]);

            let k_repeated_4d = pool.get(vec![batch, num_q_heads, max_seq_len, head_dim]);
            let v_repeated_4d = pool.get(vec![batch, num_q_heads, max_seq_len, head_dim]);

            self.repeat_kv
                .encode(encoder, &physical_k_4d, &k_repeated_4d);
            self.repeat_kv
                .encode(encoder, &physical_v_4d, &v_repeated_4d);

            (k_repeated_4d, v_repeated_4d)
        } else {
            let physical_k_4d = physical_k.view(vec![batch, num_q_heads, max_seq_len, head_dim]);
            let physical_v_4d = physical_v.view(vec![batch, num_q_heads, max_seq_len, head_dim]);
            (physical_k_4d, physical_v_4d)
        };

        let k_t = k_for_attn.permute(encoder, &self.permute, &[0, 1, 3, 2]);
        let scores = self.bmm_4d(encoder, &q_for_attn, &k_t, pool);

        let logical_key_len = scores.shape()[2] + position_offset;
        self.apply_mask.encode(
            encoder,
            &scores,
            attention_mask,
            true,
            position_offset as u32,
            logical_key_len as u32,
        );

        self.softmax.encode(encoder, &scores, self.scale_factor);

        let context = self.bmm_4d(encoder, &scores, &v_for_attn, pool);

        let context_merged = self.merge_heads(encoder, &context, pool);
        self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        )
    }

    pub fn attend(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor,
        weights: &GpuAttentionWeights,
        attention_mask: &GpuTensor,
        is_causal: bool,
        kv_cache: (&GpuTensor, &GpuTensor),
        position_offset: usize,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (cache_k, cache_v) = kv_cache;

        let q_biased = self.project(encoder, query, &weights.q_weight, &weights.q_bias, pool);
        let q_split = self.split_heads(encoder, &q_biased, pool);

        let k_transposed = cache_k.permute(encoder, &self.permute, &[0, 1, 3, 2]);

        let scores = self.bmm_4d(encoder, &q_split, &k_transposed, pool);

        let logical_key_len = scores.shape()[2] + position_offset;
        self.apply_mask.encode(
            encoder,
            &scores,
            attention_mask,
            is_causal,
            position_offset as u32,
            logical_key_len as u32,
        );
        self.softmax.encode(encoder, &scores, self.scale_factor);
        let attention_weights = scores;

        let context = self.bmm_4d(encoder, &attention_weights, cache_v, pool);

        let context_merged = self.merge_heads(encoder, &context, pool);
        self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        )
    }

    pub fn project(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        weight: &GpuTensor,
        bias: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (b, s, h) = input.dims3();

        let (_, out_features) = weight.linear_layer_dims();

        let input_2d = input.view(vec![b * s, h]);
        let proj_2d = pool.get(vec![b * s, out_features]);

        self.linear.encode(encoder, &input_2d, weight, &proj_2d);

        let output_2d = if bias.shape()[0] > 0 {
            let biased_2d = pool.get(vec![b * s, out_features]);
            self.add_bias.encode(encoder, &[&proj_2d, bias], &biased_2d);
            biased_2d
        } else {
            proj_2d
        };

        output_2d.view(vec![b, s, out_features])
    }

    pub fn split_heads(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (b, s, total_dim) = input.dims3();

        let num_heads = if total_dim == (self.num_heads * self.head_dim) as usize {
            self.num_heads as usize
        } else {
            self.num_kv_heads as usize
        };

        let head_dim = total_dim / num_heads;
        let output = pool.get(vec![b, num_heads, s, head_dim]);

        self.reshape.encode(encoder, input, &output, false);
        output
    }

    pub fn merge_heads(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (b, h, s, d) = input.dims4();
        let output = pool.get(vec![b, s, h * d]);
        self.unreshape.encode(encoder, input, &output);
        output
    }

    pub fn bmm_4d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuTensor,
        b: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (b_size, h_size, m, k1) = a.dims4();
        let (_, _, k2, n) = b.dims4();
        assert_eq!(k1, k2, "matrix dimensions incompatible for BMM");

        let a_3d = a.view(vec![b_size * h_size, m, k1]);
        let b_3d = b.view(vec![b_size * h_size, k1, n]);

        let c_3d = pool.get(vec![b_size * h_size, m, n]);
        self.bmm.encode(encoder, &[&a_3d, &b_3d], &c_3d);

        c_3d.view(vec![b_size, h_size, m, n])
    }

    fn apply_padding_mask(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scores: &GpuTensor,
        mask: &GpuTensor,
        pool: &mut GpuTensorPool,
        position_offset: u32,
        logical_key_len: u32,
    ) {
        self.apply_mask.encode(
            encoder,
            scores,
            mask,
            false,
            position_offset,
            logical_key_len,
        );
    }

    /// Pre-computes K and V from encoder states, with K already transposed for BMM.
    pub fn precompute_cross_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        key_value: &GpuTensor,
        weights: &GpuAttentionWeights,
        pool: &mut GpuTensorPool,
    ) -> (GpuTensor, GpuTensor) {
        let k_proj = self.project(encoder, key_value, &weights.k_weight, &weights.k_bias, pool);
        let v_proj = self.project(encoder, key_value, &weights.v_weight, &weights.v_bias, pool);

        let k_heads = self.split_heads(encoder, &k_proj, pool);
        let v_heads = self.split_heads(encoder, &v_proj, pool);

        let k_optimized = k_heads.permute(encoder, &self.permute, &[0, 1, 3, 2]);

        (k_optimized, v_heads)
    }

    /// Forward pass using pre-computed K/V (skips K/V projection and K permutation).
    pub fn forward_cross_precomputed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor,
        precomputed_k: &GpuTensor,
        precomputed_v: &GpuTensor,
        weights: &GpuAttentionWeights,
        attention_mask: Option<&GpuTensor>,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let q_proj = self.project(encoder, query, &weights.q_weight, &weights.q_bias, pool);

        let q_heads = self.split_heads(encoder, &q_proj, pool);

        let scores = self.bmm_4d(encoder, &q_heads, precomputed_k, pool);

        if let Some(mask) = attention_mask {
            let position_offset = 0;
            let logical_key_len = precomputed_k.shape()[3] as u32;
            self.apply_padding_mask(
                encoder,
                &scores,
                mask,
                pool,
                position_offset,
                logical_key_len,
            );
        }

        self.softmax.encode(encoder, &scores, self.scale_factor);

        let context = self.bmm_4d(encoder, &scores, precomputed_v, pool);

        let context_merged = self.merge_heads(encoder, &context, pool);
        self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        )
    }
}

#[cfg(test)]
mod tests;
