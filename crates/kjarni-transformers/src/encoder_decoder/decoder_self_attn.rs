//! Causal self-attention for decoder models.

use anyhow::Result;
use ndarray::{Array2, Array3, Array4, ArrayView3, Axis};

use crate::activations::softmax_4d_inplace;
use crate::linear_layer::LinearLayer;
use crate::utils::linear_algebra::{
    apply_attention_mask, matmul_4d, matmul_4d_context, matmul_4d_decode,
};
use crate::utils::masks::apply_causal_mask;

pub struct DecoderSelfAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale_factor: f32,
    pub scale_qk: bool,
}

impl DecoderSelfAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        q: LinearLayer,
        k: LinearLayer,
        v: LinearLayer,
        o: LinearLayer,
    ) -> Self {
        let head_dim = hidden_size / num_heads;

        Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            num_heads,
            head_dim,
            scale_factor: 1.0 / (head_dim as f32).sqrt(),
            scale_qk: true,
        }
    }

    pub fn with_no_qk_scaling(mut self) -> Self {
        self.scale_qk = false;
        self
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
        past_kv: Option<(ArrayView3<f32>, ArrayView3<f32>)>,
        position_bias: Option<&Array4<f32>>,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        let (batch, seq_len, _) = hidden_states.dim();
        let hidden_size = self.num_heads * self.head_dim;

        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((batch * seq_len, hidden_size))?;

        let q = self.q_proj.matmul(&hidden_2d);
        let k = self.k_proj.matmul(&hidden_2d);
        let v = self.v_proj.matmul(&hidden_2d);

        let q_heads = q
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .to_owned();

        let k_3d = k.into_shape_with_order((batch, seq_len, hidden_size))?;
        let v_3d = v.into_shape_with_order((batch, seq_len, hidden_size))?;

        let (full_k, full_v) = if let Some((cached_k, cached_v)) = past_kv {
            let full_k = ndarray::concatenate![Axis(1), cached_k, k_3d.view()]
                .as_standard_layout()
                .to_owned();
            let full_v = ndarray::concatenate![Axis(1), cached_v, v_3d.view()]
                .as_standard_layout()
                .to_owned();
            (full_k, full_v)
        } else {
            (k_3d.clone(), v_3d.clone())
        };

        let total_seq = full_k.shape()[1];

        let k_heads = full_k
            .view()
            .into_shape_with_order((batch, total_seq, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let k_t = k_heads.permuted_axes([0, 1, 3, 2]).to_owned();

        let mut scores = if seq_len == 1 {
            matmul_4d_decode(&q_heads, &k_t)
        } else {
            matmul_4d(&q_heads, &k_t)
        };

        if self.scale_qk {
            scores.mapv_inplace(|x| x * self.scale_factor);
        }

        if let Some(bias) = position_bias {
            scores += bias;
        }

        if let Some(mask) = attention_mask {
            scores = apply_attention_mask(scores, mask);
        }

        if seq_len > 1 {
            let cache_len = past_kv.map_or(0, |(k, _)| k.shape()[1]);
            apply_causal_mask(&mut scores, cache_len);
        }

        softmax_4d_inplace(&mut scores);

        let v_heads = full_v
            .view()
            .into_shape_with_order((batch, total_seq, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .to_owned();

        let context = if seq_len == 1 {
            matmul_4d_context(&scores, &v_heads)
        } else {
            matmul_4d(&scores, &v_heads)
        };

        let context_flat = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((batch * seq_len, hidden_size))?
            .to_owned();

        let output = self.o_proj.matmul(&context_flat.view());
        let output_3d = output.into_shape_with_order((batch, seq_len, hidden_size))?;

        Ok((output_3d, k_3d, v_3d))
    }
}