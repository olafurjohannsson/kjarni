// Import Zip
use crate::linear_layer::LinearLayer;
use crate::utils::linear_algebra::{apply_attention_mask, softmax_inplace};
use crate::utils::linear_algebra::{matmul_4d, matmul_4d_context, matmul_4d_decode};
use anyhow::Result;
use ndarray::{Array2, Array3, Array4};

// Standard large negative value for masking (avoids NaN in softmax)
const MASK_VALUE: f32 = -1e9;


// ============================================================================
//  2. Decoder Cross-Attention
// ============================================================================

pub struct DecoderCrossAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,

    pub num_heads: usize,
    pub head_dim: usize,
    pub scale_factor: f32,
    pub scale_qk: bool,
}

impl DecoderCrossAttention {
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
    /// Pre-computes K and V from the encoder output.
    pub fn precompute_encoder_kv(
        &self,
        encoder_hidden_states: &Array3<f32>,
    ) -> Result<(Array4<f32>, Array4<f32>)> {
        let (batch, seq_len, _) = encoder_hidden_states.dim();
        let enc_2d = encoder_hidden_states
            .view()
            .into_shape_with_order((batch * seq_len, self.num_heads * self.head_dim))?;

        let k = self.k_proj.matmul(&enc_2d);
        let v = self.v_proj.matmul(&enc_2d);

        // K: Transpose to [B, H, D, S] for efficient MatMul with Q
        let k_heads_t = k
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 3, 1])
            .as_standard_layout()
            .to_owned();

        // V: Keep as [B, H, S, D] for context computation
        let v_heads = v
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .to_owned();

        Ok((k_heads_t, v_heads))
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        encoder_k_t: &Array4<f32>,
        encoder_v: &Array4<f32>,
        attention_mask: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        let (batch, seq_len, _) = hidden_states.dim();

        // Project Q from Decoder Hidden States
        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((batch * seq_len, self.num_heads * self.head_dim))?;
        let q = self.q_proj.matmul(&hidden_2d);

        let q_heads = q
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .to_owned();

        // Scores
        let mut scores = if seq_len == 1 {
            matmul_4d_decode(&q_heads, encoder_k_t)
        } else {
            matmul_4d(&q_heads, encoder_k_t)
        };
        if self.scale_qk {
            scores.mapv_inplace(|x| x * self.scale_factor);
        }
        // Apply Padding Mask (Using safe Zip broadcasting)
        if let Some(mask) = attention_mask {
            scores = apply_attention_mask(scores, mask);
        }

        softmax_inplace(&mut scores);

        // Context
        let context = if seq_len == 1 {
            matmul_4d_context(&scores, encoder_v)
        } else {
            matmul_4d(&scores, encoder_v)
        };

        // Output
        let context_flat = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((batch * seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        let output = self.o_proj.matmul(&context_flat.view());
        let output_3d = output.into_shape_with_order((batch, seq_len, self.num_heads * self.head_dim))?;

        Ok(output_3d)
    }
}
