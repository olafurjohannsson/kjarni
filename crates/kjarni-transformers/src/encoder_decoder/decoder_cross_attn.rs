// Import Zip
use crate::linear_layer::LinearLayer;
use crate::utils::linear_algebra::{matmul_4d, matmul_4d_context, matmul_4d_decode};
use anyhow::Result;
use ndarray::{Array2, Array3, Array4, Axis, Zip};
use rayon::prelude::*;

// Standard large negative value for masking (avoids NaN in softmax)
const MASK_VALUE: f32 = -1e9;

// ============================================================================
//  Shared Utilities
// ============================================================================

/// Safe implementation of masking using NDArray Zip (handles memory strides correctly)
/// This logic matches the "Old Code" apply_padding_mask exactly.
pub fn apply_attention_mask(mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
    let (batch, heads, seq_q, seq_k) = scores.dim();

    // FIX: Only check Sequence Length match.
    // We allow Batch size mismatch to support Beam Search broadcasting (e.g., Mask [1, S] -> Scores [4, S])
    if mask.shape()[1] != seq_k {
        return scores;
    }

    // Expand mask: [MaskBatch, SeqK] → [MaskBatch, 1, 1, SeqK]
    let mask_expanded = mask.view().insert_axis(Axis(1)).insert_axis(Axis(1));

    // Broadcast and apply.
    // Zip handles non-contiguous memory layouts and broadcasting automatically.
    if let Some(broadcast_mask) = mask_expanded.broadcast((batch, heads, seq_q, seq_k)) {
        Zip::from(&mut scores)
            .and(&broadcast_mask)
            .par_for_each(|s, &m| {
                // Assuming mask 0.0 == Padding (Masked out)
                if m == 0.0 {
                    *s = MASK_VALUE;
                }
            });
    }

    scores
}

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
        }
    }

    /// Pre-computes K and V from the encoder output.
    pub fn precompute_encoder_kv(
        &self,
        encoder_hidden_states: &Array3<f32>,
    ) -> Result<(Array4<f32>, Array4<f32>)> {
        let (batch, seq_len, _) = encoder_hidden_states.dim();
        let enc_2d = encoder_hidden_states
            .view()
            .into_shape((batch * seq_len, self.num_heads * self.head_dim))?;

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
        scores.mapv_inplace(|x| x * self.scale_factor);

        // Apply Padding Mask (Using safe Zip broadcasting)
        if let Some(mask) = attention_mask {
            scores = apply_attention_mask(scores, mask);
        }

        self.softmax_inplace(&mut scores);

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

    fn softmax_inplace(&self, x: &mut Array4<f32>) {
        for mut batch in x.outer_iter_mut() {
            for mut head in batch.outer_iter_mut() {
                for mut row in head.outer_iter_mut() {
                    let max = row.fold(MASK_VALUE, |a, &b| a.max(b));
                    let mut sum = 0.0;
                    for v in row.iter_mut() {
                        *v = (*v - max).exp();
                        sum += *v;
                    }
                    if sum > 0.0 {
                        for v in row.iter_mut() {
                            *v /= sum;
                        }
                    }
                }
            }
        }
    }
}
