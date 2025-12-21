// Import Zip
use crate::linear_layer::LinearLayer;
use crate::utils::linear_algebra::{matmul_4d, matmul_4d_context, matmul_4d_decode};
use anyhow::Result;
use ndarray::{s, Array2, Array3, Array4, Axis, Zip};
use rayon::prelude::*;
use crate::utils::masks::apply_causal_mask;

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
//  1. Decoder Self-Attention
// ============================================================================

pub struct DecoderSelfAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,

    pub num_heads: usize,
    pub head_dim: usize,
    pub scale_factor: f32,
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
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        let (batch, seq_len, _) = hidden_states.dim();
        let hidden_size = self.num_heads * self.head_dim;

        // 1. Flatten & Project
        let hidden_2d = hidden_states
            .view()
            .into_shape((batch * seq_len, hidden_size))?;
        let q = self.q_proj.matmul(&hidden_2d);
        let k = self.k_proj.matmul(&hidden_2d);
        let v = self.v_proj.matmul(&hidden_2d);

        let q_heads = q
            .into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .to_owned();

        // Keep K/V in 3D [Batch, Seq, Hidden] for caching
        let k_3d = k.into_shape_with_order((batch, seq_len, hidden_size))?;
        let v_3d = v.into_shape_with_order((batch, seq_len, hidden_size))?;

        // 2. Update Cache
        let (full_k, full_v) = if let Some((cached_k, cached_v)) = past_kv {
            // Force standard layout after concatenation to ensure reshape safety
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

        // 3. Prepare for Attention
        let total_seq = full_k.shape()[1];

        let k_heads = full_k
            .view()
            .into_shape_with_order((batch, total_seq, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        // Transpose K for matmul [Batch, Heads, Dim, TotalSeq]
        let k_t = k_heads.permuted_axes([0, 1, 3, 2]).to_owned();

        // 4. Scores
        let mut scores = if seq_len == 1 {
            matmul_4d_decode(&q_heads, &k_t)
        } else {
            matmul_4d(&q_heads, &k_t)
        };
        scores.mapv_inplace(|x| x * self.scale_factor);

        if let Some(mask) = attention_mask {
            scores = apply_attention_mask(scores, mask);
        }

        if seq_len > 1 {
            let cache_len = past_kv.map_or(0, |(k, _)| k.shape()[1]);
            apply_causal_mask(&mut scores, cache_len);
        }

        self.softmax_inplace(&mut scores);

        // 5. Context
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

        // 6. Output Projection
        let context_flat = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((batch * seq_len, hidden_size))?
            .to_owned();

        let output = self.o_proj.matmul(&context_flat.view());
        let output_3d = output.into_shape_with_order((batch, seq_len, hidden_size))?;

        Ok((output_3d, k_3d, v_3d))
    }

    fn softmax_inplace(&self, x: &mut Array4<f32>) {
        for mut batch in x.outer_iter_mut() {
            for mut head in batch.outer_iter_mut() {
                for mut row in head.outer_iter_mut() {
                    // Use MASK_VALUE as baseline to avoid NaN if row is all masked
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
