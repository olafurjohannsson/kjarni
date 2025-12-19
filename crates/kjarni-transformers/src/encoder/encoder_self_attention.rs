use crate::linear_layer_old::LinearLayer;
use crate::utils::linear_algebra::matmul_4d;
use anyhow::Result;
use ndarray::{Array2, Array3, Array4, Axis, Zip};
use rayon::prelude::*;

const MASK_VALUE: f32 = -1e9;

pub struct EncoderSelfAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub out_proj: LinearLayer,

    pub num_heads: usize,
    pub head_dim: usize,
    pub scale_factor: f32,
    pub scale_qk: bool,  // T5 doesn't scale by sqrt(d)
}

impl EncoderSelfAttention {
    /// Pure constructor: Takes pre-loaded layers.
    /// Matches DecoderSelfAttention signature.
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
            out_proj: o,
            num_heads,
            head_dim,
            scale_factor: 1.0 / (head_dim as f32).sqrt(),
            scale_qk: true,
        }
    }
    // Builder method for T5-style attention (no scaling)
    pub fn with_no_qk_scaling(mut self) -> Self {
        self.scale_qk = false;
        self
    }

    /// Forward pass with optional position bias
    /// 
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch, seq, hidden]
    /// * `attention_mask` - Padding mask [batch, seq], 0 = masked
    /// * `position_bias` - Optional relative position bias [1, heads, seq_q, seq_k] (T5/ALiBi)
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_bias: Option<&Array4<f32>>,
    ) -> Result<Array3<f32>> {
        let (batch, seq_len, _) = hidden_states.dim();
        let hidden_dim = self.num_heads * self.head_dim;

        // 1. Flatten & Project
        let hidden_2d = hidden_states.view().into_shape_with_order((batch * seq_len, hidden_dim))?;
        
        let q = self.q_proj.matmul(&hidden_2d);
        let k = self.k_proj.matmul(&hidden_2d);
        let v = self.v_proj.matmul(&hidden_2d);

        // 2. Reshape & Permute
        let q_heads = q.into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]).to_owned();
        
        // K transposed for efficient Q@K
        let k_heads_t = k.into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 3, 1]).to_owned();

        let v_heads = v.into_shape_with_order((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]).to_owned();

        // 3. Scores
        let mut scores = matmul_4d(&q_heads, &k_heads_t);
        
    
        // 4. Scale (BERT/RoBERTa/BART) or not (T5)
        if self.scale_qk {
            scores.mapv_inplace(|x| x * self.scale_factor);
        }

        // 5. Add position bias if provided (T5/ALiBi)
        if let Some(bias) = position_bias {
            // bias is [1, heads, seq_q, seq_k], broadcasts to [batch, heads, seq_q, seq_k]
            scores = scores + bias;
        }

        // 6. Mask
        scores = apply_attention_mask(scores, attention_mask);

        // 7. Softmax
        self.softmax_inplace(&mut scores);

        // 8. Context
        let context = matmul_4d(&scores, &v_heads);

        // 9. Output
        let context_flat = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((batch * seq_len, hidden_dim))?
            .to_owned();

        let output = self.out_proj.matmul(&context_flat.view());

        Ok(output.into_shape_with_order((batch, seq_len, hidden_dim))?)
    }

    pub fn softmax_inplace(&self, x: &mut Array4<f32>) {
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
                        for v in row.iter_mut() { *v /= sum; }
                    }
                }
            }
        }
    }
}

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