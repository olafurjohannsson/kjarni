use crate::weights::ModelWeights;
use anyhow::{anyhow, Result};
use ndarray::{Array2, Array4};

pub struct T5RelativePositionBias {
    pub bias_table: Array2<f32>, // Shape: [num_buckets, num_heads]
    pub num_buckets: usize,
    pub max_distance: usize,
    pub is_bidirectional: bool,
}

impl T5RelativePositionBias {
    pub fn new(
        weights: &ModelWeights,
        prefix: &str,
        is_bidirectional: bool,
        num_buckets: usize,
        max_distance: usize,
    ) -> Result<Self> {
        // T5 weights are typically [buckets, heads]
        // We check all common naming conventions for T5 (Safetensors/HF style)
        let bias_table = weights
            .get_array2(&format!(
                "{}.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                prefix
            ))
            .or_else(|_| {
                weights.get_array2(&format!(
                    "{}.layer.0.SelfAttention.relative_attention_bias.weight",
                    prefix
                ))
            })
            .or_else(|_| {
                weights.get_array2(&format!(
                    "{}.layers.0.self_attn.relative_attention_bias.weight",
                    prefix
                ))
            })
            .map_err(|_| {
                anyhow!(
                    "Could not find T5 relative attention bias weights for prefix: {}",
                    prefix
                )
            })?;

        Ok(Self {
            bias_table,
            num_buckets,
            max_distance,
            is_bidirectional,
        })
    }

    pub fn compute(&self, query_len: usize, key_len: usize) -> Result<Array4<f32>> {
        let num_heads = self.bias_table.ncols();
        let mut bias = Array4::<f32>::zeros((1, num_heads, query_len, key_len));

        for q in 0..query_len {
            for k in 0..key_len {
                // T5 relative position: k - q
                let rel_pos = k as i32 - q as i32;
                let bucket = self.relative_position_bucket(rel_pos);

                for h in 0..num_heads {
                    // Indexing into [bucket, head]
                    bias[[0, h, q, k]] = self.bias_table[[bucket, h]];
                }
            }
        }

        Ok(bias)
    }
    pub fn compute_with_offset(
        &self,
        query_len: usize,
        key_len: usize,
        query_offset: usize,
    ) -> Result<Array4<f32>> {
        let num_heads = self.bias_table.ncols();
        let mut bias = Array4::<f32>::zeros((1, num_heads, query_len, key_len));

        for q in 0..query_len {
            for k in 0..key_len {
                // Absolute position of query token 'q' is q + offset
                let abs_q = q + query_offset;
                let abs_k = k;

                // rel_pos = key_pos - query_pos
                let rel_pos = abs_k as i32 - abs_q as i32;
                let bucket = self.relative_position_bucket(rel_pos);

                for h in 0..num_heads {
                    bias[[0, h, q, k]] = self.bias_table[[bucket, h]];
                }
            }
        }
        Ok(bias)
    }
    fn relative_position_bucket(&self, relative_position: i32) -> usize {
        let num_buckets = self.num_buckets as i32;
        let max_distance = self.max_distance as i32;

        // T5 Logic from Mesh Tensorflow / HuggingFace
        // Standard relative position is (k - q), but bucket logic often expects (q - k)
        // Let's stick to the canonical implementation:

        let mut n = -relative_position; // Convert (k-q) to (q-k)
        let mut ret = 0;

        if self.is_bidirectional {
            let num_buckets_half = num_buckets / 2;
            // In T5, "Future" is usually the first half, "Past" is the second half
            // If n > 0 (Query > Key, i.e., Past), we add the offset
            if n > 0 {
                ret += num_buckets_half;
            }
            n = n.abs();
        } else {
            n = (-n).max(0);
        }

        let max_exact = num_buckets / 2;
        let is_small = n < max_exact;

        let val = if is_small {
            n
        } else {
            // T5's specific log-bucket formula
            // val = max_exact + (log(n / max_exact) / log(max_distance / max_exact) * (num_buckets - max_exact))
            let n_float = n as f32;
            let max_exact_float = max_exact as f32;
            let max_dist_float = max_distance as f32;

            let log_ratio = (n_float / max_exact_float).ln() / (max_dist_float / max_exact_float).ln();
            let bucket = max_exact_float + (log_ratio * (num_buckets - max_exact) as f32);

            bucket.min((num_buckets - 1) as f32) as i32
        };

        (ret + val) as usize
    }
}
