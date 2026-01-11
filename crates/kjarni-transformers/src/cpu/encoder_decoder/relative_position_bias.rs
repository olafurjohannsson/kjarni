use crate::weights::ModelWeights;
use anyhow::{Result, anyhow};
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
        let mut num_buckets = self.num_buckets as i32; // Local mutable copy
        let max_distance = self.max_distance as i32;

        let mut n = -relative_position; 
        let mut ret = 0;

        if self.is_bidirectional {
            num_buckets /= 2; // FIX: Reduce buckets for half-space
            if n < 0 {
                ret += num_buckets; // Add offset (e.g. +16)
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

#[cfg(test)]
mod t5_bias_tests {
    use super::*;
    use crate::cpu::encoder_decoder::relative_position_bias::T5RelativePositionBias;
    use crate::weights::ModelWeights;
    use anyhow::Result;
    use ndarray::{Array2, Array4};
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use tempfile::NamedTempFile;

    // Reuse the helper from previous tests
    fn create_model_weights(
        weights_map: HashMap<String, Vec<f32>>,
        shapes: HashMap<String, Vec<usize>>,
    ) -> Result<(ModelWeights, NamedTempFile)> {
        let file = NamedTempFile::new()?;
        let stored_data: Vec<(String, Vec<usize>, Vec<u8>)> = weights_map
            .into_iter()
            .map(|(k, v)| {
                let shape = shapes.get(&k).unwrap().clone();
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                (k, shape, bytes)
            })
            .collect();

        let mut tensors = HashMap::new();
        for (k, shape, bytes) in &stored_data {
            tensors.insert(
                k.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)?,
            );
        }
        safetensors::serialize_to_file(&tensors, &None, file.path())?;
        let weights = ModelWeights::from_file(file.path())?;
        Ok((weights, file))
    }

    // ==========================================
    // GOLDEN VALUES
    // ==========================================

    // bias_table Shape: [32, 2]
    fn get_bias_table_data() -> Vec<f32> {
        vec![
            0.000000, 0.010000, 0.020000, 0.030000, 0.040000, 0.050000, 0.060000, 0.070000,
            0.080000, 0.090000, 0.100000, 0.110000, 0.120000, 0.130000, 0.140000, 0.150000,
            0.160000, 0.170000, 0.180000, 0.190000, 0.200000, 0.210000, 0.220000, 0.230000,
            0.240000, 0.250000, 0.260000, 0.270000, 0.280000, 0.290000, 0.300000, 0.310000,
            0.320000, 0.330000, 0.340000, 0.350000, 0.360000, 0.370000, 0.380000, 0.390000,
            0.400000, 0.410000, 0.420000, 0.430000, 0.440000, 0.450000, 0.460000, 0.470000,
            0.480000, 0.490000, 0.500000, 0.510000, 0.520000, 0.530000, 0.540000, 0.550000,
            0.560000, 0.570000, 0.580000, 0.590000, 0.600000, 0.610000, 0.620000, 0.630000,
        ]
    }

    // buckets_short Shape: [4, 4]
    // Indices generated by Python
    fn get_buckets_short() -> Vec<usize> {
        vec![0, 17, 18, 19, 1, 0, 17, 18, 2, 1, 0, 17, 3, 2, 1, 0]
    }

    // bias_short Shape: [1, 2, 4, 4]
    fn get_bias_short_golden() -> Vec<f32> {
        vec![
            // Head 0 (Even indices of table)
            0.000000, 0.340000, 0.360000, 0.380000, 0.020000, 0.000000, 0.340000, 0.360000,
            0.040000, 0.020000, 0.000000, 0.340000, 0.060000, 0.040000, 0.020000, 0.000000,
            // Head 1 (Odd indices of table)
            0.010000, 0.350000, 0.370000, 0.390000, 0.030000, 0.010000, 0.350000, 0.370000,
            0.050000, 0.030000, 0.010000, 0.350000, 0.070000, 0.050000, 0.030000, 0.010000,
        ]
    }

    // buckets_long Shape: [1, 20]
    fn get_buckets_long() -> Vec<usize> {
        vec![
            0, 17, 18, 19, 20, 21, 22, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26,
        ]
    }

    // buckets_offset Shape: [1, 6]
    fn get_buckets_offset() -> Vec<usize> {
        vec![5, 4, 3, 2, 1, 0]
    }

    // ==========================================
    // TESTS
    // ==========================================

    #[test]
    fn test_weight_loading_prefixes() -> Result<()> {
        let mut weights_map = HashMap::new();
        let mut shapes = HashMap::new();

        let data = get_bias_table_data();
        let shape = vec![32, 2];

        // Case 1: Standard HF T5
        weights_map.insert(
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
            data.clone(),
        );
        shapes.insert(
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
            shape.clone(),
        );

        let (mw, _tmp) = create_model_weights(weights_map, shapes)?;

        let bias = T5RelativePositionBias::new(&mw, "encoder", true, 32, 128);
        assert!(bias.is_ok(), "Failed to load standard HF prefix");

        Ok(())
    }

    #[test]
    fn test_bucket_logic_short() -> Result<()> {
        let buckets = 32;
        let max_dist = 128;

        let mut weights_map = HashMap::new();
        let mut shapes = HashMap::new();
        weights_map.insert(
            "test.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
            get_bias_table_data(),
        );
        shapes.insert(
            "test.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
            vec![32, 2],
        );
        let (mw, _tmp) = create_model_weights(weights_map, shapes)?;

        let bias_model = T5RelativePositionBias::new(&mw, "test", true, buckets, max_dist)?;

        // Compute 4x4
        let result = bias_model.compute(4, 4)?; // [1, 2, 4, 4]
        let golden_flat = get_bias_short_golden();
        let golden = Array4::from_shape_vec((1, 2, 4, 4), golden_flat)?;

        let diff = (&result - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Short Sequence Max Diff: {}", max_diff);
        assert!(max_diff < 1e-5);

        Ok(())
    }

    #[test]
    fn test_bucket_logic_long() -> Result<()> {
        // Test logarithmic binning behavior via indices verification
        let buckets = 32;
        let max_dist = 128;

        // Debug Table: value = index (0.0, 0.0, 1.0, 1.0...)
        // Head 0 and Head 1 will both return the bucket index.
        let mut weights_map = HashMap::new();
        let mut shapes = HashMap::new();

        let debug_table: Vec<f32> = (0..32 * 2).map(|i| (i / 2) as f32).collect();

        weights_map.insert(
            "test.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
            debug_table,
        );
        shapes.insert(
            "test.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
            vec![32, 2],
        );
        let (mw, _tmp) = create_model_weights(weights_map, shapes)?;

        let bias_model = T5RelativePositionBias::new(&mw, "test", true, buckets, max_dist)?;

        // Compute 1x20 (Row 0)
        let result = bias_model.compute(1, 20)?; // [1, 2, 1, 20]

        let expected_buckets = get_buckets_long();

        for (k, &expected_bucket) in expected_buckets.iter().enumerate() {
            // Head 0, Q=0, K=k
            let actual_val = result[[0, 0, 0, k]];
            assert_eq!(actual_val as usize, expected_bucket, "Mismatch at K={}", k);
        }

        Ok(())
    }

    #[test]
    fn test_compute_with_offset() -> Result<()> {
        let buckets = 32;
        let max_dist = 128;

        let mut weights_map = HashMap::new();
        let mut shapes = HashMap::new();
        let debug_table: Vec<f32> = (0..32 * 2).map(|i| (i / 2) as f32).collect();

        weights_map.insert(
            "test.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
            debug_table,
        );
        shapes.insert(
            "test.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
            vec![32, 2],
        );
        let (mw, _tmp) = create_model_weights(weights_map, shapes)?;

        let bias_model = T5RelativePositionBias::new(&mw, "test", true, buckets, max_dist)?;

        // Scenario: Decoding 6th token (index 5)
        // Cache History: 0..5 (length 6)
        // Query is just index 5 (length 1), but with offset 5 (so abs pos 5)
        // Key is indices 0..5 (abs pos 0..5)
        let result = bias_model.compute_with_offset(1, 6, 5)?; // [1, 2, 1, 6]

        let expected_buckets = get_buckets_offset(); // [5, 4, 3, 2, 1, 0]

        for (k, &expected_bucket) in expected_buckets.iter().enumerate() {
            let actual_val = result[[0, 0, 0, k]];
            assert_eq!(
                actual_val as usize, expected_bucket,
                "Offset Mismatch at K={}",
                k
            );
        }

        Ok(())
    }
}
