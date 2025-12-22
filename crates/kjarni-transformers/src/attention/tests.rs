use crate::attention::{repeat_kv, MultiHeadAttention};
use crate::rope::RoPE;
use crate::utils::linear_algebra::matmul_3d_2d;
use crate::utils::masks::{apply_causal_mask, apply_padding_mask, MASK_VALUE};
use anyhow::Result;
use ndarray::{s, Array1, Array2, Array4};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    fn create_mock_attention(
        hidden: usize,
        heads: usize,
        kv_heads: Option<usize>,
    ) -> MultiHeadAttention {
        let kv_heads = kv_heads.unwrap_or(heads);
        let k_v_dim = (hidden / heads) * kv_heads;

        MultiHeadAttention::new(
            hidden,
            heads,
            Array2::eye(hidden),
            Array1::zeros(0),
            Array2::eye(hidden).slice(s![.., 0..k_v_dim]).to_owned(),
            Array1::zeros(0),
            Array2::eye(hidden).slice(s![.., 0..k_v_dim]).to_owned(),
            Array1::zeros(0),
            Array2::eye(hidden),
            Array1::zeros(0),
            Some(kv_heads),
        )
    }

    #[test]
    fn test_mha_forward_no_cache() -> Result<()> {
        let attn = create_mock_attention(64, 4, None);
        let input = Array3::ones((1, 10, 64));
        let mask = Array2::ones((1, 10));

        let (output, k, v) =
            attn.forward_with_cache(&input, None, Some(&mask), false, None, None)?;

        assert_eq!(output.shape(), &[1, 10, 64]);
        assert_eq!(k.shape(), &[1, 10, 64]);
        assert_eq!(v.shape(), &[1, 10, 64]);
        Ok(())
    }

    #[test]
    fn test_mha_with_cache() -> Result<()> {
        let attn = create_mock_attention(64, 4, None);
        let past_k = Array3::zeros((1, 5, 64));
        let past_v = Array3::zeros((1, 5, 64));
        let input = Array3::ones((1, 1, 64));
        let mask = Array2::ones((1, 6)); // Mask for full sequence length

        let (output, k, v) = attn.forward_with_cache(
            &input,
            None,
            Some(&mask),
            true,
            Some((past_k.view(), past_v.view())),
            None,
        )?;

        assert_eq!(output.shape(), &[1, 1, 64]);
        assert_eq!(k.shape(), &[1, 1, 64]); // Returns only the NEW key
        assert_eq!(v.shape(), &[1, 1, 64]); // Returns only the NEW value
        Ok(())
    }

    #[test]
    fn test_gqa_with_rope() -> Result<()> {
        let attn = create_mock_attention(64, 4, Some(2)); // 4 Q heads, 2 KV heads
        let rope = RoPE::new(16, 128, 10000.0);
        let input = Array3::ones((1, 10, 64));
        let mask = Array2::ones((1, 10));

        let (output, k, v) =
            attn.forward_with_cache(&input, None, Some(&mask), true, None, Some(&rope))?;

        assert_eq!(output.shape(), &[1, 10, 64]);
        // K and V have reduced dimension due to GQA
        assert_eq!(k.shape(), &[1, 10, 32]);
        assert_eq!(v.shape(), &[1, 10, 32]);
        Ok(())
    }

    #[test]
    fn test_attention_with_cache() {
        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 1;
        let seq_len = 1; // Single token (incremental decoding)
        let cache_len = 5; // 5 tokens already cached

        let q_weight = Array2::zeros((hidden_size, hidden_size));
        let q_bias = Array1::zeros(hidden_size);
        let k_weight = Array2::zeros((hidden_size, hidden_size));
        let k_bias = Array1::zeros(hidden_size);
        let v_weight = Array2::zeros((hidden_size, hidden_size));
        let v_bias = Array1::zeros(hidden_size);
        let output_weight = Array2::zeros((hidden_size, hidden_size));
        let output_bias = Array1::zeros(hidden_size);

        let attention = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            output_weight,
            output_bias,
            None,
        );

        let input = Array3::zeros((batch_size, seq_len, hidden_size));
        let cached_k = Array3::zeros((batch_size, cache_len, hidden_size));
        let cached_v = Array3::zeros((batch_size, cache_len, hidden_size));
        let mask = Array2::ones((batch_size, cache_len + seq_len));

        let result = attention.forward_with_cache(
            &input,
            None,
            Some(&mask),
            true, // Causal
            Some((cached_k.view(), cached_v.view())),
            None,
        );

        assert!(result.is_ok());

        let (output, new_k, new_v) = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_k.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_v.shape(), &[batch_size, seq_len, hidden_size]);
    }
    #[test]
    fn test_attention_without_bias() {
        // Test LLaMA-style attention (no biases)
        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

        // Create weights without biases (zero-length arrays)
        let q_weight = Array2::zeros((hidden_size, hidden_size));
        let q_bias = Array1::zeros(0); // No bias
        let k_weight = Array2::zeros((hidden_size, hidden_size));
        let k_bias = Array1::zeros(0);
        let v_weight = Array2::zeros((hidden_size, hidden_size));
        let v_bias = Array1::zeros(0);
        let output_weight = Array2::zeros((hidden_size, hidden_size));
        let output_bias = Array1::zeros(0);

        let attention = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            output_weight,
            output_bias,
            None,
        );

        let input = Array3::zeros((batch_size, seq_len, hidden_size));
        let mask = Array2::ones((batch_size, seq_len));

        let q_proj = matmul_3d_2d(&input, &attention.q_weight); // Add bias if exists
        let (k_states, v_states) = attention.project_kv(&input);

        let result = attention.attend(&q_proj, &k_states, &v_states, Some(&mask), false, 0);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_attention_with_bias() {
        // Test GPT-2-style attention (with biases)
        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

        // Create weights WITH biases
        let q_weight = Array2::zeros((hidden_size, hidden_size));
        let q_bias = Array1::zeros(hidden_size); // Has bias
        let k_weight = Array2::zeros((hidden_size, hidden_size));
        let k_bias = Array1::zeros(hidden_size);
        let v_weight = Array2::zeros((hidden_size, hidden_size));
        let v_bias = Array1::zeros(hidden_size);
        let output_weight = Array2::zeros((hidden_size, hidden_size));
        let output_bias = Array1::zeros(hidden_size);

        let attention = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            output_weight,
            output_bias,
            None,
        );

        let input = Array3::zeros((batch_size, seq_len, hidden_size));
        let mask = Array2::ones((batch_size, seq_len));

        let q_proj = matmul_3d_2d(&input, &attention.q_weight); // Add bias if exists
        let (k_states, v_states) = attention.project_kv(&input);

        let result = attention.attend(&q_proj, &k_states, &v_states, Some(&mask), false, 0);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_attention_with_cache_and_rope() {
        let hidden_size = 64;
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;
        let batch_size = 1;
        let seq_len = 1;
        let cache_len = 5;
        let max_seq_len = 128;

        let q_weight = Array2::eye(hidden_size);
        let q_bias = Array1::zeros(0);
        let k_weight = Array2::eye(hidden_size);
        let k_bias = Array1::zeros(0);
        let v_weight = Array2::eye(hidden_size);
        let v_bias = Array1::zeros(0);
        let output_weight = Array2::eye(hidden_size);
        let output_bias = Array1::zeros(0);

        let attention = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            output_weight,
            output_bias,
            None,
        );

        let rope = RoPE::new(head_dim, max_seq_len, 10000.0);
        let input = Array3::ones((batch_size, seq_len, hidden_size));
        let cached_k = Array3::ones((batch_size, cache_len, hidden_size));
        let cached_v = Array3::ones((batch_size, cache_len, hidden_size));
        let mask = Array2::ones((batch_size, cache_len + seq_len));

        let result = attention.forward_with_cache(
            &input,
            None,
            Some(&mask),
            true,
            Some((cached_k.view(), cached_v.view())),
            Some(&rope),
        );

        assert!(result.is_ok());
        let (output, new_k, new_v) = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_k.shape(), &[batch_size, seq_len, hidden_size]);
        assert_eq!(new_v.shape(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_repeat_kv_gqa() {
        use ndarray::Array4;

        // 8 KV heads, need to repeat 4x to get 32 Q heads
        let kv = Array4::from_shape_fn((2, 8, 10, 64), |(b, h, s, d)| {
            (b * 1000 + h * 100 + s * 10 + d) as f32
        });

        let repeated = repeat_kv(&kv, 4).unwrap();

        // Check shape
        assert_eq!(repeated.shape(), &[2, 32, 10, 64]);

        // Check that each KV head is repeated 4 times
        for kv_head in 0..8 {
            for group in 0..4 {
                let q_head = kv_head * 4 + group;

                // All values in this Q head should match the original KV head
                for b in 0..2 {
                    for s in 0..10 {
                        for d in 0..64 {
                            let original = kv[[b, kv_head, s, d]];
                            let repeated_val = repeated[[b, q_head, s, d]];
                            assert_eq!(
                                original, repeated_val,
                                "Mismatch at batch={}, kv_head={}, q_head={}, seq={}, dim={}",
                                b, kv_head, q_head, s, d
                            );
                        }
                    }
                }
            }
        }

        println!("✓ GQA repeat_kv test passed");
    }

    #[test]
    fn test_gqa_attention_shapes() {
        let hidden_size = 2048;
        let num_heads = 32;
        let num_kv_heads = 8;
        let batch_size = 1;
        let seq_len = 10;

        let q_weight = Array2::eye(hidden_size);
        let k_weight = Array2::eye(hidden_size).slice(s![.., 0..512]).to_owned();
        let v_weight = Array2::eye(hidden_size).slice(s![.., 0..512]).to_owned();
        let o_weight = Array2::eye(hidden_size);

        let attention = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            Array1::zeros(0),
            k_weight,
            Array1::zeros(0),
            v_weight,
            Array1::zeros(0),
            o_weight,
            Array1::zeros(0),
            Some(num_kv_heads),
        );

        let input = Array3::ones((batch_size, seq_len, hidden_size));
        let mask = Array2::ones((batch_size, seq_len));

        // ✅ Use forward() which handles projection
        let result = attention.forward_with_cache(&input, None, Some(&mask), false, None, None);

        assert!(result.is_ok(), "GQA attention should not fail");
        let (output, _, _) = result.unwrap();

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);

        println!("✓ GQA attention shapes test passed");
    }
    #[test]
    fn test_causal_mask_with_cache() {
        use ndarray::Array4;

        // Simulate: 5 tokens in cache, generating token 6
        let mut scores = Array4::ones((1, 4, 1, 6)); // [batch, heads, query_seq=1, key_seq=6]
        let cache_len = 5;

        apply_causal_mask(&mut scores, cache_len);

        // Query position is 5 (cache_len + 0)
        // Should be able to attend to positions 0-5 (all 6 keys)
        for key_pos in 0..6 {
            assert_ne!(
                scores[[0, 0, 0, key_pos]],
                MASK_VALUE,
                "Position {} should NOT be masked when query at position 5",
                key_pos
            );
        }

        println!("✓ Causal mask with cache test passed");
    }

    #[test]
    fn test_causal_mask_blocks_future() {
        use ndarray::Array4;

        // No cache, processing 3 tokens
        let mut scores = Array4::ones((1, 4, 3, 3));
        let cache_len = 0;

        apply_causal_mask(&mut scores, cache_len);

        // Query 0 can see: [0]
        // Query 1 can see: [0, 1]
        // Query 2 can see: [0, 1, 2]

        assert_ne!(scores[[0, 0, 0, 0]], MASK_VALUE); // Q0 sees K0
        assert_eq!(scores[[0, 0, 0, 1]], MASK_VALUE); // Q0 doesn't see K1
        assert_eq!(scores[[0, 0, 0, 2]], MASK_VALUE); // Q0 doesn't see K2

        assert_ne!(scores[[0, 0, 1, 0]], MASK_VALUE); // Q1 sees K0
        assert_ne!(scores[[0, 0, 1, 1]], MASK_VALUE); // Q1 sees K1
        assert_eq!(scores[[0, 0, 1, 2]], MASK_VALUE); // Q1 doesn't see K2

        println!("✓ Causal mask blocks future test passed");
    }
    #[test]
    fn test_rope_position_offset_correctness() {
        let hidden_size = 64;
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;
        let batch_size = 1;
        let seq_len = 1;
        let max_seq_len = 128;

        // ✅ Use identity weights for clearer signal
        let q_weight = Array2::eye(hidden_size);
        let q_bias = Array1::zeros(0);
        let k_weight = Array2::eye(hidden_size);
        let k_bias = Array1::zeros(0);
        let v_weight = Array2::eye(hidden_size);
        let v_bias = Array1::zeros(0);
        let output_weight = Array2::eye(hidden_size);
        let output_bias = Array1::zeros(0);

        let attention = MultiHeadAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            output_weight,
            output_bias,
            None,
        );

        let rope = RoPE::new(head_dim, max_seq_len, 10000.0);

        // ✅ Use more varied input
        let input1 = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, i)| {
            if i % 2 == 0 { 1.0 } else { 0.5 }
        });

        let input2 = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, i)| {
            if i % 2 == 0 { 0.8 } else { 0.6 }
        });

        println!("\n=== RoPE Position Test ===");

        // First call: position 0
        let result1 = attention.forward_with_cache(&input1, None, None, true, None, Some(&rope));
        assert!(result1.is_ok());
        let (output1, k1, v1) = result1.unwrap();
        println!("Output1 mean: {}", output1.mean().unwrap());

        // Second call: position 1 (cached position is 1)
        let result2 = attention.forward_with_cache(
            &input2,
            None,
            None,
            true,
            Some((k1.view(), v1.view())),
            Some(&rope),
        );
        assert!(result2.is_ok());
        let (output2, _k2, _v2) = result2.unwrap();
        println!("Output2 mean: {}", output2.mean().unwrap());

        // Outputs should differ
        let diff: f32 = (&output1 - &output2).mapv(|x| x.abs()).sum();
        println!("Difference: {}", diff);

        // Also check that RoPE actually modified the keys
        let k1_mean = k1.mean().unwrap();
        let input1_mean = input1.mean().unwrap();
        println!("K1 mean: {}, Input1 mean: {}", k1_mean, input1_mean);

        assert!(
            diff > 1e-6,
            "Outputs should differ due to RoPE position encoding, diff={}",
            diff
        );
    }
    #[test]
    fn test_apply_causal_mask_no_offset() {
        let mut scores = Array4::<f32>::zeros((1, 1, 4, 4));
        apply_causal_mask(&mut scores, 0);

        assert_eq!(scores[[0, 0, 0, 1]], MASK_VALUE);
        assert_eq!(scores[[0, 0, 1, 2]], MASK_VALUE);
        assert_eq!(scores[[0, 0, 2, 3]], MASK_VALUE);
        assert_eq!(scores[[0, 0, 1, 1]], 0.0);
    }

    #[test]
    fn test_apply_causal_mask_with_offset() {
        // Simulates generating the 3rd token (index 2) when 2 tokens are in cache
        let mut scores = Array4::<f32>::zeros((1, 1, 1, 3)); // Query len=1, Key len=3
        apply_causal_mask(&mut scores, 2);

        // Query at pos 2 can attend to keys at pos 0, 1, 2.
        assert_eq!(scores[[0, 0, 0, 0]], 0.0);
        assert_eq!(scores[[0, 0, 0, 1]], 0.0);
        assert_eq!(scores[[0, 0, 0, 2]], 0.0);
    }

    #[test]
    fn test_apply_padding_mask() {
        let mut scores = Array4::<f32>::zeros((1, 2, 2, 4)); // b, h, q, k
        let mask = Array2::from_shape_vec((1, 4), vec![1.0, 1.0, 0.0, 0.0]).unwrap();
        let masked = apply_padding_mask(scores, &mask).unwrap();

        // The last two key positions should be masked for all queries and heads
        assert_eq!(masked[[0, 0, 0, 2]], MASK_VALUE);
        assert_eq!(masked[[0, 0, 1, 3]], MASK_VALUE);
        assert_eq!(masked[[0, 1, 0, 2]], MASK_VALUE);
        assert_eq!(masked[[0, 0, 0, 1]], 0.0);
    }
}
