use anyhow::Result;
use ndarray::{s, Array1, Array2, Array3, Array4};

use super::*;
use crate::rope::RoPE;
use crate::utils::linear_algebra::matmul_3d_2d;
use crate::utils::masks::{apply_causal_mask, apply_padding_mask, MASK_VALUE};

#[cfg(test)]
mod tests {
    use super::*;

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
        let mask = Array2::ones((1, 6));

        let (output, k, v) = attn.forward_with_cache(
            &input,
            None,
            Some(&mask),
            true,
            Some((past_k.view(), past_v.view())),
            None,
        )?;

        assert_eq!(output.shape(), &[1, 1, 64]);
        assert_eq!(k.shape(), &[1, 1, 64]);
        assert_eq!(v.shape(), &[1, 1, 64]);
        Ok(())
    }

    #[test]
    fn test_gqa_with_rope() -> Result<()> {
        let attn = create_mock_attention(64, 4, Some(2));
        let rope = RoPE::new(16, 128, 10000.0);
        let input = Array3::ones((1, 10, 64));
        let mask = Array2::ones((1, 10));

        let (output, k, v) =
            attn.forward_with_cache(&input, None, Some(&mask), true, None, Some(&rope))?;

        assert_eq!(output.shape(), &[1, 10, 64]);
        assert_eq!(k.shape(), &[1, 10, 32]);
        assert_eq!(v.shape(), &[1, 10, 32]);
        Ok(())
    }

    #[test]
    fn test_attention_with_cache() {
        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 1;
        let seq_len = 1;
        let cache_len = 5;

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
            true,
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
        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

        let q_weight = Array2::zeros((hidden_size, hidden_size));
        let q_bias = Array1::zeros(0);
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

        let q_proj = matmul_3d_2d(&input, &attention.q_weight);
        let (k_states, v_states) = attention.project_kv(&input);

        let result = attention.attend(&q_proj, &k_states, &v_states, Some(&mask), false, 0);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_attention_with_bias() {
        let hidden_size = 64;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 10;

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
        let mask = Array2::ones((batch_size, seq_len));

        let q_proj = matmul_3d_2d(&input, &attention.q_weight);
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
        let kv = Array4::from_shape_fn((2, 8, 10, 64), |(b, h, s, d)| {
            (b * 1000 + h * 100 + s * 10 + d) as f32
        });

        let repeated = repeat_kv(&kv, 4).unwrap();

        assert_eq!(repeated.shape(), &[2, 32, 10, 64]);

        for kv_head in 0..8 {
            for group in 0..4 {
                let q_head = kv_head * 4 + group;

                for b in 0..2 {
                    for s in 0..10 {
                        for d in 0..64 {
                            let original = kv[[b, kv_head, s, d]];
                            let repeated_val = repeated[[b, q_head, s, d]];
                            assert_eq!(
                                original, repeated_val,
                                "mismatch at batch={}, kv_head={}, q_head={}, seq={}, dim={}",
                                b, kv_head, q_head, s, d
                            );
                        }
                    }
                }
            }
        }
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

        let result = attention.forward_with_cache(&input, None, Some(&mask), false, None, None);

        assert!(result.is_ok(), "GQA attention should not fail");
        let (output, _, _) = result.unwrap();

        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_causal_mask_with_cache() {
        let mut scores = Array4::ones((1, 4, 1, 6));
        let cache_len = 5;

        apply_causal_mask(&mut scores, cache_len);

        for key_pos in 0..6 {
            assert_ne!(
                scores[[0, 0, 0, key_pos]],
                MASK_VALUE,
                "position {} should not be masked when query at position 5",
                key_pos
            );
        }
    }

    #[test]
    fn test_causal_mask_blocks_future() {
        let mut scores = Array4::ones((1, 4, 3, 3));
        let cache_len = 0;

        apply_causal_mask(&mut scores, cache_len);

        assert_ne!(scores[[0, 0, 0, 0]], MASK_VALUE);
        assert_eq!(scores[[0, 0, 0, 1]], MASK_VALUE);
        assert_eq!(scores[[0, 0, 0, 2]], MASK_VALUE);

        assert_ne!(scores[[0, 0, 1, 0]], MASK_VALUE);
        assert_ne!(scores[[0, 0, 1, 1]], MASK_VALUE);
        assert_eq!(scores[[0, 0, 1, 2]], MASK_VALUE);
    }

    #[test]
    fn test_rope_position_offset_correctness() {
        let hidden_size = 64;
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;
        let batch_size = 1;
        let seq_len = 1;
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

        let input1 = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, i)| {
            if i % 2 == 0 {
                1.0
            } else {
                0.5
            }
        });

        let input2 = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, i)| {
            if i % 2 == 0 {
                0.8
            } else {
                0.6
            }
        });

        let result1 = attention.forward_with_cache(&input1, None, None, true, None, Some(&rope));
        assert!(result1.is_ok());
        let (output1, k1, v1) = result1.unwrap();

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

        let diff: f32 = (&output1 - &output2).mapv(|x| x.abs()).sum();

        assert!(
            diff > 1e-6,
            "outputs should differ due to RoPE position encoding, diff={}",
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
        let mut scores = Array4::<f32>::zeros((1, 1, 1, 3));
        apply_causal_mask(&mut scores, 2);

        assert_eq!(scores[[0, 0, 0, 0]], 0.0);
        assert_eq!(scores[[0, 0, 0, 1]], 0.0);
        assert_eq!(scores[[0, 0, 0, 2]], 0.0);
    }

    #[test]
    fn test_apply_padding_mask() {
        let scores = Array4::<f32>::zeros((1, 2, 2, 4));
        let mask = Array2::from_shape_vec((1, 4), vec![1.0, 1.0, 0.0, 0.0]).unwrap();
        let masked = apply_padding_mask(scores, &mask).unwrap();

        assert_eq!(masked[[0, 0, 0, 2]], MASK_VALUE);
        assert_eq!(masked[[0, 0, 1, 3]], MASK_VALUE);
        assert_eq!(masked[[0, 1, 0, 2]], MASK_VALUE);
        assert_eq!(masked[[0, 0, 0, 1]], 0.0);
    }
}