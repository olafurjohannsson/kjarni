
    use super::*;
    use crate::feedforward::{FeedForward, StdFeedForward, SwiGluFeedForward};
    use crate::linear_layer_old::LinearLayer;
    use crate::normalization::{LayerNorm, RMSNorm};
    use crate::rope::RoPE;
    use ndarray::{Array1, Array2, Array3, s};
    use std::sync::Arc;

    #[test]
    fn test_decoder_layer_with_rope_and_gqa() {
        let hidden_size = 2048;
        let num_heads = 32;
        let num_kv_heads = 8;
        let head_dim = hidden_size / num_heads; // 64
        let kv_dim = num_kv_heads * head_dim; // 512
        let intermediate_size = 8192;

        // Attention weights: [out_features, in_features]
        let q_weight = LinearLayer::from(Array2::<f32>::zeros((hidden_size, hidden_size)));
        let k_weight = LinearLayer::from(Array2::<f32>::zeros((kv_dim, hidden_size)));
        let v_weight = LinearLayer::from(Array2::<f32>::zeros((kv_dim, hidden_size)));
        let o_weight = LinearLayer::from(Array2::<f32>::zeros((hidden_size, hidden_size)));

        let attention = DecoderAttention::new(
            hidden_size,
            num_heads,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            Some(num_kv_heads),
        );

        let rope = Arc::new(RoPE::new(head_dim, 128, 10000.0));

        // FFN weights: [out_features, in_features]
        let gate_weight = LinearLayer::from(Array2::<f32>::zeros((intermediate_size, hidden_size)));
        let up_weight = LinearLayer::from(Array2::<f32>::zeros((intermediate_size, hidden_size)));
        let down_weight = LinearLayer::from(Array2::<f32>::zeros((hidden_size, intermediate_size)));

        let feedforward =
            FeedForward::SwiGLU(SwiGluFeedForward::new(gate_weight, up_weight, down_weight));

        let norm1 = Normalization::RMSNorm(RMSNorm::new(Array1::ones(hidden_size), 1e-5));
        let norm2 = Normalization::RMSNorm(RMSNorm::new(Array1::ones(hidden_size), 1e-5));

        let layer = DecoderLayer {
            self_attn: attention,
            self_attn_layer_norm: norm1,
            feedforward,
            ffn_layer_norm: norm2,
            is_prenorm: true,
            rope: Some(rope),
        };

        // Test 1: Prefill (no cache)
        let input = Array3::ones((1, 10, hidden_size));
        let mask = Array2::ones((1, 10));

        let result = layer.forward(&input, &mask, 0, None);
        assert!(result.is_ok(), "Prefill should succeed");

        let (output, (k, v)) = result.unwrap();
        assert_eq!(output.shape(), &[1, 10, hidden_size]);
        assert_eq!(k.shape(), &[1, 10, kv_dim]);
        assert_eq!(v.shape(), &[1, 10, kv_dim]);

        // Test 2: Generate (with cache)
        let input2 = Array3::ones((1, 1, hidden_size));
        let mask2 = Array2::ones((1, 11));

        let result2 = layer.forward(&input2, &mask2, 10, Some((k.view(), v.view())));
        assert!(result2.is_ok(), "Generation should succeed");

        let (output2, (k2, v2)) = result2.unwrap();
        assert_eq!(output2.shape(), &[1, 1, hidden_size]);
        assert_eq!(k2.shape(), &[1, 1, kv_dim]);
        assert_eq!(v2.shape(), &[1, 1, kv_dim]);

        println!("✓ Decoder layer integration test passed");
    }
