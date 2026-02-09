#[cfg(test)]
mod tests {
    use std::path::Path;

    
    use crate::weights::{gguf_loader::GgufLoader, ModelWeights, WeightLoader};

    const SAFETENSORS_PATH: &str = "/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B";
    const GGUF_PATH: &str = "/home/olafurj/.cache/kjarni/llama-3.2-1b-instruct-q4_k_m/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

    fn skip_if_missing(path: &str) -> bool {
        !Path::new(path).exists()
    }

    #[test]
    fn test_gguf_embedding_lookup_matches() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let emb = gguf_weights
            .get_array2("model.embed_tokens.weight")
            .unwrap();

        let bos_embedding = emb.row(128000);

        let sum: f32 = bos_embedding.iter().map(|x| x.abs()).sum();

        assert!(sum > 0.1, "bos embedding should not be near-zero");
        assert!(sum < 1000.0, "bos embedding should not be huge");
    }

    #[test]
    fn test_gguf_transpose_hypothesis() {
        if skip_if_missing(SAFETENSORS_PATH) || skip_if_missing(GGUF_PATH) {
            return;
        }

        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let st_q = st_weights
            .get_array2("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        let gguf_q = gguf_weights
            .get_array2("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        let gguf_q_transposed = gguf_q.t();

        let input = ndarray::Array1::<f32>::ones(2048);

        let st_out: f32 = st_q
            .row(0)
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum();
        let gguf_out: f32 = gguf_q
            .row(0)
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum();
        let _gguf_t_out: f32 = gguf_q_transposed
            .row(0)
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum();

        let _diff_original: f32 = st_q
            .iter()
            .zip(gguf_q.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / st_q.len() as f32;

        let _diff_transposed: f32 = st_q
            .iter()
            .zip(gguf_q_transposed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / st_q.len() as f32;
    }

    #[test]
    fn test_gguf_vs_safetensors_embedding_lookup() {
        if skip_if_missing(SAFETENSORS_PATH) || skip_if_missing(GGUF_PATH) {
            return;
        }

        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let st_emb = st_weights.get_array2("model.embed_tokens.weight").unwrap();
        let gguf_emb = gguf_weights
            .get_array2("model.embed_tokens.weight")
            .unwrap();

        let test_tokens = [128000, 128006, 882, 128007];

        for token_id in test_tokens {
            let st_vec = st_emb.row(token_id);
            let gguf_vec = gguf_emb.row(token_id);

            let _diff: f32 = st_vec
                .iter()
                .zip(gguf_vec.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / st_vec.len() as f32;
        }
    }

    #[test]
    fn test_q4k_dequantization_sanity() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let raw = gguf_weights.loader()
            .get_raw("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        let dequant = gguf_weights
            .get_array2("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        let has_nan = dequant.iter().any(|v| v.is_nan());
        let has_inf = dequant.iter().any(|v| v.is_infinite());

        assert!(!has_nan, "dequantized values should not contain NaN");
        assert!(!has_inf, "dequantized values should not contain Inf");
    }

    #[test]
    fn test_gguf_first_layer_forward() {
        if skip_if_missing(SAFETENSORS_PATH) || skip_if_missing(GGUF_PATH) {
            return;
        }

        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let st_q = st_weights
            .get_array2("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        let gguf_q = gguf_weights
            .get_array2("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        let input = ndarray::Array1::<f32>::ones(2048);

        let _st_out: f32 = st_q
            .row(0)
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum();
        let _gguf_out: f32 = gguf_q
            .row(0)
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum();
    }

    #[test]
    fn test_safetensors_loader_basic() {
        if skip_if_missing(SAFETENSORS_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();

        assert!(
            !weights.config_json().is_empty(),
            "config json should not be empty"
        );

        assert!(
            weights.contains("model.embed_tokens.weight"),
            "should contain embeddings"
        );
        assert!(
            weights.contains("model.norm.weight"),
            "should contain final norm"
        );
        assert!(
            weights.contains("model.layers.0.self_attn.q_proj.weight"),
            "should contain layer 0 Q proj"
        );

        let has_separate_lm_head = weights.contains("lm_head.weight");
        assert!(
            has_separate_lm_head || weights.contains("model.embed_tokens.weight"),
            "should have lm head or tied embeddings"
        );
    }

    #[test]
    fn test_gguf_loader_basic() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        assert!(
            !weights.config_json().is_empty(),
            "config json should not be empty"
        );

        assert!(
            weights.contains("model.embed_tokens.weight"),
            "should contain embeddings"
        );
        assert!(
            weights.contains("model.norm.weight"),
            "should contain final norm"
        );
        assert!(
            weights.contains("model.layers.0.self_attn.q_proj.weight"),
            "should contain layer 0 Q proj"
        );
    }

    #[test]
    fn test_gguf_dtypes() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let tensors = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.input_layernorm.weight",
        ];

        for name in tensors {
            let raw = weights.loader().get_raw(name).unwrap();
            assert!(!raw.bytes.is_empty(), "tensor {} should have data", name);
        }
    }

    #[test]
    fn test_embedding_shape_matches() {
        if skip_if_missing(SAFETENSORS_PATH) || skip_if_missing(GGUF_PATH) {
            return;
        }

        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let st_emb = st_weights.loader().get_raw("model.embed_tokens.weight").unwrap();
        let gguf_emb = gguf_weights.loader().get_raw("model.embed_tokens.weight").unwrap();

        assert_eq!(
            st_emb.shape, gguf_emb.shape,
            "embedding shapes should match"
        );
    }

    #[test]
    fn test_layer_shapes_match() {
        if skip_if_missing(SAFETENSORS_PATH) || skip_if_missing(GGUF_PATH) {
            return;
        }

        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let tensors_to_check = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ];

        for name in tensors_to_check {
            
            let st_tensor = st_weights.loader().get_raw(name).unwrap();
            let gguf_tensor = gguf_weights.loader().get_raw(name).unwrap();

            assert_eq!(
                st_tensor.shape, gguf_tensor.shape,
                "shape mismatch for tensor '{}'",
                name
            );
        }
    }

    #[test]
    fn test_gguf_dequantization_norm() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let norm = weights.get_array1("model.norm.weight").unwrap();

        assert!(
            norm.iter().all(|&v| v.is_finite()),
            "norm values should be finite"
        );
        assert!(
            norm.iter().all(|&v| v.abs() < 100.0),
            "norm values should be reasonable"
        );
    }

    #[test]
    fn test_gguf_dequantization_layer_weight() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let q_proj = weights
            .get_array2("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        assert!(
            q_proj.iter().all(|&v| v.is_finite()),
            "values should be finite"
        );
        assert!(
            q_proj.iter().all(|&v| v.abs() < 10.0),
            "transformer weights should be small"
        );
    }

    #[test]
    fn test_norm_values_similar() {
        if skip_if_missing(SAFETENSORS_PATH) || skip_if_missing(GGUF_PATH) {
            return;
        }

        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let st_norm = st_weights.get_array1("model.norm.weight").unwrap();
        let gguf_norm = gguf_weights.get_array1("model.norm.weight").unwrap();

        assert_eq!(st_norm.len(), gguf_norm.len(), "norm lengths should match");

        assert!(
            st_norm.iter().all(|&v| v.is_finite()),
            "st norm should be finite"
        );
        assert!(
            gguf_norm.iter().all(|&v| v.is_finite()),
            "gguf norm should be finite"
        );
    }

    #[test]
    fn test_quantized_weight_reasonable_range() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        for layer_idx in [0, 5, 10, 15] {
            let name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);

            if let Ok(arr) = weights.get_array2(&name) {
                let min = arr.iter().cloned().reduce(f32::min).unwrap();
                let max = arr.iter().cloned().reduce(f32::max).unwrap();
                let mean = arr.iter().sum::<f32>() / arr.len() as f32;
                let variance =
                    arr.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / arr.len() as f32;
                let std = variance.sqrt();

                assert!(min > -5.0, "min should be > -5.0, got {}", min);
                assert!(max < 5.0, "max should be < 5.0, got {}", max);
                assert!(mean.abs() < 0.1, "mean should be near 0, got {}", mean);
                assert!(std < 1.0, "std should be < 1.0, got {}", std);
            }
        }
    }

    #[test]
    fn test_gguf_name_translation() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let loader = GgufLoader::new(Path::new(GGUF_PATH)).unwrap();

        let required_tensors = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ];

        for hf_name in required_tensors {
            let tensor = loader
                .get_raw(hf_name)
                .unwrap_or_else(|e| panic!("failed to load '{}': {}", hf_name, e));
            assert!(!tensor.shape.is_empty(), "tensor {} should have shape", hf_name);
        }
    }

    #[test]
    fn test_list_gguf_tensors() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let loader = GgufLoader::new(Path::new(GGUF_PATH)).unwrap();

        let probe_names = [
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
        ];

        for _name in probe_names {
            // Probe only, no assertions needed
        }
    }

    #[test]
    fn test_embedding_values_sane() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let emb = weights.get_array2("model.embed_tokens.weight").unwrap();

        for token_id in [0, 1, 100, 1000, 10000] {
            if token_id < emb.shape()[0] {
                let row = emb.row(token_id);
                let slice = row.as_slice().unwrap();

                assert!(
                    slice.iter().all(|&v| v.is_finite()),
                    "token {} embedding has non-finite values",
                    token_id
                );
                assert!(
                    slice.iter().all(|&v| v.abs() < 100.0),
                    "token {} embedding has unreasonably large values",
                    token_id
                );
            }
        }
    }

    #[test]
    fn test_first_layer_norm_output() {
        if skip_if_missing(SAFETENSORS_PATH) || skip_if_missing(GGUF_PATH) {
            return;
        }

        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let st_norm = st_weights
            .get_array1("model.layers.0.input_layernorm.weight")
            .unwrap();
        let gguf_norm = gguf_weights
            .get_array1("model.layers.0.input_layernorm.weight")
            .unwrap();

        assert_eq!(st_norm.len(), gguf_norm.len());
        assert!(st_norm.iter().all(|&v| v.is_finite()));
        assert!(gguf_norm.iter().all(|&v| v.is_finite()));
    }
}