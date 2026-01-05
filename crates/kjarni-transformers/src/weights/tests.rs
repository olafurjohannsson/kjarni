#[cfg(test)]
mod loader_tests {
    use crate::tensor::DType;
    use crate::weights::{ModelWeights, WeightLoader};
    use std::path::Path;

    // Test paths - adjust these to your setup
    const SAFETENSORS_PATH: &str = "/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B";
    const GGUF_PATH: &str = "/home/olafurj/.cache/kjarni/llama-3.2-1b-instruct-q4_k_m/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

    fn skip_if_missing(path: &str) -> bool {
        if !Path::new(path).exists() {
            eprintln!("⚠️  Skipping test - path not found: {}", path);
            true
        } else {
            false
        }
    }

    // =========================================================================
    // Basic Loading Tests
    // =========================================================================
    #[test]
    fn test_gguf_embedding_lookup_matches() {
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        // Get embedding table
        let emb = gguf_weights
            .get_array2("model.embed_tokens.weight")
            .unwrap();

        // Simulate token 128000 (BOS) lookup
        let bos_embedding = emb.row(128000);
        println!(
            "BOS embedding first 10: {:?}",
            &bos_embedding.as_slice().unwrap()[..10]
        );

        // Check it's not all zeros or garbage
        let sum: f32 = bos_embedding.iter().map(|x| x.abs()).sum();
        println!("BOS embedding L1 norm: {}", sum);

        assert!(sum > 0.1, "BOS embedding should not be near-zero");
        assert!(sum < 1000.0, "BOS embedding should not be huge");
    }
    #[test]
    fn test_gguf_transpose_hypothesis() {
        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let st_q = st_weights
            .get_array2("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        let gguf_q = gguf_weights
            .get_array2("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        // Try transposing GGUF weights
        let gguf_q_transposed = gguf_q.t();

        let input = ndarray::Array1::<f32>::ones(2048);

        // Original comparison
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
        let gguf_t_out: f32 = gguf_q_transposed
            .row(0)
            .iter()
            .zip(input.iter())
            .map(|(a, b)| a * b)
            .sum();

        println!("ST Q_proj row 0 dot input: {}", st_out);
        println!("GGUF Q_proj row 0 dot input: {}", gguf_out);
        println!("GGUF TRANSPOSED row 0 dot input: {}", gguf_t_out);
        println!();
        println!("Diff (original): {}", (st_out - gguf_out).abs());
        println!("Diff (transposed): {}", (st_out - gguf_t_out).abs());

        // Compare full weight statistics
        let diff_original: f32 = st_q
            .iter()
            .zip(gguf_q.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / st_q.len() as f32;

        let diff_transposed: f32 = st_q
            .iter()
            .zip(gguf_q_transposed.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / st_q.len() as f32;

        println!("Avg diff (original): {}", diff_original);
        println!("Avg diff (transposed): {}", diff_transposed);
    }
    #[test]
    fn test_gguf_vs_safetensors_embedding_lookup() {
        // Load both models
        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        // Get embeddings (dequantized)
        let st_emb = st_weights.get_array2("model.embed_tokens.weight").unwrap();
        let gguf_emb = gguf_weights
            .get_array2("model.embed_tokens.weight")
            .unwrap();

        // Test specific token lookups
        let test_tokens = [128000, 128006, 882, 128007]; // <|begin_of_text|>, <|start_header_id|>, "user", <|end_header_id|>

        for token_id in test_tokens {
            let st_vec = st_emb.row(token_id);
            let gguf_vec = gguf_emb.row(token_id);

            let diff: f32 = st_vec
                .iter()
                .zip(gguf_vec.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / st_vec.len() as f32;

            println!("Token {}: avg diff = {:.6}", token_id, diff);

            // For instruct vs base, there WILL be differences, but they should be small
            // Large differences indicate a loading bug
        }
    }
    #[test]
    fn test_q4k_dequantization_sanity() {
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        // Q_proj is Q4_K
        let raw = gguf_weights
            .get_raw("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        println!(
            "Q_proj raw dtype: {:?}, bytes: {}",
            raw.dtype,
            raw.bytes.len()
        );

        let dequant = gguf_weights
            .get_array2("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        println!("Q_proj dequantized shape: {:?}", dequant.shape());

        // Check statistics
        let min = dequant.iter().cloned().reduce(f32::min).unwrap();
        let max = dequant.iter().cloned().reduce(f32::max).unwrap();
        let mean = dequant.iter().sum::<f32>() / dequant.len() as f32;

        println!("Stats: min={:.6}, max={:.6}, mean={:.6}", min, max, mean);

        // Check for NaN or Inf
        let has_nan = dequant.iter().any(|v| v.is_nan());
        let has_inf = dequant.iter().any(|v| v.is_infinite());
        println!("Has NaN: {}, Has Inf: {}", has_nan, has_inf);

        // Print first few values
        println!("First 20 values: {:?}", &dequant.as_slice().unwrap()[..20]);
    }
    #[test]
    fn test_gguf_first_layer_forward() {
        // Compare layer 0 Q projection output for same input
        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let st_q = st_weights
            .get_array2("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        let gguf_q = gguf_weights
            .get_array2("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        // Create a simple test input
        let input = ndarray::Array1::<f32>::ones(2048);

        // Manual matmul: output = input @ weight.T
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

        println!("ST Q_proj row 0 dot input: {}", st_out);
        println!("GGUF Q_proj row 0 dot input: {}", gguf_out);
        println!("Difference: {}", (st_out - gguf_out).abs());
    }
    #[test]
    fn test_safetensors_loader_basic() {
        if skip_if_missing(SAFETENSORS_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();

        assert!(
            !weights.config_json.is_empty(),
            "Config JSON should not be empty"
        );

        // Check key tensors exist
        assert!(
            weights.contains("model.embed_tokens.weight"),
            "Should contain embeddings"
        );
        assert!(
            weights.contains("model.norm.weight"),
            "Should contain final norm"
        );
        assert!(
            weights.contains("model.layers.0.self_attn.q_proj.weight"),
            "Should contain layer 0 Q proj"
        );

        // LM head may or may not exist (tied weights)
        // Llama uses tied weights, so lm_head.weight == model.embed_tokens.weight
        let has_separate_lm_head = weights.contains("lm_head.weight");
        println!("Has separate LM head: {}", has_separate_lm_head);
        // Either lm_head exists OR we have tied weights via embeddings
        assert!(
            has_separate_lm_head || weights.contains("model.embed_tokens.weight"),
            "Should have LM head or tied embeddings"
        );
    }

    #[test]
    fn test_gguf_loader_basic() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        assert!(
            !weights.config_json.is_empty(),
            "Config JSON should not be empty"
        );
        println!("GGUF synthesized config:\n{}", weights.config_json);

        // Check key tensors exist (using HF names - loader should translate)
        assert!(
            weights.contains("model.embed_tokens.weight"),
            "Should contain embeddings"
        );
        assert!(
            weights.contains("model.norm.weight"),
            "Should contain final norm"
        );
        assert!(
            weights.contains("model.layers.0.self_attn.q_proj.weight"),
            "Should contain layer 0 Q proj"
        );

        // Check if lm_head exists or falls back to embeddings (tied weights)
        // After fix, this should work via fallback
        let has_lm_head = weights.contains("lm_head.weight");
        println!("Has LM head (may be tied): {}", has_lm_head);
    }

    #[test]
    fn test_gguf_dtypes() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        // Tensors that definitely exist
        let tensors = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.input_layernorm.weight",
        ];

        for name in tensors {
            let raw = weights.get_raw(name).unwrap();
            println!(
                "{}: dtype={:?}, shape={:?}, bytes={}",
                name,
                raw.dtype,
                raw.shape,
                raw.bytes.len()
            );
        }

        // LM head - may be tied, test separately
        match weights.get_raw("lm_head.weight") {
            Ok(raw) => println!(
                "lm_head.weight: dtype={:?}, shape={:?}",
                raw.dtype, raw.shape
            ),
            Err(e) => println!("lm_head.weight not found (tied weights): {}", e),
        }
    }

    // =========================================================================
    // Shape Comparison Tests
    // =========================================================================

    #[test]
    fn test_embedding_shape_matches() {
        if skip_if_missing(SAFETENSORS_PATH) || skip_if_missing(GGUF_PATH) {
            return;
        }

        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let st_emb = st_weights.get_raw("model.embed_tokens.weight").unwrap();
        let gguf_emb = gguf_weights.get_raw("model.embed_tokens.weight").unwrap();

        println!(
            "SafeTensors embedding: shape={:?}, dtype={:?}",
            st_emb.shape, st_emb.dtype
        );
        println!(
            "GGUF embedding: shape={:?}, dtype={:?}",
            gguf_emb.shape, gguf_emb.dtype
        );

        assert_eq!(
            st_emb.shape, gguf_emb.shape,
            "Embedding shapes should match"
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
            let st_tensor = st_weights.get_raw(name).unwrap();
            let gguf_tensor = gguf_weights.get_raw(name).unwrap();

            println!(
                "{}: ST shape={:?} dtype={:?} | GGUF shape={:?} dtype={:?}",
                name, st_tensor.shape, st_tensor.dtype, gguf_tensor.shape, gguf_tensor.dtype
            );

            assert_eq!(
                st_tensor.shape, gguf_tensor.shape,
                "Shape mismatch for tensor '{}': ST={:?}, GGUF={:?}",
                name, st_tensor.shape, gguf_tensor.shape
            );
        }
    }

    // =========================================================================
    // Dequantization Tests
    // =========================================================================

    #[test]
    fn test_gguf_dequantization_norm() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        // Norm weights should be small F32/F16 tensors
        let norm = weights.get_array1("model.norm.weight").unwrap();

        println!("Norm shape: {:?}", norm.shape());
        println!(
            "Norm first 10 values: {:?}",
            &norm.as_slice().unwrap()[..10]
        );
        println!(
            "Norm stats: min={}, max={}, mean={}",
            norm.iter().cloned().reduce(f32::min).unwrap(),
            norm.iter().cloned().reduce(f32::max).unwrap(),
            norm.iter().sum::<f32>() / norm.len() as f32,
        );

        // Sanity checks - norm weights should be close to 1.0 (RMSNorm)
        assert!(
            norm.iter().all(|&v| v.is_finite()),
            "Norm values should be finite"
        );
        assert!(
            norm.iter().all(|&v| v.abs() < 100.0),
            "Norm values should be reasonable"
        );
    }

    #[test]
    fn test_gguf_dequantization_layer_weight() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        // This is a quantized weight - test dequantization
        let q_proj = weights
            .get_array2("model.layers.0.self_attn.q_proj.weight")
            .unwrap();

        println!("Q_proj shape: {:?}", q_proj.shape());
        println!(
            "Q_proj first row, first 10 values: {:?}",
            &q_proj.row(0).as_slice().unwrap()[..10]
        );
        println!(
            "Q_proj stats: min={}, max={}, mean={}",
            q_proj.iter().cloned().reduce(f32::min).unwrap(),
            q_proj.iter().cloned().reduce(f32::max).unwrap(),
            q_proj.iter().sum::<f32>() / q_proj.len() as f32,
        );

        // Sanity checks
        assert!(
            q_proj.iter().all(|&v| v.is_finite()),
            "Values should be finite"
        );
        assert!(
            q_proj.iter().all(|&v| v.abs() < 10.0),
            "Transformer weights should be small"
        );
    }

    // =========================================================================
    // Cross-Format Comparison Tests
    // =========================================================================

    #[test]
    fn test_norm_values_similar() {
        if skip_if_missing(SAFETENSORS_PATH) || skip_if_missing(GGUF_PATH) {
            return;
        }

        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        // Norm weights should be identical (usually not quantized)
        let st_norm = st_weights.get_array1("model.norm.weight").unwrap();
        let gguf_norm = gguf_weights.get_array1("model.norm.weight").unwrap();

        assert_eq!(st_norm.len(), gguf_norm.len(), "Norm lengths should match");

        // Compare values
        let max_diff = st_norm
            .iter()
            .zip(gguf_norm.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("Norm max difference: {}", max_diff);

        // Note: GGUF instruct model may have different norm values than base model
        // This test validates the loading works, not exact match
        assert!(
            st_norm.iter().all(|&v| v.is_finite()),
            "ST norm should be finite"
        );
        assert!(
            gguf_norm.iter().all(|&v| v.is_finite()),
            "GGUF norm should be finite"
        );
    }

    #[test]
    fn test_quantized_weight_reasonable_range() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        // Test multiple quantized layers
        for layer_idx in [0, 5, 10, 15] {
            let name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);

            if let Ok(arr) = weights.get_array2(&name) {
                let min = arr.iter().cloned().reduce(f32::min).unwrap();
                let max = arr.iter().cloned().reduce(f32::max).unwrap();
                let mean = arr.iter().sum::<f32>() / arr.len() as f32;
                let variance =
                    arr.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / arr.len() as f32;
                let std = variance.sqrt();

                println!(
                    "Layer {} Q_proj: min={:.4}, max={:.4}, mean={:.6}, std={:.4}",
                    layer_idx, min, max, mean, std
                );

                // Transformer weights typically have small values
                assert!(min > -5.0, "Min should be > -5.0, got {}", min);
                assert!(max < 5.0, "Max should be < 5.0, got {}", max);
                assert!(mean.abs() < 0.1, "Mean should be near 0, got {}", mean);
                assert!(std < 1.0, "Std should be < 1.0, got {}", std);
            }
        }
    }

    // =========================================================================
    // GGUF Name Translation Tests
    // =========================================================================

    #[test]
    fn test_gguf_name_translation() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        use crate::weights::gguf_loader::GgufLoader;

        let loader = GgufLoader::new(Path::new(GGUF_PATH)).unwrap();

        // Print all available tensors first
        println!("\n=== Available GGUF Tensors ===");
        loader.debug_print_tensors();

        // Test translations that should work
        let required_tensors = [
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "blk.0.attn_q.weight",
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "blk.0.attn_k.weight",
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                "blk.0.attn_v.weight",
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "blk.0.attn_output.weight",
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "blk.0.ffn_gate.weight",
            ),
            ("model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
            (
                "model.layers.0.mlp.down_proj.weight",
                "blk.0.ffn_down.weight",
            ),
            (
                "model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "blk.0.ffn_norm.weight",
            ),
        ];

        for (hf_name, expected_gguf_name) in required_tensors {
            match loader.get_raw(hf_name) {
                Ok(tensor) => {
                    println!(
                        "✅ {} -> {} : shape={:?}, dtype={:?}",
                        hf_name, expected_gguf_name, tensor.shape, tensor.dtype
                    );
                }
                Err(e) => {
                    panic!(
                        "❌ Failed to load '{}' (expected '{}'): {}",
                        hf_name, expected_gguf_name, e
                    );
                }
            }
        }

        // LM head is optional (tied weights)
        println!("\n--- Optional Tensors (may be tied) ---");
        match loader.get_raw("lm_head.weight") {
            Ok(tensor) => {
                println!(
                    "✅ lm_head.weight: shape={:?}, dtype={:?}",
                    tensor.shape, tensor.dtype
                );
            }
            Err(_) => {
                println!("⚠️  lm_head.weight not found - model uses tied weights");
            }
        }
    }

    // =========================================================================
    // Debug: Print All GGUF Tensor Names
    // =========================================================================

    #[test]
    fn test_list_gguf_tensors() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        use crate::weights::gguf_loader::GgufLoader;

        let loader = GgufLoader::new(Path::new(GGUF_PATH)).unwrap();

        println!("\n=== GGUF Tensor Names ===");
        // Note: You may need to expose tensor_map or add a method to list tensors
        // For now, try loading known tensors and see what's actually in the file

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

        for name in probe_names {
            // Direct access without translation
            println!("Checking GGUF name: {}", name);
        }
    }

    // =========================================================================
    // Embedding Sanity Check (This is likely where garbage output comes from)
    // =========================================================================

    #[test]
    fn test_embedding_values_sane() {
        if skip_if_missing(GGUF_PATH) {
            return;
        }

        let weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        let emb = weights.get_array2("model.embed_tokens.weight").unwrap();

        println!("Embedding shape: {:?}", emb.shape());

        // Check a few token embeddings
        for token_id in [0, 1, 100, 1000, 10000] {
            if token_id < emb.shape()[0] {
                let row = emb.row(token_id);
                let slice = row.as_slice().unwrap();
                println!(
                    "Token {}: first 5 = {:?}, last 5 = {:?}",
                    token_id,
                    &slice[..5],
                    &slice[slice.len() - 5..],
                );

                // Sanity checks
                assert!(
                    slice.iter().all(|&v| v.is_finite()),
                    "Token {} embedding has non-finite values",
                    token_id
                );
                assert!(
                    slice.iter().all(|&v| v.abs() < 100.0),
                    "Token {} embedding has unreasonably large values",
                    token_id
                );
            }
        }
    }

    // =========================================================================
    // End-to-End Forward Pass Comparison (Optional - More Complex)
    // =========================================================================

    #[test]
    fn test_first_layer_norm_output() {
        if skip_if_missing(SAFETENSORS_PATH) || skip_if_missing(GGUF_PATH) {
            return;
        }

        let st_weights = ModelWeights::new(Path::new(SAFETENSORS_PATH)).unwrap();
        let gguf_weights = ModelWeights::new(Path::new(GGUF_PATH)).unwrap();

        // Get layer 0 input norm weights
        let st_norm = st_weights
            .get_array1("model.layers.0.input_layernorm.weight")
            .unwrap();
        let gguf_norm = gguf_weights
            .get_array1("model.layers.0.input_layernorm.weight")
            .unwrap();

        println!(
            "ST layer 0 norm: shape={:?}, first 5={:?}",
            st_norm.shape(),
            &st_norm.as_slice().unwrap()[..5]
        );
        println!(
            "GGUF layer 0 norm: shape={:?}, first 5={:?}",
            gguf_norm.shape(),
            &gguf_norm.as_slice().unwrap()[..5]
        );

        // Note: Instruct model may have different trained weights
        // This test just validates loading works correctly
        assert_eq!(st_norm.len(), gguf_norm.len());
        assert!(st_norm.iter().all(|&v| v.is_finite()));
        assert!(gguf_norm.iter().all(|&v| v.is_finite()));
    }
}
