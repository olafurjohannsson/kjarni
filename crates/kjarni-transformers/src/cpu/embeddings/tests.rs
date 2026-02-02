#[cfg(test)]
mod embeddings_tests {
    use super::*;
    use crate::{EmbeddingConfig, EmbeddingData, LoadedEmbeddings};
    use crate::gpu::{GpuTensor, GpuTensorPool};
    use crate::models::base::ModelInput; // <--- Using your actual input enum
    use crate::tensor::{CpuTensor, DType};
    use crate::weights::ModelWeights;
    use crate::{Embeddings, WgpuContext};
    use anyhow::Result;
    use ndarray::{Array2, Array3, ArrayView2, arr2, s};
    use std::path::Path;
    use std::sync::Arc;

    // =========================================================================
    //  Helpers & Mocks
    // =========================================================================

    fn create_dummy_weights(
        tensors: Vec<(&str, Vec<f32>, Vec<usize>)>,
    ) -> (tempfile::TempDir, ModelWeights) {
        use safetensors::serialize;
        use safetensors::tensor::{Dtype, TensorView};
        use std::io::Write;

        let dir = tempfile::TempDir::new().unwrap();
        let model_path = dir.path().join("model.safetensors");
        let config_json = r#"{ "hidden_size": 64, "vocab_size": 100 }"#;
        std::fs::write(dir.path().join("config.json"), config_json).unwrap();

        let mut buffers = Vec::new();
        for (name, data, shape) in tensors {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            buffers.push((name.to_string(), bytes, shape));
        }

        let mut data_map = std::collections::HashMap::new();
        for (name, bytes, shape) in &buffers {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap();
            data_map.insert(name.clone(), view);
        }

        let serialized = serialize(&data_map, &None).unwrap();
        std::fs::write(&model_path, &serialized).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        (dir, weights)
    }

    fn assert_tensors_are_close(a: &Array3<f32>, b: &Array3<f32>, epsilon: f32) {
        assert_eq!(a.shape(), b.shape(), "Array shapes do not match");
        for (val_a, val_b) in a.iter().zip(b.iter()) {
            assert!(
                (val_a - val_b).abs() < epsilon,
                "Values differ: {} vs {} (diff {})",
                val_a,
                val_b,
                (val_a - val_b).abs()
            );
        }
    }

    // =========================================================================
    //  Unit Tests: CPU Math Logic (Verify Correctness)
    // =========================================================================

    #[test]
    fn test_cpu_embeddings_math() {
        // 1. Setup Data
        let word_emb = Array2::<f32>::from_elem((10, 4), 1.0); // All 1.0
        let pos_emb = Array2::<f32>::from_elem((10, 4), 0.5); // All 0.5

        let embeddings =
            Embeddings::new(EmbeddingData::F32(Arc::new(word_emb)), Some(pos_emb), None);

        let input_ids = Array2::<u32>::zeros((1, 2)); // [0, 0]

        // 2. Run Forward
        // 1.0 (Word) + 0.5 (Pos) = 1.5
        let output = embeddings.forward(&input_ids, None, 0, false);

        assert_eq!(output[[0, 0, 0]], 1.5);
        assert_eq!(output[[0, 1, 0]], 1.5);
    }

    #[test]
    fn test_position_embedding_broadcasting_batch() {
        // Setup: word embeddings = 1.0, position embeddings = 0.5
        let word_emb = Array2::<f32>::from_elem((10, 4), 1.0); // 10 words, hidden 4
        let pos_emb = Array2::<f32>::from_elem((10, 4), 0.5);  // 10 positions, hidden 4

        let embeddings =
            Embeddings::new(EmbeddingData::F32(Arc::new(word_emb)), Some(pos_emb), None);

        let input_ids = Array2::<u32>::zeros((2, 3)); // batch 2, seq 3

        // Forward pass
        let output = embeddings.forward(&input_ids, None, 0, false);

        // Each element = word (1.0) + pos (0.5) = 1.5
        for b in 0..2 {
            for s in 0..3 {
                for h in 0..4 {
                    assert_eq!(output[[b, s, h]], 1.5);
                }
            }
        }
    }
    #[test]
    fn test_position_embedding_with_offset() {
        // Setup: word embeddings = 1.0, position embeddings = increasing values
        let word_emb = Array2::<f32>::from_elem((10, 2), 1.0); 
        let pos_emb = Array2::<f32>::from_shape_fn((10, 2), |(i, j)| i as f32 * 0.1 + j as f32);

        let embeddings =
            Embeddings::new(EmbeddingData::F32(Arc::new(word_emb)), Some(pos_emb), None);

        let input_ids = Array2::<u32>::zeros((1, 4)); // batch 1, seq 4

        // Use position_offset = 2 → slice positions 2..6 (but max 10)
        let output = embeddings.forward(&input_ids, None, 2, false);

        // Check first sequence position: word=1.0 + pos_emb[2] = 1.0 + 0.2, 1.0 + 1.2
        assert_eq!(output[[0, 0, 0]], 1.0 + 0.2);
        assert_eq!(output[[0, 0, 1]], 1.0 + 1.2);

        // Check last sequence position: word=1.0 + pos_emb[5] = 1.0 + 0.5, 1.0 + 1.5
        assert_eq!(output[[0, 3, 0]], 1.0 + 0.5);
        assert_eq!(output[[0, 3, 1]], 1.0 + 1.5);
    }

    // =========================================================================
    //  Integration Tests: LoadedEmbeddings (The Decision Matrix)
    // =========================================================================

    // Scenario 2: Hybrid (GPU Weights, CPU Input)
    // Verifies automatic upload of tokens
    #[tokio::test]
    async fn test_scenario_hybrid_gpu_weights() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (_dir, weights) = create_dummy_weights(vec![("w", vec![2.0; 64], vec![64, 1])]);
        let config = EmbeddingConfig::builder("w", 1).build();

        let loaded = LoadedEmbeddings::new(Some(&context), &weights, config, false, true, None)?;

        // CPU Input
        let input_cpu = arr2(&[[0u32]]);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        let result = loaded.embed(
            &mut encoder,
            &mut pool,
            ModelInput::from_array(input_cpu.view()), // CPU View
            None,
            0,
        )?;

        context.queue.submit(Some(encoder.finish()));
        let output = result.to_ndarray_3d::<f32>().await?;

        assert_eq!(output[[0, 0, 0]], 2.0);
        Ok(())
    }

    #[test]
    fn test_scaling_embeddings() {
        let hidden_size = 4;
        let word_emb = Array2::<f32>::from_elem((5, hidden_size), 1.0);
        let embeddings = Embeddings::new(EmbeddingData::F32(Arc::new(word_emb)), None, None);

        let input_ids = Array2::<u32>::zeros((1, 2));

        let output = embeddings.forward(&input_ids, None, 0, true); // scaling enabled
        let scale = (hidden_size as f32).sqrt();

        for b in 0..1 {
            for s in 0..2 {
                for h in 0..hidden_size {
                    assert_eq!(output[[b, s, h]], 1.0 * scale);
                }
            }
        }
    }

    #[test]
    fn test_token_type_embeddings() {
        let hidden_size = 3;
        let word_emb = Array2::<f32>::from_elem((5, hidden_size), 1.0);
        let token_type_emb = Array2::<f32>::from_shape_fn((2, hidden_size), |(i, j)| i as f32 + 0.1 * j as f32);

        let embeddings = Embeddings::new(
            EmbeddingData::F32(Arc::new(word_emb)),
            None,
            Some(token_type_emb),
        );

        let input_ids = Array2::<u32>::zeros((1, 2));
        let type_ids = Array2::<u32>::from_elem((1, 2), 1); // Use second row

        let output = embeddings.forward(&input_ids, Some(&type_ids), 0, false);

        // Expected: word=1.0 + token_type_emb[1]
        for b in 0..1 {
            for s in 0..2 {
                for h in 0..hidden_size {
                    let expected = 1.0 + (1.0 + 0.1 * h as f32);
                    assert_eq!(output[[b, s, h]], expected);
                }
            }
        }
    }

    #[test]
    fn test_batch_sequence_broadcasting() {
        let word_emb = Array2::<f32>::from_elem((5, 2), 1.0);
        let pos_emb = Array2::<f32>::from_elem((5, 2), 0.5);

        let embeddings = Embeddings::new(EmbeddingData::F32(Arc::new(word_emb)), Some(pos_emb), None);

        let input_ids = Array2::<u32>::zeros((3, 4)); // batch 3, seq 4
        let output = embeddings.forward(&input_ids, None, 0, false);

        for b in 0..3 {
            for s in 0..4 {
                for h in 0..2 {
                    assert_eq!(output[[b, s, h]], 1.5);
                }
            }
        }
    }

    #[test]
    fn test_position_offset_clamping() {
        let word_emb = Array2::<f32>::from_elem((5, 2), 1.0);
        let pos_emb = Array2::<f32>::from_elem((3, 2), 0.5); // shorter than sequence

        let embeddings = Embeddings::new(EmbeddingData::F32(Arc::new(word_emb)), Some(pos_emb), None);
        let input_ids = Array2::<u32>::zeros((1, 4));
        
        let output = embeddings.forward(&input_ids, None, 1, false); // offset 1

        // Only positions 1..3 are added (2 positions)
        assert_eq!(output[[0, 0, 0]], 1.0 + 0.5); // first seq position added
        assert_eq!(output[[0, 1, 0]], 1.0 + 0.5); // second seq position added
        assert_eq!(output[[0, 2, 0]], 1.0);       // beyond pos_emb -> only word embedding
        assert_eq!(output[[0, 3, 0]], 1.0);       // beyond pos_emb
    }


    // Scenario 1: Pure GPU
    // Weights: GPU
    // Input:   ModelInput::TokensGpu
    // Action:  Compute on GPU
    #[tokio::test]
    async fn test_scenario_pure_gpu() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (_dir, weights) = create_dummy_weights(vec![("w", vec![1.0; 100], vec![100, 1])]);
        let config = EmbeddingConfig::builder("w", 1).build();

        let loaded = LoadedEmbeddings::new(Some(&context), &weights, config, false, true, None)?;

        // Prepare GPU Input
        let input_cpu = arr2(&[[0u32, 1]]);
        let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        let result = loaded.embed(
            &mut encoder,
            &mut pool,
            ModelInput::TokensGpu(&input_gpu),
            None,
            0,
        )?;

        context.queue.submit(Some(encoder.finish()));
        let output = result.to_ndarray_3d::<f32>().await?;

        assert_eq!(output[[0, 0, 0]], 1.0);
        Ok(())
    }

    // Scenario 2: Hybrid Upload
    // Weights: GPU
    // Input:   ModelInput::TokensCpu
    // Action:  Upload Tokens -> Compute on GPU
    #[tokio::test]
    async fn test_scenario_hybrid_upload() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (_dir, weights) = create_dummy_weights(vec![("w", vec![2.0; 100], vec![100, 1])]);
        let config = EmbeddingConfig::builder("w", 1).build();

        let loaded = LoadedEmbeddings::new(Some(&context), &weights, config, false, true, None)?;

        let input_cpu = arr2(&[[0u32, 1]]); // Array2

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        let result = loaded.embed(
            &mut encoder,
            &mut pool,
            ModelInput::from_array(input_cpu.view()), // Pass View
            None,
            0,
        )?;

        context.queue.submit(Some(encoder.finish()));
        let output = result.to_ndarray_3d::<f32>().await?;

        assert_eq!(output[[0, 0, 0]], 2.0);
        Ok(())
    }

    // Scenario 3: CPU Offload
    // Weights: CPU (No GPU weights loaded!)
    // Input:   ModelInput::TokensCpu
    // Action:  Compute on CPU -> Upload Result to GPU
    #[tokio::test]
    async fn test_scenario_cpu_offload() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (_dir, weights) = create_dummy_weights(vec![("w", vec![3.0; 100], vec![100, 1])]);
        let config = EmbeddingConfig::builder("w", 1).build();

        let loaded = LoadedEmbeddings::new(
            Some(&context),
            &weights,
            config,
            true,  // Load CPU
            false, // NO GPU
            None,
        )?;

        let input_cpu = arr2(&[[0u32]]);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        let result = loaded.embed(
            &mut encoder,
            &mut pool,
            ModelInput::from_array(input_cpu.view()),
            None,
            0,
        )?;

        context.queue.submit(Some(encoder.finish()));
        let output = result.to_ndarray_3d::<f32>().await?;

        assert_eq!(output[[0, 0, 0]], 3.0);
        Ok(())
    }

    // Scenario 4: Hidden State Passthrough (Optimization)
    // Weights: Irrelevant
    // Input:   ModelInput::HiddenGpu
    // Action:  Return as is
    #[tokio::test]
    async fn test_scenario_hidden_passthrough() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (_dir, weights) = create_dummy_weights(vec![("w", vec![0.0], vec![1, 1])]);
        let config = EmbeddingConfig::builder("w", 1).build();
        let loaded = LoadedEmbeddings::new(Some(&context), &weights, config, true, true, None)?;

        // Fake pre-computed hidden states
        let hidden_cpu = Array3::<f32>::from_elem((1, 1, 1), 99.0);
        let hidden_gpu = GpuTensor::from_ndarray(&context, &hidden_cpu)?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        let result = loaded.embed(
            &mut encoder,
            &mut pool,
            ModelInput::from_gpu_hidden(&hidden_gpu),
            None,
            0,
        )?;

        // Should return the exact same tensor ID/Buffer if optimized,
        // or at least same data
        let output = result.to_ndarray_3d::<f32>().await?;
        assert_eq!(output[[0, 0, 0]], 99.0);
        Ok(())
    }

    // Scenario 5: Full BERT (Token Types)
    // Weights: GPU
    // Input:   TokensGpu + TypesGpu
    #[tokio::test]
    async fn test_scenario_token_types() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (_dir, weights) = create_dummy_weights(vec![
            ("w", vec![1.0; 50], vec![50, 1]), // Word = 1.0
            ("t", vec![5.0; 2], vec![2, 1]),   // Type = 5.0
        ]);
        let config = EmbeddingConfig::builder("w", 1).type_embedding("t").build();

        let loaded = LoadedEmbeddings::new(Some(&context), &weights, config, false, true, None)?;

        let input_cpu = arr2(&[[0u32]]);
        let types_cpu = arr2(&[[0u32]]);
        let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
        let types_gpu = GpuTensor::from_ndarray(&context, &types_cpu)?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        // Pass ModelInput::TokensGpu for both
        let result = loaded.embed(
            &mut encoder,
            &mut pool,
            ModelInput::from_gpu_tokens(&input_gpu),
            Some(ModelInput::from_gpu_tokens(&types_gpu)),
            0,
        )?;

        context.queue.submit(Some(encoder.finish()));
        let output = result.to_ndarray_3d::<f32>().await?;

        // 1.0 (Word) + 5.0 (Type) = 6.0
        assert_eq!(output[[0, 0, 0]], 6.0);
        Ok(())
    }

    // Scenario 6: Pure CPU Execution (Decoder Backend Use Case)
    // This tests the `embed_cpu` method specifically used by CpuDecoderBackend
    #[tokio::test]
    async fn test_scenario_pure_cpu_backend() -> Result<()> {
        // Note: No WgpuContext needed!
        let (_dir, weights) = create_dummy_weights(vec![("w", vec![7.0; 100], vec![100, 1])]);
        let config = EmbeddingConfig::builder("w", 1).build();

        // Load CPU only
        let loaded = LoadedEmbeddings::new(None, &weights, config, true, false, None)?;

        let input_tokens = arr2(&[[0u32, 1]]);

        // Direct CPU call (no command encoder needed)
        let output = loaded.embed_cpu(&input_tokens, None, 0)?;

        assert_eq!(output[[0, 0, 0]], 7.0);
        Ok(())
    }
    // Scenario 5: Full BERT Parity (GPU vs CPU)
    // Verifies complex logic (Word + Pos + Type) matches on both devices
    #[tokio::test]
    async fn test_gpu_cpu_parity_bert() -> Result<()> {
        let context = WgpuContext::new().await?;
        let hidden = 32;
        let vocab = 50;

        // Create random-ish data
        let word_data: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32) * 0.01).collect();
        let type_data = vec![0.5f32; 2 * hidden];

        let (_dir, weights) = create_dummy_weights(vec![
            ("w", word_data, vec![vocab, hidden]),
            ("t", type_data, vec![2, hidden]),
        ]);

        let config = EmbeddingConfig::builder("w", hidden)
            .type_embedding("t")
            .build();

        // Load BOTH
        let loaded = LoadedEmbeddings::new(Some(&context), &weights, config, true, true, None)?;

        let input_ids = arr2(&[[1u32, 5]]);
        let type_ids = arr2(&[[0u32, 1]]);

        // 1. Run CPU
        let cpu_out = loaded.embed_cpu(&input_ids, Some(&type_ids), 0)?;

        // 2. Run GPU
        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        let gpu_tensor = loaded.embed(
            &mut encoder,
            &mut pool,
            ModelInput::from_array(input_ids.view()),
            Some(ModelInput::from_array(type_ids.view())), // Pass CPU types, should auto-upload
            0,
        )?;

        context.queue.submit(Some(encoder.finish()));
        let gpu_out = gpu_tensor.to_ndarray_3d::<f32>().await?;

        // 3. Compare
        assert_tensors_are_close(&cpu_out, &gpu_out, 1e-5);
        Ok(())
    }
    #[test]
    fn test_q8_0_lifecycle_correctness() {
        // 1. Setup: Create known F32 weights
        let hidden_size = 32; // Must be multiple of 32 for Q8_0
        let vocab_size = 4;

        // Create a pattern: Token 0 = 1.0, Token 1 = 2.0, etc.
        let mut word_data = Vec::new();
        for i in 0..vocab_size {
            word_data.extend(vec![i as f32; hidden_size]);
        }

        let (_dir, weights) = create_dummy_weights(vec![(
            "q8.weight",
            word_data.clone(),
            vec![vocab_size, hidden_size],
        )]);

        // 2. Load: Request Q8_0 quantization
        let embeddings = Embeddings::from_weights(
            &weights,
            "q8.weight",
            None,
            None,
            Some(DType::Q8_0), // <--- Force Q8_0
        )
        .expect("Failed to load/quantize Q8_0");

        // 3. Verify Internal Storage (Optional reflection check)
        match embeddings.word_embeddings {
            EmbeddingData::Q8_0(_) => println!("Confirmed loaded as Q8_0"),
            _ => panic!("Failed to load as Q8_0"),
        }

        // 4. Run Forward (Triggers dequantization)
        let input_ids = arr2(&[[0u32, 1, 3]]); // Skip 2
        let output = embeddings.forward(&input_ids, None, 0, false);

        // 5. Assert Values
        // Q8_0 is lossy but precise for simple integers like 1.0, 2.0
        // Token 0 -> 0.0
        assert!((output[[0, 0, 0]] - 0.0).abs() < 1e-2);
        // Token 1 -> 1.0
        assert!((output[[0, 1, 0]] - 1.0).abs() < 1e-2);
        // Token 3 -> 3.0
        assert!((output[[0, 2, 0]] - 3.0).abs() < 1e-2);
    }

    #[test]
    fn test_bf16_lifecycle_correctness() {
        let hidden_size = 16;
        let vocab_size = 2;
        let word_data = vec![0.5f32; vocab_size * hidden_size]; // All 0.5

        let (_dir, weights) = create_dummy_weights(vec![(
            "bf.weight",
            word_data,
            vec![vocab_size, hidden_size],
        )]);

        // Request BF16 conversion on load
        let embeddings =
            Embeddings::from_weights(&weights, "bf.weight", None, None, Some(DType::BF16))
                .expect("Failed to load BF16");

        match embeddings.word_embeddings {
            EmbeddingData::BF16(_) => println!("Confirmed loaded as BF16"),
            _ => panic!("Failed to load as BF16"),
        }

        let input_ids = arr2(&[[0u32, 1]]);
        let output = embeddings.forward(&input_ids, None, 0, false);

        // BF16 precision is low, but 0.5 is exactly representable
        assert!((output[[0, 0, 0]] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_scaling_and_position_offset_logic() {
        let hidden_size = 4;
        let vocab_size = 2;

        // Word = 1.0
        let word_data = vec![1.0f32; vocab_size * hidden_size];
        // Pos = [0.1, 0.2, 0.3, ...]
        let pos_data = vec![
            0.1f32, 0.1, 0.1, 0.1, // Pos 0
            0.2f32, 0.2, 0.2, 0.2, // Pos 1
            0.3f32, 0.3, 0.3, 0.3,
        ]; // Pos 2

        let (_dir, weights) = create_dummy_weights(vec![
            ("w", word_data, vec![vocab_size, hidden_size]),
            ("p", pos_data, vec![3, hidden_size]),
        ]);

        let embeddings = Embeddings::from_weights(&weights, "w", Some("p"), None, None).unwrap();

        let input_ids = arr2(&[[0u32]]);

        // CASE 1: Scale = true, Offset = 0
        // Expected: (1.0 * sqrt(4)) + 0.1 (Pos 0) = 2.0 + 0.1 = 2.1
        let out1 = embeddings.forward(&input_ids, None, 0, true);
        assert!((out1[[0, 0, 0]] - 2.1).abs() < 1e-5);

        // CASE 2: Scale = false, Offset = 1
        // Expected: 1.0 + 0.2 (Pos 1) = 1.2
        let out2 = embeddings.forward(&input_ids, None, 1, false);
        assert!((out2[[0, 0, 0]] - 1.2).abs() < 1e-5);
    }

    #[test]
    fn test_batch_processing_correctness() {
        // Verify batch > 1 works for Q8_0
        let hidden_size = 32;
        let vocab_size = 10;
        let word_data = vec![1.0f32; vocab_size * hidden_size];

        let (_dir, weights) =
            create_dummy_weights(vec![("w", word_data, vec![vocab_size, hidden_size])]);

        let embeddings =
            Embeddings::from_weights(&weights, "w", None, None, Some(DType::Q8_0)).unwrap();

        let input_ids = arr2(&[
            [0u32, 1], // Batch 0
            [2u32, 3], // Batch 1
        ]);

        let output = embeddings.forward(&input_ids, None, 0, false);

        assert_eq!(output.shape(), &[2, 2, 32]);
        // Check random spot in batch 1
        assert!((output[[1, 0, 0]] - 1.0).abs() < 1e-2);
    }

    #[test]
    fn test_cpu_embeddings_basic_math() {
        // 1. Setup Data
        let word_emb = Array2::<f32>::from_elem((10, 4), 1.0); // All 1.0
        let pos_emb = Array2::<f32>::from_elem((10, 4), 0.5); // All 0.5

        let embeddings =
            Embeddings::new(EmbeddingData::F32(Arc::new(word_emb)), Some(pos_emb), None);

        let input_ids = Array2::<u32>::zeros((1, 2)); // [0, 0]

        // 2. Run Forward
        // 1.0 (Word) + 0.5 (Pos) = 1.5
        let output = embeddings.forward(&input_ids, None, 0, false);

        assert_eq!(output[[0, 0, 0]], 1.5);
        assert_eq!(output[[0, 1, 0]], 1.5);
    }

    #[test]
    fn test_scaling_and_offsets() {
        // Gemma style: scale by sqrt(hidden)
        let hidden = 4;
        let word_emb = Array2::<f32>::ones((10, hidden)); // 1.0

        // Position 0 = 0.1, Position 1 = 0.2
        let mut pos_data = Vec::new();
        pos_data.extend(vec![0.1f32; hidden]);
        pos_data.extend(vec![0.2f32; hidden]);
        let pos_emb = Array2::from_shape_vec((2, hidden), pos_data).unwrap();

        let embeddings =
            Embeddings::new(EmbeddingData::F32(Arc::new(word_emb)), Some(pos_emb), None);

        let input_ids = Array2::<u32>::zeros((1, 1));

        // Case 1: No Scale, Offset 0 -> 1.0 + 0.1 = 1.1
        let out1 = embeddings.forward(&input_ids, None, 0, false);
        assert!((out1[[0, 0, 0]] - 1.1).abs() < 1e-6);

        // Case 2: Scaled, Offset 0 -> (1.0 * 2.0) + 0.1 = 2.1
        let out2 = embeddings.forward(&input_ids, None, 0, true);
        assert!((out2[[0, 0, 0]] - 2.1).abs() < 1e-6);

        // Case 3: No Scale, Offset 1 -> 1.0 + 0.2 = 1.2
        let out3 = embeddings.forward(&input_ids, None, 1, false);
        assert!((out3[[0, 0, 0]] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_q8_0_lifecycle() {
        // Test that we can load F32, quantize to Q8_0, and read back correct values
        let hidden = 32; // Multiple of 32 required
        let vocab = 2;
        // Token 0 = 10.0, Token 1 = -5.0
        let mut data = vec![10.0f32; hidden];
        data.extend(vec![-5.0f32; hidden]);

        let (_dir, weights) = create_dummy_weights(vec![("w", data, vec![vocab, hidden])]);

        // Force Q8_0 quantization
        let embeddings =
            Embeddings::from_weights(&weights, "w", None, None, Some(DType::Q8_0)).unwrap();

        // Verify it's actually Q8
        match embeddings.word_embeddings {
            EmbeddingData::Q8_0(_) => {}
            _ => panic!("Expected Q8_0 data"),
        }

        let input = arr2(&[[0u32, 1]]);
        let output = embeddings.forward(&input, None, 0, false);

        // Check values (Q8 has some loss, but small integers are usually exact)
        assert!((output[[0, 0, 0]] - 10.0).abs() < 0.1);
        assert!((output[[0, 1, 0]] - -5.0).abs() < 0.1);
    }

    #[test]
    fn test_vocab_bounds() {
        let word_emb = Array2::<f32>::zeros((10, 4));
        let embeddings = Embeddings::new(EmbeddingData::F32(Arc::new(word_emb)), None, None);

        let bad_input = arr2(&[[100u32]]);
        // Should NOT panic
        let output = embeddings.forward(&bad_input, None, 0, false);
        // Verify it ignored the input (remained zeros)
        assert_eq!(output[[0, 0, 0]], 0.0);
    }
}
