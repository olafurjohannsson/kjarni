use crate::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use crate::gpu_ops::{GpuTensor, GpuTensorPool};
use crate::traits::{ModelConfig, ModelLayout, ModelMetadata};
use crate::embeddings::Embeddings;
use crate::normalization::Normalization;
// Unified Traits
use crate::WgpuContext;

use anyhow::Result;
use ndarray::{arr2, s, Array2, Array3};

struct TestConfig {
    extra_pos_embeddings: usize,
    scale_embed: bool,
}

// Replaces TransformerConfig and LanguageModelConfig
impl ModelConfig for TestConfig {
    fn model_type(&self) -> &str {
        "test_encoder"
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: 384,
            num_layers: 6,
            num_attention_heads: 12,
            num_kv_heads: 12,
            head_dim: 384 / 12,
            vocab_size: 30522,
            max_seq_len: 512,
            norm_eps: 1e-12,
            activation: crate::activations::Activation::Gelu,
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: self.scale_embed,
            extra_pos_embeddings: self.extra_pos_embeddings,
            is_prenorm: false,
            transpose_ffn_weights: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        ModelLayout {
            token_embedding: "embeddings.word_embeddings.weight".to_string(),
            position_embedding: Some("embeddings.position_embeddings.weight".to_string()),
            token_type_embedding: Some("embeddings.token_type_embeddings.weight".to_string()),
            embedding_norm: Some("embeddings.LayerNorm.weight".to_string()),
            embedding_norm_bias: "embeddings.LayerNorm.bias".to_string(),
            final_norm: "".to_string(),
            lm_head: "".to_string(),

            attn_q: "".to_string(),
            attn_k: "".to_string(),
            attn_v: "".to_string(),
            attn_o: "".to_string(),
            attn_norm: "".to_string(),
            attn_q_bias: String::new(),
            attn_k_bias: String::new(),
            attn_v_bias: String::new(),
            attn_o_bias: String::new(),
            attn_norm_bias: String::new(),

            ffn_gate: None,
            ffn_up: "".to_string(),
            ffn_down: "".to_string(),
            ffn_norm: "".to_string(),
            ffn_up_bias: String::new(),
            ffn_down_bias: String::new(),
            ffn_norm_bias: String::new(),

            cross_attn_q: None,
            cross_attn_k: None,
            cross_attn_v: None,
            cross_attn_o: None,
            cross_attn_norm: None,
        }
    }
}

// Helper for float comparison
fn assert_tensors_are_close(a: &Array3<f32>, b: &Array3<f32>, epsilon: f32) {
    assert_eq!(a.shape(), b.shape(), "Array shapes do not match");
    for (val_a, val_b) in a.iter().zip(b.iter()) {
        assert!(
            (val_a - val_b).abs() < epsilon,
            "Values differ: {} vs {}",
            val_a,
            val_b
        );
    }
}
#[tokio::test]
async fn test_bart_embeddings_golden_parity() -> Result<()> {
    use crate::weights::ModelWeights;
    use ndarray::Array2;
    use std::path::Path;
    let path_str = "/home/olafurj/.cache/kjarni/olafuraron_distilbart-cnn-12-6/";
    let path = Path::new(path_str);
    assert!(path.exists());
    if !path.exists() {
        println!(
            "SKIPPING TEST: Weights not found at {}. Please download them to run golden test.",
            path_str
        );
        return Ok(());
    }

    let weights = ModelWeights::new(path)?;

    // 2. Load Weights (Standard BART naming)
    let word_emb = weights.get_array2("model.shared.weight")?;
    let pos_emb = weights.get_array2("model.encoder.embed_positions.weight")?;

    // BART-specific: No token type embeddings
    let embeddings = Embeddings::new(word_emb, Some(pos_emb), None);

    // 3. Define Inputs (From your Python Checkpoint 1)
    // "Rust is a multi-paradigm..."
    let input_ids_vec = vec![0, 46541, 16, 10, 3228, 12, 5489, 625, 35045, 6];
    let input_ids = Array2::from_shape_vec((1, 10), input_ids_vec)?;

    // 4. Run Forward
    // CONFIG CHECK: extra_pos_embeddings=2, scale_embeddings=False
    let output = embeddings.forward(
        &input_ids, None, 2,     // position_offset (BART starts at 2)
        false, // scale_embeddings (DistilBART is false)
    );

    // 5. Assert against Ground Truth (From your Python Checkpoint 1.5)
    let expected_first_10 = vec![
        0.011993408,
        -0.13934326,
        0.058532715,
        -0.042541504,
        -0.061767578,
        -0.002746582,
        0.0048828125,
        -0.037017822,
        -0.015655518,
        -0.053588867,
    ];

    let actual_slice = output.slice(s![0, 0, 0..10]);
    println!("Golden Test - Actual Output: {:?}", actual_slice);

    for (i, &expected) in expected_first_10.iter().enumerate() {
        let actual = actual_slice[i];
        let diff = (actual - expected).abs();

        // Tolerance 1e-5 is usually safe for F32 comparisons across PyTorch/Rust
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: expected {}, got {} (diff {})",
            i,
            expected,
            actual,
            diff
        );
    }

    println!("✅ Golden Test Passed: Embeddings match PyTorch exactly.");
    Ok(())
}
#[tokio::test]
async fn test_gpu_vs_cpu_embeddings_parity() -> Result<()> {
    // --- 1. Setup Common Data and Config ---
    let context = WgpuContext::new().await?;
    let config = TestConfig {
        extra_pos_embeddings: 2,
        scale_embed: false,
    };

    // Create mock inputs on CPU
    let input_ids_cpu: Array2<u32> = arr2(&[[10, 20, 30], [40, 50, 60]]);
    let token_type_ids_cpu: Array2<u32> = arr2(&[[0, 0, 1], [1, 1, 0]]);

    // Keep the hardcoded path for now as requested
    let p = "/home/olafurj/.cache/edgegpt/sentence-transformers_all-MiniLM-L6-v2/";
    let weights = crate::weights::ModelWeights::new(Path::new(p))?;

    // --- 2. Run CPU Path (Expected Result) ---
    let (word_w, pos_w, type_w) = config.get_embedding_weight_names();
    let token_type_embeddings = if let Some(name) = type_w {
        Some(weights.get_array2(name)?)
    } else {
        None
    };

    let cpu_embeddings = Embeddings::new(
        weights.get_array2(word_w)?,
        Some(weights.get_array2(pos_w)?),
        token_type_embeddings,
    );
    let expected_output = cpu_embeddings.forward(
        &input_ids_cpu,
        Some(&token_type_ids_cpu),
        config.extra_pos_embeddings(),
        config.scale_embeddings(),
    );

    // --- 3. Setup GPU Modules and Inputs ---
    let gpu_embedding_weights = GpuEmbeddingWeights::new(&context, &weights, &config)?;
    let gpu_embeddings = GpuEmbeddings::new(&context)?;

    // Upload inputs
    let input_ids_gpu = GpuTensor::from_ndarray(&context, &input_ids_cpu)?;
    let token_type_ids_gpu = GpuTensor::from_ndarray(&context, &token_type_ids_cpu)?;

    // --- START CORRECTION ---

    // 1. Create the encoder and pool directly for the test.
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(context.clone());

    // 2. Call the encode function with the raw &mut encoder and &mut pool.
    let output_gpu = gpu_embeddings.encode(
        &mut encoder,
        &gpu_embedding_weights,
        &input_ids_gpu,
        Some(&token_type_ids_gpu),
        0, // position_offset
        config,
        &mut pool,
    )?;

    // 3. Submit the work and advance the pool's frame.
    context.queue.submit(Some(encoder.finish()));
    pool.next_frame();

    // --- END CORRECTION ---

    // --- 4. Verify Results ---
    let actual_output = output_gpu.to_ndarray_3d().await?;

    // Using a slightly more relaxed tolerance for embeddings is often wise,
    // as it involves multiple additions which can accumulate small errors.
    assert_tensors_are_close(&expected_output, &actual_output, 1e-5);

    Ok(())
}
#[tokio::test]
async fn test_gpu_vs_cpu_embeddings_parity_no_token_type_ids() -> Result<()> {
    // --- 1. Setup Common Data and Config ---
    let context = WgpuContext::new().await?;
    let config = TestConfig {
        extra_pos_embeddings: 2,
        scale_embed: false,
    }; // Test BART-like settings

    // Create mock inputs on CPU
    let input_ids_cpu: Array2<u32> = arr2(&[[10, 20, 30], [40, 50, 60]]);

    let p = "/home/olafurj/.cache/edgegpt/sentence-transformers_all-MiniLM-L6-v2/";

    // --- 2. Run CPU Path (Expected Result) ---
    let mut weights = crate::weights::ModelWeights::new(Path::new(p))?;
    let (word_w, pos_w, type_w) = config.get_embedding_weight_names();
    let token_type_embeddings = match type_w {
        Some(name) => Some(weights.get_array2(name)?), // Load if present
        None => None,
    };

    let cpu_embeddings = Embeddings::new(
        weights.get_array2(word_w)?,
        Some(weights.get_array2(pos_w)?),
        token_type_embeddings,
    );
    let expected_output = cpu_embeddings.forward(
        &input_ids_cpu,
        None,
        config.extra_pos_embeddings(),
        config.scale_embeddings(),
    );

    let gpu_embedding_weights = GpuEmbeddingWeights::new(&context, &weights, &config)?;
    let gpu_embeddings = GpuEmbeddings::new(&context)?;

    // Upload inputs
    let input_ids_gpu = GpuTensor::from_ndarray(&context, &input_ids_cpu)?;
    let mut pool = GpuTensorPool::new(context.clone());

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let output_gpu = gpu_embeddings.encode(
        &mut encoder,
        &gpu_embedding_weights,
        &input_ids_gpu,
        None,
        0,
        &config,
        &mut pool,
    )?;
    context.queue.submit(Some(encoder.finish()));
    pool.next_frame();

    let actual_output = output_gpu.to_ndarray_3d().await?;

    // --- 4. Verification ---
    assert_tensors_are_close(&expected_output, &actual_output, 1e-4);

    Ok(())
}
#[test]
fn test_embeddings_with_position() {
    // GPT-2 / BERT style
    let word_emb = Array2::ones((100, 64));
    let pos_emb = Array2::ones((512, 64));

    let embeddings = Embeddings::new(word_emb, Some(pos_emb), None);

    let input_ids = Array2::zeros((2, 10));
    let output = embeddings.forward(&input_ids, None, 0, false);

    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
fn test_embeddings_without_position() {
    // LLaMA / RoPE style
    let word_emb = Array2::ones((100, 64));

    let embeddings = Embeddings::new(word_emb, None, None);

    let input_ids = Array2::zeros((2, 10));
    let output = embeddings.forward(&input_ids, None, 0, false);

    assert_eq!(output.shape(), &[2, 10, 64]);
    // Should work without position embeddings
}

#[test]
fn test_embeddings_with_token_types() {
    let word_emb = Array2::ones((100, 64));
    let pos_emb = Array2::ones((512, 64));
    let token_type_emb = Array2::ones((2, 64));

    let embeddings = Embeddings::new(word_emb, Some(pos_emb), Some(token_type_emb));

    let input_ids = Array2::zeros((2, 10));
    let token_type_ids = Array2::zeros((2, 10));
    let output = embeddings.forward(&input_ids, Some(&token_type_ids), 0, false);

    assert_eq!(output.shape(), &[2, 10, 64]);
}

#[test]
#[should_panic(expected = "exceeds max position embeddings")]
fn test_sequence_too_long() {
    let word_emb = Array2::ones((100, 64));
    let pos_emb = Array2::ones((10, 64)); // Only 10 positions

    let embeddings = Embeddings::new(word_emb, Some(pos_emb), None);

    let input_ids = Array2::zeros((2, 20)); // 20 tokens - too long!
    let _ = embeddings.forward(&input_ids, None, 0, false);
}

#[test]
fn test_llama_long_sequence() {
    // LLaMA without position embeddings can handle any length
    let word_emb = Array2::ones((100, 64));

    let embeddings = Embeddings::new(word_emb, None, None);

    let input_ids = Array2::zeros((2, 1000)); // Very long sequence
    let output = embeddings.forward(&input_ids, None, 0, false);

    assert_eq!(output.shape(), &[2, 1000, 64]);
    // Should work fine - RoPE handles position in attention layer
}

#[test]
#[should_panic(expected = "out of vocabulary range")]
fn test_invalid_token_id() {
    let word_emb = Array2::ones((100, 64));
    let pos_emb = Array2::ones((512, 64));

    let embeddings = Embeddings::new(word_emb, Some(pos_emb), None);

    let mut input_ids = Array2::zeros((2, 10));
    input_ids[[0, 0]] = 150 as u32; // Out of vocab range [0, 100)

    let _ = embeddings.forward(&input_ids, None, 0, false);
}
