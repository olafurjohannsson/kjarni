use crate::sentence_encoder::SentenceEncoder;
use crate::text_generation::{Gpt2Config, Gpt2Model};
use crate::generation::Generator;
use anyhow::Result;
use edgetransformers::cache::{Cache, CpuKVCache, GpuKVCache};
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::decoder::{CpuTransformerDecoder, GpuTransformerDecoder};
use edgetransformers::gpu_context::WgpuContext;
use edgetransformers::models::ModelType;
use edgetransformers::models::base::{GenerationConfig, DecodingStrategy};
use edgetransformers::models::{DecoderLanguageModel, EncoderLanguageModel, LanguageModel};
use edgetransformers::traits::Device;
use edgetransformers::traits::{Decoder, DecoderArchitecture, DecoderOutput};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array, Array2, Array3, Array4, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::path::PathBuf;
use std::sync::Arc;

const TOLERANCE: f32 = 1e-3; // Allow small numerical differences

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

fn compare_vectors(name: &str, cpu: &[f32], gpu: &[f32], tolerance: f32) -> bool {
    println!("\n=== Comparing: {} ===", name);

    if cpu.len() != gpu.len() {
        println!("❌ Length mismatch: CPU {} vs GPU {}", cpu.len(), gpu.len());
        return false;
    }

    // Statistics
    let cpu_min = cpu.iter().cloned().fold(f32::INFINITY, f32::min);
    let cpu_max = cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let cpu_mean = cpu.iter().sum::<f32>() / cpu.len() as f32;

    let gpu_min = gpu.iter().cloned().fold(f32::INFINITY, f32::min);
    let gpu_max = gpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let gpu_mean = gpu.iter().sum::<f32>() / gpu.len() as f32;

    println!(
        "CPU: Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        cpu_min, cpu_max, cpu_mean
    );
    println!(
        "GPU: Min: {:.6}, Max: {:.6}, Mean: {:.6}",
        gpu_min, gpu_max, gpu_mean
    );

    // Element-wise comparison
    let mut max_diff = 0.0f32;
    let mut num_mismatches = 0;
    let mut sum_abs_diff = 0.0f32;

    for (i, (&c, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        sum_abs_diff += diff;

        if diff > max_diff {
            max_diff = diff;
        }

        if diff > tolerance {
            num_mismatches += 1;
            if num_mismatches <= 10 {
                println!(
                    "  Mismatch at [{}]: CPU={:.6}, GPU={:.6}, diff={:.6}",
                    i, c, g, diff
                );
            }
        }
    }

    let mean_abs_diff = sum_abs_diff / cpu.len() as f32;
    let cosine_sim = cosine_similarity(cpu, gpu);

    println!("Max diff: {:.6}", max_diff);
    println!("Mean abs diff: {:.6}", mean_abs_diff);
    println!("Cosine similarity: {:.6}", cosine_sim);
    println!(
        "Mismatches (>{:.1e}): {} / {}",
        tolerance,
        num_mismatches,
        cpu.len()
    );

    // First 10 values
    println!("CPU first 10: {:?}", &cpu[..10.min(cpu.len())]);
    println!("GPU first 10: {:?}", &gpu[..10.min(gpu.len())]);

    if num_mismatches == 0 {
        println!("✅ PASS");
        true
    } else {
        println!("❌ FAIL ({} mismatches)", num_mismatches);
        false
    }
}

#[tokio::test]
async fn test_full_text_generation_parity() -> Result<()> {
    println!("\n--- Full End-to-End CPU vs. GPU Parity Test ---");

    // --- 1. Common Setup ---
    let model_type = ModelType::DistilGpt2;
    let prompt = "Alan Turing was a"; // Use a slightly shorter prompt for faster testing
    let config = GenerationConfig {
        max_new_tokens: Some(3), // Generate a small number of tokens to keep the test fast
        strategy: DecodingStrategy::Greedy,
        ..Default::default()
    };

    // --- 2. Generate with CPU Backend ---
    println!("\n[1/2] Generating text with CPU backend...");

    // Create a TextGenerator for the CPU. No WgpuContext is needed.
    let cpu_generator = Gpt2Model::from_registry(model_type, None, Device::Cpu, None).await?;
    let cpu_gen = Generator::new(Box::new(cpu_generator));
    let cpu_generated_text = cpu_gen.generate(prompt, &config).await?;
    println!("- CPU Output: '{}'", cpu_generated_text);

    // --- 3. Generate with GPU Backend ---
    println!("\n[2/2] Generating text with GPU backend...");

    // Create a WgpuContext and a TextGenerator for the GPU.
    let context = Arc::new(edgetransformers::WgpuContext::new().await?);
    let gpu_generator =
        Gpt2Model::from_registry(model_type, None, Device::Wgpu, Some(context)).await?;

    let gpu_gen = Generator::new(Box::new(gpu_generator));

    let gpu_generated_text = gpu_gen.generate(prompt, &config).await?;
    println!("- GPU Output: '{}'", gpu_generated_text);

    // --- 4. Assert Equivalence ---
    println!("\nComparing outputs...");
    assert_eq!(
        cpu_generated_text, gpu_generated_text,
        "The final generated text from CPU and GPU backends did not match!"
    );

    println!("\n✅ Full text generation parity test passed!");

    Ok(())
}

/// A helper function to compare two tensors for approximate equality.
fn assert_tensors_approx_equal(a: &Array3<f32>, b: &Array3<f32>, tolerance: f32) {
    assert_eq!(a.shape(), b.shape(), "Tensor shapes do not match");
    for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (val_a - val_b).abs() < tolerance,
            "Tensor values differ at index {}: a={}, b={}",
            i,
            val_a,
            val_b
        );
    }
    println!("✓ Tensors are approximately equal.");
}

/// This is the most critical test. It ensures that for a given set of weights
/// and inputs, the CPU and GPU decoders produce nearly identical outputs.
/// It tests both the priming (prompt processing) and a single generation step.
#[tokio::test]
async fn test_full_forward_pass_parity() -> Result<()> {
    // --- 1. Arrange ---
    let model_dir = PathBuf::from("/home/olafurj/.cache/edgegpt/distilgpt2/");
    let weights = ModelWeights::new(&model_dir)?;
    let mut config1 = serde_json::from_str::<Gpt2Config>(&weights.config_json)?;
    config1.set_model_type("distilgpt2".to_string());
    let config: Arc<dyn DecoderArchitecture + Send + Sync> = Arc::new(config1);
    let context = Arc::new(WgpuContext::new().await?);
    let cpu_decoder =
        TransformerDecoder::Cpu(CpuTransformerDecoder::new(&weights, config.clone(), None)?);
    let gpu_decoder = TransformerDecoder::Gpu(GpuTransformerDecoder::new(
        &weights,
        config.clone(),
        context.clone(),
    )?);

    let batch_size = 1;
    let prompt_len = 11;
    let max_len = 32;
    let tolerance = 1e-3;

    let input_ids: Array2<u32> = Array::random((batch_size, prompt_len), Uniform::new(0, 50256));

    // --- 2. Act & Assert: Priming Pass (cache = None) ---
    println!("\n--- Testing Priming Pass Parity ---");

    let mut full_attention_mask = Array2::zeros((batch_size, max_len));
    full_attention_mask
        .slice_mut(s![.., 0..prompt_len])
        .fill(1.0);

    // When cache is None, BOTH decoders expect a mask sliced to the input length.
    let sliced_mask_priming = full_attention_mask.slice(s![.., 0..prompt_len]).to_owned();

    let cpu_output_priming = cpu_decoder
        .forward(&input_ids, &sliced_mask_priming, None)
        .await?;
    let gpu_output_priming = gpu_decoder
        .forward(&input_ids, &sliced_mask_priming, None) // ✅ FIX 1: Use sliced mask
        .await?;

    assert_tensors_approx_equal(
        &cpu_output_priming.last_hidden_state,
        &gpu_output_priming.last_hidden_state,
        tolerance,
    );

    // --- 3. Act & Assert: Generation Step (cache = Some) ---
    println!("\n--- Testing Generation Step Parity ---");

    // Set up caches
    let mut cpu_cache = CpuKVCache::new(
        config.num_hidden_layers(),
        batch_size,
        max_len,
        config.hidden_size(),
    );
    let mut gpu_cache = GpuKVCache::new(
        &context,
        config.num_hidden_layers(),
        batch_size,
        config.num_attention_heads(),
        config.hidden_size() / config.num_attention_heads(),
        max_len,
    )?;

    // Manually "prime" the caches.
    // The CPU still needs a sliced mask.
    cpu_decoder
        .forward(&input_ids, &sliced_mask_priming, Some(&mut cpu_cache))
        .await?;

    // ✅ FIX 2: When priming WITH a cache, the GPU expects the FULL physical mask.
    gpu_decoder
        .forward(&input_ids, &full_attention_mask, Some(&mut gpu_cache))
        .await?;

    // Prepare inputs for a single new token
    let next_token_id = Array2::from_elem((batch_size, 1), 500);
    let current_len = prompt_len;
    full_attention_mask[[0, current_len]] = 1.0;

    // For the actual generation step, CPU gets a sliced mask, GPU gets the full mask.
    let cpu_mask_gen = full_attention_mask
        .slice(s![.., 0..current_len + 1])
        .to_owned();
    let gpu_mask_gen = full_attention_mask.clone();

    let cpu_output_gen = cpu_decoder
        .forward(&next_token_id, &cpu_mask_gen, Some(&mut cpu_cache))
        .await?;
    let gpu_output_gen = gpu_decoder
        .forward(&next_token_id, &gpu_mask_gen, Some(&mut gpu_cache))
        .await?;

    assert_tensors_approx_equal(
        &cpu_output_gen.last_hidden_state,
        &gpu_output_gen.last_hidden_state,
        tolerance,
    );

    Ok(())
}
