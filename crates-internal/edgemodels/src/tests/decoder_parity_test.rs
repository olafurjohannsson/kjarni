
use crate::sentence_encoder::SentenceEncoder;
use crate::text_generation::{TextGenerator, Gpt2Config};
use edgetransformers::models::base::{GenerationConfig, SamplingStrategy};
use anyhow::Result;
use edgetransformers::gpu_context::WgpuContext;
use edgetransformers::models::ModelType;
use edgetransformers::models::{EncoderLanguageModel, LanguageModel, DecoderLanguageModel};
use edgetransformers::traits::Device;
use ndarray::Array2;
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
        sampling_strategy: SamplingStrategy::Greedy,
        ..Default::default()
    };

    // --- 2. Generate with CPU Backend ---
    println!("\n[1/2] Generating text with CPU backend...");
    
    // Create a TextGenerator for the CPU. No WgpuContext is needed.
    let cpu_generator = TextGenerator::from_registry(
        model_type,
        None,
        Device::Cpu,
        None,
    ).await?;
    
    let cpu_generated_text = cpu_generator.generate(prompt, &config).await?;
    println!("- CPU Output: '{}'", cpu_generated_text);
    
    // --- 3. Generate with GPU Backend ---
    println!("\n[2/2] Generating text with GPU backend...");
    
    // Create a WgpuContext and a TextGenerator for the GPU.
    let context = Arc::new(edgetransformers::WgpuContext::new().await);
    let gpu_generator = TextGenerator::from_registry(
        model_type,
        None, 
        Device::Wgpu,
        Some(context), 
    ).await?;
    
    let gpu_generated_text = gpu_generator.generate(prompt, &config).await?;
    println!("- GPU Output: '{}'", gpu_generated_text);

    // --- 4. Assert Equivalence ---
    println!("\nComparing outputs...");
    assert_eq!(
        cpu_generated_text,
        gpu_generated_text,
        "The final generated text from CPU and GPU backends did not match!"
    );

    println!("\n✅ Full text generation parity test passed!");
    
    Ok(())
}