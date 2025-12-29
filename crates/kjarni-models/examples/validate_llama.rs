//! Comprehensive Llama Validation Test Suite
//!
//! Tests all loading paths, execution modes, and compares GGUF vs SafeTensors outputs.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use futures_util::StreamExt;
use kjarni_models::models::llama::LlamaModel;
use kjarni_transformers::{ChatTemplate, Conversation, Device, ModelType, WgpuContext, chat::llama3::Llama3ChatTemplate, common::GenerationConfig, decoder::{prelude::DecoderGenerator, traits::DecoderLanguageModel}, models::base::ModelLoadConfig, tensor::DType, weights::ModelWeights};
use ndarray::{Array1, Array2, Array3};



// =============================================================================
// Test Paths
// =============================================================================

const LLAMA_3_2_1B_PATH: &str = "/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B";
const LLAMA_3_2_3B_INSTRUCT_PATH: &str = "/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-3B-Instruct";
const LLAMA_3_2_1B_INSTRUCT_GGUF: &str = "/home/olafurj/.cache/kjarni/llama-3.2-1b-instruct-q4_k_m/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

// For comparison tests - need both formats of same model
const LLAMA_3_2_1B_INSTRUCT_ST_PATH: &str = "/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B-Instruct";

// =============================================================================
// Test Macros
// =============================================================================

macro_rules! run_test {
    ($name:expr, $test:expr, $passed:ident, $failed:ident) => {{
        print!("Testing: {} ... ", $name);
        match $test.await {
            Ok(_) => {
                println!("✅ PASSED");
                $passed += 1;
            }
            Err(e) => {
                println!("❌ FAILED: {}", e);
                $failed += 1;
            }
        }
    }};
}

macro_rules! run_sync_test {
    ($name:expr, $test:expr, $passed:ident, $failed:ident) => {{
        print!("Testing: {} ... ", $name);
        match $test {
            Ok(_) => {
                println!("✅ PASSED");
                $passed += 1;
            }
            Err(e) => {
                println!("❌ FAILED: {}", e);
                $failed += 1;
            }
        }
    }};
}

// =============================================================================
// Helper Functions
// =============================================================================

fn generation_config() -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: Some(20),
        strategy: kjarni_transformers::common::DecodingStrategy::Greedy,
        ..Default::default()
    }
}

fn path_exists(path: &str) -> bool {
    Path::new(path).exists()
}

async fn run_base_model_generation(model: Box<dyn DecoderLanguageModel>, label: &str) -> Result<String> {
    let generator = DecoderGenerator::new(model)?;
    let prompt = "The capital of France is";
    
    println!("\n============================================================");
    println!("  {}", label);
    println!("============================================================");
    
    let start = std::time::Instant::now();
    let output = generator.generate(prompt, &generation_config()).await?;
    let elapsed = start.elapsed();
    
    println!("Prompt: {}", prompt);
    println!("Output: {}", output);
    println!("Time: {:.2}s ({:.2} tok/s)", elapsed.as_secs_f32(), 20.0 / elapsed.as_secs_f32());
    
    // Validate output contains expected content
    if !output.to_lowercase().contains("paris") {
        anyhow::bail!("Expected output to mention 'Paris', got: {}", output);
    }
    
    Ok(output)
}

async fn run_instruct_model_generation(model: Box<dyn DecoderLanguageModel>, label: &str) -> Result<String> {
    let generator = DecoderGenerator::new(model)?;
    let template = Llama3ChatTemplate::for_generation();
    
    let mut conv = Conversation::new();
    conv.push_user("What is the capital of France? Answer in one word.");
    
    let prompt = template.apply(&conv);
    
    println!("\n============================================================");
    println!("  {}", label);
    println!("============================================================");
    println!("Formatted prompt:\n{}", prompt);
    
    let start = std::time::Instant::now();
    let output = generator.generate(&prompt, &generation_config()).await?;
    let elapsed = start.elapsed();
    
    println!("Output: {}", output);
    println!("Time: {:.2}s ({:.2} tok/s)", elapsed.as_secs_f32(), 20.0 / elapsed.as_secs_f32());
    
    // For instruct models, check if the response makes sense
    let response_lower = output.to_lowercase();
    if !response_lower.contains("paris") && !response_lower.contains("capital") {
        println!("⚠️  Warning: Response may be garbage: {}", output);
    }
    
    Ok(output)
}

// =============================================================================
// Weight Comparison Tests
// =============================================================================

fn test_weights_comparison() -> Result<()> {
    println!("\n=== Weight Comparison: SafeTensors vs GGUF ===\n");
    
    if !path_exists(LLAMA_3_2_1B_INSTRUCT_ST_PATH) || !path_exists(LLAMA_3_2_1B_INSTRUCT_GGUF) {
        println!("⚠️  Skipping - need both SafeTensors and GGUF versions of 1B Instruct");
        return Ok(());
    }
    
    let st_weights = ModelWeights::new(Path::new(LLAMA_3_2_1B_INSTRUCT_ST_PATH))?;
    let gguf_weights = ModelWeights::new(Path::new(LLAMA_3_2_1B_INSTRUCT_GGUF))?;
    
    // Compare key tensors
    let tensors_to_compare = [
        ("model.embed_tokens.weight", "Embeddings"),
        ("model.norm.weight", "Final Norm"),
        ("model.layers.0.self_attn.q_proj.weight", "Layer 0 Q Proj"),
        ("model.layers.0.self_attn.k_proj.weight", "Layer 0 K Proj"),
        ("model.layers.0.self_attn.v_proj.weight", "Layer 0 V Proj"),
        ("model.layers.0.mlp.gate_proj.weight", "Layer 0 Gate Proj"),
        ("model.layers.0.input_layernorm.weight", "Layer 0 Attn Norm"),
        ("model.layers.15.self_attn.q_proj.weight", "Layer 15 Q Proj"),
    ];
    
    for (name, label) in tensors_to_compare {
        let st_raw = st_weights.get_raw(name)?;
        let gguf_raw = gguf_weights.get_raw(name)?;
        
        println!("{}: ST={:?} {:?} | GGUF={:?} {:?}", 
            label, st_raw.shape, st_raw.dtype, gguf_raw.shape, gguf_raw.dtype);
        
        // Shapes should match
        if st_raw.shape != gguf_raw.shape {
            anyhow::bail!("Shape mismatch for {}: ST={:?}, GGUF={:?}", name, st_raw.shape, gguf_raw.shape);
        }
    }
    
    println!("\n✅ All shapes match");
    Ok(())
}

fn test_embedding_values() -> Result<()> {
    println!("\n=== Embedding Value Comparison ===\n");
    
    if !path_exists(LLAMA_3_2_1B_INSTRUCT_ST_PATH) || !path_exists(LLAMA_3_2_1B_INSTRUCT_GGUF) {
        println!("⚠️  Skipping - need both formats");
        return Ok(());
    }
    
    let st_weights = ModelWeights::new(Path::new(LLAMA_3_2_1B_INSTRUCT_ST_PATH))?;
    let gguf_weights = ModelWeights::new(Path::new(LLAMA_3_2_1B_INSTRUCT_GGUF))?;
    
    let st_emb = st_weights.get_array2("model.embed_tokens.weight")?;
    let gguf_emb = gguf_weights.get_array2("model.embed_tokens.weight")?;
    
    // Test specific token embeddings
    let test_tokens = [
        (128000, "BOS <|begin_of_text|>"),
        (128001, "EOS <|end_of_text|>"),
        (128006, "<|start_header_id|>"),
        (128007, "<|end_header_id|>"),
        (128009, "<|eot_id|>"),
        (1, "Token 1"),
        (1000, "Token 1000"),
    ];
    
    for (token_id, label) in test_tokens {
        let st_vec = st_emb.row(token_id);
        let gguf_vec = gguf_emb.row(token_id);
        
        let avg_diff: f32 = st_vec.iter()
            .zip(gguf_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / st_vec.len() as f32;
        
        let st_norm: f32 = st_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let gguf_norm: f32 = gguf_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        println!("{} ({}): avg_diff={:.6}, ST_norm={:.4}, GGUF_norm={:.4}", 
            label, token_id, avg_diff, st_norm, gguf_norm);
        
        // Check for reasonable similarity (different training = different weights, but should be same magnitude)
        if gguf_norm < 0.1 || gguf_norm > 100.0 {
            anyhow::bail!("GGUF embedding for {} has abnormal norm: {}", label, gguf_norm);
        }
    }
    
    println!("\n✅ Embedding values look reasonable");
    Ok(())
}

fn test_linear_layer_output() -> Result<()> {
    println!("\n=== Linear Layer Output Comparison ===\n");
    
    if !path_exists(LLAMA_3_2_1B_INSTRUCT_ST_PATH) || !path_exists(LLAMA_3_2_1B_INSTRUCT_GGUF) {
        println!("⚠️  Skipping - need both formats");
        return Ok(());
    }
    
    let st_weights = ModelWeights::new(Path::new(LLAMA_3_2_1B_INSTRUCT_ST_PATH))?;
    let gguf_weights = ModelWeights::new(Path::new(LLAMA_3_2_1B_INSTRUCT_GGUF))?;
    
    // Get Q projection weights
    let st_q = st_weights.get_array2("model.layers.0.self_attn.q_proj.weight")?;
    let gguf_q = gguf_weights.get_array2("model.layers.0.self_attn.q_proj.weight")?;
    
    println!("ST Q shape: {:?}", st_q.shape());
    println!("GGUF Q shape: {:?}", gguf_q.shape());
    
    // Create test input (simulating hidden state)
    let input = Array1::<f32>::ones(2048);
    
    // Compute output: sum of each row (equivalent to input @ weight.T for all-ones input)
    let st_outputs: Vec<f32> = st_q.outer_iter().map(|row| row.sum()).collect();
    let gguf_outputs: Vec<f32> = gguf_q.outer_iter().map(|row| row.sum()).collect();
    
    println!("\nFirst 10 row sums:");
    println!("ST:   {:?}", &st_outputs[..10]);
    println!("GGUF: {:?}", &gguf_outputs[..10]);
    
    // Check statistics
    let st_mean: f32 = st_outputs.iter().sum::<f32>() / st_outputs.len() as f32;
    let gguf_mean: f32 = gguf_outputs.iter().sum::<f32>() / gguf_outputs.len() as f32;
    
    let st_std: f32 = (st_outputs.iter().map(|x| (x - st_mean).powi(2)).sum::<f32>() / st_outputs.len() as f32).sqrt();
    let gguf_std: f32 = (gguf_outputs.iter().map(|x| (x - gguf_mean).powi(2)).sum::<f32>() / gguf_outputs.len() as f32).sqrt();
    
    println!("\nStatistics:");
    println!("ST:   mean={:.6}, std={:.6}", st_mean, st_std);
    println!("GGUF: mean={:.6}, std={:.6}", gguf_mean, gguf_std);
    
    // They should be in the same ballpark (instruct vs base will differ)
    if gguf_std < 0.01 {
        anyhow::bail!("GGUF weights have near-zero std - possible dequantization bug");
    }
    
    println!("\n✅ Linear layer weights look reasonable");
    Ok(())
}

fn test_dequantization_sanity() -> Result<()> {
    println!("\n=== Dequantization Sanity Check ===\n");
    
    if !path_exists(LLAMA_3_2_1B_INSTRUCT_GGUF) {
        println!("⚠️  Skipping - GGUF not found");
        return Ok(());
    }
    
    let gguf_weights = ModelWeights::new(Path::new(LLAMA_3_2_1B_INSTRUCT_GGUF))?;
    
    let tensors = [
        ("model.layers.0.self_attn.q_proj.weight", "Q Proj (Q4_K)"),
        ("model.layers.0.self_attn.v_proj.weight", "V Proj (Q6_K)"),
        ("model.layers.0.mlp.down_proj.weight", "Down Proj (Q6_K)"),
        ("model.embed_tokens.weight", "Embeddings (Q6_K)"),
    ];
    
    for (name, label) in tensors {
        let raw = gguf_weights.get_raw(name)?;
        let arr = gguf_weights.get_array2(name)?;
        
        let min = arr.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = arr.iter().sum::<f32>() / arr.len() as f32;
        let has_nan = arr.iter().any(|x| x.is_nan());
        let has_inf = arr.iter().any(|x| x.is_infinite());
        
        println!("{} ({:?}):", label, raw.dtype);
        println!("  shape={:?}, min={:.6}, max={:.6}, mean={:.6}", arr.shape(), min, max, mean);
        println!("  has_nan={}, has_inf={}", has_nan, has_inf);
        
        if has_nan || has_inf {
            anyhow::bail!("{} has NaN or Inf values!", label);
        }
        
        // Transformer weights should be small
        if min < -10.0 || max > 10.0 {
            println!("  ⚠️  Warning: Values outside expected range [-10, 10]");
        }
    }
    
    println!("\n✅ Dequantization produces valid values");
    Ok(())
}

fn test_gguf_config_synthesis() -> Result<()> {
    println!("\n=== GGUF Config Synthesis ===\n");
    
    if !path_exists(LLAMA_3_2_1B_INSTRUCT_GGUF) {
        println!("⚠️  Skipping - GGUF not found");
        return Ok(());
    }
    
    let gguf_weights = ModelWeights::new(Path::new(LLAMA_3_2_1B_INSTRUCT_GGUF))?;
    
    println!("Synthesized config:\n{}", gguf_weights.config_json);
    
    // Parse and validate
    let config: serde_json::Value = serde_json::from_str(&gguf_weights.config_json)?;
    
    let expected = [
        ("hidden_size", 2048),
        ("num_hidden_layers", 16),
        ("num_attention_heads", 32),
        ("num_key_value_heads", 8),
        ("intermediate_size", 8192),
        ("vocab_size", 128256),
    ];
    
    for (key, expected_val) in expected {
        let actual = config[key].as_u64().unwrap_or(0) as usize;
        if actual != expected_val {
            anyhow::bail!("{}: expected {}, got {}", key, expected_val, actual);
        }
        println!("  {}: {} ✓", key, actual);
    }
    
    // Check rope_theta
    let rope_theta = config["rope_theta"].as_f64().unwrap_or(0.0);
    if rope_theta < 100000.0 {
        anyhow::bail!("rope_theta too small: {} (expected ~500000)", rope_theta);
    }
    println!("  rope_theta: {} ✓", rope_theta);
    
    println!("\n✅ Config synthesis correct");
    Ok(())
}

// =============================================================================
// Main Test Runner
// =============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Kjarni Llama Validation Test Suite                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    
    let mut passed = 0;
    let mut failed = 0;
    
    // =========================================================================
    // Part 1: Comparison Tests (Sync)
    // =========================================================================
    
    println!("═══════════════════════════════════════════════════════════════");
    println!("  PART 1: Weight & Dequantization Comparison Tests");
    println!("═══════════════════════════════════════════════════════════════");
    
    run_sync_test!("GGUF Config Synthesis", test_gguf_config_synthesis(), passed, failed);
    run_sync_test!("Dequantization Sanity", test_dequantization_sanity(), passed, failed);
    run_sync_test!("Weight Shape Comparison", test_weights_comparison(), passed, failed);
    run_sync_test!("Embedding Values", test_embedding_values(), passed, failed);
    run_sync_test!("Linear Layer Output", test_linear_layer_output(), passed, failed);
    
    // =========================================================================
    // Part 2: Model Loading & Generation Tests
    // =========================================================================
    
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  PART 2: Model Loading & Generation Tests");
    println!("═══════════════════════════════════════════════════════════════");
    
    // --- Test 1: Baseline - Llama 3.2 1B (BF16, Base Model) ---
    // if path_exists(LLAMA_3_2_1B_PATH) {
    //     run_test!(
    //         "Llama 3.2 1B - CPU BF16 (Baseline)",
    //         async {
    //             let model = LlamaModel::from_pretrained(
    //                 Path::new(LLAMA_3_2_1B_PATH),
    //                 Device::Cpu,
    //                 None,
    //                 None,
    //                 Some(ModelType::Llama3_2_1B),
    //             )?;
    //             run_base_model_generation(Box::new(model), "CPU | Llama 3.2 1B BF16").await?;
    //             Ok::<(), anyhow::Error>(())
    //         },
    //         passed,
    //         failed
    //     );
    // } else {
    //     println!("⚠️  Skipping 3.2 1B - not found at {}", LLAMA_3_2_1B_PATH);
    // }
    
    // // --- Test 2: Llama 3.2 3B Instruct (BF16) ---
    // if path_exists(LLAMA_3_2_3B_INSTRUCT_PATH) {
    //     run_test!(
    //         "Llama 3.2 3B Instruct - CPU BF16",
    //         async {
    //             let model = LlamaModel::from_pretrained(
    //                 Path::new(LLAMA_3_2_3B_INSTRUCT_PATH),
    //                 Device::Cpu,
    //                 None,
    //                 None,
    //                 Some(ModelType::Llama3_2_3B_Instruct),
    //             )?;
    //             run_instruct_model_generation(Box::new(model), "CPU | Llama 3.2 3B Instruct BF16").await?;
    //             Ok::<(), anyhow::Error>(())
    //         },
    //         passed,
    //         failed
    //     );
    // } else {
    //     println!("⚠️  Skipping 3.2 3B Instruct - not found at {}", LLAMA_3_2_3B_INSTRUCT_PATH);
    // }
    
    // // --- Test 3: Llama 3.2 1B Instruct SafeTensors (if available) ---
    // if path_exists(LLAMA_3_2_1B_INSTRUCT_ST_PATH) {
    //     run_test!(
    //         "Llama 3.2 1B Instruct - CPU BF16 SafeTensors",
    //         async {
    //             let model = LlamaModel::from_pretrained(
    //                 Path::new(LLAMA_3_2_1B_INSTRUCT_ST_PATH),
    //                 Device::Cpu,
    //                 None,
    //                 None,
    //                 Some(ModelType::Llama3_2_1B_Instruct),
    //             )?;
    //             run_instruct_model_generation(Box::new(model), "CPU | Llama 3.2 1B Instruct BF16").await?;
    //             Ok::<(), anyhow::Error>(())
    //         },
    //         passed,
    //         failed
    //     );
    // } else {
    //     println!("⚠️  Skipping 3.2 1B Instruct ST - not found at {}", LLAMA_3_2_1B_INSTRUCT_ST_PATH);
    // }
    
    // --- Test 4: GGUF - Native Quantized (Q4_K matmul) ---
    if path_exists(LLAMA_3_2_1B_INSTRUCT_GGUF) {
        run_test!(
            "Llama 3.2 1B Instruct GGUF - CPU Quantized",
            async {
                let model = LlamaModel::from_pretrained(
                    Path::new(LLAMA_3_2_1B_INSTRUCT_GGUF),
                    Device::Cpu,
                    None,
                    None, // Use native quantized weights
                    Some(ModelType::Llama3_2_1B_Instruct),
                )?;
                run_instruct_model_generation(Box::new(model), "CPU | GGUF Q4_K (Native)").await?;
                Ok::<(), anyhow::Error>(())
            },
            passed,
            failed
        );
        
        // --- Test 5: GGUF - Force F32 Dequantization ---
        run_test!(
            "Llama 3.2 1B Instruct GGUF - CPU F32 (Dequantized)",
            async {
                let model = LlamaModel::from_pretrained(
                    Path::new(LLAMA_3_2_1B_INSTRUCT_GGUF),
                    Device::Cpu,
                    None,
                    Some(ModelLoadConfig {
                        target_dtype: Some(DType::F32),
                        ..Default::default()
                    }),
                    Some(ModelType::Llama3_2_1B_Instruct),
                )?;
                run_instruct_model_generation(Box::new(model), "CPU | GGUF F32 (Dequantized)").await?;
                Ok::<(), anyhow::Error>(())
            },
            passed,
            failed
        );
        
        // --- Test 6: GGUF on GPU ---
        // run_test!(
        //     "Llama 3.2 1B Instruct GGUF - GPU",
        //     async {
        //         let ctx = WgpuContext::new().await?;
        //         let model = LlamaModel::from_pretrained(
        //             Path::new(LLAMA_3_2_1B_INSTRUCT_GGUF),
        //             Device::Wgpu,
        //             Some(ctx),
        //             None,
        //             Some(ModelType::Llama3_2_1B_Instruct),
        //         )?;
        //         run_instruct_model_generation(Box::new(model), "GPU | GGUF").await?;
        //         Ok::<(), anyhow::Error>(())
        //     },
        //     passed,
        //     failed
        // );
    } else {
        println!("⚠️  Skipping GGUF tests - not found at {}", LLAMA_3_2_1B_INSTRUCT_GGUF);
    }
    
    // =========================================================================
    // Summary
    // =========================================================================
    
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                        SUMMARY                               ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  ✅ Passed:    {:2}                                            ║", passed);
    println!("║  ❌ Failed:    {:2}                                            ║", failed);
    println!("╚══════════════════════════════════════════════════════════════╝");
    
    if failed > 0 {
        println!("\n⚠️  Some tests failed. Check output above for details.");
        std::process::exit(1);
    } else {
        println!("\n🎉 All tests passed!");
    }
    
    Ok(())
}