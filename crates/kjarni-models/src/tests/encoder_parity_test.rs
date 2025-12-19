//! Unit tests comparing CPU and GPU encoder outputs
//!
//! This test helps identify where GPU implementation diverges from CPU.

use crate::models::bart::model::BartModel;
use crate::sentence_encoder::SentenceEncoder;
use anyhow::Result;
use kjarni_transformers::encoder::prelude::*;
use kjarni_transformers::encoder::traits::CpuEncoder;
use kjarni_transformers::gpu_context::WgpuContext;
use kjarni_transformers::gpu_ops::{GpuFrameContext, GpuTensor};
use kjarni_transformers::models::ModelType;
use kjarni_transformers::traits::Device;
use ndarray::Array2;
use ndarray::Array3;

const TOLERANCE: f32 = 1e-3;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

fn compare_vectors(name: &str, cpu: &[f32], gpu: &[f32], tolerance: f32) -> bool {
    println!("\n=== Comparing: {} ===", name);

    if cpu.len() != gpu.len() {
        println!("Length mismatch: CPU {} vs GPU {}", cpu.len(), gpu.len());
        return false;
    }
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

    println!("CPU first 10: {:?}", &cpu[..10.min(cpu.len())]);
    println!("GPU first 10: {:?}", &gpu[..10.min(gpu.len())]);

    if num_mismatches == 0 {
        println!(" PASS");
        true
    } else {
        println!("FAIL ({} mismatches)", num_mismatches);
        false
    }
}

#[tokio::test]
async fn test_bart_encoder_step_by_step_parity() -> Result<()> {
    let ctx = WgpuContext::new().await?;

    let model_type = ModelType::DistilBartCnn;
    let cpu_model = BartModel::from_registry(model_type, None, Device::Cpu, None).await?;
    let gpu_model =
        BartModel::from_registry(model_type, None, Device::Wgpu, Some(ctx.clone())).await?;

    let tokens: Vec<u32> = vec![0, 100, 200, 300, 400, 2];
    let input_ids_cpu: Array2<u32> = Array2::from_shape_vec((1, tokens.len()), tokens.clone())?;
    let mask_cpu = Array2::<f32>::ones((1, tokens.len()));

    let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids_cpu)?;
    let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask_cpu)?;

    let cpu_encoder = cpu_model.bart_cpu_encoder().expect("No CPU encoder");
    let gpu_encoder = gpu_model.bart_gpu_encoder().expect("No GPU encoder");

    fn assert_close(cpu: &Array3<f32>, gpu: &Array3<f32>, atol: f32, name: &str) {
        let max_diff = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("[{}] Max diff: {:.6}", name, max_diff);
        println!(
            "  CPU first 5: {:?}",
            cpu.iter().take(5).collect::<Vec<_>>()
        );
        println!(
            "  GPU first 5: {:?}",
            gpu.iter().take(5).collect::<Vec<_>>()
        );

        if max_diff > atol {
            panic!("[FAIL] {} - max_diff {} > atol {}", name, max_diff, atol);
        }
        println!("[PASS] {}\n", name);
    }

    println!("\n=== STEP 1: EMBEDDINGS ===");
    let cpu_embed = cpu_encoder.embed(&input_ids_cpu, None);

    let pool = ctx.get_inference_pool();
    {
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let gpu_embed = gpu_encoder.embed(
            enc,
            pool_ref,
            GpuEncoderInput::TokensGpu(&input_ids_gpu),
            None,
        )?;
        frame.finish();

        let gpu_embed_cpu = gpu_embed.to_ndarray_3d::<f32>().await?;
        assert_close(&cpu_embed, &gpu_embed_cpu, 1e-4, "Embeddings");
    }

    println!("=== STEP 2: EMBED + LAYERNORM ===");
    let cpu_embed_ln = cpu_encoder.embed_and_normalize(&input_ids_cpu, None);

    {
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let gpu_embed_ln = gpu_encoder.embed_and_normalize(
            enc,
            pool_ref,
            GpuEncoderInput::TokensGpu(&input_ids_gpu),
            None,
        )?;
        frame.finish();

        let gpu_embed_ln_cpu = gpu_embed_ln.to_ndarray_3d::<f32>().await?;
        assert_close(&cpu_embed_ln, &gpu_embed_ln_cpu, 1e-4, "Embed+LayerNorm");
    }

    println!("=== STEP 3: AFTER LAYER 0 ===");
    let h = cpu_encoder.embed_and_normalize(&input_ids_cpu, None);

    let cpu_layer0 = cpu_encoder.forward_layers(&h, &mask_cpu, 0, 1)?;

    {
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let input_gpu = gpu_encoder.embed_and_normalize(
            enc,
            pool_ref,
            GpuEncoderInput::TokensGpu(&input_ids_gpu),
            None,
        )?;

        let gpu_layer0 = gpu_encoder.forward_layers(enc, pool_ref, &input_gpu, &mask_gpu, 0, 1)?;
        frame.finish();

        let gpu_layer0_cpu = gpu_layer0.to_ndarray_3d::<f32>().await?;
        assert_close(&cpu_layer0, &gpu_layer0_cpu, 1e-4, "After Layer 0");
    }
    for n in 1..=12 {
        println!("=== AFTER LAYER {} ===", n);
        let cpu_layer_n = cpu_encoder.forward_layers(&h, &mask_cpu, 0, n)?;

        {
            let mut pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();
            let input_gpu = gpu_encoder.embed_and_normalize(
                enc,
                pool_ref,
                GpuEncoderInput::TokensGpu(&input_ids_gpu),
                None,
            )?;
            let gpu_layer_n =
                gpu_encoder.forward_layers(enc, pool_ref, &input_gpu, &mask_gpu, 0, n)?;
            frame.finish();

            let gpu_layer_n_cpu = gpu_layer_n.to_ndarray_3d::<f32>().await?;

            let max_diff = cpu_layer_n
                .iter()
                .zip(gpu_layer_n_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("[After Layer {}] Max diff: {:.6}", n, max_diff);

            if max_diff > 0.01 {
                println!(
                    "  CPU first 5: {:?}",
                    cpu_layer_n.iter().take(5).collect::<Vec<_>>()
                );
                println!(
                    "  GPU first 5: {:?}",
                    gpu_layer_n_cpu.iter().take(5).collect::<Vec<_>>()
                );
                panic!("Divergence found at layer {}", n);
            }
        }
    }
    println!("=== SANITY CHECK: debug_n_layers(12) vs forward() ===");

    let cpu_debug_12 = cpu_encoder.forward_layers(&h, &mask_cpu, 0, 12)?;
    let cpu_forward = cpu_encoder.forward(&input_ids_cpu, &mask_cpu, None)?;
    let cpu_internal_diff = cpu_debug_12
        .iter()
        .zip(cpu_forward.last_hidden_state.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "CPU debug_n_layers(12) vs forward() diff: {:.6}",
        cpu_internal_diff
    );
    {
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();
        let input_gpu = gpu_encoder.embed_and_normalize(
            enc,
            pool_ref,
            GpuEncoderInput::TokensGpu(&input_ids_gpu),
            None,
        )?;
        let gpu_debug_12 =
            gpu_encoder.forward_layers(enc, pool_ref, &input_gpu, &mask_gpu, 0, 12)?;
        let gpu_forward = gpu_encoder.forward(
            enc,
            pool_ref,
            GpuEncoderInput::TokensGpu(&input_ids_gpu),
            &mask_gpu,
            None,
        )?;
        frame.finish();

        let gpu_debug_12_cpu = gpu_debug_12.to_ndarray_3d::<f32>().await?;
        let gpu_forward_cpu = gpu_forward.last_hidden_state.to_ndarray_3d::<f32>().await?;

        let gpu_internal_diff = gpu_debug_12_cpu
            .iter()
            .zip(gpu_forward_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "GPU debug_n_layers(12) vs forward() diff: {:.6}",
            gpu_internal_diff
        );
    }
    println!("=== STEP 4: FULL ENCODER ===");
    let cpu_full = cpu_encoder.forward(&input_ids_cpu, &mask_cpu, None)?;

    {
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let gpu_full = gpu_encoder.forward(
            enc,
            pool_ref,
            GpuEncoderInput::TokensGpu(&input_ids_gpu),
            &mask_gpu,
            None,
        )?;
        frame.finish();

        let gpu_full_cpu = gpu_full.last_hidden_state.to_ndarray_3d::<f32>().await?;
        assert_close(
            &cpu_full.last_hidden_state,
            &gpu_full_cpu,
            1e-4,
            "Full Encoder",
        );
    }

    println!("✓ All steps passed!");
    Ok(())
}

#[tokio::test]
async fn test_encoder_cpu_gpu_parity() -> Result<()> {
    println!("Testing CPU vs GPU Encoder Parity");

    // Load CPU encoder
    println!("Loading CPU encoder...");
    let cpu_encoder =
        SentenceEncoder::from_registry(ModelType::MiniLML6V2, None, Device::Cpu, None).await?;
    println!("CPU encoder loaded\n");

    // Load GPU encoder
    println!("Loading GPU encoder...");
    let context = WgpuContext::new().await?;
    let gpu_encoder =
        SentenceEncoder::from_registry(ModelType::MiniLML6V2, None, Device::Wgpu, Some(context))
            .await?;
    println!("GPU encoder loaded\n");

    // Test sentences
    let test_sentences = ["The cat sits on the mat", "Machine learning is fascinating"];

    println!("Test sentences:");
    for (i, sentence) in test_sentences.iter().enumerate() {
        println!("  {}: {}", i + 1, sentence);
    }
    println!();

    // Encode on CPU
    println!("Encoding on CPU...");
    let cpu_embeddings = cpu_encoder.encode_batch(&test_sentences).await?;
    println!("CPU encoding complete\n");

    // Encode on GPU
    println!("Encoding on GPU...");
    let gpu_embeddings = gpu_encoder.encode_batch(&test_sentences).await?;
    println!("GPU encoding complete\n");

    // Compare results
    let mut all_pass = true;

    for (i, (cpu_emb, gpu_emb)) in cpu_embeddings.iter().zip(gpu_embeddings.iter()).enumerate() {
        let pass = compare_vectors(
            &format!("Sentence {} embedding", i + 1),
            cpu_emb,
            gpu_emb,
            TOLERANCE,
        );
        all_pass = all_pass && pass;
    }

    // Compute and compare cosine similarities
    if test_sentences.len() >= 2 {
        println!("\n=== Cosine Similarities ===");

        let cpu_sim = cosine_similarity(&cpu_embeddings[0], &cpu_embeddings[1]);
        let gpu_sim = cosine_similarity(&gpu_embeddings[0], &gpu_embeddings[1]);

        println!("CPU similarity: {:.6}", cpu_sim);
        println!("GPU similarity: {:.6}", gpu_sim);
        println!("Difference: {:.6}", (cpu_sim - gpu_sim).abs());

        if (cpu_sim - gpu_sim).abs() > 0.01 {
            println!("Similarity mismatch!");
            all_pass = false;
        } else {
            println!("Similarities match");
        }
    }

    if all_pass {
        println!("ALL TESTS PASSED");
        Ok(())
    } else {
        println!("SOME TESTS FAILED");

        Err(anyhow::anyhow!("CPU-GPU parity test failed"))
    }
}

#[tokio::test]
async fn test_simple_input() -> Result<()> {
    println!("Testing Simple Input: Single Word");

    let cpu_encoder =
        SentenceEncoder::from_registry(ModelType::MiniLML6V2, None, Device::Cpu, None).await?;

    let context = WgpuContext::new().await?;
    let gpu_encoder =
        SentenceEncoder::from_registry(ModelType::MiniLML6V2, None, Device::Wgpu, Some(context))
            .await?;

    let simple_text = "hello";

    println!("Input: \"{}\"", simple_text);

    let cpu_emb = cpu_encoder.encode(simple_text).await?;
    let gpu_emb = gpu_encoder.encode(simple_text).await?;

    let pass = compare_vectors("Simple input embedding", &cpu_emb, &gpu_emb, TOLERANCE);

    if pass {
        println!("\nSimple input test PASSED");
        Ok(())
    } else {
        println!("\nSimple input test FAILED");
        Err(anyhow::anyhow!("Simple input test failed"))
    }
}

#[tokio::test]
async fn test_identical_sentences() -> Result<()> {
    println!("Testing Identical Sentences");

    let cpu_encoder =
        SentenceEncoder::from_registry(ModelType::MiniLML6V2, None, Device::Cpu, None).await?;

    let context = WgpuContext::new().await?;
    let gpu_encoder =
        SentenceEncoder::from_registry(ModelType::MiniLML6V2, None, Device::Wgpu, Some(context))
            .await?;

    let text = "This is a test sentence";
    let sentences = [text, text]; // Same sentence twice

    let cpu_embeddings = cpu_encoder.encode_batch(&sentences).await?;
    let gpu_embeddings = gpu_encoder.encode_batch(&sentences).await?;

    // CPU should produce identical embeddings
    let cpu_self_sim = cosine_similarity(&cpu_embeddings[0], &cpu_embeddings[1]);
    println!("CPU self-similarity: {:.6}", cpu_self_sim);
    assert!(
        (cpu_self_sim - 1.0).abs() < 1e-5,
        "CPU should produce identical embeddings"
    );

    // GPU should produce identical embeddings
    let gpu_self_sim = cosine_similarity(&gpu_embeddings[0], &gpu_embeddings[1]);
    println!("GPU self-similarity: {:.6}", gpu_self_sim);
    assert!(
        (gpu_self_sim - 1.0).abs() < 1e-5,
        "GPU should produce identical embeddings"
    );

    // Compare CPU vs GPU
    let pass1 = compare_vectors(
        "First embedding",
        &cpu_embeddings[0],
        &gpu_embeddings[0],
        TOLERANCE,
    );
    let pass2 = compare_vectors(
        "Second embedding",
        &cpu_embeddings[1],
        &gpu_embeddings[1],
        TOLERANCE,
    );

    if pass1 && pass2 {
        println!("\nIdentical sentences test PASSED");
        Ok(())
    } else {
        println!("\nIdentical sentences test FAILED");
        Err(anyhow::anyhow!("Identical sentences test failed"))
    }
}
