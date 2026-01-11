//! Unit tests comparing CPU and GPU encoder outputs
//!
//! This test helps identify where GPU implementation diverges from CPU.

use crate::models::bart::model::BartModel;
use crate::models::sentence_encoder::SentenceEncoder;
use anyhow::Result;
use kjarni_transformers::cpu::encoder::prelude::*;
use kjarni_transformers::cpu::encoder::traits::CpuEncoder;
use kjarni_transformers::encoder_decoder::EncoderDecoderLanguageModel;
use kjarni_transformers::encoder_decoder::traits::{CpuCrossDecoder, GpuCrossDecoder};
use kjarni_transformers::gpu_ops::{GpuFrameContext, GpuTensor, Kernel};
use kjarni_transformers::models::ModelType;
use kjarni_transformers::models::base::ModelInput;
use kjarni_transformers::traits::Device;
use kjarni_transformers::{WgpuContext, activations};
use ndarray::Array2;
use ndarray::Array3;
use ndarray::Array4;
use ndarray::s;

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
    let cpu_model = BartModel::from_registry(model_type, None, Device::Cpu, None, None).await?;
    let gpu_model =
        BartModel::from_registry(model_type, None, Device::Wgpu, Some(ctx.clone()), None).await?;

    // 1. Check Activation

    assert_eq!(
        cpu_model.meta().activation,
        gpu_model.meta().activation,
        "Activation mismatch"
    );
    assert_eq!(
        cpu_model.meta().activation,
        activations::Activation::Gelu,
        "BART must use Gelu"
    );

    // 2. Check LayerNorm Epsilon (Top Suspect for 0.0019 drift)
    assert_eq!(
        cpu_model.meta().norm_eps,
        gpu_model.meta().norm_eps,
        "Norm Epsilon mismatch"
    );
    // Standard BART often uses 1e-12 or 1e-5 depending on the checkpoint
    log::info!("Metadata Epsilon: {}", cpu_model.meta().norm_eps);

    // 3. Check Normalization Style
    assert_eq!(
        cpu_model.meta().is_prenorm,
        gpu_model.meta().is_prenorm,
        "Pre-norm flag mismatch"
    );
    assert_eq!(
        cpu_model.meta().is_prenorm,
        false,
        "BART must be Post-Norm (is_prenorm=false)"
    );

    // 4. Check Attention Scaling
    assert_eq!(
        cpu_model.meta().head_dim,
        gpu_model.meta().head_dim,
        "Head Dim mismatch"
    );
    assert_eq!(
        cpu_model.meta().head_dim,
        1024 / 16,
        "BART-Large Head Dim should be 64"
    );

    // 5. Check Transposition Flags
    assert_eq!(
        cpu_model.meta().transpose_attention_weights,
        gpu_model.meta().transpose_attention_weights,
        "Transpose Attn mismatch"
    );
    assert_eq!(
        cpu_model.meta().transpose_ffn_weights,
        gpu_model.meta().transpose_ffn_weights,
        "Transpose FFN mismatch"
    );

    // 6. Check Position Offsets
    assert_eq!(
        cpu_model.meta().extra_pos_embeddings,
        gpu_model.meta().extra_pos_embeddings,
        "Extra Pos mismatch"
    );
    assert_eq!(
        cpu_model.meta().extra_pos_embeddings,
        2,
        "BART must use position offset 2"
    );
    let layout = cpu_model.layout().clone();
    let gpu_layout = gpu_model.layout().clone();

    // Get the nested layouts for both CPU and GPU models.
    let cpu_encoder_layout = layout
        .encoder
        .as_ref()
        .expect("CPU model requires encoder layout");
    let cpu_decoder_layout = layout
        .decoder
        .as_ref()
        .expect("CPU model requires decoder layout");
    let gpu_encoder_layout = gpu_layout
        .encoder
        .as_ref()
        .expect("GPU model requires encoder layout");
    let gpu_decoder_layout = gpu_layout
        .decoder
        .as_ref()
        .expect("GPU model requires decoder layout");

    // 1. Check Shared weights
    assert_eq!(
        cpu_model.layout().token_embedding,
        gpu_model.layout().token_embedding
    );
    assert_eq!(cpu_model.layout().lm_head, gpu_model.layout().lm_head);

    // 2. Check for "Llama leaks" (Ensure SwiGLU isn't accidentally enabled in either encoder or decoder)
    assert!(
        cpu_encoder_layout.layer.ffn.gate_weight.is_none(),
        "BART encoder should not have a SwiGLU gate"
    );
    assert!(
        cpu_decoder_layout.layer.ffn.gate_weight.is_none(),
        "BART decoder should not have a SwiGLU gate"
    );
    assert!(
        gpu_encoder_layout.layer.ffn.gate_weight.is_none(),
        "BART GPU encoder should not have a SwiGLU gate"
    );
    assert!(
        gpu_decoder_layout.layer.ffn.gate_weight.is_none(),
        "BART GPU decoder should not have a SwiGLU gate"
    );

    // 3. Check Attention Bias names (Ensure they aren't empty)
    // We check one from the encoder and one from the decoder to be thorough.
    assert!(
        cpu_encoder_layout.layer.self_attn.q_bias.is_some(),
        "BART encoder requires attention biases"
    );
    assert!(
        cpu_decoder_layout.layer.self_attn.norm_bias.is_some(),
        "BART decoder requires norm biases"
    );
    // Also check the GPU side for consistency
    assert!(
        gpu_encoder_layout.layer.self_attn.q_bias.is_some(),
        "BART GPU encoder requires attention biases"
    );
    assert!(
        gpu_decoder_layout.layer.self_attn.norm_bias.is_some(),
        "BART GPU decoder requires norm biases"
    );

    // 4. Check Cross-Attention Templates (If used in the test)
    let cpu_cross_attn_layout = cpu_decoder_layout.layer.cross_attn.as_ref();
    let gpu_cross_attn_layout = gpu_decoder_layout.layer.cross_attn.as_ref();

    if cpu_cross_attn_layout.is_some() {
        assert!(
            gpu_cross_attn_layout.is_some(),
            "GPU layout should also have cross-attention"
        );
        let cpu_cross = cpu_cross_attn_layout.unwrap();
        let gpu_cross = gpu_cross_attn_layout.unwrap();

        assert_eq!(cpu_cross.q_weight, gpu_cross.q_weight);
        assert_eq!(cpu_cross.q_bias, gpu_cross.q_bias);
    }

    log::info!("✓ Metadata and Layout verified for both CPU and GPU");

    let tokens: Vec<u32> = vec![0, 100, 200, 300, 400, 2];
    let input_ids_cpu: Array2<u32> = Array2::from_shape_vec((1, tokens.len()), tokens.clone())?;
    let mask_cpu = Array2::<f32>::ones((1, tokens.len()));

    let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids_cpu)?;
    let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask_cpu)?;

    let cpu_encoder = cpu_model.pipeline.cpu_encoder().expect("No CPU encoder");
    let gpu_encoder = gpu_model.pipeline.gpu_encoder().expect("No GPU encoder");

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
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let gpu_embed =
            gpu_encoder.embed(enc, pool_ref, ModelInput::TokensGpu(&input_ids_gpu), None)?;
        frame.finish();

        let gpu_embed_cpu = gpu_embed.to_ndarray_3d::<f32>().await?;
        assert_close(&cpu_embed, &gpu_embed_cpu, 1e-4, "Embeddings");
    }

    println!("=== STEP 2: EMBED + LAYERNORM ===");
    let cpu_embed_ln = cpu_encoder.embed_and_normalize(&input_ids_cpu, None);

    {
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let gpu_embed_ln = gpu_encoder.embed_and_normalize(
            enc,
            pool_ref,
            ModelInput::TokensGpu(&input_ids_gpu),
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
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let input_gpu = gpu_encoder.embed_and_normalize(
            enc,
            pool_ref,
            ModelInput::TokensGpu(&input_ids_gpu),
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
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();
            let input_gpu = gpu_encoder.embed_and_normalize(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&input_ids_gpu),
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
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();
        let input_gpu = gpu_encoder.embed_and_normalize(
            enc,
            pool_ref,
            ModelInput::TokensGpu(&input_ids_gpu),
            None,
        )?;
        let gpu_debug_12 =
            gpu_encoder.forward_layers(enc, pool_ref, &input_gpu, &mask_gpu, 0, 12)?;
        let gpu_forward = gpu_encoder.forward(
            enc,
            pool_ref,
            ModelInput::TokensGpu(&input_ids_gpu),
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
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let gpu_full = gpu_encoder.forward(
            enc,
            pool_ref,
            ModelInput::TokensGpu(&input_ids_gpu),
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
        SentenceEncoder::from_registry(ModelType::MiniLML6V2, None, Device::Cpu, None, None)
            .await?;
    println!("CPU encoder loaded\n");

    // Load GPU encoder
    println!("Loading GPU encoder...");
    let context = WgpuContext::new().await?;
    let gpu_encoder = SentenceEncoder::from_registry(
        ModelType::MiniLML6V2,
        None,
        Device::Wgpu,
        Some(context),
        None,
    )
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
        SentenceEncoder::from_registry(ModelType::MiniLML6V2, None, Device::Cpu, None, None)
            .await?;

    let context = WgpuContext::new().await?;
    let gpu_encoder = SentenceEncoder::from_registry(
        ModelType::MiniLML6V2,
        None,
        Device::Wgpu,
        Some(context),
        None,
    )
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
        SentenceEncoder::from_registry(ModelType::MiniLML6V2, None, Device::Cpu, None, None)
            .await?;

    let context = WgpuContext::new().await?;
    let gpu_encoder = SentenceEncoder::from_registry(
        ModelType::MiniLML6V2,
        None,
        Device::Wgpu,
        Some(context),
        None,
    )
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

#[tokio::test]
async fn test_bart_decoder_step_by_step_parity() -> Result<()> {
    let ctx = WgpuContext::new().await?;
    {
        let model_type = ModelType::DistilBartCnn;
        let cpu_model = BartModel::from_registry(model_type, None, Device::Cpu, None, None).await?;
        let gpu_model =
            BartModel::from_registry(model_type, None, Device::Wgpu, Some(ctx.clone()), None)
                .await?;

        let cpu_encoder = cpu_model.pipeline.cpu_encoder().expect("No CPU encoder");
        let gpu_encoder = gpu_model.pipeline.gpu_encoder().expect("No GPU encoder");
        let cpu_decoder = cpu_model.pipeline.cpu_decoder().expect("No CPU decoder");
        let gpu_decoder = gpu_model.pipeline.gpu_decoder().expect("No GPU decoder");

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
        fn assert_close_4d(cpu: &Array4<f32>, gpu: &Array4<f32>, atol: f32, name: &str) {
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

        // ========================================================================
        // SETUP: Get encoder hidden states (we know this works)
        // ========================================================================
        let encoder_tokens: Vec<u32> = vec![0, 100, 200, 300, 400, 2];
        let encoder_input_ids =
            Array2::from_shape_vec((1, encoder_tokens.len()), encoder_tokens.clone())?;
        let encoder_mask = Array2::<f32>::ones((1, encoder_tokens.len()));

        // CPU encoder forward
        let cpu_encoder_output = cpu_encoder.forward(&encoder_input_ids, &encoder_mask, None)?;
        let cpu_encoder_hidden = cpu_encoder_output.last_hidden_state;

        // GPU encoder forward
        let encoder_input_ids_gpu = GpuTensor::from_ndarray(&ctx, &encoder_input_ids)?;
        let encoder_mask_gpu = GpuTensor::from_ndarray(&ctx, &encoder_mask)?;

        let pool = ctx.get_inference_pool();
        let gpu_encoder_hidden = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let output = gpu_encoder.forward(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&encoder_input_ids_gpu),
                &encoder_mask_gpu,
                None,
            )?;
            frame.finish();
            output.last_hidden_state
        };

        println!("✓ Encoder outputs ready\n");

        // ========================================================================
        // DECODER TESTS
        // ========================================================================

        // Decoder input: just the BOS token
        let decoder_tokens: Vec<u32> = vec![2]; // decoder_start_token_id for BART
        let decoder_input_ids =
            Array2::from_shape_vec((1, decoder_tokens.len()), decoder_tokens.clone())?;
        let decoder_input_ids_gpu = GpuTensor::from_ndarray(&ctx, &decoder_input_ids)?;

        // Decoder attention mask (causal)
        let decoder_mask = Array2::<f32>::ones((1, 1));
        let decoder_mask_gpu = GpuTensor::from_ndarray(&ctx, &decoder_mask)?;

        let position_offset = 0usize;

        println!("=== DECODER STEP 1: EMBEDDINGS ===");
        let cpu_decoder_embed = cpu_decoder.embed(&decoder_input_ids, position_offset);
        println!("CPU decoder embed shape: {:?}", cpu_decoder_embed.shape());
        println!(
            "CPU decoder embed first 5: {:?}",
            cpu_decoder_embed.iter().take(5).collect::<Vec<_>>()
        );

        {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let gpu_decoder_embed = gpu_decoder.embed(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&decoder_input_ids_gpu),
                position_offset,
            )?;
            frame.finish();

            let gpu_decoder_embed_cpu = gpu_decoder_embed.to_ndarray_3d::<f32>().await?;
            println!(
                "GPU decoder embed shape: {:?}",
                gpu_decoder_embed_cpu.shape()
            );
            println!(
                "GPU decoder embed first 5: {:?}",
                gpu_decoder_embed_cpu.iter().take(5).collect::<Vec<_>>()
            );

            assert_close(
                &cpu_decoder_embed,
                &gpu_decoder_embed_cpu,
                1e-4,
                "Decoder Embeddings",
            );
        }

        println!("=== DECODER STEP 2: EMBED + LAYERNORM ===");
        let cpu_decoder_embed_ln =
            cpu_decoder.embed_and_normalize(&decoder_input_ids, position_offset)?;
        println!(
            "CPU decoder embed+ln first 5: {:?}",
            cpu_decoder_embed_ln.iter().take(5).collect::<Vec<_>>()
        );

        {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let gpu_decoder_embed_ln = gpu_decoder.embed_and_normalize(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&decoder_input_ids_gpu),
                position_offset,
            )?;
            frame.finish();

            let gpu_decoder_embed_ln_cpu = gpu_decoder_embed_ln.to_ndarray_3d::<f32>().await?;
            println!(
                "GPU decoder embed+ln first 5: {:?}",
                gpu_decoder_embed_ln_cpu.iter().take(5).collect::<Vec<_>>()
            );

            assert_close(
                &cpu_decoder_embed_ln,
                &gpu_decoder_embed_ln_cpu,
                1e-4,
                "Decoder Embed+LayerNorm",
            );
        }

        println!("=== DECODER STEP 3: PRECOMPUTE CROSS-ATTENTION KV ===");
        let cpu_cross_kv = cpu_decoder.precompute_cross_attention_kv(&cpu_encoder_hidden)?;
        println!("CPU cross KV layers: {}", cpu_cross_kv.0.len());
        if let Some((k, v)) = cpu_cross_kv.0.first() {
            println!(
                "CPU cross K[0] shape: {:?}, first 5: {:?}",
                k.shape(),
                k.iter().take(5).collect::<Vec<_>>()
            );
            println!(
                "CPU cross V[0] shape: {:?}, first 5: {:?}",
                v.shape(),
                v.iter().take(5).collect::<Vec<_>>()
            );
        }

        let gpu_cross_kv = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let cross_kv =
                gpu_decoder.precompute_cross_attention_kv(enc, pool_ref, &gpu_encoder_hidden)?;
            frame.finish();
            cross_kv
        };

        // Compare cross KV caches
        println!("GPU cross KV layers: {}", gpu_cross_kv.0.len());
        if let Some((k_gpu, v_gpu)) = gpu_cross_kv.0.first() {
            let k_cpu_arr = k_gpu.to_ndarray_4d::<f32>().await?;
            let v_cpu_arr = v_gpu.to_ndarray_4d::<f32>().await?;
            println!(
                "GPU cross K[0] shape: {:?}, first 5: {:?}",
                k_cpu_arr.shape(),
                k_cpu_arr.iter().take(5).collect::<Vec<_>>()
            );
            println!(
                "GPU cross V[0] shape: {:?}, first 5: {:?}",
                v_cpu_arr.shape(),
                v_cpu_arr.iter().take(5).collect::<Vec<_>>()
            );

            if let Some((cpu_k, cpu_v)) = cpu_cross_kv.0.first() {
                assert_close_4d(cpu_k, &k_cpu_arr, 1e-4, "Cross K[0]");
                assert_close_4d(cpu_v, &v_cpu_arr, 1e-4, "Cross V[0]");
            }
        }

        println!("=== DECODER STEP 4: LAYER 0 DETAILED BREAKDOWN ===");

        let decoder_hidden_cpu =
            cpu_decoder.embed_and_normalize(&decoder_input_ids, position_offset)?;

        // Get layer 0 references
        let cpu_layer0 = &cpu_decoder.layers()[0];
        let gpu_layer0 = &gpu_decoder.layers()[0];

        println!("=== DECODER STEP 4a-DETAIL: SELF-ATTENTION INTERNALS ===");

        // ============================================================
        // Check Q, K, V projections BEFORE attention
        // ============================================================
        println!("--- Checking Q/K/V Projections ---");

        let decoder_hidden_cpu =
            cpu_decoder.embed_and_normalize(&decoder_input_ids, position_offset)?;

        // CPU: Get raw Q, K, V (before split heads)
        let cpu_q = cpu_layer0.self_attn.q_proj.matmul(
            &decoder_hidden_cpu
                .view()
                .into_shape_with_order((1, 1024))
                .unwrap(),
        );
        let cpu_k = cpu_layer0.self_attn.k_proj.matmul(
            &decoder_hidden_cpu
                .view()
                .into_shape_with_order((1, 1024))
                .unwrap(),
        );
        let cpu_v = cpu_layer0.self_attn.v_proj.matmul(
            &decoder_hidden_cpu
                .view()
                .into_shape_with_order((1, 1024))
                .unwrap(),
        );

        println!(
            "CPU Q first 5: {:?}",
            cpu_q.iter().take(5).collect::<Vec<_>>()
        );
        println!(
            "CPU K first 5: {:?}",
            cpu_k.iter().take(5).collect::<Vec<_>>()
        );
        println!(
            "CPU V first 5: {:?}",
            cpu_v.iter().take(5).collect::<Vec<_>>()
        );

        // GPU: Get raw Q, K, V
        let (gpu_q, gpu_k, gpu_v) = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let gpu_decoder_hidden = gpu_decoder.embed_and_normalize(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&decoder_input_ids_gpu),
                position_offset,
            )?;

            // Project Q, K, V using the layer's weights
            let q_out = gpu_layer0.self_attn.ops.project(
                enc,
                &gpu_decoder_hidden,
                &gpu_layer0.self_attn_weights.q_weight,
                &gpu_layer0.self_attn_weights.q_bias,
                pool_ref,
            );
            let k_out = gpu_layer0.self_attn.ops.project(
                enc,
                &gpu_decoder_hidden,
                &gpu_layer0.self_attn_weights.k_weight,
                &gpu_layer0.self_attn_weights.k_bias,
                pool_ref,
            );
            let v_out = gpu_layer0.self_attn.ops.project(
                enc,
                &gpu_decoder_hidden,
                &gpu_layer0.self_attn_weights.v_weight,
                &gpu_layer0.self_attn_weights.v_bias,
                pool_ref,
            );

            frame.finish();

            (
                q_out.to_ndarray_3d::<f32>().await?,
                k_out.to_ndarray_3d::<f32>().await?,
                v_out.to_ndarray_3d::<f32>().await?,
            )
        };

        println!(
            "GPU Q first 5: {:?}",
            gpu_q.iter().take(5).collect::<Vec<_>>()
        );
        println!(
            "GPU K first 5: {:?}",
            gpu_k.iter().take(5).collect::<Vec<_>>()
        );
        println!(
            "GPU V first 5: {:?}",
            gpu_v.iter().take(5).collect::<Vec<_>>()
        );

        // Compare
        let q_diff = cpu_q
            .iter()
            .zip(gpu_q.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let k_diff = cpu_k
            .iter()
            .zip(gpu_k.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let v_diff = cpu_v
            .iter()
            .zip(gpu_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("[Q Projection] Max diff: {:.6}", q_diff);
        println!("[K Projection] Max diff: {:.6}", k_diff);
        println!("[V Projection] Max diff: {:.6}", v_diff);

        if q_diff > 0.001 {
            panic!("[FAIL] Q projection diverged!");
        }
        if k_diff > 0.001 {
            panic!("[FAIL] K projection diverged!");
        }
        if v_diff > 0.001 {
            panic!("[FAIL] V projection diverged!");
        }

        println!("[PASS] Q/K/V Projections match\n");

        // ============================================================
        // Check attention scores (Q @ K^T)
        // ============================================================
        println!("--- Checking Attention Scores ---");

        // CPU: Compute scores manually
        // Q, K are [1, 1024], need to reshape to [1, 16, 1, 64] for heads
        let num_heads = 16;
        let head_dim = 64;

        let cpu_q_heads = cpu_q
            .view()
            .into_shape_with_order((1, num_heads, 1, head_dim))
            .unwrap();
        let cpu_k_heads = cpu_k
            .view()
            .into_shape_with_order((1, num_heads, 1, head_dim))
            .unwrap();

        // For single token: scores = Q @ K^T = [1, 16, 1, 64] @ [1, 16, 64, 1] = [1, 16, 1, 1]
        // Scale factor
        let scale = 1.0 / (head_dim as f32).sqrt();
        println!("Attention scale factor: {}", scale);

        // Manual score computation for head 0
        let q_h0: Vec<f32> = cpu_q_heads.slice(s![0, 0, 0, ..]).iter().cloned().collect();
        let k_h0: Vec<f32> = cpu_k_heads.slice(s![0, 0, 0, ..]).iter().cloned().collect();
        let cpu_score_h0: f32 = q_h0
            .iter()
            .zip(k_h0.iter())
            .map(|(q, k)| q * k)
            .sum::<f32>()
            * scale;
        println!("CPU attention score (head 0): {:.6}", cpu_score_h0);

        // ============================================================
        // STEP 4a: SELF-ATTENTION ONLY (with residual + norm)
        // ============================================================
        println!("--- 4a: Self-Attention Block ---");

        // let (cpu_after_self_attn, (cpu_new_k, cpu_new_v)) = cpu_layer0.self_attention(
        //     &decoder_hidden_cpu,
        //     Some(&decoder_mask),
        //     None, // No past KV
        // )?;
        let (attn_out, new_k, new_v) =
            cpu_layer0
                .self_attn
                .forward(&decoder_hidden_cpu, Some(&decoder_mask), None, None)?;
        let hidden_states_after_add = &decoder_hidden_cpu + &attn_out;
        let final_output = cpu_layer0
            .self_attn_layer_norm
            .forward(&hidden_states_after_add);

        let (cpu_after_self_attn, (cpu_new_k, cpu_new_v)) = (final_output, (new_k, new_v));

        println!(
            "CPU after self-attn+norm shape: {:?}",
            cpu_after_self_attn.shape()
        );
        println!(
            "CPU after self-attn+norm first 5: {:?}",
            cpu_after_self_attn.iter().take(5).collect::<Vec<_>>()
        );

        let gpu_after_self_attn = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let gpu_decoder_hidden = gpu_decoder.embed_and_normalize(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&decoder_input_ids_gpu),
                position_offset,
            )?;

            // GPU self-attention: residual + attn + norm
            let residual = &gpu_decoder_hidden;

            // let (self_attn_output, _new_k, _new_v) = gpu_layer0.self_attn.forward_seq2seq(
            //     enc,
            //     residual,
            //     &gpu_layer0.self_attn_weights,
            //     &decoder_mask_gpu,
            //     None, // No past KV
            //     0,    // cache_len
            //     pool_ref,
            // )?;
            let output = gpu_layer0.self_attn.forward(
                enc,
                residual,
                &gpu_layer0.self_attn_weights,
                &decoder_mask_gpu,
                None, // No past KV
                0,    // cache_len
                pool_ref,
            )?;
            let self_attn_output = output.hidden_states;
            let _new_k = output.new_k;
            let _new_v = output.new_v;

            // Add residual
            let after_add = pool_ref.get(residual.shape().to_vec());
            gpu_layer0
                .add
                .encode(enc, &[residual, &self_attn_output], &after_add);

            // LayerNorm
            let after_norm = pool_ref.get(after_add.shape().to_vec());
            gpu_layer0.self_attn_norm.encode(
                enc,
                &gpu_layer0.self_attn_norm_weights,
                &after_add,
                &after_norm,
            );

            frame.finish();
            after_norm.to_ndarray_3d::<f32>().await?
        };

        println!(
            "GPU after self-attn+norm shape: {:?}",
            gpu_after_self_attn.shape()
        );
        println!(
            "GPU after self-attn+norm first 5: {:?}",
            gpu_after_self_attn.iter().take(5).collect::<Vec<_>>()
        );

        let self_attn_diff = cpu_after_self_attn
            .iter()
            .zip(gpu_after_self_attn.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("[Self-Attention Block] Max diff: {:.6}", self_attn_diff);

        if self_attn_diff > 0.001 {
            panic!(
                "[FAIL] Self-Attention Block diverged! max_diff = {}",
                self_attn_diff
            );
        }

        println!("--- 4b: Cross-Attention Block ---");
        let cross_attn_output = cpu_layer0.cross_attn.forward(
            &cpu_after_self_attn,
            &cpu_cross_kv.0[0].0,
            &cpu_cross_kv.0[0].1,
            None, // encoder_attn_mask
        )?;
        let hidden_states_after_add = &cpu_after_self_attn + &cross_attn_output;
        let cpu_after_cross_attn = cpu_layer0
            .cross_attn_layer_norm
            .forward(&hidden_states_after_add);
        println!(
            "CPU after cross-attn+norm shape: {:?}",
            cpu_after_cross_attn.shape()
        );
        println!(
            "CPU after cross-attn+norm first 5: {:?}",
            cpu_after_cross_attn.iter().take(5).collect::<Vec<_>>()
        );

        let gpu_after_cross_attn = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            // Recreate GPU state up to this point
            let gpu_decoder_hidden = gpu_decoder.embed_and_normalize(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&decoder_input_ids_gpu),
                position_offset,
            )?;

            // Redo self-attention to get to the same state
            let residual = &gpu_decoder_hidden;
            // let (self_attn_output, _, _) = gpu_layer0.self_attn.forward_seq2seq(
            //     enc,
            //     residual,
            //     &gpu_layer0.self_attn_weights,
            //     &decoder_mask_gpu,
            //     None,
            //     0,
            //     pool_ref,
            // )?;
            let self_attn_output = gpu_layer0
                .self_attn
                .forward(
                    enc,
                    residual,
                    &gpu_layer0.self_attn_weights,
                    &decoder_mask_gpu,
                    None,
                    0,
                    pool_ref,
                )?
                .hidden_states;

            let after_add1 = pool_ref.get(residual.shape().to_vec());
            gpu_layer0
                .add
                .encode(enc, &[residual, &self_attn_output], &after_add1);
            let after_self_attn_norm = pool_ref.get(after_add1.shape().to_vec());
            gpu_layer0.self_attn_norm.encode(
                enc,
                &gpu_layer0.self_attn_norm_weights,
                &after_add1,
                &after_self_attn_norm,
            );

            // NOW: Cross-attention
            let residual = &after_self_attn_norm;
            let (gpu_cross_k, gpu_cross_v) = &gpu_cross_kv.0[0];

            // let cross_attn_output = gpu_layer0.cross_attn.forward_cross_precomputed(
            //     enc,
            //     residual,
            //     gpu_cross_k,
            //     gpu_cross_v,
            //     &gpu_layer0.cross_attn_weights,
            //     None, // encoder_attn_mask
            //     pool_ref,
            // );
            let cross_attn_output = gpu_layer0.cross_attn.forward(
                enc,
                residual,
                &gpu_cross_kv.0[0],
                &gpu_layer0.cross_attn_weights,
                None, // encoder_attn_mask
                pool_ref,
            );

            // Add residual
            let after_add2 = pool_ref.get(residual.shape().to_vec());
            gpu_layer0
                .add
                .encode(enc, &[residual, &cross_attn_output], &after_add2);

            // LayerNorm
            let after_cross_norm = pool_ref.get(after_add2.shape().to_vec());
            gpu_layer0.cross_attn_norm.encode(
                enc,
                &gpu_layer0.cross_attn_norm_weights,
                &after_add2,
                &after_cross_norm,
            );

            frame.finish();
            after_cross_norm.to_ndarray_3d::<f32>().await?
        };

        println!(
            "GPU after cross-attn+norm shape: {:?}",
            gpu_after_cross_attn.shape()
        );
        println!(
            "GPU after cross-attn+norm first 5: {:?}",
            gpu_after_cross_attn.iter().take(5).collect::<Vec<_>>()
        );

        let cross_attn_diff = cpu_after_cross_attn
            .iter()
            .zip(gpu_after_cross_attn.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("[Cross-Attention Block] Max diff: {:.6}", cross_attn_diff);

        if cross_attn_diff > 0.001 {
            panic!(
                "[FAIL] Cross-Attention Block diverged! max_diff = {}",
                cross_attn_diff
            );
        }
        println!("[PASS] Cross-Attention Block\n");

        // ============================================================
        // STEP 4c: FFN ONLY (with residual + norm)
        // ============================================================
        println!("--- 4c: FFN Block ---");

        let cpu_after_ffn = cpu_layer0.feed_forward(&cpu_after_cross_attn)?;
        println!("CPU after FFN+norm shape: {:?}", cpu_after_ffn.shape());
        println!(
            "CPU after FFN+norm first 5: {:?}",
            cpu_after_ffn.iter().take(5).collect::<Vec<_>>()
        );

        let gpu_after_ffn = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            // Recreate GPU state up to cross-attention output
            let gpu_decoder_hidden = gpu_decoder.embed_and_normalize(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&decoder_input_ids_gpu),
                position_offset,
            )?;

            // Self-attention
            let residual = &gpu_decoder_hidden;
            // let (self_attn_output, _, _) = gpu_layer0.self_attn.forward_seq2seq(
            //     enc,
            //     residual,
            //     &gpu_layer0.self_attn_weights,
            //     &decoder_mask_gpu,
            //     None,
            //     0,
            //     pool_ref,
            // )?;
            let self_attn_output = gpu_layer0
                .self_attn
                .forward(
                    enc,
                    residual,
                    &gpu_layer0.self_attn_weights,
                    &decoder_mask_gpu,
                    None,
                    0,
                    pool_ref,
                )?
                .hidden_states;
            let after_add1 = pool_ref.get(residual.shape().to_vec());
            gpu_layer0
                .add
                .encode(enc, &[residual, &self_attn_output], &after_add1);
            let after_self_attn_norm = pool_ref.get(after_add1.shape().to_vec());
            gpu_layer0.self_attn_norm.encode(
                enc,
                &gpu_layer0.self_attn_norm_weights,
                &after_add1,
                &after_self_attn_norm,
            );

            // Cross-attention
            let residual = &after_self_attn_norm;

            let (gpu_cross_k, gpu_cross_v) = &gpu_cross_kv.0[0];

            // let cross_attn_output = gpu_layer0.cross_attn.forward_cross_precomputed(
            //     enc,
            //     residual,
            //     gpu_cross_k,
            //     gpu_cross_v,
            //     &gpu_layer0.cross_attn_weights,
            //     None,
            //     pool_ref,
            // );
            let cross_attn_output = gpu_layer0.cross_attn.forward(
                enc,
                residual,
                &gpu_cross_kv.0[0],
                &gpu_layer0.cross_attn_weights,
                None,
                pool_ref,
            );
            let after_add2 = pool_ref.get(residual.shape().to_vec());
            gpu_layer0
                .add
                .encode(enc, &[residual, &cross_attn_output], &after_add2);
            let after_cross_norm = pool_ref.get(after_add2.shape().to_vec());
            gpu_layer0.cross_attn_norm.encode(
                enc,
                &gpu_layer0.cross_attn_norm_weights,
                &after_add2,
                &after_cross_norm,
            );

            // NOW: FFN
            let residual = &after_cross_norm;
            let ffn_output = pool_ref.get(residual.shape().to_vec());
            gpu_layer0.feedforward.encode(
                enc,
                &gpu_layer0.ff_weights,
                residual,
                &ffn_output,
                pool_ref,
            );

            // Add residual
            let after_add3 = pool_ref.get(residual.shape().to_vec());
            gpu_layer0
                .add
                .encode(enc, &[residual, &ffn_output], &after_add3);

            // Final LayerNorm
            let final_output = pool_ref.get(after_add3.shape().to_vec());
            gpu_layer0.ffn_norm.encode(
                enc,
                &gpu_layer0.ffn_norm_weights,
                &after_add3,
                &final_output,
            );

            frame.finish();
            final_output.to_ndarray_3d::<f32>().await?
        };

        println!("GPU after FFN+norm shape: {:?}", gpu_after_ffn.shape());
        println!(
            "GPU after FFN+norm first 5: {:?}",
            gpu_after_ffn.iter().take(5).collect::<Vec<_>>()
        );

        let ffn_diff = cpu_after_ffn
            .iter()
            .zip(gpu_after_ffn.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("[FFN Block] Max diff: {:.6}", ffn_diff);

        if ffn_diff > 0.001 {
            panic!("[FAIL] FFN Block diverged! max_diff = {}", ffn_diff);
        }
        println!("[PASS] FFN Block\n");

        // ============================================================
        // STEP 4d: COMPARE WITH FULL LAYER FORWARD
        // ============================================================
        println!("--- 4d: Full Layer 0 Forward (should match 4c) ---");

        // CPU full forward (what we had before)
        let cpu_layer0_full = cpu_decoder.forward_layers(
            &decoder_hidden_cpu,
            &cpu_encoder_hidden,
            Some(&decoder_mask),
            None,
            Some(&cpu_cross_kv),
            0,
            1,
        )?;

        let full_vs_step_diff = cpu_layer0_full
            .last_hidden_state
            .iter()
            .zip(cpu_after_ffn.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "CPU full forward vs step-by-step diff: {:.6}",
            full_vs_step_diff
        );

        if full_vs_step_diff > 1e-6 {
            println!("WARNING: CPU step-by-step doesn't match full forward!");
        }

        println!("\n=== BREAKDOWN COMPLETE ===\n");

        println!("=== DECODER STEP 4: FORWARD LAYER 0 (no self-attn cache) ===");
        let decoder_hidden =
            cpu_decoder.embed_and_normalize(&decoder_input_ids, position_offset)?;

        let cpu_layer0_output = cpu_decoder.forward_layers(
            &decoder_hidden,
            &cpu_encoder_hidden,
            Some(&decoder_mask),
            None, // No cache
            Some(&cpu_cross_kv),
            0,
            1, // Layer 0 only
        )?;
        println!(
            "CPU layer 0 output shape: {:?}",
            cpu_layer0_output.last_hidden_state.shape()
        );
        println!(
            "CPU layer 0 first 5: {:?}",
            cpu_layer0_output
                .last_hidden_state
                .iter()
                .take(5)
                .collect::<Vec<_>>()
        );

        {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let gpu_decoder_hidden = gpu_decoder.embed_and_normalize(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&decoder_input_ids_gpu),
                position_offset,
            )?;

            let gpu_layer0_output = gpu_decoder.forward_layers(
                enc,
                pool_ref,
                &gpu_decoder_hidden,
                &gpu_encoder_hidden,
                &decoder_mask_gpu,
                position_offset,
                None, // No cache
                Some(&gpu_cross_kv),
                0,
                1,
            )?;
            frame.finish();

            let gpu_layer0_cpu = gpu_layer0_output
                .last_hidden_state
                .to_ndarray_3d::<f32>()
                .await?;
            println!("GPU layer 0 output shape: {:?}", gpu_layer0_cpu.shape());
            println!(
                "GPU layer 0 first 5: {:?}",
                gpu_layer0_cpu.iter().take(5).collect::<Vec<_>>()
            );

            assert_close(
                &cpu_layer0_output.last_hidden_state,
                &gpu_layer0_cpu,
                1e-4,
                "Decoder Layer 0",
            );
        }

        println!("=== DECODER STEP 5: FORWARD ALL LAYERS ===");
        let num_layers = cpu_decoder.num_layers();

        for n in 1..=num_layers {
            let cpu_layer_n = cpu_decoder.forward_layers(
                &decoder_hidden,
                &cpu_encoder_hidden,
                Some(&decoder_mask),
                None,
                Some(&cpu_cross_kv),
                0,
                n,
            )?;

            {
                let pool_guard = pool.lock().await;
                let mut frame = GpuFrameContext::new(&ctx, pool_guard);
                let (enc, pool_ref) = frame.resources();

                let gpu_decoder_hidden = gpu_decoder.embed_and_normalize(
                    enc,
                    pool_ref,
                    ModelInput::TokensGpu(&decoder_input_ids_gpu),
                    position_offset,
                )?;

                let gpu_layer_n = gpu_decoder.forward_layers(
                    enc,
                    pool_ref,
                    &gpu_decoder_hidden,
                    &gpu_encoder_hidden,
                    &decoder_mask_gpu,
                    position_offset,
                    None,
                    Some(&gpu_cross_kv),
                    0,
                    n,
                )?;
                frame.finish();

                let gpu_layer_n_cpu = gpu_layer_n.last_hidden_state.to_ndarray_3d::<f32>().await?;

                let max_diff = cpu_layer_n
                    .last_hidden_state
                    .iter()
                    .zip(gpu_layer_n_cpu.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);

                println!("[Decoder Layer 0..{}] Max diff: {:.6}", n, max_diff);

                if max_diff > 0.01 {
                    println!(
                        "  CPU first 5: {:?}",
                        cpu_layer_n
                            .last_hidden_state
                            .iter()
                            .take(5)
                            .collect::<Vec<_>>()
                    );
                    println!(
                        "  GPU first 5: {:?}",
                        gpu_layer_n_cpu.iter().take(5).collect::<Vec<_>>()
                    );
                    panic!("Decoder divergence at layer {}", n);
                }
            }
        }

        println!("=== DECODER STEP 6: LM HEAD PROJECTION ===");
        let cpu_final_hidden = cpu_decoder
            .forward_layers(
                &decoder_hidden,
                &cpu_encoder_hidden,
                Some(&decoder_mask),
                None,
                Some(&cpu_cross_kv),
                0,
                num_layers,
            )?
            .last_hidden_state;

        // Use the model's project_to_logits
        let cpu_logits = cpu_model
            .encoder_decoder_cpu_ops()
            .unwrap()
            .project_to_logits(&cpu_final_hidden)?;

        println!("CPU logits shape: {:?}", cpu_logits.shape());
        println!(
            "CPU logits first 5: {:?}",
            cpu_logits.iter().take(5).collect::<Vec<_>>()
        );

        // Find argmax
        let cpu_argmax = cpu_logits
            .slice(s![0, 0, ..])
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        println!("CPU predicted token: {}", cpu_argmax);

        {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let gpu_decoder_hidden = gpu_decoder.embed_and_normalize(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&decoder_input_ids_gpu),
                position_offset,
            )?;

            let gpu_final_hidden = gpu_decoder
                .forward_layers(
                    enc,
                    pool_ref,
                    &gpu_decoder_hidden,
                    &gpu_encoder_hidden,
                    &decoder_mask_gpu,
                    position_offset,
                    None,
                    Some(&gpu_cross_kv),
                    0,
                    num_layers,
                )?
                .last_hidden_state;

            let gpu_logits = gpu_model
                .encoder_decoder_gpu_ops()
                .unwrap()
                .project_to_logits(&mut frame, &gpu_final_hidden)?;

            frame.finish();

            let gpu_logits_cpu = gpu_logits.to_ndarray_3d::<f32>().await?;
            println!("GPU logits shape: {:?}", gpu_logits_cpu.shape());
            println!(
                "GPU logits first 5: {:?}",
                gpu_logits_cpu.iter().take(5).collect::<Vec<_>>()
            );

            let gpu_argmax = gpu_logits_cpu
                .slice(s![0, 0, ..])
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            println!("GPU predicted token: {}", gpu_argmax);

            assert_eq!(cpu_argmax, gpu_argmax, "First predicted token mismatch!");

            // Check logits are close
            let max_logit_diff = cpu_logits
                .iter()
                .zip(gpu_logits_cpu.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("Max logits diff: {:.6}", max_logit_diff);
        }

        println!("\n✓ All decoder steps passed!");
        
    }
    kjarni_transformers::weights::clear_mmap_cache();
    Ok(())
}
