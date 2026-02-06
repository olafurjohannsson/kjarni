use anyhow::Result;

use kjarni_transformers::{
    WgpuContext,
    common::DecodingStrategy,
    cpu::encoder::{CpuEncoderOps, traits::EncoderLanguageModel},
    encoder_decoder::{
        CpuBackend, CpuSeq2SeqState, EncoderDecoderGenerator,
        traits::{CpuCrossDecoder, EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel},
    },
    gpu::encoder_decoder::backend::{GpuEncoderDecoderBackend, GpuSeq2SeqState},
    models::{ModelType, base::ModelInput},
};

use kjarni_models::models::bart::model::BartModel;
use kjarni_transformers::Device;
use kjarni_transformers::gpu::{GpuFrameContext, GpuTensor};
use kjarni_transformers::models::base::LanguageModel;
use ndarray::{ArrayViewD, IxDyn};
use std::sync::Arc;
async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}
use anyhow::anyhow;

use std::io;
use std::io::Write;

fn assert_all_close(a: &ArrayViewD<f32>, b: &ArrayViewD<f32>, rtol: f32, atol: f32, context: &str) {
    if a.shape() != b.shape() {
        panic!(
            "[DEBUG FAIL] {} shape mismatch! CPU: {:?}, GPU: {:?}",
            context,
            a.shape(),
            b.shape()
        );
    }
    let mut max_abs_diff = 0.0;
    for (a_val, b_val) in a.iter().zip(b.iter()) {
        max_abs_diff = f32::max(max_abs_diff, (a_val - b_val).abs());
    }

    // A simplified check focusing on absolute tolerance, which is often more stable for comparing GPU/CPU.
    if max_abs_diff > atol {
        log::error!("[DEBUG FAIL] {} mismatch!", context);
        log::error!("  Max absolute difference: {}", max_abs_diff);
        log::error!("  CPU First 5: {:?}", a.iter().take(5).collect::<Vec<_>>());
        log::error!("  GPU First 5: {:?}", b.iter().take(5).collect::<Vec<_>>());
        panic!("{} mismatch", context);
    } else {
        log::info!(
            "[DEBUG PASS] {} matches. Max absolute difference: {}",
            context,
            max_abs_diff
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // --- 1. SETUP ---
    log::info!("Loading models for CPU and GPU...");
    let ctx = get_test_context().await;
    let model_type = ModelType::BartLargeCnn;

    let cpu_model = BartModel::from_registry(model_type, None, Device::Cpu, None, None).await?;
    let gpu_model =
        BartModel::from_registry(model_type, None, Device::Wgpu, Some(ctx.clone()), None).await?;

    let cpu_backend = CpuBackend;
    let gpu_backend = GpuEncoderDecoderBackend::new(ctx.clone())?;

    let article = "Rust is a multi-paradigm, general-purpose programming language.";
    let num_beams = 4; // Use the same number of beams for both

    let encoding = cpu_model
        .tokenizer()
        .encode(article, true)
        .map_err(|e| anyhow!(e))?;
    let tokens = encoding.get_ids();

    // === STEP 0: CONFIG CHECK ===
    log::info!("\n--- STEP 0: CONFIG CHECK ---");
    log::info!(
        "CPU model normalize_embedding config: {}",
        cpu_model.config.normalize_embedding
    );
    log::info!(
        "GPU model normalize_embedding config: {}",
        gpu_model.config.normalize_embedding
    );

    // === STEP 0a: ENCODER RAW EMBEDDINGS (before layernorm) ===
    log::info!("\n--- STEP 0a: ENCODER RAW EMBEDDINGS ---");

    let input_ids_2d =
        ndarray::Array2::from_shape_vec((1, tokens.len()), tokens.iter().map(|&t| t).collect())?;

    // CPU: use the pipeline's encoder embeddings
    let cpu_enc_embed = cpu_model.embed_tokens(&input_ids_2d, None, 2)?;
    println!("CPU encoder embed shape: {:?}", cpu_enc_embed.shape());
    println!(
        "CPU encoder embed first 5: {:?}",
        cpu_enc_embed.iter().take(5).collect::<Vec<_>>()
    );

    // GPU
    let gpu_encoder = gpu_model.pipeline.gpu_encoder().expect("No GPU encoder");
    let gpu_ops = gpu_model.encoder_gpu_ops().expect("No GPU OPS");
    let input_gpu_enc = GpuTensor::from_ndarray(&ctx, &input_ids_2d)?;
    {
        let pool = ctx.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        // let gpu_enc_embed = gpu_encoder.embed(
        //     enc,
        //     pool_ref,
        //     ModelInput::TokensGpu(&input_gpu_enc),
        //     None,
        // )?;
        let gpu_enc_embed = gpu_ops.embed_tokens(
            enc,
            pool_ref,
            ModelInput::TokensGpu(&input_gpu_enc),
            None,
            0,
        )?;

        frame.finish();

        let gpu_enc_embed_cpu = gpu_enc_embed.to_ndarray_3d::<f32>().await?;
        println!("GPU encoder embed shape: {:?}", gpu_enc_embed_cpu.shape());
        println!(
            "GPU encoder embed first 5: {:?}",
            gpu_enc_embed_cpu.iter().take(5).collect::<Vec<_>>()
        );

        assert_all_close(
            &cpu_enc_embed.view().into_dyn(),
            &gpu_enc_embed_cpu.view().into_dyn(),
            1e-3,
            1e-4,
            "Encoder Raw Embeddings",
        );
    }

    // === STEP 0b: ENCODER EMBED + LAYERNORM ===
    log::info!("\n--- STEP 0b: ENCODER EMBED + LAYERNORM ---");

    // CPU: apply embed_norm manually
    let cpu_encoder = cpu_model.pipeline.cpu_encoder().expect("No CPU encoder");
    let cpu_enc_normed = cpu_encoder.embed_norm(&cpu_enc_embed)?;
    println!("CPU encoder normed shape: {:?}", cpu_enc_normed.shape());
    println!(
        "CPU encoder normed first 5: {:?}",
        cpu_enc_normed.iter().take(5).collect::<Vec<_>>()
    );

    // GPU
    {
        let pool = ctx.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        // let gpu_enc_normed = gpu_encoder.embed_and_normalize(
        //     enc,
        //     pool_ref,
        //     ModelInput::TokensGpu(&input_gpu_enc),
        //     None,
        // )?;

        // let tensor = pool_ref.get(vec![
        //     cpu_enc_normed.shape()[0],
        //     cpu_enc_normed.shape()[1],
        //     cpu_enc_normed.shape()[2],
        // ]);
        let gpu_enc_embed = gpu_ops.embed_tokens(
            enc,
            pool_ref,
            ModelInput::TokensGpu(&input_gpu_enc),
            None,
            0,
        )?;

        // Then apply embed_norm
        let gpu_enc_normed = gpu_encoder.embed_norm(enc, pool_ref, &gpu_enc_embed)?;

        frame.finish();

        let gpu_enc_normed_cpu = gpu_enc_normed.to_ndarray_3d::<f32>().await?;
        println!("GPU encoder normed shape: {:?}", gpu_enc_normed_cpu.shape());
        println!(
            "GPU encoder normed first 5: {:?}",
            gpu_enc_normed_cpu.iter().take(5).collect::<Vec<_>>()
        );

        assert_all_close(
            &cpu_enc_normed.view().into_dyn(),
            &gpu_enc_normed_cpu.view().into_dyn(),
            1e-3,
            1e-4,
            "Encoder Embed + LayerNorm",
        );
    }

    // === STEP 0c: LAYER-BY-LAYER ENCODER CHECK ===
log::info!("\n--- STEP 0c: LAYER-BY-LAYER ENCODER CHECK ---");

let attention_mask_cpu = ndarray::Array2::<f32>::ones((1, tokens.len()));
let attention_mask_gpu = GpuTensor::from_ndarray(&ctx, &attention_mask_cpu)?;

// Start with normalized embeddings
let mut cpu_hidden = cpu_enc_normed.clone();

// GPU: Run ALL layers in a single frame to avoid cross-frame aliasing
let num_layers = cpu_encoder.num_layers();

for layer_idx in 0..num_layers {
    // CPU: run single layer
    cpu_hidden = cpu_encoder.forward_layers(
        &cpu_hidden,
        &attention_mask_cpu,
        layer_idx,
        layer_idx + 1,
    )?;
}

// GPU: run all layers in one frame
let gpu_final = {
    let pool = ctx.get_inference_pool();
    let pool_guard = pool.lock().await;
    let mut frame = GpuFrameContext::new(&ctx, pool_guard);
    let (enc, pool_ref) = frame.resources();

    // Embed + normalize
    let embed = gpu_ops.embed_tokens(
        enc,
        pool_ref,
        ModelInput::TokensGpu(&input_gpu_enc),
        None,
        0,
    )?;
    let mut gpu_hidden = gpu_encoder.embed_norm(enc, pool_ref, &embed)?;

    // All layers
    for layer_idx in 0..num_layers {
        gpu_hidden = gpu_encoder.forward_layers(
            enc,
            pool_ref,
            &gpu_hidden,
            &attention_mask_gpu,
            layer_idx,
            layer_idx + 1,
        )?;
    }

    frame.finish();
    gpu_hidden.to_ndarray_3d::<f32>().await?
};

// Compare final output
let max_diff = cpu_hidden
    .iter()
    .zip(gpu_final.iter())
    .map(|(a, b)| (a - b).abs())
    .fold(0.0f32, f32::max);

log::info!("Final encoder output max diff: {:.6}", max_diff);

assert_all_close(
    &cpu_hidden.view().into_dyn(),
    &gpu_final.view().into_dyn(),
    1e-3,
    1e-3,  // Slightly higher tolerance for accumulated error over 12 layers
    "Encoder All Layers",
);

    // Now continue with your existing STEP 1...

    // --- 2. ENCODER PARITY CHECK ---
    log::info!("\n--- STEP 1: CHECKING ENCODER PARITY ---");
    let cpu_encoder_state_enum = cpu_backend.encode(&cpu_model, tokens, num_beams).await?;
    let gpu_encoder_state_enum = gpu_backend.encode(&gpu_model, tokens, num_beams).await?;

    let cpu_hs = match &cpu_encoder_state_enum {
        CpuSeq2SeqState::EncoderState { hidden_states, .. } => hidden_states,
        _ => panic!("Expected CpuSeq2SeqState::EncoderState"),
    };

    let gpu_hs_gpu = match &gpu_encoder_state_enum {
        GpuSeq2SeqState::EncoderOutput { hidden_states, .. } => hidden_states,
        _ => panic!("Expected GpuSeq2SeqState::EncoderOutput"),
    };
    let gpu_hs_cpu = gpu_hs_gpu.to_ndarray_3d::<f32>().await?;

    assert_all_close(
        &cpu_hs.view().into_dyn(),
        &gpu_hs_cpu.view().into_dyn(),
        1e-3,
        1e-4,
        "Encoder Hidden States",
    );

    // --- DECODER EMBEDDINGS PARITY ---
    // Prepare identical inputs for the first decode step
    let decoder_start_token_id = cpu_model.decoder_start_token_id();
    let initial_decoder_tokens = vec![decoder_start_token_id; num_beams];
    log::info!("\n--- STEP 2a: DECODER EMBEDDINGS ---");

    let cpu_decoder = cpu_model.pipeline.cpu_decoder().expect("No CPU decoder");
    let gpu_decoder = gpu_model.pipeline.gpu_decoder().expect("No GPU decoder");

    // Single token input for first decode step
    let decoder_input_cpu =
        ndarray::Array2::from_shape_vec((num_beams, 1), initial_decoder_tokens.clone())?;
    let decoder_input_gpu = GpuTensor::from_ndarray(&ctx, &decoder_input_cpu)?;

    let position_offset = 0; // First decode step

    // CPU decoder embeddings
    let cpu_dec_embed = cpu_decoder.embed(&decoder_input_cpu, position_offset);
    println!("CPU decoder embed shape: {:?}", cpu_dec_embed.shape());
    println!(
        "CPU decoder embed first 5: {:?}",
        cpu_dec_embed.iter().take(5).collect::<Vec<_>>()
    );

    // GPU decoder embeddings
    {
        let pool = ctx.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let gpu_dec_embed = gpu_decoder.embed(
            enc,
            pool_ref,
            ModelInput::TokensGpu(&decoder_input_gpu),
            position_offset,
        )?;

        frame.finish();

        let gpu_dec_embed_cpu = gpu_dec_embed.to_ndarray_3d::<f32>().await?;
        println!("GPU decoder embed shape: {:?}", gpu_dec_embed_cpu.shape());
        println!(
            "GPU decoder embed first 5: {:?}",
            gpu_dec_embed_cpu.iter().take(5).collect::<Vec<_>>()
        );

        assert_all_close(
            &cpu_dec_embed.view().into_dyn(),
            &gpu_dec_embed_cpu.view().into_dyn(),
            1e-3,
            1e-4,
            "Decoder Embeddings",
        );
    }

    // --- DECODER EMBED + LAYERNORM PARITY ---
    log::info!("\n--- STEP 2b: DECODER EMBED + LAYERNORM ---");
    let cpu_dec_ln = cpu_decoder.embed_and_normalize(&decoder_input_cpu, position_offset)?;

    {
        let pool = ctx.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let gpu_dec_ln = gpu_decoder.embed_and_normalize(
            enc,
            pool_ref,
            ModelInput::TokensGpu(&decoder_input_gpu),
            position_offset,
        )?;
        frame.finish();

        let gpu_dec_ln_cpu = gpu_dec_ln.to_ndarray_3d::<f32>().await?;

        assert_all_close(
            &cpu_dec_ln.view().into_dyn(),
            &gpu_dec_ln_cpu.view().into_dyn(),
            1e-3,
            1e-4,
            "Decoder Embed + LayerNorm",
        );
    }

    // --- 3. DECODER STEP PARITY CHECK ---
    log::info!("\n--- STEP 2: CHECKING DECODER STEP PARITY ---");

    let mut cpu_cache = cpu_model.new_cache(1, 142, num_beams)?;
    let mut gpu_cache = gpu_model.new_cache(1, 142, num_beams)?;

    let cpu_decoder_tokens_enum =
        cpu_backend.create_token_tensor(&initial_decoder_tokens, num_beams)?;
    let gpu_decoder_tokens_enum =
        gpu_backend.create_token_tensor(&initial_decoder_tokens, num_beams)?;

    // Run both backends
    let cpu_logits = cpu_backend
        .decode_step(
            &cpu_model,
            &cpu_decoder_tokens_enum,
            &cpu_encoder_state_enum,
            cpu_cache.as_mut(),
        )
        .await?;

    let gpu_logits = gpu_backend
        .decode_step(
            &gpu_model,
            &gpu_decoder_tokens_enum,
            &gpu_encoder_state_enum,
            gpu_cache.as_mut(),
        )
        .await?;

    assert_all_close(
        &cpu_logits.view().into_dyn(),
        &gpu_logits.view().into_dyn(),
        1e-3,
        1e-4,
        "Decoder Step 1 Logits",
    );

    log::info!("\n--- ALL PARITY CHECKS PASSED ---");
    log::info!(
        "The GPU implementation appears to be numerically correct. Now running full generation..."
    );

    // --- 4. RUN FULL GENERATION (if checks pass) ---
    let generator = EncoderDecoderGenerator::new(Box::new(gpu_model))?;
    let generation_config = generator.model.get_default_generation_config();
    let summary = generator
        .generate(article, Some(&generation_config))
        .await?;

    println!("\n--- FINAL GPU GENERATED SUMMARY ---");
    println!("Summary: {}", summary);

    Ok(())
}
