use anyhow::Result;
use std::sync::Arc;
use ndarray::{ArrayViewD, IxDyn};
// use edgemodels::generation::seq2seq::Seq2SeqGenerator; // Your new generator
// use edgemodels::generation::seq2seq2::Seq2SeqGenerator as Seq2SeqGeneratorNew; // Your new generator
use edgemodels::generation::encoder_decoder::Seq2SeqGenerator;
use edgemodels::generation::encoder_decoder::{CpuBackend, GpuBackend};
use edgemodels::generation::encoder_decoder::CpuTensor;
use edgemodels::generation::encoder_decoder::GpuSeq2SeqTensor;
use edgetransformers::gpu_ops::tensor::GpuTensor;
use edgetransformers::encoder_decoder::traits::EncoderDecoderLanguageModel;
use edgetransformers::encoder_decoder::traits::EncoderDecoderGenerationBackend;
// use edgemodels::generation::seq2seq::Seq2SeqGenerator as Seq2SeqGeneratorOld;
use edgetransformers::models::base::LanguageModel;
use edgemodels::generation::DecodingStrategy;
// use edgemodels::seq2seq::AnySeq2SeqModel;
use edgemodels::models::bart::model::BartModel;
use edgetransformers::{Device, ModelType, WgpuContext};
async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}
use anyhow::anyhow;
use std::io;
use std::io::Write;
fn assert_all_close(a: &ArrayViewD<f32>, b: &ArrayViewD<f32>, rtol: f32, atol: f32, context: &str) {
    if a.shape() != b.shape() {
        panic!("[DEBUG FAIL] {} shape mismatch! CPU: {:?}, GPU: {:?}", context, a.shape(), b.shape());
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
        log::info!("[DEBUG PASS] {} matches. Max absolute difference: {}", context, max_abs_diff);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
     env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // --- 1. SETUP ---
    log::info!("Loading models for CPU and GPU...");
    let ctx = get_test_context().await;
    let model_type = ModelType::DistilBartCnn;
    
    let cpu_model = BartModel::from_registry(model_type, None, Device::Cpu, None).await?;
    let gpu_model = BartModel::from_registry(model_type, None, Device::Wgpu, Some(ctx.clone())).await?;

    let cpu_backend = CpuBackend;
    let gpu_backend = GpuBackend::new(ctx)?;
    
    let article = "Rust is a multi-paradigm, general-purpose programming language.";
    let num_beams = 4; // Use the same number of beams for both

    let encoding = cpu_model.tokenizer().encode(article, true).map_err(|e| anyhow!(e))?;
    let tokens = encoding.get_ids();

    // --- 2. ENCODER PARITY CHECK ---
    log::info!("\n--- STEP 1: CHECKING ENCODER PARITY ---");
    let cpu_encoder_state_enum = cpu_backend.encode(&cpu_model, tokens, num_beams).await?;
    let gpu_encoder_state_enum = gpu_backend.encode(&gpu_model, tokens, num_beams).await?;

    let cpu_hs = match &cpu_encoder_state_enum {
        CpuTensor::EncoderState { state, .. } => state,
        _ => panic!("Expected CpuTensor::EncoderState"),
    };

    let gpu_hs_gpu = match &gpu_encoder_state_enum {
        GpuSeq2SeqTensor::EncoderOutput { hidden_states, .. } => hidden_states,
        _ => panic!("Expected GpuSeq2SeqTensor::EncoderOutput"),
    };
    let gpu_hs_cpu = gpu_hs_gpu.to_ndarray_3d::<f32>().await?;
    
    assert_all_close(&cpu_hs.view().into_dyn(), &gpu_hs_cpu.view().into_dyn(), 1e-3, 1e-4, "Encoder Hidden States");

    // --- 3. DECODER STEP PARITY CHECK ---
    log::info!("\n--- STEP 2: CHECKING DECODER STEP PARITY ---");
    
    // Prepare identical inputs for the first decode step
    let decoder_start_token_id = cpu_model.decoder_start_token_id();
    let initial_decoder_tokens = vec![decoder_start_token_id; num_beams];
    
    let mut cpu_cache = cpu_model.new_cache(1, 142, num_beams)?;
    let mut gpu_cache = gpu_model.new_cache(1, 142, num_beams)?;
    
    let cpu_decoder_tokens_enum = cpu_backend.create_token_tensor(&initial_decoder_tokens, num_beams)?;
    let gpu_decoder_tokens_enum = gpu_backend.create_token_tensor(&initial_decoder_tokens, num_beams)?;

    // Run both backends
    let cpu_logits = cpu_backend.decode_step(
        &cpu_model,
        &cpu_decoder_tokens_enum,
        &cpu_encoder_state_enum,
        cpu_cache.as_mut(),
    ).await?;

    let gpu_logits = gpu_backend.decode_step(
        &gpu_model,
        &gpu_decoder_tokens_enum,
        &gpu_encoder_state_enum,
        gpu_cache.as_mut(),
    ).await?;
    
    assert_all_close(&cpu_logits.view().into_dyn(), &gpu_logits.view().into_dyn(), 1e-3, 1e-4, "Decoder Step 1 Logits");

    log::info!("\n--- ALL PARITY CHECKS PASSED ---");
    log::info!("The GPU implementation appears to be numerically correct. Now running full generation...");

    // --- 4. RUN FULL GENERATION (if checks pass) ---
    let generator = Seq2SeqGenerator::new(Box::new(gpu_model))?;
    let generation_config = generator.model.get_default_generation_config();
    let summary = generator.generate(article, Some(&generation_config)).await?;

    println!("\n--- FINAL GPU GENERATED SUMMARY ---");
    println!("Summary: {}", summary);

    Ok(())
}