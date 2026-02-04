use anyhow::Result;
use kjarni_transformers::cache::*;
use kjarni_models::models::whisper::WhisperModel;
use kjarni_transformers::audio::{compute_mel_spectrogram, load_audio_for_whisper};
use kjarni_transformers::cache::CpuBeamKVCache;
use kjarni_transformers::cpu::encoder::CpuEncoderOps;
use kjarni_transformers::encoder_decoder::traits::{CpuEncoderDecoderOps, EncoderDecoderLanguageModel};
use kjarni_transformers::{Device, LanguageModel, ModelType};
use ndarray::{Array2, s};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // 1. Load audio file
    let audio_path = "./crates/kjarni-models/examples/hideyowife.wav";
    log::info!("Loading audio from: {}", audio_path);
    
    let audio_samples = load_audio_for_whisper(audio_path)?;
    log::info!("Loaded {} samples ({:.2}s at 16kHz)", 
        audio_samples.len(), 
        audio_samples.len() as f32 / 16000.0
    );

    // 2. Load Whisper model
    log::info!("Loading Whisper model...");
    let model_type = ModelType::WhisperSmall;
    let model = WhisperModel::from_registry(
        model_type,
        None,
        Device::Cpu,
        None,
        None,
    )
    .await?;
    log::info!("Model loaded successfully!");

    // 3. Compute mel spectrogram
    let mel_config = model.expected_mel_config();
    log::info!("Computing mel spectrogram with config: {:?}", mel_config);
    
    let mel_spectrogram = compute_mel_spectrogram(&audio_samples, &mel_config)?;
    log::info!("Mel spectrogram shape: {:?}", mel_spectrogram.dim());

    // 4. Add batch dimension and ensure contiguous layout
    let mel_batch = mel_spectrogram
        .insert_axis(ndarray::Axis(0))
        .as_standard_layout()
        .to_owned();
    log::info!("Mel batch shape: {:?}", mel_batch.dim());

    // 5. Run encoder
    log::info!("Running encoder...");
let encoder_hidden_states = {
    let embedded = model.embed_audio(&mel_batch)?;
    let embedded = embedded.as_standard_layout().to_owned();
    log::info!("After audio frontend shape: {:?}", embedded.dim());
    
    // Remove the embed_norm call - it's now handled correctly by the encoder
    let seq_len = embedded.dim().1;
    let attention_mask = Array2::<f32>::ones((1, seq_len));
    
    let encoder = model.encoder();
    let output = encoder.forward(&embedded, &attention_mask)?;
    output.last_hidden_state
};
    log::info!("Encoder output shape: {:?}", encoder_hidden_states.dim());
    log::info!("Encoder stats - min: {:.4}, max: {:.4}, mean: {:.4}", 
        encoder_hidden_states.iter().cloned().fold(f32::INFINITY, f32::min),
        encoder_hidden_states.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        encoder_hidden_states.iter().sum::<f32>() / encoder_hidden_states.len() as f32
    );

    // 6. Decode
    log::info!("Decoding...");
    let transcription = greedy_decode(&model, &encoder_hidden_states, 100)?;
    
    println!("\n========================================");
    println!("Transcription: {}", transcription);
    println!("========================================\n");

    Ok(())
}
fn greedy_decode(
    model: &WhisperModel,
    encoder_hidden_states: &ndarray::Array3<f32>,
    max_length: usize,
) -> Result<String> {
    let tokenizer = model.tokenizer();
    let eos_token_id = model.eos_token_id().unwrap_or(50257);
    
    // Whisper special tokens start at 50257
    // We should only generate regular text tokens (< 50257) during transcription
    const FIRST_SPECIAL_TOKEN: u32 = 50257;
    
    let prompt_tokens: Vec<u32> = vec![
        50258,  // <|startoftranscript|>
        50259,  // <|en|>
        50359,  // <|transcribe|>
        50363,  // <|notimestamps|>
    ];
    
    let decoder_ops = model.encoder_decoder_cpu_ops()
        .ok_or_else(|| anyhow::anyhow!("CPU decoder not available"))?;
    
    let mut cache = model.new_cache(1, max_length, 0)?;
    let cpu_cache = cache
        .as_any_mut()
        .downcast_mut::<CpuBeamKVCache>()
        .ok_or_else(|| anyhow::anyhow!("Expected CpuBeamKVCache"))?;
    
    let cross_kv_cache = decoder_ops.decoder().precompute_cross_attention_kv(encoder_hidden_states)?;
    
    let encoder_seq_len = encoder_hidden_states.dim().1;
    let encoder_padding_mask = Array2::<f32>::ones((1, encoder_seq_len));
    
    let decoder = decoder_ops.decoder();
    
    // Process prompt
    log::info!("Processing prompt tokens: {:?}", prompt_tokens);
    let decoder_input_ids = Array2::from_shape_vec(
        (1, prompt_tokens.len()), 
        prompt_tokens.clone()
    )?;
    let decoder_padding_mask = Array2::<f32>::ones((1, prompt_tokens.len()));
    
    let output = decoder.forward(
        &decoder_input_ids,
        encoder_hidden_states,
        Some(&decoder_padding_mask),
        Some(&encoder_padding_mask),
        Some(cpu_cache),
        Some(&cross_kv_cache),
    )?;
    
    for (i, (k, v)) in output.new_self_attn_kv.into_iter().enumerate() {
        cpu_cache.update(i, &k, &v)?;
    }
    cpu_cache.increment_len(prompt_tokens.len());
    
    // Get first predicted token (suppress special tokens)
    let logits = decoder_ops.project_to_logits(&output.last_hidden_state)?;
    let last_logits = logits.slice(ndarray::s![0, -1, ..]);
    
    // Log top predictions including special tokens for debugging
    let mut all_indexed: Vec<(usize, f32)> = last_logits.iter().cloned().enumerate().collect();
    all_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    log::info!("After prompt, top 10 (all): {:?}", &all_indexed[..10]);
    
    // Find best NON-SPECIAL token
    let mut next_token_id = last_logits
        .iter()
        .enumerate()
        .filter(|(idx, _)| (*idx as u32) < FIRST_SPECIAL_TOKEN)
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(eos_token_id);
    
    log::info!("First non-special token: {} ({})", 
        next_token_id,
        tokenizer.decode(&[next_token_id], false).unwrap_or_default()
    );
    
    let mut generated_ids: Vec<u32> = vec![next_token_id];
    
    if let Ok(token_str) = tokenizer.decode(&[next_token_id], false) {
        print!("{}", token_str);
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }
    
    // Generate remaining tokens
    for step in 0..(max_length - prompt_tokens.len()) {
        if next_token_id == eos_token_id {
            log::info!("EOS reached at step {}", step);
            break;
        }
        
        let decoder_input_ids = Array2::from_shape_vec((1, 1), vec![next_token_id])?;
        let decoder_padding_mask = Array2::<f32>::ones((1, 1));
        
        let output = decoder.forward(
            &decoder_input_ids,
            encoder_hidden_states,
            Some(&decoder_padding_mask),
            Some(&encoder_padding_mask),
            Some(cpu_cache),
            Some(&cross_kv_cache),
        )?;
        
        for (i, (k, v)) in output.new_self_attn_kv.into_iter().enumerate() {
            cpu_cache.update(i, &k, &v)?;
        }
        cpu_cache.increment_len(1);
        
        let logits = decoder_ops.project_to_logits(&output.last_hidden_state)?;
        let last_logits = logits.slice(ndarray::s![0, -1, ..]);
        
        // Find best non-special token (or EOS)
        next_token_id = last_logits
            .iter()
            .enumerate()
            .filter(|(idx, _)| {
                let id = *idx as u32;
                id < FIRST_SPECIAL_TOKEN || id == eos_token_id
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(eos_token_id);
        
        generated_ids.push(next_token_id);
        
        if let Ok(token_str) = tokenizer.decode(&[next_token_id], false) {
            print!("{}", token_str);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    println!();
    
    let transcription = tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {}", e))?;
    
    Ok(transcription)
}