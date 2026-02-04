use anyhow::Result;
use kjarni_models::models::whisper::WhisperModel;
use kjarni_transformers::audio::{MelConfig, compute_mel_spectrogram, load_audio_for_whisper};
use kjarni_transformers::cpu::encoder::CpuEncoderOps;
use kjarni_transformers::encoder_decoder::traits::{
    CpuEncoderDecoderOps, EncoderDecoderLanguageModel,
};
use kjarni_transformers::{Device, LanguageModel, ModelType};
use ndarray::{Array2, s};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // 1. Load audio file
    let audio_path = "/home/olafurj/dev/edgebert/crates/kjarni-models/examples/hideyowife.wav";
    log::info!("Loading audio from: {}", audio_path);

    let audio_samples = load_audio_for_whisper(audio_path)?;
    log::info!(
        "Loaded {} samples ({:.2}s at 16kHz)",
        audio_samples.len(),
        audio_samples.len() as f32 / 16000.0
    );

    // 2. Load Whisper model
    log::info!("Loading Whisper model...");
    let model_type = ModelType::WhisperSmall; // Start small for testing
    let model = WhisperModel::from_registry(
        model_type,
        None, // Default cache directory
        Device::Cpu,
        None, // No WgpuContext for CPU
        None, // Default ModelLoadConfig
    )
    .await?;
    log::info!("Model loaded successfully!");

    // 3. Compute mel spectrogram
    let mel_config = model.expected_mel_config();
    log::info!("Computing mel spectrogram with config: {:?}", mel_config);

    let mel_spectrogram = compute_mel_spectrogram(&audio_samples, &mel_config)?;
    log::info!("Mel spectrogram shape: {:?}", mel_spectrogram.dim());

    // 4. Add batch dimension: [n_mels, frames] -> [1, n_mels, frames]
    let mel_batch = mel_spectrogram
        .insert_axis(ndarray::Axis(0))
        .as_standard_layout()
        .to_owned();
    log::info!("Mel batch shape: {:?}", mel_batch.dim());

    // 5. Run through audio frontend + encoder
    log::info!("Running encoder...");
    let encoder_hidden_states = {
        // embed_audio runs through AudioConvFrontend
        let embedded = model.embed_audio(&mel_batch)?;
        log::info!("After audio frontend shape: {:?}", embedded.dim());

        // Create attention mask (all ones for audio - no padding)
        let seq_len = embedded.dim().1;
        let attention_mask = Array2::<f32>::ones((1, seq_len));

        // Run through encoder transformer layers
        let encoder = model.encoder();
        let output = encoder.forward(&embedded, &attention_mask)?;
        output.last_hidden_state
    };
    log::info!("Encoder output shape: {:?}", encoder_hidden_states.dim());

    // 6. Decode (greedy for simplicity)
    log::info!("Decoding...");
    let transcription = greedy_decode(&model, &encoder_hidden_states, 100)?;

    println!("\n========================================");
    println!("Transcription: {}", transcription);
    println!("========================================\n");

    Ok(())
}

/// Simple greedy decoding loop
fn greedy_decode(
    model: &WhisperModel,
    encoder_hidden_states: &ndarray::Array3<f32>,
    max_length: usize,
) -> Result<String> {
    use ndarray::Array2;

    let tokenizer = model.tokenizer();
    let decoder_start_token_id = model.decoder_start_token_id();
    let eos_token_id = model.eos_token_id().unwrap_or(50257); // Whisper EOS

    // Whisper's typical start sequence for English transcription:
    // <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
    // For now, just use decoder_start_token_id
    let mut generated_ids: Vec<u32> = vec![decoder_start_token_id];

    // Get decoder ops
    let decoder_ops = model
        .encoder_decoder_cpu_ops()
        .ok_or_else(|| anyhow::anyhow!("CPU decoder not available"))?;

    // Create cache for KV caching
    let mut cache = model.new_cache(1, max_length, 0)?;

    // Encoder sequence length for encoder padding mask
    let encoder_seq_len = encoder_hidden_states.dim().1;
    let encoder_padding_mask = Array2::<f32>::ones((1, encoder_seq_len));

    for step in 0..max_length {
        // Build decoder input
        let decoder_input_ids =
            Array2::from_shape_vec((1, generated_ids.len()), generated_ids.clone())?;

        // Decoder padding mask (causal mask is handled internally)
        // For now: all ones (no padding)
        let decoder_seq_len = generated_ids.len();
        let decoder_padding_mask = Array2::<f32>::ones((1, decoder_seq_len));

        // Run decoder
        let decoder = decoder_ops.decoder();
        let output = decoder.forward(
            &decoder_input_ids,
            encoder_hidden_states,
            Some(&decoder_padding_mask),
            Some(&encoder_padding_mask),
            Some(cache.as_mut()),
            None, // cross_kv_cache - let decoder manage this
        )?;

        // Project to logits
        let logits = decoder_ops.project_to_logits(&output.last_hidden_state)?;

        // Get last token logits and find argmax
        let last_logits = logits.slice(s![0, -1, ..]);
        let next_token_id = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(eos_token_id);

        // Check for EOS
        if next_token_id == eos_token_id {
            log::info!("EOS reached at step {}", step);
            break;
        }

        generated_ids.push(next_token_id);

        // Debug: print token as we go
        if let Ok(token_str) = tokenizer.decode(&[next_token_id], false) {
            print!("{}", token_str);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    println!(); // newline after streaming output

    // Decode tokens to string (skip the start token)
    let output_ids: Vec<u32> = generated_ids[1..].to_vec();
    let transcription = tokenizer
        .decode(&output_ids, true) // skip_special_tokens = true
        .map_err(|e| anyhow::anyhow!("Tokenizer decode error: {}", e))?;

    Ok(transcription)
}
