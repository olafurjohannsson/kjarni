// whisper/model.rs

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3, s};
use tokenizers::Tokenizer;

use super::config::WhisperConfig;

use kjarni_transformers::{
    LanguageModel, ModelType, WgpuContext, audio::{AudioConvFrontend, MelConfig, compute_mel_spectrogram}, cache::{Cache, CpuBeamKVCache}, common::{
        BeamSearchParams, DecodingStrategy, GenerationConfig, HFGenerationConfig,
        HFGenerationDefaults, ModelGenerationDefaults,
    }, cpu::{
        encoder::{CpuEncoderOps, prelude::*, traits::CpuEncoder},
        encoder_decoder::{
            cpu_decoder::{Seq2SeqCPUDecoder, Seq2SeqDecoderConfig},
            cpu_encoder::{Seq2SeqCPUEncoder, Seq2SeqEncoderConfig},
        },
    }, encoder_decoder::traits::{
        CpuCrossDecoder, CpuEncoderDecoderOps, EncoderDecoderLanguageModel, GpuCrossDecoder,
    }, models::base::ModelLoadConfig, pipeline::{EncoderDecoderModelFactory, EncoderDecoderPipeline}, traits::{Device, InferenceModel, ModelConfig as _, ModelLayout, ModelMetadata}, weights::ModelWeights
};

// =============================================================================
// WhisperModel
// =============================================================================

pub struct WhisperModel {
    tokenizer: Tokenizer,
    config: Arc<WhisperConfig>,

    // Audio frontend: conv1 + GELU + conv2 + GELU + positional embeddings
    audio_frontend: AudioConvFrontend,

    pipeline: EncoderDecoderPipeline,

    // Transformer encoder (reuses Seq2Seq infrastructure)
    encoder: Seq2SeqCPUEncoder,

    // Transformer decoder with cross-attention
    decoder: Seq2SeqCPUDecoder,

    // LM head weights for projecting to vocab
    lm_head_weight: Array2<f32>,

    // Mel spectrogram configuration
    mel_config: MelConfig,

    // Generation configuration
    generation_config: HFGenerationConfig,
}


impl EncoderDecoderModelFactory for WhisperModel {
    type Config = WhisperConfig;

    fn load_config(weights: &ModelWeights) -> Result<Arc<Self::Config>> {
        let config: WhisperConfig = serde_json::from_str(&weights.config_json())?;
        Ok(Arc::new(config))
    }

    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        _layout: &ModelLayout,
        config: &Arc<WhisperConfig>,
        load_config: &ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(
        Option<Box<dyn CpuEncoder>>,
        Option<Box<dyn GpuEncoder>>,
        Option<Box<dyn CpuCrossDecoder>>,
        Option<Box<dyn GpuCrossDecoder>>,
    )> {
        let mut cpu_enc = None;
        let mut cpu_dec = None;
        let gpu_enc = None; // TODO: GPU T5
        let gpu_dec = None;

        // CPU Backends
        if device.is_cpu() || load_config.offload_embeddings {
            // T5 Encoder
            let enc_config = Seq2SeqEncoderConfig::t5();
            cpu_enc = Some(Box::new(Seq2SeqCPUEncoder::new(
                weights,
                config.as_ref(),
                enc_config,
                *load_config,
            )?) as Box<dyn CpuEncoder>);

            // T5 Decoder
            let dec_config = Seq2SeqDecoderConfig::t5();
            cpu_dec = Some(Box::new(Seq2SeqCPUDecoder::new(
                weights,
                config.as_ref(),
                dec_config,
                *load_config,
            )?) as Box<dyn CpuCrossDecoder>);
        } else if device.is_gpu() {
            todo!()
        }

        // TODO: GPU backends for T5
        // if let Some(ctx) = context { ... }

        Ok((cpu_enc, gpu_enc, cpu_dec, gpu_dec))
    }

    fn new_from_pipeline(
        pipeline: EncoderDecoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<WhisperConfig>,
        _: Option<HFGenerationDefaults>,
        generation_config: HFGenerationConfig,
    ) -> Self {
        Self {
            pipeline,
            tokenizer,
            config,
            generation_config,
        }
    }
}

impl WhisperModel {
    // =========================================================================
    // Construction
    // =========================================================================

    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        _context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
       kjarni_transformers::pipeline::Seq2SeqLoader::load_from_registry::<Self>(
            model_type,
            cache_dir,
            device,
            context,
            load_config,
        )
        .await
    }

    pub fn from_weights(
        weights: ModelWeights,
        config: Arc<WhisperConfig>,
        tokenizer: Tokenizer,
        generation_config: HFGenerationConfig,
        load_config: ModelLoadConfig,
    ) -> Result<Self> {
        // 1. Load audio conv frontend
        let audio_frontend = AudioConvFrontend::from_weights(
            &weights,
            "model.encoder",
            config.max_source_positions,
        )?;

        // 2. Load encoder transformer layers
        let enc_config = Seq2SeqEncoderConfig {
            num_layers: config.encoder_layers,
            hidden_size: config.d_model,
            num_attention_heads: config.encoder_attention_heads,
            intermediate_size: config.encoder_ffn_dim,
            activation: config.activation_function.clone(),
            layer_norm_eps: 1e-5,
            // Whisper encoder doesn't use token embeddings - we handle that separately
            skip_token_embedding: true,
            use_bias: true,
            is_prenorm: true,
        };

        let encoder = Seq2SeqCPUEncoder::new(
            &weights,
            config.as_ref(),
            enc_config,
            load_config,
        )?;

        // 3. Load decoder transformer layers
        let dec_config = Seq2SeqDecoderConfig {
            num_layers: config.decoder_layers,
            hidden_size: config.d_model,
            num_attention_heads: config.decoder_attention_heads,
            intermediate_size: config.decoder_ffn_dim,
            activation: config.activation_function.clone(),
            layer_norm_eps: 1e-5,
            use_bias: true,
            is_prenorm: true,
            has_cross_attention: true,
            max_position_embeddings: config.max_target_positions,
        };

        let decoder = Seq2SeqCPUDecoder::new(
            &weights,
            config.as_ref(),
            dec_config,
            load_config,
        )?;

        // 4. Load LM head
        // Whisper typically uses "proj_out.weight" or ties with decoder embeddings
        let lm_head_weight = weights
            .get_array2("proj_out.weight")
            .or_else(|_| weights.get_array2("model.decoder.embed_tokens.weight"))
            .map_err(|e| anyhow!("Failed to load LM head: {}", e))?;

        // 5. Mel config for Whisper
        let mel_config = MelConfig::whisper();

        Ok(Self {
            tokenizer,
            config,
            audio_frontend,
            encoder,
            decoder,
            lm_head_weight,
            mel_config,
            generation_config,
        })
    }

    // =========================================================================
    // Audio Encoding
    // =========================================================================

    /// Encode mel spectrogram to hidden states.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram `[n_mels, frames]` (typically `[80, 3000]`)
    ///
    /// # Returns
    /// Encoder hidden states `[1, seq_len, hidden_size]`
    pub fn encode_mel(&self, mel: &Array2<f32>) -> Result<Array3<f32>> {
        // Add batch dimension: [n_mels, frames] -> [1, n_mels, frames]
        let mel_batch = mel.clone().insert_axis(ndarray::Axis(0));
        self.encode_mel_batch(&mel_batch)
    }

    /// Encode batched mel spectrograms.
    ///
    /// # Arguments
    /// * `mel` - Mel spectrograms `[batch, n_mels, frames]`
    ///
    /// # Returns
    /// Encoder hidden states `[batch, seq_len, hidden_size]`
    pub fn encode_mel_batch(&self, mel: &Array3<f32>) -> Result<Array3<f32>> {
        // 1. Conv frontend: [batch, n_mels, frames] -> [batch, frames/2, hidden_size]
        let hidden_states = self.audio_frontend.forward(mel)?;

        // 2. Create attention mask (all ones for audio - no padding)
        let (batch_size, seq_len, _) = hidden_states.dim();
        let attention_mask = Array2::<f32>::ones((batch_size, seq_len));

        // 3. Transformer encoder layers
        let encoder_output = self.encoder.forward_layers(
            &hidden_states,
            &attention_mask,
            0,
            self.encoder.num_layers(),
        )?;

        // 4. Final layer norm
        self.encoder.final_norm(&encoder_output)
    }

    /// Encode raw audio samples to hidden states.
    ///
    /// Convenience method that computes mel spectrogram first.
    ///
    /// # Arguments
    /// * `audio` - Raw audio samples (16kHz, mono, normalized to [-1, 1])
    ///
    /// # Returns
    /// Encoder hidden states `[1, seq_len, hidden_size]`
    pub fn encode_audio(&self, audio: &[f32]) -> Result<Array3<f32>> {
        let mel = compute_mel_spectrogram(audio, &self.mel_config)?;
        self.encode_mel(&mel)
    }

    // =========================================================================
    // Decoding
    // =========================================================================

    /// Decode with cross-attention to encoder states.
    ///
    /// # Arguments
    /// * `decoder_input_ids` - Decoder token IDs `[batch, seq_len]`
    /// * `encoder_hidden_states` - Encoder output `[batch, enc_seq_len, hidden_size]`
    /// * `attention_mask` - Causal mask for decoder self-attention
    /// * `position_offset` - Position offset for KV cache
    /// * `cache` - Optional KV cache
    ///
    /// # Returns
    /// Decoder hidden states `[batch, seq_len, hidden_size]`
    pub fn decode(
        &self,
        decoder_input_ids: &Array2<u32>,
        encoder_hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        // Embed decoder tokens
        let decoder_hidden = self.decoder.embed(decoder_input_ids, position_offset)?;

        // Forward through decoder with cross-attention
        self.decoder.forward_with_cross_attention(
            &decoder_hidden,
            encoder_hidden_states,
            attention_mask,
            position_offset,
            cache,
        )
    }

    // =========================================================================
    // Projection
    // =========================================================================

    /// Project hidden states to vocabulary logits.
    fn project_to_vocab(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq_len, hidden_size) = hidden_states.dim();
        let vocab_size = self.lm_head_weight.dim().0;

        let mut logits = Array3::<f32>::zeros((batch, seq_len, vocab_size));

        for b in 0..batch {
            for s in 0..seq_len {
                let hidden = hidden_states.slice(s![b, s, ..]);
                for v in 0..vocab_size {
                    let weight = self.lm_head_weight.slice(s![v, ..]);
                    logits[[b, s, v]] = hidden.dot(&weight);
                }
            }
        }

        Ok(logits)
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the mel spectrogram configuration.
    pub fn mel_config(&self) -> &MelConfig {
        &self.mel_config
    }

    /// Get model config.
    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    /// Get metadata.
    pub fn meta(&self) -> ModelMetadata {
        self.config.metadata()
    }

    /// Get layout.
    pub fn layout(&self) -> ModelLayout {
        self.config.layout()
    }

    // =========================================================================
    // Supported Models
    // =========================================================================

    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::WhisperTiny,
        ModelType::WhisperBase,
        ModelType::WhisperSmall,
        ModelType::WhisperMedium,
        ModelType::WhisperLarge,
        ModelType::WhisperLargeV2,
        ModelType::WhisperLargeV3,
    ];
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl InferenceModel for WhisperModel {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn context(&self) -> Option<Arc<WgpuContext>> {
        None
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl LanguageModel for WhisperModel {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn context_size(&self) -> usize {
        self.config.max_target_positions
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn hidden_size(&self) -> usize {
        self.config.d_model
    }

    fn num_heads(&self) -> usize {
        self.config.decoder_attention_heads
    }

    fn num_layers(&self) -> usize {
        self.config.decoder_layers
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.generation_config
            .eos_token_id
            .as_ref()
            .map(|e| e.primary())
            .or(Some(self.config.eos_token_id))
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(self.config.bos_token_id)
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.generation_config
            .pad_token_id
            .or(Some(self.config.pad_token_id))
    }

    fn forced_bos_token_id(&self) -> Option<u32> {
        self.generation_config.forced_bos_token_id
    }

    fn forced_eos_token_id(&self) -> Option<u32> {
        self.generation_config.forced_eos_token_id
    }

    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        num_beams: usize,
    ) -> Result<Box<dyn Cache>> {
        let effective_batch = if num_beams > 0 { num_beams } else { batch_size };
        Ok(Box::new(CpuBeamKVCache::new(
            self.config.decoder_layers,
            effective_batch,
            max_len,
            self.config.d_model,
        )))
    }
}

impl CpuEncoderOps for WhisperModel {
    fn encoder(&self) -> &dyn CpuEncoder {
        &self.encoder
    }
}

impl CpuEncoderDecoderOps for WhisperModel {
    fn decoder(&self) -> &dyn CpuCrossDecoder {
        &self.decoder
    }

    fn get_decoder_mask(&self, seq_len: usize, past_len: usize) -> Option<Array2<f32>> {
        Some(kjarni_transformers::utils::create_causal_mask(seq_len, past_len))
    }

    fn broadcast_encoder_states(
        &self,
        encoder_hidden_states: &Array3<f32>,
        num_beams: usize,
    ) -> Result<Array3<f32>> {
        Ok(encoder_hidden_states
            .broadcast((
                num_beams,
                encoder_hidden_states.shape()[1],
                encoder_hidden_states.shape()[2],
            ))
            .ok_or_else(|| anyhow!("Failed to broadcast encoder state"))?
            .to_owned())
    }

    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        self.project_to_vocab(hidden_states)
    }
}

#[async_trait]
impl EncoderDecoderLanguageModel for WhisperModel {
    fn get_generation_config_for_input(&self, _input: &str) -> GenerationConfig {
        self.get_default_generation_config()
    }

    fn encoder_decoder_cpu_ops(&self) -> Option<&dyn CpuEncoderDecoderOps> {
        Some(self)
    }

    fn encoder_decoder_gpu_ops(&self) -> Option<&dyn kjarni_transformers::encoder_decoder::traits::GpuEncoderDecoderOps> {
        None
    }

    fn decoder_start_token_id(&self) -> u32 {
        self.generation_config
            .decoder_start_token_id
            .unwrap_or(self.config.decoder_start_token_id)
    }

    fn get_default_generation_config(&self) -> GenerationConfig {
        GenerationConfig {
            max_length: self.config.max_target_positions,
            min_length: 0,
            no_repeat_ngram_size: 0,
            repetition_penalty: 1.0,
            max_new_tokens: Some(448), // Whisper default
            add_bos_token: false,
            strategy: DecodingStrategy::Greedy, // Whisper typically uses greedy
        }
    }
}

// =============================================================================
// Audio-Specific Encoder Trait
// =============================================================================

/// Extension trait for audio-capable models.
pub trait AudioEncoderOps: Send + Sync {
    /// Encode mel spectrogram to hidden states.
    fn encode_mel(&self, mel: &Array2<f32>) -> Result<Array3<f32>>;

    /// Encode mel spectrogram batch.
    fn encode_mel_batch(&self, mel: &Array3<f32>) -> Result<Array3<f32>>;

    /// Encode raw audio samples.
    fn encode_audio(&self, audio: &[f32]) -> Result<Array3<f32>>;

    /// Get mel config.
    fn mel_config(&self) -> &MelConfig;
}

impl AudioEncoderOps for WhisperModel {
    fn encode_mel(&self, mel: &Array2<f32>) -> Result<Array3<f32>> {
        WhisperModel::encode_mel(self, mel)
    }

    fn encode_mel_batch(&self, mel: &Array3<f32>) -> Result<Array3<f32>> {
        WhisperModel::encode_mel_batch(self, mel)
    }

    fn encode_audio(&self, audio: &[f32]) -> Result<Array3<f32>> {
        WhisperModel::encode_audio(self, audio)
    }

    fn mel_config(&self) -> &MelConfig {
        WhisperModel::mel_config(self)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_config() {
        let mel_config = MelConfig::whisper();
        assert_eq!(mel_config.sample_rate, 16000);
        assert_eq!(mel_config.n_fft, 400);
        assert_eq!(mel_config.hop_length, 160);
        assert_eq!(mel_config.n_mels, 80);
    }
}