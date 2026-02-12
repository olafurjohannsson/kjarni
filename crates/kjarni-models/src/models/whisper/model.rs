// whisper/model.rs

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use tokenizers::Tokenizer;

use super::config::WhisperConfig;

use kjarni_transformers::{
    LanguageModel, ModelType, WgpuContext,
    audio::{AudioConvFrontend, MelConfig},
    cache::{Cache, CpuBeamKVCache},
    common::{
        DecodingStrategy, GenerationConfig, HFGenerationConfig, HFGenerationDefaults,
    },
    cpu::{
        encoder::{CpuEncoderOps, GpuEncoderOps, prelude::*, traits::CpuEncoder},
        encoder_decoder::{
            cpu_decoder::{Seq2SeqCPUDecoder, Seq2SeqDecoderConfig},
            cpu_encoder::{Seq2SeqCPUEncoder, Seq2SeqEncoderConfig},
        },
    },
    encoder_decoder::traits::{
            CpuCrossDecoder, CpuEncoderDecoderOps, EncoderDecoderLanguageModel, GpuCrossDecoder,
            GpuEncoderDecoderOps,
        },
    gpu::{GpuTensor, GpuTensorPool},
    models::base::{ModelInput, ModelLoadConfig},
    pipeline::{EncoderDecoderModelFactory, EncoderDecoderPipeline, EncoderDecoderPipelineBuilder},
    traits::{Device, InferenceModel, ModelConfig as _, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};


pub struct WhisperModel {
    tokenizer: Tokenizer,
    config: Arc<WhisperConfig>,
    pipeline: EncoderDecoderPipeline,
    audio_frontend: AudioConvFrontend,
    generation_config: HFGenerationConfig,
}

impl WhisperModel {
    /// Load from Hugging Face model registry.
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        let info = model_type.info();
        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory")
                .join("kjarni")
        });
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        // Download model files
        kjarni_transformers::models::download_model_files(
            &model_dir,
            &info.paths,
            kjarni_transformers::models::registry::WeightsFormat::SafeTensors,
            true,
        )
        .await?;

        // Setup context for GPU if needed
        let context = if device.is_gpu() && context.is_none() {
            Some(WgpuContext::new().await?)
        } else {
            context
        };

        let generation_config = HFGenerationConfig::load_or_default(&model_dir);
        let load_config = load_config.unwrap_or_default();

        Self::from_pretrained(&model_dir, device, context, load_config, generation_config)
    }

    /// Load from local pretrained model directory.
    pub fn from_pretrained(
        model_path: &std::path::Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: ModelLoadConfig,
        generation_config: HFGenerationConfig,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;

        // Load config
        let config: WhisperConfig = serde_json::from_str(&weights.config_json())?;
        let config = Arc::new(config);
        let meta = config.metadata();
        let layout = config.layout();

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;

        // Load audio frontend (Whisper-specific)
        let audio_frontend = AudioConvFrontend::from_weights(
            &weights,
            "model.encoder",
            config.max_source_positions,
        )?;

        // Build encoder/decoder backends
        let (cpu_enc, gpu_enc, cpu_dec, gpu_dec) = Self::build_backends(
            &weights,
            &meta,
            &layout,
            &config,
            &load_config,
            context.as_ref(),
            device,
        )?;

        // Build pipeline with audio encoder flag
        let mut builder = EncoderDecoderPipelineBuilder::new(&weights, config.clone())
            .with_load_config(load_config)
            .with_context(context)
            .with_encoder_backends(cpu_enc, gpu_enc)
            .with_decoder_backends(cpu_dec, gpu_dec);

        builder.is_audio_encoder = true; // Skip encoder token embeddings

        let pipeline = builder.build()?;

        Ok(Self {
            tokenizer,
            config,
            pipeline,
            audio_frontend,
            generation_config,
        })
    }

    /// Returns the expected mel spectrogram configuration.
    pub fn expected_mel_config(&self) -> MelConfig {
        MelConfig {
            n_mels: self.config.num_mel_bins,
            ..MelConfig::whisper()
        }
    }

    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    pub fn meta(&self) -> ModelMetadata {
        self.config.metadata()
    }

    pub fn layout(&self) -> ModelLayout {
        self.config.layout()
    }

    pub const SUPPORTED_MODELS: &'static [ModelType] = &[
        // ModelType::WhisperTiny,
        // ModelType::WhisperBase,
        ModelType::WhisperSmall,
        // ModelType::WhisperMedium,
        // ModelType::WhisperLarge,
        // ModelType::WhisperLargeV2,
        ModelType::WhisperLargeV3,
    ];
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
        _context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(
        Option<Box<dyn CpuEncoder>>,
        Option<Box<dyn GpuEncoder>>,
        Option<Box<dyn CpuCrossDecoder>>,
        Option<Box<dyn GpuCrossDecoder>>,
    )> {
        let mut cpu_enc = None;
        let mut cpu_dec = None;
        let gpu_enc = None; // TODO: GPU Whisper
        let gpu_dec = None;

        if device.is_cpu() || load_config.offload_embeddings {
            let enc_config = Seq2SeqEncoderConfig::whisper();
            cpu_enc = Some(Box::new(Seq2SeqCPUEncoder::new(
                weights,
                config.as_ref(),
                enc_config,
                *load_config,
            )?) as Box<dyn CpuEncoder>);

            let dec_config = Seq2SeqDecoderConfig::whisper();
            cpu_dec = Some(Box::new(Seq2SeqCPUDecoder::new(
                weights,
                config.as_ref(),
                dec_config,
                *load_config,
            )?) as Box<dyn CpuCrossDecoder>);
        } else if device.is_gpu() {
            todo!("GPU Whisper not yet implemented")
        }

        Ok((cpu_enc, gpu_enc, cpu_dec, gpu_dec))
    }

    fn new_from_pipeline(
        pipeline: EncoderDecoderPipeline,
        tokenizer: Tokenizer,
        config: Arc<WhisperConfig>,
        _: Option<HFGenerationDefaults>,
        generation_config: HFGenerationConfig,
    ) -> Self {
        panic!(
            "WhisperModel::new_from_pipeline is not supported. \
             Use WhisperModel::from_registry() or from_pretrained() instead."
        );
    }
}


// Trait Implementations


impl InferenceModel for WhisperModel {
    fn device(&self) -> Device {
        self.pipeline.plan().layers
    }

    fn context(&self) -> Option<Arc<WgpuContext>> {
        self.pipeline.context().cloned()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CpuEncoderOps for WhisperModel {
    fn encoder(&self) -> &dyn CpuEncoder {
        self.pipeline.cpu_encoder().expect("CPU Encoder not active")
    }

    fn embed_tokens(
        &self,
        _input_ids: &Array2<u32>,
        _token_type_ids: Option<&Array2<u32>>,
        _pos: usize,
    ) -> Result<Array3<f32>> {
        Err(anyhow!(
            "Whisper encoder uses audio input, not tokens. Use embed_audio() instead."
        ))
    }

    fn embed_audio(&self, mel: &Array3<f32>) -> Result<Array3<f32>> {
        self.audio_frontend.forward(mel)
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

impl CpuEncoderDecoderOps for WhisperModel {
    fn decoder(&self) -> &dyn CpuCrossDecoder {
        self.pipeline.cpu_decoder().expect("CPU Decoder not active")
    }

    fn embed_decoder_tokens(
        &self,
        input_ids: &Array2<u32>,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        self.pipeline
            .decoder_embeddings()
            .embed_cpu(input_ids, None, position_offset)
    }

    fn get_decoder_mask(&self, seq_len: usize, past_len: usize) -> Option<Array2<f32>> {
        Some(kjarni_transformers::utils::create_causal_mask(
            seq_len, past_len,
        ))
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
        self.pipeline.lm_head().forward_cpu(hidden_states)
    }
}

impl GpuEncoderOps for WhisperModel {
    fn encoder(&self) -> &dyn GpuEncoder {
        self.pipeline.gpu_encoder().expect("GPU Encoder not active")
    }
    fn embed_tokens(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input_ids: ModelInput<'_>,
        token_type_ids: Option<ModelInput<'_>>,
        pos: usize,
    ) -> Result<GpuTensor> {
        unimplemented!()
        // self.pipeline()
        //     .embeddings()
        //     .embed(cmd_encoder, pool, input_ids, token_type_ids, pos)
    }
}

#[async_trait]
impl EncoderLanguageModel for WhisperModel {
    fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
        if self.pipeline.cpu_encoder().is_some() {
            Some(self)
        } else {
            None
        }
    }

    fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
        if self.pipeline.gpu_encoder().is_some() {
            Some(self)
        } else {
            None
        }
    }
}

impl GpuEncoderDecoderOps for WhisperModel {
    fn decoder(&self) -> &dyn GpuCrossDecoder {
        self.pipeline.gpu_decoder().expect("GPU Decoder not active")
    }

    fn embed_decoder_tokens(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        self.pipeline
            .decoder_embeddings()
            .embed(encoder, pool, input, None, position_offset)
    }

    fn broadcast_encoder_states(
        &self,
        frame: &mut kjarni_transformers::gpu::GpuFrameContext,
        encoder_hidden_states: &kjarni_transformers::gpu::GpuTensor,
        num_beams: usize,
    ) -> Result<kjarni_transformers::gpu::GpuTensor> {
        let broadcast = self
            .pipeline
            .gpu_broadcast()
            .ok_or_else(|| anyhow!("No broadcast kernel"))?;
        let (encoder_cmd, _pool) = frame.resources();

        let mut expanded_shape = encoder_hidden_states.shape().to_vec();
        if expanded_shape.get(0) != Some(&1) {
            return Err(anyhow!(
                "Cannot broadcast encoder states with batch size != 1"
            ));
        }
        expanded_shape[0] = num_beams;

        let expanded_states = kjarni_transformers::gpu::GpuTensor::uninitialized(
            self.context().as_ref().unwrap(),
            expanded_shape,
            encoder_hidden_states.dtype(),
            "expanded_encoder_states",
        );

        broadcast.encode(encoder_cmd, encoder_hidden_states, &expanded_states, 0);
        Ok(expanded_states)
    }

    fn project_to_logits(
        &self,
        frame: &mut kjarni_transformers::gpu::GpuFrameContext,
        hidden_states: &kjarni_transformers::gpu::GpuTensor,
    ) -> Result<kjarni_transformers::gpu::GpuTensor> {
        let (encoder_cmd, pool) = frame.resources();
        self.pipeline
            .lm_head()
            .forward_gpu(encoder_cmd, pool, hidden_states)
    }
}

#[async_trait]
impl EncoderDecoderLanguageModel for WhisperModel {
    fn get_generation_config_for_input(&self, _input: &str) -> GenerationConfig {
        self.get_default_generation_config()
    }

    fn encoder_decoder_cpu_ops(&self) -> Option<&dyn CpuEncoderDecoderOps> {
        if self.pipeline.cpu_decoder().is_some() {
            Some(self)
        } else {
            None
        }
    }

    fn encoder_decoder_gpu_ops(&self) -> Option<&dyn GpuEncoderDecoderOps> {
        None // TODO: GPU support
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
            strategy: DecodingStrategy::Greedy,
            speculation: None,
        }
    }
}


// Tests


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_mel_config() {
        // Just verify the config is constructed correctly
        let mel_config = MelConfig::whisper();
        assert_eq!(mel_config.sample_rate, 16000);
        assert_eq!(mel_config.n_fft, 400);
        assert_eq!(mel_config.hop_length, 160);
        assert_eq!(mel_config.n_mels, 80);
    }
}
