use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use tokenizers::Tokenizer;

use crate::models::t5::T5Task;

use super::config::T5Config;

use kjarni_transformers::{
    LanguageModel, ModelType, WgpuContext,
    cache::{Cache, CpuBeamKVCache, GpuBeamKVCache},
    common::{
        BeamSearchParams, DecodingStrategy, GenerationConfig, HFGenerationConfig,
        HFGenerationDefaults, ModelGenerationDefaults,
    },
    cpu::{
        encoder::{CpuEncoderOps, GpuEncoderOps, prelude::*, traits::CpuEncoder},
        encoder_decoder::{
            cpu_decoder::{Seq2SeqCPUDecoder, Seq2SeqDecoderConfig},
            cpu_encoder::{Seq2SeqCPUEncoder, Seq2SeqEncoderConfig},
        },
    },
    encoder_decoder::{
        config::TranslationParams,
        traits::{
            CpuCrossDecoder, CpuEncoderDecoderOps, EncoderDecoderLanguageModel, GpuCrossDecoder,
            GpuEncoderDecoderOps,
        },
    },
    models::base::ModelLoadConfig,
    pipeline::{EncoderDecoderModelFactory, EncoderDecoderPipeline},
    traits::{Device, InferenceModel, ModelConfig as _, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

pub struct T5Model {
    tokenizer: Tokenizer,
    config: Arc<T5Config>,
    pub pipeline: EncoderDecoderPipeline,
    generation_config: HFGenerationConfig,
}

impl T5Model {
    pub fn meta(&self) -> ModelMetadata {
        self.config.metadata()
    }

    pub fn layout(&self) -> ModelLayout {
        self.config.layout()
    }
}

impl EncoderDecoderModelFactory for T5Model {
    type Config = T5Config;

    fn load_config(weights: &ModelWeights) -> Result<Arc<Self::Config>> {
        let config: T5Config = serde_json::from_str(&weights.config_json())?;
        Ok(Arc::new(config))
    }

    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        _layout: &ModelLayout,
        config: &Arc<T5Config>,
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
        config: Arc<T5Config>,
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

impl T5Model {
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
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

    /// Get generation config for a specific task
    pub fn get_generation_config_for_task(&self, task: &T5Task) -> GenerationConfig {
        // Start with base config from generation_config.json
        let mut config = self
            .generation_config
            .to_generation_config(&ModelGenerationDefaults::for_t5());

        // Apply task-specific overrides
        if let Some(params) = &self.config.task_specific_params {
            match task {
                T5Task::Summarization => {
                    if let Some(task_cfg) = &params.summarization {
                        config.max_length = task_cfg.max_length;
                        config.min_length = task_cfg.min_length;
                        config.no_repeat_ngram_size = task_cfg.no_repeat_ngram_size;
                        config.strategy = DecodingStrategy::BeamSearch(BeamSearchParams {
                            num_beams: task_cfg.num_beams,
                            length_penalty: task_cfg.length_penalty,
                            early_stopping: task_cfg.early_stopping,
                        });
                    }
                }
                T5Task::TranslationEnToDe => {
                    if let Some(task_cfg) = &params.translation_en_to_de {
                        self.apply_translation_params(&mut config, task_cfg);
                    }
                }
                T5Task::TranslationEnToFr => {
                    if let Some(task_cfg) = &params.translation_en_to_fr {
                        self.apply_translation_params(&mut config, task_cfg);
                    }
                }
                T5Task::TranslationEnToRo => {
                    if let Some(task_cfg) = &params.translation_en_to_ro {
                        self.apply_translation_params(&mut config, task_cfg);
                    }
                }
                T5Task::TranslationCustom { .. } | T5Task::Question | T5Task::Unknown => {
                    // Use base config with translation-safe defaults
                    config.min_length = 0;
                    config.no_repeat_ngram_size = 0;
                }
            }
        }

        config
    }

    fn apply_translation_params(&self, config: &mut GenerationConfig, params: &TranslationParams) {
        config.max_length = params.max_length;
        config.min_length = 0; // Translation should not force min length
        config.no_repeat_ngram_size = 0; // Translation may need repetition
        config.strategy = DecodingStrategy::BeamSearch(BeamSearchParams {
            num_beams: params.num_beams,
            length_penalty: 1.0,
            early_stopping: params.early_stopping,
        });
    }

    const SUPPORTED_MODELS: &'static [ModelType] = &[
        // ModelType::FlanT5Small,
        ModelType::FlanT5Base,
        ModelType::FlanT5Large,
        // Add more as needed
    ];
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl InferenceModel for T5Model {
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

impl CpuEncoderOps for T5Model {
    fn encoder(&self) -> &dyn CpuEncoder {
        self.pipeline.cpu_encoder().expect("CPU Encoder not active")
    }
    fn embed_tokens(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
        pos: usize,
    ) -> Result<Array3<f32>> {
        self.pipeline
            .embeddings()
            .embed_cpu(input_ids, token_type_ids, pos)
    }
}

impl GpuEncoderOps for T5Model {
    fn encoder(&self) -> &dyn GpuEncoder {
        self.pipeline.gpu_encoder().expect("GPU Encoder not active")
    }
}

#[async_trait]
impl EncoderLanguageModel for T5Model {
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

impl LanguageModel for T5Model {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn context_size(&self) -> usize {
        self.config.metadata().max_seq_len
    }

    // fn forced_eos_token_id(&self) -> Option<u32> {
    //     Some(self.config.eos_token_id)
    // }
    fn forced_eos_token_id(&self) -> Option<u32> {
        self.generation_config.forced_eos_token_id
    }

    fn pad_token_id(&self) -> Option<u32> {
        self.generation_config.pad_token_id.or(Some(0))
    }

    // fn pad_token_id(&self) -> Option<u32> {
    //     Some(self.config.pad_token_id)
    // }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn hidden_size(&self) -> usize {
        self.config.d_model
    }

    fn num_heads(&self) -> usize {
        self.config.num_heads
    }

    fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    // fn eos_token_id(&self) -> Option<u32> {
    //     Some(self.config.eos_token_id)
    // }
    fn eos_token_id(&self) -> Option<u32> {
        self.generation_config
            .eos_token_id
            .as_ref()
            .map(|e| e.primary())
            .or(Some(1)) // T5 default
    }

    fn forced_bos_token_id(&self) -> Option<u32> {
        self.generation_config.forced_bos_token_id
    }

    fn bos_token_id(&self) -> Option<u32> {
        None // T5 uses decoder_start_token_id instead
    }

    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        num_beams: usize,
    ) -> Result<Box<dyn Cache>> {
        let num_decoder_layers = self
            .config
            .num_decoder_layers
            .unwrap_or(self.config.num_layers);

        match self.pipeline.plan().layers {
            Device::Cpu => {
                let effective_batch = if num_beams > 0 { num_beams } else { batch_size };
                Ok(Box::new(CpuBeamKVCache::new(
                    num_decoder_layers,
                    effective_batch,
                    max_len,
                    self.config.d_model,
                )))
            }
            Device::Wgpu => {
                let context = self
                    .context()
                    .ok_or_else(|| anyhow!("GPU cache requires WgpuContext"))?;
                Ok(Box::new(GpuBeamKVCache::new(
                    &context,
                    num_decoder_layers,
                    num_beams,
                    self.config.num_heads,
                    self.config.d_kv,
                    max_len,
                )?))
            }
        }
    }
}

impl CpuEncoderDecoderOps for T5Model {
    fn decoder(&self) -> &dyn CpuCrossDecoder {
        self.pipeline.cpu_decoder().expect("CPU Decoder not active")
    }
    fn get_decoder_mask(&self, seq_len: usize, past_len: usize) -> Option<Array2<f32>> {
        // Creates a triangular matrix where 1.0 is allowed and 0.0 is masked
        // This prevents the decoder from "looking ahead"
        Some(kjarni_transformers::utils::create_causal_mask(
            seq_len, past_len,
        ))
    }

    fn broadcast_encoder_states(
        &self,
        encoder_hidden_states: &ndarray::Array3<f32>,
        num_beams: usize,
    ) -> Result<ndarray::Array3<f32>> {
        Ok(encoder_hidden_states
            .broadcast((
                num_beams,
                encoder_hidden_states.shape()[1],
                encoder_hidden_states.shape()[2],
            ))
            .ok_or_else(|| anyhow!("Failed to broadcast encoder state"))?
            .to_owned())
    }

    fn project_to_logits(
        &self,
        hidden_states: &ndarray::Array3<f32>,
    ) -> Result<ndarray::Array3<f32>> {
        self.pipeline.lm_head().forward_cpu(hidden_states)
    }
}

impl GpuEncoderDecoderOps for T5Model {
    fn decoder(&self) -> &dyn GpuCrossDecoder {
        self.pipeline.gpu_decoder().expect("GPU Decoder not active")
    }

    fn broadcast_encoder_states(
        &self,
        frame: &mut kjarni_transformers::gpu_ops::GpuFrameContext,
        encoder_hidden_states: &kjarni_transformers::gpu_ops::GpuTensor,
        num_beams: usize,
    ) -> Result<kjarni_transformers::gpu_ops::GpuTensor> {
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

        let expanded_states = kjarni_transformers::gpu_ops::GpuTensor::uninitialized(
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
        frame: &mut kjarni_transformers::gpu_ops::GpuFrameContext,
        hidden_states: &kjarni_transformers::gpu_ops::GpuTensor,
    ) -> Result<kjarni_transformers::gpu_ops::GpuTensor> {
        let (encoder_cmd, pool) = frame.resources();
        self.pipeline
            .lm_head()
            .forward_gpu(encoder_cmd, pool, hidden_states)
    }
}

#[async_trait]
impl EncoderDecoderLanguageModel for T5Model {
    /// Get generation config for a specific input (auto-detects task)
    fn get_generation_config_for_input(&self, input: &str) -> GenerationConfig {
        let task = T5Task::from_input(input);
        self.get_generation_config_for_task(&task)
    }
    fn encoder_decoder_cpu_ops(&self) -> Option<&dyn CpuEncoderDecoderOps> {
        if self.pipeline.cpu_decoder().is_some() {
            Some(self)
        } else {
            None
        }
    }

    fn encoder_decoder_gpu_ops(&self) -> Option<&dyn GpuEncoderDecoderOps> {
        if self.pipeline.gpu_decoder().is_some() {
            Some(self)
        } else {
            None
        }
    }

    // fn decoder_start_token_id(&self) -> u32 {
    //     self.config.decoder_start_token_id
    // }
    fn decoder_start_token_id(&self) -> u32 {
        self.generation_config.decoder_start_token_id.unwrap_or(0)
    }

    fn get_default_generation_config(&self) -> GenerationConfig {
        // Try to load from task specific params
        if let Some(params) = &self.config.task_specific_params {
            if let Some(summary) = &params.summarization {
                return GenerationConfig {
                    max_length: summary.max_length,
                    min_length: summary.min_length,
                    no_repeat_ngram_size: summary.no_repeat_ngram_size,
                    repetition_penalty: 1.0,
                    max_new_tokens: None,
                    add_bos_token: false,
                    strategy: DecodingStrategy::BeamSearch(BeamSearchParams {
                        num_beams: summary.num_beams,
                        length_penalty: summary.length_penalty,
                        early_stopping: summary.early_stopping,
                    }),
                };
            }
        }

        // T5 Safe Defaults
        GenerationConfig {
            max_length: 512,
            min_length: 0,
            no_repeat_ngram_size: 3,
            repetition_penalty: 1.0,
            max_new_tokens: None,
            add_bos_token: false,
            strategy: DecodingStrategy::BeamSearch(BeamSearchParams {
                num_beams: 4,
                length_penalty: 1.0, // T5 typically uses 1.0
                early_stopping: true,
            }),
        }
    }
}
