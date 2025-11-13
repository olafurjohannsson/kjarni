use crate::seq2seq::configs::BartConfig;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::TransformerConfig;
use edgetransformers::encoder_decoder::TransformerEncoderDecoder;
use edgetransformers::models::base::EncoderDecoderLanguageModel;
use edgetransformers::models::download_model_files;
use edgetransformers::models::{
    ModelArchitecture, ModelType,
    base::{GenerationConfig, GenerationStrategy, SamplingStrategy},
};
use edgetransformers::prelude::*;
use edgetransformers::traits::{
    CrossAttentionDecoder, DecoderOutput, Encoder, EncoderDecoderArchitecture, EncoderOutput,
    LanguageModelConfig, TransformerModel,
};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array1, Array2, Array3, s};
use std::ops::AddAssign;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
pub mod configs;

pub struct Seq2SeqModel {
    model: TransformerEncoderDecoder,
    tokenizer: Tokenizer,
    config: Arc<BartConfig>, // Use concrete config type now
    lm_head: Array2<f32>,    // [vocab_size, hidden_size]
    final_logits_bias: Option<Array1<f32>>,
}

impl Seq2SeqModel {
    /// Supported seq2seq model types
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::BartLargeCnn,
        ModelType::DistilBartCnn,
        ModelType::T5Small,
        ModelType::MarianEnIs,
    ];

    /// Create seq2seq model from HuggingFace model registry.
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!("Seq2seq: Unsupported model type: {:?}", model_type));
        }
        if model_type.info().architecture != ModelArchitecture::EncoderDecoder {
            return Err(anyhow!(
                "Model {:?} is not an encoder-decoder model.",
                model_type
            ));
        }

        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("edgetransformers")
        });
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        download_model_files(&model_dir, &model_type.info().paths).await?;

        Self::from_pretrained(&model_dir, model_type, device, context)
    }

    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let tokenizer =
            Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|e| anyhow!(e))?;

        let config: Arc<BartConfig> = Arc::new(serde_json::from_str(&weights.config_json)?);
        let model = TransformerEncoderDecoder::new(&weights, config.clone(), device, context)?;

        // Load the LM head WITHOUT transposing. Shape is [vocab_size, hidden_size].
        let lm_head = weights.get_array2(config.get_lm_head_name())?;

        let final_logits_bias = config
            .get_final_logits_bias_name()
            .and_then(|name| weights.get_array1(name).ok());

        Ok(Self {
            model,
            tokenizer,
            config,
            lm_head,
            final_logits_bias,
        })
    }
}

// --- TRAIT IMPLEMENTATIONS ---

impl TransformerModel for Seq2SeqModel {
    fn device(&self) -> Device {
        self.model.device()
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        self.model.context()
    }
}

impl LanguageModel for Seq2SeqModel {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn config(&self) -> &dyn LanguageModelConfig {
        self.config.as_ref()
    }

    fn new_cache(&self, batch_size: usize, max_len: usize) -> Result<Box<dyn Cache>> {
        // This logic is now correctly encapsulated in the model.
        Ok(Box::new(CpuKVCache::new(
            self.config.num_decoder_layers(),
            batch_size,
            max_len,
            self.config.hidden_size(),
        )))
    }
}

#[async_trait]
impl EncoderDecoderLanguageModel for Seq2SeqModel {
    fn encoder(&self) -> &dyn Encoder<Input = Array2<f32>, Output = EncoderOutput> {
        self.model.encoder()
    }

    fn decoder(&self) -> &dyn CrossAttentionDecoder<Input = Array2<f32>, Output = DecoderOutput> {
        self.model.decoder()
    }

    fn decoder_start_token_id(&self) -> u32 {
        self.config.decoder_start_token_id
    }
    fn generation_config_from_preset(&self) -> GenerationConfig {
        if let Some(bart_config) = self.config.as_any().downcast_ref::<BartConfig>() {
            // 3. If the downcast succeeds, we now have a `&BartConfig` and can access its fields directly.
            if let Some(params) = &bart_config.task_specific_params {
                let summary_params = &params.summarization;
                return GenerationConfig {
                    num_beams: summary_params.num_beams,
                    max_length: summary_params.max_length,
                    min_length: summary_params.min_length,
                    length_penalty: summary_params.length_penalty,
                    early_stopping: summary_params.early_stopping,
                    no_repeat_ngram_size: summary_params.no_repeat_ngram_size,
                    sampling_strategy: SamplingStrategy::BeamSearch,
                    ..GenerationConfig::default()
                };
            }
        }
        println!(
            "Warning: Could not find 'task_specific_params.summarization' in config or config is not a BartConfig. Using default generation settings."
        );
        GenerationConfig {
            sampling_strategy: SamplingStrategy::BeamSearch,
            num_beams: 4, // A reasonable default for summarization
            max_length: 142,
            min_length: 56,
            ..GenerationConfig::default()
        }
    }
    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {


        let (batch, seq, hidden) = hidden_states.dim();

        assert_eq!(self.lm_head.shape()[0], self.vocab_size());
        assert_eq!(self.lm_head.shape()[1], self.hidden_size());

        let hidden_2d = hidden_states.to_shape((batch * seq, hidden))?;
        let hidden_transposed = hidden_2d.t();
        
        let mut logits_2d = self.lm_head.dot(&hidden_transposed);
        logits_2d = logits_2d.t().as_standard_layout().to_owned();

        if let Some(bias) = &self.final_logits_bias {
            logits_2d.add_assign(bias);
        }
        logits_2d
            .into_shape_with_order((batch, seq, self.config.vocab_size()))
            .map_err(|e| anyhow!(e))
    }
}
