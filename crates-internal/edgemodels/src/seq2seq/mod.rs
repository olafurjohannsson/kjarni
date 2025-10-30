use crate::generation::{
    apply_no_repeat_ngram, apply_repetition_penalty, log_softmax_1d, sample_token,
};
use anyhow::{Result, anyhow};
use edgetransformers::CpuKVCache;
use edgetransformers::encoder_decoder::TransformerEncoderDecoder;
use edgetransformers::models::base::{BeamHypothesis, GenerationConfig, SamplingStrategy};
use edgetransformers::models::{ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::traits::EncoderOutput as EncoderOutputTrait;
use edgetransformers::traits::{
    CrossAttentionDecoder, EncoderDecoderArchitecture, EncoderOutput, LanguageModelConfig,
    TransformerModel,
};
use edgetransformers::weights::ModelWeights;
use edgetransformers::{LanguageModel, Seq2SeqLanguageModel};
use ndarray::{Array1, Array2, s};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
mod configs;
use async_trait::async_trait;
pub use configs::BartConfig;

pub struct Seq2SeqModel {
    model: TransformerEncoderDecoder, // Renamed for clarity
    tokenizer: Tokenizer,
    config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
    lm_head: Array2<f32>, // [hidden_size, vocab_size]
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

        Self::download_model_files(&model_dir, &model_type.info().paths).await?;

        Self::from_pretrained(&model_dir, model_type, device, context)
    }
    /// Creates a GenerationConfig based on the "summarization" preset in the model's config.json.
    ///
    /// This is the preferred way to get generation parameters, as it uses the values
    /// the model was trained and evaluated with. It falls back to a reasonable default
    /// if the preset is not found in the config file.
    pub fn generation_config_from_preset(&self) -> GenerationConfig {
        // --- THIS IS THE CORRECTED LOGIC ---

        // 1. Use the `as_any` helper to get a `&dyn Any`.
        // 2. Attempt to downcast it to our concrete `BartConfig` type.
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

        // --- END CORRECTION ---

        // If downcasting or finding params fails, return the default.
        println!(
            "Warning: Could not find 'task_specific_params.summarization' in config or config is not a BartConfig. Using default generation settings."
        );
        GenerationConfig::default()
    }

    pub async fn summarize(&self, input_text: &str) -> Result<String> {
        // 1. Get the best configuration from the model's preset.
        let config = self.generation_config_from_preset();

        // 2. Call the powerful, explicit trait method with that config.
        self.generate(input_text, &config).await
    }

    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
        println!(
            "--- Loaded config.json ---\n{}\n--------------------------",
            &weights.config_json
        );
        let config: Arc<dyn EncoderDecoderArchitecture + Send + Sync> = match model_type {
            ModelType::BartLargeCnn | ModelType::DistilBartCnn => {
                Arc::new(serde_json::from_str::<BartConfig>(&weights.config_json)?)
            }
            _ => {
                return Err(anyhow!(
                    "Seq2seq: Unsupported config for model: {:?}",
                    model_type
                ));
            }
        };

        let model = TransformerEncoderDecoder::new(&weights, config.clone(), device, context)?;

        // *** THE FIX IS HERE ***
        // Load the LM head WITHOUT transposing it. Its shape is [vocab_size, hidden_size].
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
    async fn download_model_files(
        model_dir: &Path,
        paths: &edgetransformers::models::ModelPaths,
    ) -> Result<()> {
        tokio::fs::create_dir_all(model_dir).await?;
        let files = [
            ("model.safetensors", paths.weights_url),
            ("tokenizer.json", paths.tokenizer_url),
            ("config.json", paths.config_url),
        ];
        for (filename, url) in files {
            let local_path = model_dir.join(filename);
            if !local_path.exists() {
                println!("Downloading {}...", filename);
                let response = reqwest::get(url).await?;
                anyhow::ensure!(response.status().is_success(), "Failed to download {}", url);
                tokio::fs::write(&local_path, &response.bytes().await?).await?;
            }
        }
        Ok(())
    }
}

impl TransformerModel for Seq2SeqModel {
    fn device(&self) -> Device {
        self.model.device()
    }
}

impl LanguageModel for Seq2SeqModel {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn config(&self) -> &dyn LanguageModelConfig {
        self.config.as_ref()
    }
}

#[async_trait]
impl Seq2SeqLanguageModel for Seq2SeqModel {
    /// Encodes the input text and returns the raw hidden states from the encoder.
    async fn encode_input(&self, text: &str) -> Result<EncoderOutput> {
        let encoding = self
            .tokenizer()
            .encode(text, true)
            .map_err(|e| anyhow!(e))?;
        let input_ids = Array2::from_shape_vec(
            (1, encoding.len()),
            encoding.get_ids().iter().map(|&id| id as f32).collect(),
        )?;
        let attention_mask = Array2::ones(input_ids.dim());

        match &self.model {
            TransformerEncoderDecoder::Cpu(cpu_model) => {
                cpu_model
                    .encoder()
                    .forward(&input_ids, &attention_mask, None) // TODO: token_type_ids ???
                    .await
            }
            TransformerEncoderDecoder::Gpu(_) => todo!("GPU path for encode_input not implemented"),
        }
    }

    /// Generates text autoregressively from pre-computed encoder hidden states.
    async fn generate_from_encoding(
        &self,
        encoder_output: &EncoderOutputTrait,
        encoder_attention_mask: &Array2<f32>,
        config: &GenerationConfig,
    ) -> Result<String> {
        let batch_size = encoder_output.last_hidden_state.shape()[0];
        let eos_token_id = self.config.eos_token_id();
        let decoder_start_token_id = self.config.decoder_start_token_id();

        // 1. Initialize Beams
        // Create an empty cache. Each beam will get a clone of this to start.
        let initial_cache = CpuKVCache::new(self.config.num_decoder_layers(), batch_size);
        let mut beams = vec![BeamHypothesis {
            tokens: vec![decoder_start_token_id],
            score: 0.0,
            cache: initial_cache,
        }];
        let mut completed_beams: Vec<BeamHypothesis> = Vec::new();

        // 2. Main Generation Loop
        for _ in 0..config.max_length {
            if beams.is_empty() {
                break;
            }
            let mut all_new_candidates: Vec<BeamHypothesis> = Vec::new();

            for hypo in &beams {
                // --- Prepare inputs for the model's `forward` call ---
                let last_token = *hypo.tokens.last().unwrap();
                let decoder_input_ids =
                    Array2::from_shape_vec((batch_size, 1), vec![last_token as f32])?;

                // This mask covers the ENTIRE sequence generated so far for this hypothesis.
                // This is the `decoder_attention_mask_from_generator` argument.
                let decoder_attention_mask = Array2::ones((batch_size, hypo.tokens.len()));

                // Each beam needs its own mutable cache for the forward pass.
                let mut current_cache = hypo.cache.clone();

                // --- THE CORRECT FORWARD CALL ---
                let decoder_output = self
                    .model
                    .forward(
                        &Array2::zeros((0, 0)), // Dummy, not used when `encoder_output_opt` is Some
                        &decoder_input_ids,
                        encoder_attention_mask,
                        &decoder_attention_mask,
                        Some(&mut current_cache), // Pass the mutable cache for this beam
                        Some(encoder_output),     // Pass the pre-computed encoder output
                    )
                    .await?;
                // --- END FORWARD CALL ---

                let last_hidden_state = decoder_output.last_hidden_state.slice(s![0, -1, ..]);
                let mut logits: Array1<f32> = self.lm_head.dot(&last_hidden_state);
                if let Some(bias) = &self.final_logits_bias {
                    logits += bias;
                }

                // Apply penalties using the generic helpers
                logits = apply_repetition_penalty(logits, &hypo.tokens, config.repetition_penalty);
                logits = apply_no_repeat_ngram(logits, &hypo.tokens, config.no_repeat_ngram_size);
                let log_probs = log_softmax_1d(&logits);

                // Get top `num_beams` candidates from this single hypothesis
                let mut top_candidates: Vec<(u32, f32)> = log_probs
                    .iter()
                    .enumerate()
                    .map(|(id, &lp)| (id as u32, lp))
                    .collect();
                top_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                top_candidates.truncate(config.num_beams);

                // Create new, full hypotheses from the candidates
                for (token_id, token_log_prob) in top_candidates {
                    let mut new_tokens = hypo.tokens.clone();
                    new_tokens.push(token_id);
                    all_new_candidates.push(BeamHypothesis {
                        tokens: new_tokens,
                        score: hypo.score + token_log_prob,
                        // The new hypothesis gets the updated cache from the forward pass
                        cache: current_cache.clone(),
                    });
                }
            }

            // 3. Prune and Manage Beams
            all_new_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            beams.clear();

            for candidate in all_new_candidates {
                if candidate.tokens.last() == Some(&eos_token_id) {
                    if candidate.tokens.len() >= config.min_length {
                        completed_beams.push(candidate);
                    }
                } else {
                    beams.push(candidate);
                }
                if beams.len() == config.num_beams {
                    break;
                }
            }

            if config.early_stopping && completed_beams.len() >= config.num_beams {
                break;
            }
        }

        // 4. Finalize and Decode
        let final_hypotheses = if completed_beams.is_empty() {
            beams
        } else {
            completed_beams
        };

        if final_hypotheses.is_empty() {
            return Err(anyhow!("No hypothesis was generated."));
        }

        let best_hypo = final_hypotheses
            .iter()
            .max_by(|a, b| {
                let score_a = a.score / (a.tokens.len() as f32).powf(config.length_penalty);
                let score_b = b.score / (b.tokens.len() as f32).powf(config.length_penalty);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .unwrap();

        let mut tokens_to_decode = &best_hypo.tokens[..];
        if tokens_to_decode.first() == Some(&decoder_start_token_id) {
            tokens_to_decode = &tokens_to_decode[1..];
        }
        if tokens_to_decode.last() == Some(&eos_token_id) {
            tokens_to_decode = &tokens_to_decode[..tokens_to_decode.len() - 1];
        }

        self.tokenizer()
            .decode(tokens_to_decode, true)
            .map_err(|e| anyhow!(e))
    }

    /// High-level method to generate text directly from a string.
    async fn generate(&self, input_text: &str, config: &GenerationConfig) -> Result<String> {
        let encoding = self
            .tokenizer()
            .encode(input_text, true)
            .map_err(|e| anyhow!(e))?;
        let attention_mask = Array2::ones((1, encoding.len()));
        let encoder_output = self.encode_input(input_text).await?;
        self.generate_from_encoding(&encoder_output, &attention_mask, config)
            .await
    }
}
