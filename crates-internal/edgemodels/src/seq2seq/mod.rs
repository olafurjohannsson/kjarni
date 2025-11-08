use crate::generation::{
    apply_no_repeat_ngram, apply_repetition_penalty, log_softmax_1d, sample_token,
};
use anyhow::{Result, anyhow};
use edgetransformers::CpuKVCache;
use edgetransformers::cache::Cache;
use edgetransformers::encoder_decoder::TransformerEncoderDecoder;
use edgetransformers::models::base::{BeamHypothesis, GenerationConfig, SamplingStrategy};
use edgetransformers::models::download_model_files;
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

        download_model_files(&model_dir, &model_type.info().paths).await?;

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
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|e| {
            anyhow!(
                "Failed to load tokenizer: {} - Model Path: {}",
                e,
                model_path.to_str().unwrap()
            )
        })?;
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
        println!("generate_from_encoding");
        let batch_size = encoder_output.last_hidden_state.shape()[0];
        let eos_token_id = self.config.eos_token_id();
        let decoder_start_token_id = self.config.decoder_start_token_id();

        // 1. Initialize Beams
        // Create an empty cache. Each beam will get a clone of this to start.
        //pub fn new(num_layers: usize, batch_size: usize, max_len: usize, hidden_size: usize) -> Self {
        let initial_cache: Box<dyn Cache> = Box::new(CpuKVCache::new(
            self.config.num_decoder_layers(),
            batch_size,
            config.max_length,
            self.config.hidden_size(),
        ));
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

            match self.model.device() {
                Device::Cpu => {
                    // For the CPU, we use the "smart clone" strategy.
                    for mut hypo in beams.drain(..) {
                        let last_token = *hypo.tokens.last().unwrap();
                        let decoder_input_ids =
                            Array2::from_shape_vec((batch_size, 1), vec![last_token as f32])?;
                        let decoder_attention_mask = Array2::ones((batch_size, hypo.tokens.len()));

                        // Downcast to the concrete, Clone-able CpuKVCache
                        let cpu_cache = hypo
                            .cache
                            .as_any_mut()
                            .downcast_mut::<CpuKVCache>()
                            .unwrap();

                        // Pass the mutable cache to be updated in-place
                        let decoder_output = self
                            .model
                            .forward(
                                &Array2::zeros((0, 0)),
                                &decoder_input_ids,
                                encoder_attention_mask,
                                &decoder_attention_mask,
                                Some(cpu_cache),
                                Some(encoder_output),
                            )
                            .await?;

                        // ... (Logits calculation is the same) ...
                        let last_hidden_state =
                            decoder_output.last_hidden_state.slice(s![0, -1, ..]);
                        let mut logits: Array1<f32> = self.lm_head.dot(&last_hidden_state);
                        // ...
                        let log_probs = log_softmax_1d(&logits);
                        let top_candidates = get_top_k_from_log_probs(&log_probs, config.num_beams);

                        let mut is_first_candidate = true;
                        for (token_id, token_log_prob) in top_candidates {
                            let mut new_tokens = hypo.tokens.clone();
                            new_tokens.push(token_id);

                            // Efficiently handle the cache for the new hypotheses
                            let new_cache: Box<dyn Cache> = if is_first_candidate {
                                is_first_candidate = false;
                                Box::new(cpu_cache.clone()) // Just clone the updated cache
                            } else {
                                Box::new(cpu_cache.clone())
                            };

                            all_new_candidates.push(BeamHypothesis {
                                tokens: new_tokens,
                                score: hypo.score + token_log_prob,
                                cache: new_cache,
                            });
                        }
                    }
                }
                Device::Wgpu => {
                    // For the GPU, we would use the batched forward pass and the `reorder_cache` kernel.

                    todo!("GPU batched beam search with cache reordering not implemented yet.");
                }
            }

            // 3. Prune and Manage Beams
            all_new_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            beams.clear();

            for candidate in all_new_candidates {
                if candidate.tokens.last() == Some(&eos_token_id) {
                    if candidate.tokens.len() >= config.min_length {
                        completed_beams.push(candidate);
                    } else {
                        // Still keep it as an active beam if under min_length
                        beams.push(candidate);
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

/// Selects the top `k` tokens and their log probabilities from a log probability distribution.
///
/// This is a core component of beam search, used to find the most likely next
/// tokens to expand upon.
///
/// # Arguments
/// * `log_probs`: A 1D array of log probabilities for the entire vocabulary.
/// * `k`: The number of top tokens to select.
///
/// # Returns
/// A `Vec` containing tuples of `(token_id, log_probability)`, sorted from highest
/// log probability to lowest.
pub fn get_top_k_from_log_probs(log_probs: &Array1<f32>, k: usize) -> Vec<(u32, f32)> {
    // 1. Create a vector of (index, value) pairs from the log probabilities array.
    let mut indexed_log_probs: Vec<(usize, f32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &lp)| (i, lp))
        .collect();

    // 2. Sort the vector in descending order based on the log probability (the f32 value).
    //    We use `partial_cmp` and reverse the comparison to sort from highest to lowest.
    //    `unwrap()` is safe here because f32 log_probs from a softmax will not be NaN.
    indexed_log_probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 3. Take the top `k` elements from the sorted vector.
    indexed_log_probs.truncate(k);

    // 4. Map the result to the final `(u32, f32)` format for token IDs.
    indexed_log_probs
        .into_iter()
        .map(|(i, lp)| (i as u32, lp))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_get_top_k() {
        let log_probs = arr1(&[-2.3, -1.1, -5.0, -0.5, -3.1, -0.8]);
        let k = 3;

        let top_k = get_top_k_from_log_probs(&log_probs, k);

        // The expected order is based on the log_probs:
        // 1st: -0.5 (index 3)
        // 2nd: -0.8 (index 5)
        // 3rd: -1.1 (index 1)
        let expected = vec![(3, -0.5), (5, -0.8), (1, -1.1)];

        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k, expected);
    }
}
