use crate::generation::{
    apply_no_repeat_ngram, apply_repetition_penalty, log_softmax_1d, sample_token,
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::CpuKVCache;
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::models::base::{BeamHypothesis, GenerationConfig, SamplingStrategy};
use edgetransformers::models::download_model_files;
use edgetransformers::models::project_to_vocab;
use edgetransformers::models::{DecoderLanguageModel, LanguageModel, ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::traits::{Decoder, DecoderArchitecture, DecoderOutput, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array1, Array2, s};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

mod configs;
pub use configs::Gpt2Config;

pub struct TextGenerator {
    model: TransformerDecoder,
    tokenizer: Tokenizer,
    config: Arc<dyn DecoderArchitecture + Send + Sync>,
    lm_head_t: Array2<f32>,
}

impl TextGenerator {
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::DistilGpt2,
        ModelType::Gpt2,
        ModelType::Gpt2Medium,
        ModelType::Gpt2Large,
        ModelType::Gpt2XL,
    ];

    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!(
                "Unsupported text generator model: {:?}",
                model_type
            ));
        }
        if model_type.info().architecture != ModelArchitecture::Decoder {
            return Err(anyhow!("Model {:?} is not a decoder model.", model_type));
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

        let config: Arc<dyn DecoderArchitecture + Send + Sync> = match model_type {
            ModelType::DistilGpt2
            | ModelType::Gpt2
            | ModelType::Gpt2Medium
            | ModelType::Gpt2Large
            | ModelType::Gpt2XL => {
                Arc::new(serde_json::from_str::<Gpt2Config>(&weights.config_json)?)
            }
            _ => {
                return Err(anyhow!(
                    "Configuration for {:?} not yet implemented.",
                    model_type
                ));
            }
        };

        let model = TransformerDecoder::new(&weights, config.clone(), device, context)?;

        let lm_head_t = weights
            .get_array2(config.get_lm_head_name())?
            .t()
            .to_owned();

        Ok(Self {
            model,
            tokenizer,
            config,
            lm_head_t,
        })
    }

    pub async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        // 1. Tokenize the input prompt
        let mut tokens = self
            .tokenizer()
            .encode(prompt, true)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec();
        let prompt_len = tokens.len();
        let batch_size = 1;

        // 2. Determine the final length of the generated sequence
        let max_len = if let Some(new_tokens) = config.max_new_tokens {
            prompt_len + new_tokens
        } else {
            config.max_length
        };
        let full_attention_mask = Array2::ones((batch_size, max_len));
        // 3. Initialize the Key-Value cache
        let mut cache: Box<dyn Cache> = match self.model.device() {
            Device::Cpu => Box::new(CpuKVCache::new(
                self.config.num_hidden_layers(),
                batch_size,
                max_len,                   // The maximum capacity for pre-allocation
                self.config.hidden_size(), // The hidden size for pre-allocation
            )),
            Device::Wgpu => {
                let context = self
                    .model
                    .context()
                    .ok_or_else(|| anyhow!("GPU model should have context"))?;

                Box::new(GpuKVCache::new(
                    &context,
                    self.config.num_hidden_layers(),
                    batch_size,
                    self.config.num_attention_heads(),
                    self.config.hidden_size() / self.config.num_attention_heads(),
                    max_len, // The capacity is the max length of the sequence
                )?)
            }
        };
        let mut current_pos = 0;

        // 4. Process the prompt in a single "priming" pass to fill the cache
        if prompt_len > 0 {
            let prompt_ids = Array2::from_shape_vec(
                (batch_size, prompt_len),
                tokens.iter().map(|&t| t as f32).collect(),
            )?;
            // The attention mask for the prompt is all ones
            // let attention_mask = Array2::ones((batch_size, prompt_len));
            let prompt_mask = full_attention_mask.slice(s![.., 0..prompt_len]).to_owned();
            // This forward pass populates the cache with the prompt's key-value states
            self.model
                .forward(&prompt_ids, &prompt_mask, Some(cache.as_mut()))
                .await?;
            current_pos = prompt_len;
        }

        // 5. Start the autoregressive generation loop
        loop {
            // Check stopping conditions
            if tokens.len() >= max_len {
                break;
            }
            let current_len = tokens.len();
            // Prepare the input for this step: it's only the *last* token
            let last_token = *tokens.last().unwrap_or(&self.bos_token_id().unwrap_or(0));
            let input_ids = Array2::from_shape_vec((1, 1), vec![last_token as f32])?;

            // Create a full attention mask for the *entire* sequence length (prompt + generated)
            // The `forward_with_cache` method in the TransformerLayer is responsible for making it causal.
            // let attention_mask = Array2::ones((batch_size, current_pos + 1));
            let attention_mask_view = full_attention_mask.slice(s![.., 0..current_len + 1]);
            // Perform the forward pass for the single new token
            let decoder_output = self
                .model
                .forward(
                    &input_ids,
                    &attention_mask_view.to_owned(),
                    Some(cache.as_mut()),
                )
                .await?;

            current_pos += 1;

            // Project the output hidden state to vocabulary logits
            // GPT-2's LM head is the word embedding matrix, which needs to be transposed for matmul.
            // For other models, this might differ. The `transpose_...` flags in the config
            // are for the main transformer layers, not necessarily the standalone LM head.
            let logits_3d = project_to_vocab(&decoder_output.last_hidden_state, &self.lm_head_t)?;

            // Get the logits for the new token (it's the only one in the sequence dimension)
            let mut next_token_logits = logits_3d.slice(s![0, 0, ..]).to_owned();

            // Apply repetition penalty
            next_token_logits =
                apply_repetition_penalty(next_token_logits, &tokens, config.repetition_penalty);

            // Sample the next token using the specified strategy (e.g., greedy)
            let next_token = sample_token(next_token_logits, config)?;
            tokens.push(next_token);

            // Check for End-Of-Sequence token
            if Some(next_token) == self.eos_token_id() {
                break;
            }
        }

        // 6. Decode the final sequence of tokens back to a string
        self.tokenizer()
            .decode(&tokens, true)
            .map_err(|e| anyhow!(e))
    }
}

impl TransformerModel for TextGenerator {
    fn device(&self) -> Device {
        self.model.device()
    }
}

impl LanguageModel for TextGenerator {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn config(&self) -> &dyn LanguageModelConfig {
        self.config.as_ref()
    }
}

#[async_trait]
impl DecoderLanguageModel for TextGenerator {
    fn decoder(&self) -> &dyn Decoder<Input = Array2<f32>, Output = DecoderOutput> {
        &self.model
    }
    fn lm_head(&self) -> &Array2<f32> {
        &self.lm_head_t
    }
}
