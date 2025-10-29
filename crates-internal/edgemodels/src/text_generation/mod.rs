use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array1, Array2, s};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
use crate::generation::{
    apply_no_repeat_ngram, apply_repetition_penalty, log_softmax_1d, sample_token,
};
use edgetransformers::models::base::{BeamHypothesis, GenerationConfig, SamplingStrategy};

use edgetransformers::CpuKVCache;
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::models::project_to_vocab;
use edgetransformers::models::{DecoderLanguageModel, LanguageModel, ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::traits::{Decoder, DecoderArchitecture, DecoderOutput, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;

mod configs;
pub use configs::Gpt2Config;

pub struct TextGenerator {
    model: TransformerDecoder,
    tokenizer: Tokenizer,
    config: Arc<dyn DecoderArchitecture + Send + Sync>,
    lm_head: Array2<f32>,
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

        Self::download_model_files(&model_dir, &model_type.info().paths).await?;
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

        let lm_head = weights.get_array2(config.get_lm_head_name())?;

        Ok(Self {
            model,
            tokenizer,
            config,
            lm_head,
        })
    }

    pub async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        // 1. Tokenize
        let mut tokens = self.tokenizer().encode(prompt, true).map_err(|e| anyhow!(e))?.get_ids().to_vec();
        let prompt_len = tokens.len();

        // 2. Determine effective max length
        let effective_max_length = if let Some(new_tokens) = config.max_new_tokens {
            prompt_len + new_tokens
        } else {
            config.max_length
        };
        
        // 3. Initialize Cache
        let batch_size = 1;
        let mut cache = CpuKVCache::new(self.config.num_hidden_layers(), batch_size);

        // 4. Prime the Cache with the prompt
        if prompt_len > 0 {
            let prompt_ids = Array2::from_shape_vec((batch_size, prompt_len), tokens.iter().map(|&t| t as f32).collect())?;
            let attention_mask = Array2::ones((batch_size, prompt_len));
            self.model.forward(&prompt_ids, &attention_mask, Some(&mut cache)).await?;
        }

        // 5. Autoregressive Loop
        loop {
            // --- Stop Conditions ---
            if tokens.len() >= effective_max_length { break; }

            // --- Prepare single-token input ---
            let last_token = *tokens.last().unwrap_or(&self.bos_token_id().unwrap_or(0));
            let input_ids = Array2::from_shape_vec((1, 1), vec![last_token as f32])?;
            let attention_mask = Array2::ones((batch_size, cache.get_seq_length() + 1));

            // --- Forward Pass ---
            let decoder_output = self.model.forward(&input_ids, &attention_mask, Some(&mut cache)).await?;

            // --- Get Logits (CRITICAL FIX: NO TRANSPOSE) ---
            let logits_3d = project_to_vocab(&decoder_output.last_hidden_state, 
                &self.lm_head.t().to_owned())?;
            let mut next_token_logits = logits_3d.slice(s![0, 0, ..]).to_owned();

            // --- Apply Penalties & Sample ---
            next_token_logits = apply_repetition_penalty(next_token_logits, &tokens, config.repetition_penalty);
            let next_token = sample_token(next_token_logits, config)?;
            tokens.push(next_token);

            // --- EOS Stop Condition ---
            if Some(next_token) == self.eos_token_id() && tokens.len() > config.min_length {
                break;
            }
        }

        // 6. Decode
        self.tokenizer().decode(&tokens, true).map_err(|e| anyhow!(e))
    }


    ////  WAS WORKING:
    pub async fn generate2(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        let mut tokens = self
            .tokenizer()
            .encode(prompt, true)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec();
        let batch_size = 1;
        let mut cache = CpuKVCache::new(self.config.num_hidden_layers(), batch_size);
        let vocab_size = self.config().vocab_size();
        
        let prompt_len = tokens.len();
        let mut next_token_logits: Array1<f32>;

        let effective_max_length = if let Some(new_tokens) = config.max_new_tokens {
            prompt_len + new_tokens
        } else {
            config.max_length
        };

        // --- PHASE 1: PROMPT PROCESSING ---
        if !tokens.is_empty() {
            let prompt_len = tokens.len();
            let prompt_input_ids = Array2::from_shape_vec(
                (batch_size, prompt_len),
                tokens.iter().map(|&t| t as f32).collect(),
            )?;

            // ============================ FIX 1 ============================
            // The attention mask must cover the entire prompt length.
            let attention_mask = Array2::ones((batch_size, prompt_len));
            // ===============================================================

            let decoder_output = self
                .model
                .forward(&prompt_input_ids, &attention_mask, Some(&mut cache))
                .await?;

            let logits_3d = project_to_vocab(
                &decoder_output.last_hidden_state,
                &self.lm_head.t().to_owned(),
            )?;
            next_token_logits = logits_3d.slice(s![0, -1, ..]).to_owned();
        } else {
            // Handle empty prompt, unchanged
            let bos_token_id = self.bos_token_id().unwrap_or(0);
            tokens.push(bos_token_id);
            let input_ids = Array2::from_shape_vec((1, 1), vec![bos_token_id as f32])?;
            let attention_mask = Array2::ones((1, 1));
            let decoder_output = self
                .model
                .forward(&input_ids, &attention_mask, Some(&mut cache))
                .await?;
            let logits_3d = project_to_vocab(
                &decoder_output.last_hidden_state,
                &self.lm_head.t().to_owned(),
            )?;
            next_token_logits = logits_3d.slice(s![0, 0, ..]).to_owned();
        }

        // --- PHASE 2: AUTOREGRESSIVE DECODING LOOP ---
        for _ in 0..effective_max_length {
            // --- APPLY SAMPLING LOGIC ---
            let mut bounded_logits = Array1::<f32>::from_elem(vocab_size, f32::NEG_INFINITY);
            let actual_size = next_token_logits.len().min(vocab_size);
            
            bounded_logits
                .slice_mut(s![..actual_size])
                .assign(&next_token_logits.slice(s![..actual_size]));
            
            
            let penalized_logits =
                apply_repetition_penalty(bounded_logits, &tokens, config.repetition_penalty);


            let next_token = sample_token(penalized_logits, &config)?.min(vocab_size as u32 - 1);
            tokens.push(next_token);
            if Some(next_token) == self.eos_token_id() {
                break;
            }

            // --- PREPARE FOR NEXT ITERATION ---
            let input_ids = Array2::from_shape_vec((1, 1), vec![next_token as f32])?;

            // ============================ FIX 2 ============================
            // The mask must now cover the length of the cache PLUS the new token.
            let total_len = cache.get_seq_length() + 1;
            let attention_mask = Array2::ones((batch_size, total_len));
            // ===============================================================

            let decoder_output = self
                .model
                .forward(&input_ids, &attention_mask, Some(&mut cache))
                .await?;
            let logits_3d = project_to_vocab(
                &decoder_output.last_hidden_state,
                &self.lm_head.t().to_owned(),
            )?;
            next_token_logits = logits_3d.slice(s![0, 0, ..]).to_owned();
        }

        self.tokenizer()
            .decode(&tokens, true)
            .map_err(|e| anyhow!(e))
    }
    

/// Sample a token from logits

    // pub async fn generate(
    //     &self,
    //     prompt: &str,
    //     config: &GenerationConfig, // <-- Accept GenerationConfig
    // ) -> Result<String> {
    //     let mut tokens = self
    //         .tokenizer()
    //         .encode(prompt, true)
    //         .map_err(|e| anyhow!(e))?
    //         .get_ids()
    //         .to_vec();
    //     let batch_size = 1;
    //     let mut cache = CpuKVCache::new(self.config.num_hidden_layers(), batch_size);
    //     let vocab_size = self.config().vocab_size(); // Get vocab size from the config trait

    //     // --- PROMPT PROCESSING ---
    //     if !tokens.is_empty() {
    //         let prompt_input_ids = Array2::from_shape_vec(
    //             (batch_size, tokens.len()),
    //             tokens.iter().map(|&t| t as f32).collect(),
    //         )?;
    //         // This pass just populates the cache. We don't need the output logits yet.
    //         self.model
    //             .forward(&prompt_input_ids, &Array2::ones((1, 1)), Some(&mut cache))
    //             .await?;
    //     }

    //     // --- TOKEN-BY-TOKEN GENERATION ---
    //     for _ in 0..config.max_new_tokens {
    //         // The input is ONLY the single, most recently generated token.
    //         let last_token = *tokens.last().unwrap_or(&self.bos_token_id().unwrap_or(0)); // Handle empty prompt
    //         let input_ids = Array2::from_shape_vec((1, 1), vec![last_token as f32])?;

    //         // Forward pass for the single token
    //         let decoder_output = self
    //             .model
    //             .forward(&input_ids, &Array2::ones((1, 1)), Some(&mut cache))
    //             .await?;

    //         // Project to vocab
    //         let logits_3d = project_to_vocab(
    //             &decoder_output.last_hidden_state,
    //             &self.lm_head.t().to_owned(),
    //         )?;

    //         let next_token_logits = logits_3d.slice(s![0, 0, ..]).to_owned();

    //         // --- APPLY SAMPLING LOGIC (Copied from old code) ---
    //         let mut bounded_logits = Array1::<f32>::from_elem(vocab_size, f32::NEG_INFINITY);
    //         let actual_size = next_token_logits.len().min(vocab_size);
    //         bounded_logits
    //             .slice_mut(s![..actual_size])
    //             .assign(&next_token_logits.slice(s![..actual_size]));

    //         let penalized_logits =
    //             apply_repetition_penalty(bounded_logits, &tokens, config.repetition_penalty);

    //         let next_token = sample_token(penalized_logits, &config)?.min(vocab_size as u32 - 1);

    //         tokens.push(next_token);

    //         if Some(next_token) == self.eos_token_id() {
    //             break;
    //         }
    //     }

    //     self.tokenizer()
    //         .decode(&tokens, true)
    //         .map_err(|e| anyhow!(e))
    // }

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
        &self.lm_head
    }
}
