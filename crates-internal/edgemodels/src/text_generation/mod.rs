use crate::generation2::{apply_repetition_penalty, sample_token};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::CpuKVCache;
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::models::base::GenerationConfig;
use edgetransformers::models::download_model_files;
use edgetransformers::models::project_to_vocab;
use edgetransformers::models::{DecoderLanguageModel, LanguageModel, ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::traits::{Decoder, DecoderArchitecture, DecoderOutput, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;
use log::{debug, info};
use ndarray::{Array2, s};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
mod configs;
mod llama;
mod llama_configs;
mod gpt2;
pub use gpt2::Gpt2Model;
pub use configs::Gpt2Config;
pub use llama::LlamaModel;
pub use llama_configs::LlamaConfig;

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
                let mut cfg = serde_json::from_str::<Gpt2Config>(&weights.config_json)?;
                if model_type == ModelType::DistilGpt2 {
                    cfg.set_model_type("distilgpt2".to_string()); // special handling for weight names
                };
                Arc::new(cfg)
            }
            _ => {
                return Err(anyhow!(
                    "Configuration for {:?} not yet implemented.",
                    model_type
                ));
            }
        };

        let model = TransformerDecoder::new(&weights, config.clone(), device, context, None)?;

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
        info!("Starting text generation...");
        // 1. Tokenize the input prompt
        let mut tokens = self
            .tokenizer()
            .encode(prompt, true)
            .map_err(|e| anyhow!(e))?
            .get_ids()
            .to_vec();
        let prompt_len = tokens.len();
        let batch_size = 1;
        debug!("[Setup] Initial prompt token count: {}", prompt_len);

        // 2. Determine the final length of the generated sequence
        let max_len = if let Some(new_tokens) = config.max_new_tokens {
            prompt_len + new_tokens
        } else {
            config.max_length
        };
        debug!("[Setup] Max sequence length set to: {}", max_len);
        // let mut full_attention_mask = Array2::ones((batch_size, max_len));
        let mut full_attention_mask = Array2::zeros((batch_size, max_len)); // ✅ CORRECT

        // 3. Initialize the Key-Value cache
        let mut cache: Box<dyn Cache> = match self.model.device() {
            Device::Cpu => {
                info!("Using CPU backend.");
                Box::new(CpuKVCache::new(
                    self.config.num_hidden_layers(),
                    batch_size,
                    max_len,
                    self.config.hidden_size(),
                ))
            }
            Device::Wgpu => {
                info!("Using GPU (WGPU) backend.");
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
                    max_len,
                )?)
            }
        };
        debug!("[Setup] Initialized cache with capacity: {}", max_len);

        // 4. Process the prompt in a single "priming" pass to fill the cache
        if prompt_len > 0 {
            info!("[Priming Pass] Processing initial prompt...");
            let prompt_ids = Array2::from_shape_vec(
                (batch_size, prompt_len),
                tokens.iter().map(|&t| t as f32).collect(),
            )?;
            // full_attention_mask.slice_mut(s![.., 0..prompt_len]).fill(1.0);
            // let prompt_mask = full_attention_mask.slice(s![.., 0..prompt_len]).to_owned();
            // ✅ Unmask prompt positions
            full_attention_mask
                .slice_mut(s![.., 0..prompt_len])
                .fill(1.0);

            // ✅ CREATE the sliced mask for priming
            // let prompt_mask = full_attention_mask.slice(s![.., 0..prompt_len]).to_owned();
            let mask_to_use_for_priming = match self.model.device() {
                Device::Cpu => {
                    // CPU needs the sliced mask
                    full_attention_mask.slice(s![.., 0..prompt_len]).to_owned()
                }
                Device::Wgpu => {
                    // GPU needs the full physical mask
                    full_attention_mask.clone()
                }
            };
            // --- STRATEGIC LOGGING (Priming) ---
            debug!(
                "[Priming Pass] Cache sequence length BEFORE forward: {}",
                cache.get_seq_length()
            );
            debug!(
                "[Priming Pass] `prompt_ids` shape: {:?}",
                prompt_ids.shape()
            );
            debug!(
                "[Priming Pass] `prompt_mask` shape: {:?}",
                mask_to_use_for_priming.shape()
            );

            self.model
                .forward(&prompt_ids, &mask_to_use_for_priming, Some(cache.as_mut()))
                .await?;

            debug!(
                "[Priming Pass] Cache sequence length AFTER forward: {}",
                cache.get_seq_length()
            );
        }

        // 5. Start the autoregressive generation loop
        info!("Starting autoregressive generation loop...");
        loop {
            // Check stopping conditions
            if tokens.len() >= max_len {
                info!("[Loop] Reached max length ({}), stopping.", max_len);
                break;
            }

            let current_len = tokens.len();
            let step = current_len - prompt_len + 1;
            full_attention_mask[[0, current_len]] = 1.0;
            let last_token = *tokens.last().unwrap_or(&self.bos_token_id().unwrap_or(0));
            let input_ids = Array2::from_shape_vec((1, 1), vec![last_token as f32])?;

            // let attention_mask_view = full_attention_mask.slice(s![.., 0..current_len + 1]);

            // --- STRATEGIC LOGGING (Generation Step) ---
            info!(
                "[Step {}] Generating token {}/{}...",
                step,
                current_len + 1,
                max_len
            );
            debug!(
                "[Step {}] Cache sequence length BEFORE forward: {}",
                step,
                cache.get_seq_length()
            );
            debug!("[Step {}] `input_ids` shape: {:?}", step, input_ids.shape());
            // debug!(
            //     "[Step {}] `attention_mask` shape: {:?}",
            //     step,
            //     attention_mask_view.shape()
            // );
            let generation_mask = full_attention_mask
                .slice(s![.., 0..current_len + 1])
                .to_owned();
            let mask_to_use = match self.model.device() {
                Device::Cpu => {
                    // CPU uses concatenated approach: needs sliced mask
                    full_attention_mask
                        .slice(s![.., 0..current_len + 1])
                        .to_owned()
                }
                Device::Wgpu => {
                    // GPU uses physical cache approach: needs full mask
                    full_attention_mask.clone()
                }
            };

            let decoder_output = self
                .model
                .forward(&input_ids, &mask_to_use.to_owned(), Some(cache.as_mut()))
                .await?;

            debug!(
                "[Step {}] Cache sequence length AFTER forward: {}",
                step,
                cache.get_seq_length()
            );

            let logits_3d = project_to_vocab(&decoder_output.last_hidden_state, &self.lm_head_t)?;

            let mut next_token_logits = logits_3d.slice(s![0, 0, ..]).to_owned();
            if step <= 20 {
                // Get top 5 token predictions
                let mut indexed_logits: Vec<(usize, f32)> = next_token_logits
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i, v))
                    .collect();
                indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                println!("[Step {}] Top-5 logits BEFORE penalty:", step);
                for (i, (token_id, logit)) in indexed_logits.iter().take(5).enumerate() {
                    let token_str = self
                        .tokenizer()
                        .decode(&[*token_id as u32], false)
                        .unwrap_or_default();
                    println!(
                        "  {}. Token {}: {} (logit: {:.4})",
                        i + 1,
                        token_id,
                        token_str,
                        logit
                    );
                }
            }
            next_token_logits =
                apply_repetition_penalty(next_token_logits, &tokens, config.repetition_penalty);

            let next_token = sample_token(next_token_logits, config)?;

            if step <= 20 {
                let selected_token_str = self
                    .tokenizer()
                    .decode(&[next_token], false)
                    .unwrap_or_default();
                println!(
                    "[Step {}] Selected token: {} (id: {})",
                    step, selected_token_str, next_token
                );
            }

            tokens.push(next_token);

            if Some(next_token) == self.eos_token_id() {
                info!("[Loop] End-Of-Sequence token found, stopping.");
                break;
            }
        }

        // 6. Decode the final sequence of tokens back to a string
        info!("Decoding final tokens...");
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
