use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, s};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use edgetransformers::encoder_decoder::TransformerEncoderDecoder;
use edgetransformers::models::{ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::traits::{
    CrossAttentionDecoder, EncoderDecoderArchitecture, EncoderOutput,
    LanguageModelConfig, TransformerModel,
};
use edgetransformers::weights::ModelWeights;
use edgetransformers::{CpuKVCache};
use edgetransformers::{LanguageModel, Seq2SeqLanguageModel};
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

    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let config: Arc<dyn EncoderDecoderArchitecture + Send + Sync> = match model_type {
            ModelType::BartLargeCnn | ModelType::DistilBartCnn => {
                Arc::new(serde_json::from_str::<BartConfig>(&weights.config_json)?)
            }
            _ => return Err(anyhow!("Seq2seq: Unsupported config for model: {:?}", model_type)),
        };

        let model = TransformerEncoderDecoder::new(&weights, config.clone(), device, context)?;
        
        // *** THE FIX IS HERE ***
        // Load the LM head WITHOUT transposing it. Its shape is [vocab_size, hidden_size].
        let lm_head = weights.get_array2(config.get_lm_head_name())?;
        
        let final_logits_bias = config.get_final_logits_bias_name().and_then(|name| weights.get_array1(name).ok());

        Ok(Self { model, tokenizer, config, lm_head, final_logits_bias })
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
                    .forward(&input_ids, &attention_mask)
                    .await
            }
            TransformerEncoderDecoder::Gpu(_) => todo!("GPU path for encode_input not implemented"),
        }
    }

    /// Generates text autoregressively from pre-computed encoder hidden states.
    async fn generate_from_encoding(
        &self,
        encoder_output: &EncoderOutput,
        encoder_attention_mask: &Array2<f32>,
        max_length: usize,
    ) -> Result<String> {
        let mut generated_tokens = vec![2u32]; // BART's EOS/BOS token
        let batch_size = encoder_output.last_hidden_state.shape()[0];
        let mut cache = CpuKVCache::new(self.config.num_hidden_layers(), batch_size);

        for _ in 0..max_length {
            let last_token = *generated_tokens.last().unwrap();
            let decoder_input_ids =
                Array2::from_shape_vec((batch_size, 1), vec![last_token as f32])?;
            let decoder_attention_mask = Array2::ones((batch_size, generated_tokens.len()));
            // The main forward pass now handles the decoder-only logic when encoder_output is provided.
            let decoder_output = self
                .model
                .forward(
                    &Array2::zeros((0, 0)), // Dummy encoder input
                    &decoder_input_ids,
                    encoder_attention_mask,
                    &decoder_attention_mask,
                    Some(&mut cache),
                    Some(encoder_output),
                )
                .await?;

            let last_hidden_state = decoder_output.last_hidden_state.slice(s![0, -1, ..]);
            // let mut logits = last_hidden_state.dot(&self.lm_head.t());
            let mut logits: Array1<f32> = self.lm_head.dot(&last_hidden_state);

            if let Some(bias) = &self.final_logits_bias {
                logits += bias;
            }

            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap();
            generated_tokens.push(next_token);
            if next_token == 2 {
                break;
            }
        }

        let clean_tokens = if generated_tokens.len() > 1 && generated_tokens.last() == Some(&2) {
            &generated_tokens[1..generated_tokens.len() - 1]
        } else {
            &generated_tokens[1..]
        };
        self.tokenizer()
            .decode(clean_tokens, true)
            .map_err(|e| anyhow!(e))
    }

    /// High-level method to generate text directly from a string.
    async fn generate(&self, input_text: &str, max_length: usize) -> Result<String> {
        let encoding = self
            .tokenizer()
            .encode(input_text, true)
            .map_err(|e| anyhow!(e))?;
        let attention_mask = Array2::ones((1, encoding.len()));
        let encoder_output = self.encode_input(input_text).await?;
        self.generate_from_encoding(&encoder_output, &attention_mask, max_length)
            .await
    }
}
