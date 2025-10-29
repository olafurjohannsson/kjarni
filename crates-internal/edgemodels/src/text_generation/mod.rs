use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{s, Array1, Array2};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

use edgetransformers::prelude::*;
use edgetransformers::models::{ModelType, ModelArchitecture, DecoderLanguageModel, LanguageModel};
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::traits::{DecoderArchitecture, LanguageModelConfig, Decoder, DecoderOutput};
use edgetransformers::weights::ModelWeights;
use edgetransformers::CpuKVCache;
use edgetransformers::models::project_to_vocab;

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
        if !Self::SUPPORTED_MODELS.contains(&model_type) { return Err(anyhow!("Unsupported text generator model: {:?}", model_type)); }
        if model_type.info().architecture != ModelArchitecture::Decoder { return Err(anyhow!("Model {:?} is not a decoder model.", model_type)); }

        let cache_dir = cache_dir.unwrap_or_else(|| dirs::cache_dir().expect("No cache directory found").join("edgetransformers"));
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
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|e| anyhow!(e))?;

        
        let config: Arc<dyn DecoderArchitecture + Send + Sync> = match model_type {
            ModelType::DistilGpt2 | ModelType::Gpt2 | ModelType::Gpt2Medium | ModelType::Gpt2Large | ModelType::Gpt2XL => {
                Arc::new(serde_json::from_str::<Gpt2Config>(&weights.config_json)?)
            },
            _ => return Err(anyhow!("Configuration for {:?} not yet implemented.", model_type)),
        };
        
        let model = TransformerDecoder::new(&weights, config.clone(), device, context)?;
        
        let lm_head = weights.get_array2(config.get_lm_head_name())?;

        Ok(Self { model, tokenizer, config, lm_head })
    }

    pub async fn generate(&self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        let mut tokens = self.tokenizer().encode(prompt, true).map_err(|e| anyhow!(e))?.get_ids().to_vec();
        let batch_size = 1;
        let mut cache = CpuKVCache::new(self.config.num_hidden_layers(), batch_size);

        for _ in 0..max_new_tokens {
            let start_pos = cache.get_seq_length();
            let context_tokens = &tokens[start_pos..];
            let input_ids = Array2::from_shape_vec((batch_size, context_tokens.len()), context_tokens.iter().map(|&t| t as f32).collect())?;
            
            // The attention mask must cover the entire sequence length (past + current)
            let attention_mask = Array2::ones((batch_size, tokens.len()));

            let decoder_output = self.model.forward(&input_ids, &attention_mask, Some(&mut cache)).await?;
            let logits_3d = project_to_vocab(&decoder_output.last_hidden_state, &self.lm_head.t().to_owned())?;
            
            let next_token_logits = logits_3d.slice(s![0, -1, ..]);
            
            let next_token = next_token_logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap();
            
            tokens.push(next_token);

            if next_token == self.eos_token_id().unwrap_or(50256) { // 50256 is GPT-2's EOS
                break;
            }
        }
        
        self.tokenizer().decode(&tokens, true).map_err(|e| anyhow!(e))
    }

    async fn download_model_files(model_dir: &Path, paths: &edgetransformers::models::ModelPaths) -> Result<()> {
        tokio::fs::create_dir_all(model_dir).await?;
        let files = [("model.safetensors", paths.weights_url), ("tokenizer.json", paths.tokenizer_url), ("config.json", paths.config_url)];
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
    fn device(&self) -> Device { self.model.device() }
}

impl LanguageModel for TextGenerator {
    fn tokenizer(&self) -> &Tokenizer { &self.tokenizer }
    fn config(&self) -> &dyn LanguageModelConfig { self.config.as_ref() }
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