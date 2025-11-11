//! LLaMA decoder-only language model
//!
//! Supports LLaMA 1, 2, and 3 variants with RoPE positional encoding,
//! RMSNorm, and SwiGLU activation.

pub use super::llama_configs::LlamaConfig;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::models::base::RopeScalingConfig;
use edgetransformers::models::download_model_files;
use edgetransformers::models::{ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::rope::RoPE;
use edgetransformers::traits::{
    DecoderArchitecture, DecoderOutput, LanguageModelConfig, TransformerConfig, TransformerModel,
};
use edgetransformers::weights::ModelWeights;
use edgetransformers::{CpuKVCache, LanguageModel};
use log::{debug, info};
use ndarray::{Array1, Array2, s};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// LLaMA language model for text generation
///
/// Supports autoregressive text generation with KV caching and RoPE positional encoding.
pub struct LlamaModel {
    decoder: TransformerDecoder,
    tokenizer: Tokenizer,
    config: Arc<LlamaConfig>,
    lm_head: Array2<f32>, // [vocab_size, hidden_size] - note the shape!
    rope: Arc<RoPE>,
    model_type: ModelType,
}

impl LlamaModel {
    /// Supported LLaMA model types
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::Llama3_2_1B, // Add these to your ModelType enum
                                // ModelType::Llama3_2_3B,
                                // ModelType::Llama3_8B,
                                // ModelType::Llama2_7B,
    ];

    /// Create LLaMA model from HuggingFace registry
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!(
                "LLaMA: Unsupported model type: {:?}. Supported: {:?}",
                model_type,
                Self::SUPPORTED_MODELS
            ));
        }

        let info = model_type.info();

        if info.architecture != ModelArchitecture::Decoder {
            return Err(anyhow!(
                "Model {:?} is not a decoder-only model (architecture: {:?})",
                model_type,
                info.architecture
            ));
        }

        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("edgetransformers")
        });

        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        // Download files
        println!("Model dir: {}", model_dir.display());
        download_model_files(&model_dir, &info.paths).await?;

        // Load from local path
        Self::from_pretrained(&model_dir, model_type, device, context)
    }

    pub fn get_rope_scaling(&self) -> Option<&RopeScalingConfig> {
        if let Some(llama_config) = self.config.as_any().downcast_ref::<LlamaConfig>() {
            return llama_config.rope_scaling.as_ref();
        }
        None
    }
    /// Create LLaMA model from local model directory
    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!("LLaMA: Unsupported model type: {:?}", model_type));
        }

        let weights = ModelWeights::new(model_path)?;

        let mut tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Parse config
        let config = Arc::new(LlamaConfig::from_json(&weights.config_json)?);

        debug!("--- Loaded LLaMA config ---");
        debug!("Hidden size: {}", config.hidden_size);
        debug!("Num layers: {}", config.num_hidden_layers);
        debug!("Num heads: {}", config.num_attention_heads);
        debug!("Num KV heads: {}", config.num_key_value_heads);
        debug!("Vocab size: {}", config.vocab_size);
        debug!("RoPE theta: {}", config.rope_theta);
        debug!("Uses GQA: {}", config.uses_gqa());
        debug!("---------------------------");

        // Set up tokenizer truncation
        let truncation_params = tokenizers::TruncationParams {
            max_length: config.max_position_embeddings(),
            ..Default::default()
        };
        let _ = tokenizer.with_truncation(Some(truncation_params));

        // No padding for autoregressive generation
        tokenizer.with_padding(None);

        // Create RoPE module
        // let rope = Arc::new(RoPE::new(
        //     config.head_dim(),
        //     config.max_position_embeddings(),
        //     config.rope_theta,
        // ));

        let rope = Arc::new(RoPE::new_with_scaling(
            config.head_dim(),
            config.max_position_embeddings(),
            config.rope_theta,
            config.rope_scaling.as_ref(), // ✅ Pass scaling config
        ));
        // config.rope_scaling()
        //       pub fn get_rope_scaling(&self) -> Option<&RopeScalingConfig> {
        //     if let Some(llama_config) = self.config.as_any().downcast_ref::<LlamaConfig>() {
        //         return llama_config.rope_scaling.as_ref();
        //     }
        //     None
        // }

        // Load decoder
        let decoder = TransformerDecoder::new(
            &weights,
            config.clone() as Arc<dyn DecoderArchitecture + Send + Sync>,
            device,
            context,
            Some(rope.clone()), // Pass RoPE to decoder
        )?;

        // Load LM head
        // Shape: [vocab_size, hidden_size] - already in correct orientation
        // let lm_head = weights.get_array2("lm_head.weight")?;
        let lm_head = weights.get_array2("model.embed_tokens.weight")?;

        println!("=== LM HEAD DEBUG ===");
        println!("Shape: {:?}", lm_head.shape());
        println!("Mean: {}", lm_head.mean().unwrap());
        println!("Std: {}", lm_head.std(0.0));
        println!(
            "First row (token 0) mean: {}",
            lm_head.row(0).mean().unwrap()
        );
        println!("Row 304 ('in') mean: {}", lm_head.row(304).mean().unwrap());

        println!(
            "LM head shape: {:?} (expected: [{}, {}])",
            lm_head.shape(),
            config.vocab_size(),
            config.hidden_size()
        );
        // println!("=== LM HEAD2 DEBUG ===");
        // println!("Shape: {:?}", lm_head.shape());
        // println!("Mean: {}", lm_head.mean().unwrap());
        // println!("Std: {}", lm_head.std(0.0));
        // println!(
        //     "First row (token 0) mean: {}",
        //     lm_head.row(0).mean().unwrap()
        // );
        // println!("Row 304 ('in') mean: {}", lm_head.row(304).mean().unwrap());

        // println!(
        //     "LM head shape: {:?} (expected: [{}, {}])",
        //     lm_head.shape(),
        //     config.vocab_size(),
        //     config.hidden_size()
        // );

        Ok(Self {
            decoder,
            tokenizer,
            config,
            lm_head,
            rope,
            model_type,
        })
    }

    /// Generate text from a prompt
    ///
    /// # Example
    /// ```no_run
    /// use edgemodels::text_generation::LlamaModel;
    /// # async fn example(model: &LlamaModel) -> anyhow::Result<()> {
    /// let prompt = "The capital of France is";
    /// let generated = model.generate(prompt, 20, 1.0, None).await?;
    /// println!("Generated: {}", generated);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> Result<String> {
        // Tokenize prompt
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();

        if input_ids.is_empty() || input_ids[0] != self.config.bos_token_id {
            input_ids.insert(0, self.config.bos_token_id);
        }

        let mut generated_ids = input_ids.clone();
        let prompt_len = input_ids.len();
        let max_len = prompt_len + max_new_tokens;

        // ✅ Initialize KV cache with physical size
        let mut cache = CpuKVCache::new(
            self.config.num_hidden_layers(),
            1,
            max_len, // ✅ Physical cache size
            self.config.kv_dim(),
        );

        // ✅ Create full mask (zeros = masked)
        let mut full_attention_mask = Array2::zeros((1, max_len));

        // ✅ Unmask prompt positions
        full_attention_mask
            .slice_mut(s![.., 0..prompt_len])
            .fill(1.0);

        // Process prompt (prefill)
        let input_array = Array2::from_shape_vec(
            (1, prompt_len),
            input_ids.iter().map(|&id| id as f32).collect(),
        )?;

        let priming_mask = full_attention_mask.slice(s![.., 0..prompt_len]).to_owned();

        self.decoder
            .forward(&input_array, &priming_mask, Some(&mut cache))
            .await?;

        // Generate tokens one by one
        for _ in 0..max_new_tokens {
            let last_token = *generated_ids.last().unwrap();
            let input = Array2::from_elem((1, 1), last_token as f32);
            let current_len = generated_ids.len();

            // ✅ Unmask the new position
            full_attention_mask[[0, current_len]] = 1.0;

            let generation_mask = full_attention_mask
                .slice(s![.., 0..current_len + 1])
                .to_owned();

            // Forward pass with full mask
            let output = self
                .decoder
                .forward(&input, &generation_mask, Some(&mut cache))
                .await?;

            let last_hidden = output.last_hidden_state.slice(ndarray::s![0, -1, ..]);
            let logits = self.lm_head.dot(&last_hidden);
            let next_token = self.sample_token(&logits, temperature, top_k)?;

            if next_token == self.config.eos_token_id {
                break;
            }

            generated_ids.push(next_token);
        }

        // Decode
        let output = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| anyhow!("Decoding failed: {}", e))?;

        Ok(output)
    }

    // DEBUG
pub async fn generate_with_ids(
    &self,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
) -> Result<(String, Vec<u32>)> {
    // --- 1. Tokenization and Setup ---
    let encoding = self
        .tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

    let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();

    if input_ids.is_empty() || input_ids.get(0) != Some(&self.config.bos_token_id) {
        input_ids.insert(0, self.config.bos_token_id);
    }

    let mut generated_ids = input_ids.clone();
    let prompt_len = generated_ids.len();
    let max_len = prompt_len + max_new_tokens;

    let mut cache = CpuKVCache::new(
        self.config.num_hidden_layers(),
        1,
        max_len,
        self.config.kv_dim(),
    );

    let mut full_attention_mask = Array2::zeros((1, max_len));
    
    // --- 2. Prefill Pass & First Token Generation ---
    if prompt_len > 0 {
        full_attention_mask
            .slice_mut(s![.., 0..prompt_len])
            .fill(1.0);
            
        let input_array = Array2::from_shape_vec(
            (1, prompt_len),
            input_ids.iter().map(|&id| id as f32).collect(),
        )?;
        let priming_mask = full_attention_mask.slice(s![.., 0..prompt_len]).to_owned();

        println!("\n--- RUST PREFILL STEP ---");
        println!("Prompt length: {}", prompt_len);
        println!("Cache length BEFORE forward: {}", cache.get_seq_length());

        let prefill_output = self
            .decoder
            .forward(&input_array, &priming_mask, Some(&mut cache))
            .await?;

        println!("Cache length AFTER forward: {}", cache.get_seq_length());
        
        // Generate the first token using the hidden state of the last prompt token
        let last_hidden = prefill_output.last_hidden_state.slice(s![0, -1, ..]);
        let logits = self.lm_head.dot(&last_hidden);
        let first_token = self.sample_token(&logits, temperature, top_k)?;
        
        println!("Selected first token: {}", first_token);
        
        if first_token == self.config.eos_token_id {
            let output = self.tokenizer.decode(&generated_ids, true).map_err(|e| anyhow!("Decoding failed: {}", e))?;
            return Ok((output, generated_ids));
        }
        
        generated_ids.push(first_token);
    }
    
    // --- 3. Autoregressive Generation Loop ---
    // Loop for the REMAINING (max_new_tokens - 1) tokens
    for i in 1..max_new_tokens {
        let current_len = generated_ids.len();
        let last_token = *generated_ids.last().unwrap();
        let input = Array2::from_elem((1, 1), last_token as f32);

        println!("\n--- RUST GEN STEP {} ---", i + 1);
        println!("Current total length: {}", current_len);
        println!("Processing token ID: {}", last_token);
        println!("Cache length BEFORE forward: {}", cache.get_seq_length());

        // ✅ CRITICAL FIX: Unmask the position for the token we are ABOUT to process.
        // `current_len - 1` is the index of `last_token`.
        full_attention_mask[[0, current_len - 1]] = 1.0;
        
        // Create the mask for this step. Its length will be `current_len`.
        let generation_mask = full_attention_mask
            .slice(s![.., 0..current_len])
            .to_owned();
            
        println!("Mask shape for this step: {:?}", generation_mask.shape());

        let output = self
            .decoder
            .forward(&input, &generation_mask, Some(&mut cache))
            .await?;

        println!("Cache length AFTER forward: {}", cache.get_seq_length());

        // The output hidden state has shape [1, 1, hidden_size], so we slice the middle dimension at 0.
        let last_hidden = output.last_hidden_state.slice(s![0, 0, ..]);
        let logits = self.lm_head.dot(&last_hidden);
        let next_token = self.sample_token(&logits, temperature, top_k)?;

        println!("Selected next token: {}", next_token);

        if next_token == self.config.eos_token_id {
            break;
        }

        generated_ids.push(next_token);
    }

    // --- 4. Decode ---
    let output_text = self
        .tokenizer
        .decode(&generated_ids, true)
        .map_err(|e| anyhow!("Decoding failed: {}", e))?;

    Ok((output_text, generated_ids))
}
    // DEBUG

    /// Generate with greedy decoding (deterministic)
    pub async fn generate_greedy(&self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        self.generate(prompt, max_new_tokens, 0.0, None).await
    }

    /// Sample next token from logits
    fn sample_token(
        &self,
        logits: &Array1<f32>,
        temperature: f32,
        top_k: Option<usize>,
    ) -> Result<u32> {
        if temperature == 0.0 {
            // Greedy decoding
            let max_idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .ok_or_else(|| anyhow!("Empty logits"))?;
            return Ok(max_idx as u32);
        }

        // Apply temperature
        let scaled_logits = logits.mapv(|x| x / temperature);

        // Apply top-k filtering if specified
        let filtered_logits = if let Some(k) = top_k {
            let mut indexed: Vec<(usize, f32)> = scaled_logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(k);

            let mut result = Array1::from_elem(scaled_logits.len(), f32::NEG_INFINITY);
            for (idx, val) in indexed {
                result[idx] = val;
            }
            result
        } else {
            scaled_logits
        };

        // Softmax to get probabilities
        let max_logit = filtered_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_logits = filtered_logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        let probs = exp_logits / sum_exp;

        // Sample from categorical distribution
        let rand_val: f32 = rand::random();
        let mut cumsum = 0.0;
        for (idx, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if rand_val < cumsum {
                return Ok(idx as u32);
            }
        }

        // Fallback (shouldn't happen)
        Ok((probs.len() - 1) as u32)
    }

    /// Get the decoder reference
    pub fn decoder(&self) -> &TransformerDecoder {
        &self.decoder
    }

    /// Get the RoPE module
    pub fn rope(&self) -> &Arc<RoPE> {
        &self.rope
    }

    /// Get the configuration
    pub fn llama_config(&self) -> &LlamaConfig {
        &self.config
    }
}

impl TransformerModel for LlamaModel {
    fn device(&self) -> Device {
        self.decoder.device()
    }
}

impl LanguageModel for LlamaModel {
    fn config(&self) -> &dyn LanguageModelConfig {
        self.config.as_ref()
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_config_loading() {
        let json = r#"{
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 8192,
            "vocab_size": 128256,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "bos_token_id": 128000,
            "eos_token_id": 128001
        }"#;

        let config = LlamaConfig::from_json(json).unwrap();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 16);
        assert!(config.uses_gqa());
    }

    #[tokio::test]
    async fn test_llama_generation() {
        // This would require actual model weights
        // Placeholder for integration test
    }
}
