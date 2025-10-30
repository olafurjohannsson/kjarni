//! Cross-encoder for reranking and pairwise classification
//!
//! Takes two texts as input and outputs a relevance score.
//! Used for reranking search results or computing pairwise similarity.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array1, Array2};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokenizers::utils::padding;

use edgetransformers::encoder::TransformerEncoder;
use edgetransformers::models::{ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::traits::{Encoder, EncoderArchitecture, EncoderOutput, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;

mod configs;
pub use configs::MiniLMCrossEncoderConfig;

/// Cross-encoder for computing relevance scores between text pairs
///
/// Useful for reranking search results or computing semantic similarity.
pub struct CrossEncoder {
    encoder: TransformerEncoder,
    tokenizer: Tokenizer,
    pooler: Pooler,
    classifier: ClassificationHead,
    config: Arc<dyn EncoderArchitecture + Send + Sync>,
    model_type: ModelType,
}

struct Pooler {
    dense_weight: Array2<f32>, // [hidden_size, hidden_size]
    dense_bias: Array1<f32>,   // [hidden_size]
}

impl Pooler {
    fn new(dense_weight: Array2<f32>, dense_bias: Array1<f32>) -> Self {
        Self {
            dense_weight,
            dense_bias,
        }
    }

    /// Forward pass: [batch, hidden] -> [batch, hidden]
    fn forward(&self, cls_hidden_state: &Array2<f32>) -> Array2<f32> {
        let mut pooled_output = cls_hidden_state.dot(&self.dense_weight);
        pooled_output += &self.dense_bias;
        // Apply Tanh activation
        pooled_output.mapv_inplace(f32::tanh);
        pooled_output
    }
}

/// Simple classification head (linear layer)
struct ClassificationHead {
    weight: Array2<f32>, // [hidden_size, num_labels]
    bias: Array1<f32>,   // [num_labels]
}

impl ClassificationHead {
    fn new(weight: Array2<f32>, bias: Array1<f32>) -> Self {
        Self { weight, bias }
    }

    /// Forward pass: [batch, hidden] → [batch, num_labels]
    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let output = input.dot(&self.weight);
        output + &self.bias
    }
}

impl CrossEncoder {
    /// Supported cross-encoder model types
    const SUPPORTED_MODELS: &'static [ModelType] = &[ModelType::MiniLML6V2CrossEncoder];

    /// Create cross-encoder from HuggingFace registry
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!(
                "Unsupported model type: {:?}. Supported: {:?}",
                model_type,
                Self::SUPPORTED_MODELS
            ));
        }

        let info = model_type.info();

        if info.architecture != ModelArchitecture::CrossEncoder {
            return Err(anyhow!(
                "Model {:?} is not a cross-encoder (architecture: {:?})",
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
        Self::download_model_files(&model_dir, &info.paths).await?;

        // Load from local path
        Self::from_pretrained(&model_dir, model_type, device, context)
    }

    /// Create cross-encoder from local model directory
    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!(
                "CrossEncoder: Unsupported model type: {:?}",
                model_type
            ));
        }

        let weights = ModelWeights::new(model_path)?;

        let mut tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Load encoder
        let config = match model_type {
            ModelType::MiniLML6V2CrossEncoder => {
                let config = Arc::new(MiniLMCrossEncoderConfig::from_json(&weights.config_json)?);
                config
            }
            _ => return Err(anyhow!("Unsupported cross-encoder: {:?}", model_type)),
        };

        let truncation_params = tokenizers::TruncationParams {
            max_length: config.max_position_embeddings(),
            ..Default::default()
        };

        _ = tokenizer.with_truncation(Some(truncation_params));

        let padding_params = tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding_params));

        let encoder = TransformerEncoder::new(&weights, config.clone(), device, context)?;

        // Load pooler
        let pooler_weight = weights.get_linear_weight("bert.pooler.dense.weight")?;
        let pooler_bias = weights.get_array1("bert.pooler.dense.bias")?;
        let pooler = Pooler::new(pooler_weight, pooler_bias);

        // Load classification head
        let classifier_weight = weights.get_linear_weight("classifier.weight")?;
        let classifier_bias = weights.get_array1("classifier.bias")?;
        let classifier = ClassificationHead::new(classifier_weight, classifier_bias);

        Ok(Self {
            encoder,
            tokenizer,
            pooler,
            config,
            classifier,
            model_type,
        })
    }

    /// Score a single text pair
    ///
    /// Returns a relevance score (higher = more relevant).
    ///
    /// # Example
    /// ```no_run
    /// use edgemodels::cross_encoder::CrossEncoder;
    /// # async fn example(encoder: &CrossEncoder) -> anyhow::Result<()> {
    /// let score = encoder.predict(
    ///     "What is the capital of France?",
    ///     "Paris is the capital and largest city of France."
    /// ).await?;
    /// println!("Relevance score: {:.4}", score);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn predict(&self, text1: &str, text2: &str) -> Result<f32> {
        let scores = self.predict_batch(&[(text1, text2)]).await?;
        Ok(scores[0])
    }

    /// Score multiple text pairs
    ///
    /// # Example
    /// ```no_run
    /// use edgemodels::cross_encoder::CrossEncoder;
    /// # async fn example(encoder: &CrossEncoder) -> anyhow::Result<()> {
    /// let pairs = [
    ///     ("query", "relevant document"),
    ///     ("query", "irrelevant document"),
    /// ];
    /// let scores = encoder.predict_batch(&pairs).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn predict_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

        let mut batch_input_ids = Array2::<f32>::zeros((batch_size, max_len));
        let mut batch_attention_mask = Array2::<f32>::zeros((batch_size, max_len));
        let mut batch_token_type_ids = Array2::<f32>::zeros((batch_size, max_len)); // The crucial tensor

        for (i, encoding) in encodings.iter().enumerate() {
            for (j, &id) in encoding.get_ids().iter().enumerate() {
                batch_input_ids[[i, j]] = id as f32;
            }
            for (j, &mask) in encoding.get_attention_mask().iter().enumerate() {
                batch_attention_mask[[i, j]] = mask as f32;
            }
            // This loop is the fix
            for (j, &type_id) in encoding.get_type_ids().iter().enumerate() {
                batch_token_type_ids[[i, j]] = type_id as f32;
            }
        }

        // Now pass all three tensors to the encoder
        let encoder_output = self
            .encoder
            .forward(
                &batch_input_ids,
                &batch_attention_mask,
                Some(&batch_token_type_ids),
            )
            .await?;

        // Extract [CLS] token
        let cls_embeddings = encoder_output
            .last_hidden_state
            .slice(ndarray::s![.., 0, ..])
            .to_owned();

        // --- ADD THE POOLER FORWARD PASS ---
        let pooled_output = self.pooler.forward(&cls_embeddings);

        // --- PASS THE POOLED OUTPUT TO THE CLASSIFIER ---
        let logits = self.classifier.forward(&pooled_output);

        // Extract scores
        Ok(logits.column(0).to_vec())
    }

    /// Rerank documents by relevance to a query
    ///
    /// Returns indices sorted by relevance (highest first).
    ///
    /// # Example
    /// ```no_run
    /// use edgemodels::cross_encoder::CrossEncoder;
    /// # async fn example(encoder: &CrossEncoder) -> anyhow::Result<()> {
    /// let query = "What is machine learning?";
    /// let documents = vec![
    ///     "Machine learning is a subset of AI",
    ///     "The weather is sunny today",
    ///     "Deep learning uses neural networks",
    /// ];
    ///
    /// let ranked_indices = encoder.rerank(query, &documents).await?;
    /// for (rank, &idx) in ranked_indices.iter().enumerate() {
    ///     println!("Rank {}: {}", rank + 1, documents[idx]);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<usize>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Create pairs
        let pairs: Vec<(&str, &str)> = documents.iter().map(|doc| (query, *doc)).collect();

        // Score all pairs
        let scores = self.predict_batch(&pairs).await?;

        // Sort indices by score (descending)
        let mut indexed_scores: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(indexed_scores.into_iter().map(|(idx, _)| idx).collect())
    }

    async fn download_model_files(
        model_dir: &Path,
        paths: &edgetransformers::models::ModelPaths,
    ) -> Result<()> {
        tokio::fs::create_dir_all(model_dir).await?;

        let files = [
            ("model.safetensors", &paths.weights_url),
            ("tokenizer.json", &paths.tokenizer_url),
            ("config.json", &paths.config_url),
        ];

        for (filename, url) in files {
            let local_path = model_dir.join(filename);

            if !local_path.exists() {
                println!("Downloading {}...", filename);
                let response = reqwest::get(*url).await?;

                if !response.status().is_success() {
                    return Err(anyhow!(
                        "Failed to download {}: HTTP {}",
                        filename,
                        response.status()
                    ));
                }

                let bytes = response.bytes().await?;
                tokio::fs::write(&local_path, &bytes).await?;
                println!("✓ Downloaded {}", filename);
            }
        }

        Ok(())
    }
}

impl TransformerModel for CrossEncoder {
    fn device(&self) -> Device {
        self.encoder.device()
    }
}

impl LanguageModel for CrossEncoder {
    fn config(&self) -> &dyn LanguageModelConfig {
        self.config.as_ref()
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}
