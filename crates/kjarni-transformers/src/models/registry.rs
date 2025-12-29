//! Pretrained model registry with metadata

use crate::utils::levenshtein;
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use strum_macros::EnumIter;

/// Distinguishes the architectural type of a transformer model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    /// Encoder-only transformer (BERT, RoBERTa, etc.)
    Encoder,
    /// Cross-encoder for reranking/pairwise scoring
    CrossEncoder,
    /// Decoder-only causal LM (GPT-2, GPT-3, Llama, etc.)
    Decoder,
    /// Encoder-decoder seq2seq (BART, T5, MarianMT, etc.)
    EncoderDecoder,
    /// Sequence classification model (BERT, RoBERTa, etc.)
    SequenceClassifier,
    /// Token classification model (BERT, RoBERTa, etc.)
    TokenClassifier,
    /// Question answering model (BERT, RoBERTa, etc.)
    QuestionAnswering,
    /// Text generation model (GPT-2, GPT-3, etc.)
    TextGeneration,
    /// Sentence encoder model (e.g., for embeddings)
    SentenceEncoder,
}

impl ModelArchitecture {
    /// Get display name for the architecture
    pub fn display_name(&self) -> &'static str {
        match self {
            ModelArchitecture::Encoder => "Encoder",
            ModelArchitecture::CrossEncoder => "Cross-Encoder",
            ModelArchitecture::Decoder => "Decoder",
            ModelArchitecture::EncoderDecoder => "Encoder-Decoder",
            ModelArchitecture::SequenceClassifier => "Sequence Classifier",
            ModelArchitecture::TokenClassifier => "Token Classifier",
            ModelArchitecture::QuestionAnswering => "Question Answering",
            ModelArchitecture::TextGeneration => "Text Generation",
            ModelArchitecture::SentenceEncoder => "Sentence Encoder",
        }
    }
}

/// Supported pretrained model identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumIter)]
pub enum ModelType {
    // === Encoders ===
    MiniLML6V2,
    MpnetBaseV2,
    DistilBertBaseCased,

    // === Cross Encoders ===
    MiniLML6V2CrossEncoder,

    // === Decoders ===
    DistilGpt2,
    Gpt2,
    Gpt2Medium,
    Gpt2Large,
    Gpt2XL,
    Llama3_2_1B,
    Llama3_2_1B_Instruct,
    Llama3_2_3B,
    Llama3_2_3B_Instruct,
    Llama3_8B,
    Llama3_8B_Instruct,

    // === Encoder-Decoder ===
    BartLargeCnn,
    DistilBartCnn,
    FlanT5Small,
    FlanT5Base,
    FlanT5Large,
}

/// Canonical URLs for model files
#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub weights_url: &'static str,
    pub tokenizer_url: &'static str,
    pub config_url: &'static str,
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub architecture: ModelArchitecture,
    pub paths: ModelPaths,
    pub description: &'static str,
    pub size_mb: usize,
    pub params_millions: usize,
}

impl ModelType {
    /// Find similar model names for "did you mean?" suggestions
    pub fn find_similar(query: &str) -> Vec<(String, f32)> {
        let all_names: Vec<&str> = ModelType::all().map(|m| m.cli_name()).collect();

        levenshtein::find_similar(query, &all_names, 3, 0.4)
    }

    /// Search models by name or description
    pub fn search(query: &str) -> Vec<(ModelType, f32)> {
        let query_lower = query.to_lowercase();

        let mut matches: Vec<(ModelType, f32)> = ModelType::all()
            .filter_map(|m| {
                let name = m.cli_name().to_lowercase();
                let desc = m.info().description.to_lowercase();

                // Check name similarity
                let name_sim = levenshtein::similarity(&query_lower, &name);

                // Check if query is substring of name or description
                let contains_bonus = if name.contains(&query_lower) {
                    0.5
                } else if desc.contains(&query_lower) {
                    0.3
                } else {
                    0.0
                };

                let score = name_sim + contains_bonus;

                if score > 0.3 { Some((m, score)) } else { None }
            })
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }
    pub fn is_instruct_model(&self) -> bool {
        matches!(
            self,
            ModelType::Llama3_2_1B_Instruct | ModelType::Llama3_2_3B_Instruct | ModelType::Llama3_8B_Instruct
        )
    }
    pub fn is_bart_model(&self) -> bool {
        matches!(self, ModelType::DistilBartCnn | ModelType::BartLargeCnn)
    }
    pub fn is_llama_model(&self) -> bool {
        matches!(
            self,
            ModelType::Llama3_2_1B
                | ModelType::Llama3_2_3B
                | ModelType::Llama3_2_3B_Instruct
                | ModelType::Llama3_8B
                | ModelType::Llama3_8B_Instruct
        )
    }
    pub fn is_gpt2_model(&self) -> bool {
        matches!(
            self,
            ModelType::DistilGpt2
                | ModelType::Gpt2
                | ModelType::Gpt2Medium
                | ModelType::Gpt2Large
                | ModelType::Gpt2XL
        )
    }
    /// Get the CLI-friendly name for this model
    pub fn cli_name(&self) -> &'static str {
        match self {
            // Encoders
            ModelType::MiniLML6V2 => "minilm-l6-v2",
            ModelType::MpnetBaseV2 => "mpnet-base-v2",
            ModelType::DistilBertBaseCased => "distilbert-base-cased",

            // Cross Encoders
            ModelType::MiniLML6V2CrossEncoder => "minilm-l6-v2-cross-encoder",

            // Decoders
            ModelType::DistilGpt2 => "distilgpt2",
            ModelType::Gpt2 => "gpt2",
            ModelType::Gpt2Medium => "gpt2-medium",
            ModelType::Gpt2Large => "gpt2-large",
            ModelType::Gpt2XL => "gpt2-xl",

            ModelType::Llama3_2_1B => "llama-3.2-1b",
            ModelType::Llama3_2_1B_Instruct => "llama-3.2-1b-instruct",
            ModelType::Llama3_2_3B => "llama-3.2-3b",
            ModelType::Llama3_2_3B_Instruct => "llama-3.2-3b-instruct",
            ModelType::Llama3_8B => "llama-3-8b",
            ModelType::Llama3_8B_Instruct => "llama-3-8b-instruct",

            // Encoder-Decoder
            ModelType::BartLargeCnn => "bart-large-cnn",
            ModelType::DistilBartCnn => "distilbart-cnn",
            ModelType::FlanT5Small => "flan-t5-small",
            ModelType::FlanT5Base => "flan-t5-base",
            ModelType::FlanT5Large => "flan-t5-large",
        }
    }

    /// Parse a CLI name into a ModelType
    pub fn from_cli_name(name: &str) -> Option<ModelType> {
        use strum::IntoEnumIterator;

        let normalized = name.to_lowercase();
        ModelType::iter().find(|m| m.cli_name() == normalized)
    }

    /// Get all available model types
    pub fn all() -> impl Iterator<Item=ModelType> {
        use strum::IntoEnumIterator;
        ModelType::iter()
    }

    /// Get the local cache directory for this model
    pub fn cache_dir(&self, base_dir: &Path) -> PathBuf {
        base_dir.join(self.repo_id().replace('/', "_"))
    }

    /// Check if this model is downloaded in the given cache directory
    pub fn is_downloaded(&self, base_dir: &Path) -> bool {
        let model_dir = self.cache_dir(base_dir);

        // Check for essential files
        let config_exists = model_dir.join("config.json").exists();
        let tokenizer_exists = model_dir.join("tokenizer.json").exists();

        // Weights can be either single file or sharded
        let weights_exist = model_dir.join("model.safetensors").exists()
            || model_dir.join("model.safetensors.index.json").exists();

        config_exists && tokenizer_exists && weights_exist
    }

    /// Get metadata for this model
    pub fn info(&self) -> ModelInfo {
        match self {
            // === ENCODERS ===
            ModelType::MiniLML6V2 => ModelInfo {
                architecture: ModelArchitecture::Encoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
                },
                description: "Compact sentence embedding model ideal for CPU or edge inference.",
                size_mb: 90,
                params_millions: 22,
            },

            ModelType::MpnetBaseV2 => ModelInfo {
                architecture: ModelArchitecture::Encoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/config.json",
                },
                description: "High-quality sentence embedding model.",
                size_mb: 420,
                params_millions: 110,
            },

            ModelType::DistilBertBaseCased => ModelInfo {
                architecture: ModelArchitecture::Encoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/config.json",
                },
                description: "Lightweight DistilBERT for question answering.",
                size_mb: 260,
                params_millions: 66,
            },

            // === CROSS ENCODERS ===
            ModelType::MiniLML6V2CrossEncoder => ModelInfo {
                architecture: ModelArchitecture::CrossEncoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/config.json",
                },
                description: "Compact cross-encoder for passage reranking.",
                size_mb: 90,
                params_millions: 22,
            },

            // === DECODERS ===
            ModelType::DistilGpt2 => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/distilgpt2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/distilgpt2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/distilgpt2/resolve/main/config.json",
                },
                description: "Distilled GPT-2 for lightweight text generation.",
                size_mb: 319,
                params_millions: 82,
            },

            ModelType::Gpt2 => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/gpt2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/gpt2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/gpt2/resolve/main/config.json",
                },
                description: "GPT-2 small: general-purpose text generator.",
                size_mb: 548,
                params_millions: 117,
            },

            ModelType::Gpt2Medium => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/gpt2-medium/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/gpt2-medium/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/gpt2-medium/resolve/main/config.json",
                },
                description: "Medium GPT-2 with stronger coherence.",
                size_mb: 1400,
                params_millions: 345,
            },

            ModelType::Gpt2Large => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/gpt2-large/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/gpt2-large/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/gpt2-large/resolve/main/config.json",
                },
                description: "Large GPT-2 for higher-quality completions.",
                size_mb: 3100,
                params_millions: 774,
            },

            ModelType::Gpt2XL => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/gpt2-xl/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/gpt2-xl/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/gpt2-xl/resolve/main/config.json",
                },
                description: "GPT-2 XL (1.5B parameters).",
                size_mb: 6100,
                params_millions: 1500,
            },

            ModelType::Llama3_2_1B => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/config.json",
                },
                description: "Llama 3.2 1B base model.",
                size_mb: 2500,
                params_millions: 1000,
            },

            ModelType::Llama3_2_1B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json",
                },
                description: "Llama 3.2 1B instruction-tuned.",
                size_mb: 2500,
                params_millions: 1000,
            },

            ModelType::Llama3_2_3B => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3.2-3B/resolve/main/model.safetensors.index.json",
                    tokenizer_url: "https://huggingface.co/meta-llama/Llama-3.2-3B/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/meta-llama/Llama-3.2-3B/resolve/main/config.json",
                },
                description: "Llama 3.2 3B base model.",
                size_mb: 6500,
                params_millions: 3000,
            },

            ModelType::Llama3_2_3B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/model.safetensors.index.json",
                    tokenizer_url: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json",
                },
                description: "Llama 3.2 3B instruction-tuned.",
                size_mb: 6500,
                params_millions: 3000,
            },

            ModelType::Llama3_8B => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3-8B/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/meta-llama/Llama-3-8B/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/meta-llama/Llama-3-8B/resolve/main/config.json",
                },
                description: "Llama 3 8B base model.",
                size_mb: 16000,
                params_millions: 8000,
            },

            ModelType::Llama3_8B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3-8B-Instruct/resolve/main/model.safetensors.index.json",
                    tokenizer_url: "https://huggingface.co/meta-llama/Llama-3-8B-Instruct/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/meta-llama/Llama-3-8B-Instruct/resolve/main/config.json",
                },
                description: "Llama 3 8B instruction-tuned.",
                size_mb: 16000,
                params_millions: 8000,
            },

            // === ENCODER-DECODER ===
            ModelType::BartLargeCnn => ModelInfo {
                architecture: ModelArchitecture::EncoderDecoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/config.json",
                },
                description: "BART large fine-tuned for summarization.",
                size_mb: 1600,
                params_millions: 406,
            },

            ModelType::DistilBartCnn => ModelInfo {
                architecture: ModelArchitecture::EncoderDecoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/olafuraron/distilbart-cnn-12-6/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/olafuraron/distilbart-cnn-12-6/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/olafuraron/distilbart-cnn-12-6/resolve/main/config.json",
                },
                description: "Distilled BART for fast summarization.",
                size_mb: 1000,
                params_millions: 306,
            },

            ModelType::FlanT5Small => ModelInfo {
                architecture: ModelArchitecture::EncoderDecoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/google/flan-t5-small/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/google/flan-t5-small/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/google/flan-t5-small/resolve/main/config.json",
                },
                description: "FLAN-T5 Small - Instruction-tuned T5 for multiple tasks",
                params_millions: 80,
                size_mb: 308,
            },
            ModelType::FlanT5Base => ModelInfo {
                architecture: ModelArchitecture::EncoderDecoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/google/flan-t5-base/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/google/flan-t5-base/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/google/flan-t5-base/resolve/main/config.json",
                },
                description: "FLAN-T5 Base - Instruction-tuned T5 for translation, summarization, QA",
                params_millions: 250,
                size_mb: 990,
            },
            ModelType::FlanT5Large => ModelInfo {
                architecture: ModelArchitecture::EncoderDecoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/google/flan-t5-large/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/google/flan-t5-large/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/google/flan-t5-large/resolve/main/config.json",
                },
                description: "FLAN-T5 Large - High quality instruction-tuned T5",
                params_millions: 780,
                size_mb: 3000,
            },
        }
    }

    /// Get the architecture type
    pub fn architecture(&self) -> ModelArchitecture {
        self.info().architecture
    }

    /// Get the repo ID from the URL
    pub fn repo_id(&self) -> String {
        let info = self.info();
        let url = info.paths.weights_url;
        let parts: Vec<&str> = url.split('/').collect();
        if parts.len() >= 5 {
            format!("{}/{}", parts[3], parts[4])
        } else {
            "unknown".to_string()
        }
    }
}

/// Get the default cache directory, respecting KJARNI_CACHE_DIR env var
pub fn get_default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("KJARNI_CACHE_DIR") {
        PathBuf::from(dir)
    } else {
        dirs::cache_dir()
            .expect("No cache directory found")
            .join("kjarni")
    }
}

/// Format parameters in a human-readable way (e.g., "1.5B", "345M")
pub fn format_params(millions: usize) -> String {
    if millions >= 1000 {
        format!("{:.1}B", millions as f64 / 1000.0)
    } else {
        format!("{}M", millions)
    }
}

/// Format size in a human-readable way (e.g., "1.6 GB", "420 MB")
pub fn format_size(mb: usize) -> String {
    if mb >= 1000 {
        format!("{:.1} GB", mb as f64 / 1000.0)
    } else {
        format!("{} MB", mb)
    }
}

// ... keep existing download_model_files, download_file, download_sharded_weights, is_sharded_model functions ...

/// Download model files (weights, tokenizer, config) to a local directory
pub async fn download_model_files(model_dir: &Path, paths: &ModelPaths) -> Result<()> {
    tokio::fs::create_dir_all(model_dir).await?;

    download_file(model_dir, "tokenizer.json", paths.tokenizer_url).await?;
    download_file(model_dir, "config.json", paths.config_url).await?;

    if is_sharded_model(paths.weights_url) {
        download_sharded_weights(model_dir, paths.weights_url).await?;
    } else {
        download_file(model_dir, "model.safetensors", paths.weights_url).await?;
    }

    Ok(())
}

fn is_sharded_model(url: &str) -> bool {
    url.ends_with(".index.json") || url.contains("model.safetensors.index.json")
}

async fn download_file(model_dir: &Path, filename: &str, url: &str) -> Result<()> {
    let local_path = model_dir.join(filename);

    if local_path.exists() {
        println!("  ✓ {} (cached)", filename);
        return Ok(());
    }

    println!("  ↓ {}...", filename);
    let response = reqwest::get(url).await?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Failed to download {}: HTTP {}",
            filename,
            response.status()
        ));
    }

    let bytes = response.bytes().await?;
    tokio::fs::write(&local_path, &bytes).await?;

    Ok(())
}

async fn download_sharded_weights(model_dir: &Path, index_url: &str) -> Result<()> {
    let index_filename = "model.safetensors.index.json";
    let index_path = model_dir.join(index_filename);

    if !index_path.exists() {
        println!("  ↓ {}...", index_filename);
        let response = reqwest::get(index_url).await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Failed to download index: HTTP {}",
                response.status()
            ));
        }

        let bytes = response.bytes().await?;
        tokio::fs::write(&index_path, &bytes).await?;
    } else {
        println!("  ✓ {} (cached)", index_filename);
    }

    let index_content = tokio::fs::read_to_string(&index_path).await?;
    let index: serde_json::Value = serde_json::from_str(&index_content)?;

    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow!("Invalid index.json: missing weight_map"))?;

    let mut shard_files: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str())
        .map(|s| s.to_string())
        .collect();
    shard_files.sort();
    shard_files.dedup();

    let base_url = index_url
        .rsplit_once('/')
        .map(|(base, _)| base)
        .ok_or_else(|| anyhow!("Invalid index URL"))?;

    for (i, shard_filename) in shard_files.iter().enumerate() {
        let shard_path = model_dir.join(shard_filename);

        if shard_path.exists() {
            println!(
                "  ✓ [{}/{}] {} (cached)",
                i + 1,
                shard_files.len(),
                shard_filename
            );
            continue;
        }

        let shard_url = format!("{}/{}", base_url, shard_filename);
        println!(
            "  ↓ [{}/{}] {}...",
            i + 1,
            shard_files.len(),
            shard_filename
        );

        let response = reqwest::get(&shard_url).await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Failed to download {}: HTTP {}",
                shard_filename,
                response.status()
            ));
        }

        let bytes = response.bytes().await?;
        tokio::fs::write(&shard_path, &bytes).await?;
    }

    Ok(())
}
