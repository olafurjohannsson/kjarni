//! Pretrained model registry with metadata

use anyhow::{Result, anyhow};
use std::path::Path;

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
}

/// Supported pretrained model identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    // Llama3_2_3B,
    Llama3_8B,
    Llama3_8B_Instruct,
    // Llama2_7B,

    // === Encoder-Decoder ===
    BartLargeCnn,
    DistilBartCnn,
    T5Small,
    MarianEnIs,
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
                description: "Compact sentence embedding model ideal for CPU or edge inference. Excels at semantic similarity, clustering, and retrieval tasks.",
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
                description: "High-quality sentence embedding model. Slightly heavier than MiniLM but achieves stronger semantic performance.",
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
                description: "Lightweight DistilBERT fine-tuned for question answering on SQuAD. Suitable for CPU inference.",
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
                description: "Compact cross-encoder trained for passage reranking (MS MARCO). Best used to rerank top candidates from bi-encoder retrieval.",
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
                description: "Distilled GPT-2 for lightweight text generation. Great for CPU or edge inference where low latency is required.",
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
                description: "GPT-2 small: good general-purpose text generator. Suitable for research and CPU inference.",
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
                description: "Medium GPT-2 model offering stronger coherence and context handling. Recommended for GPU inference.",
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
                description: "Large GPT-2 variant for higher-quality completions. Requires GPU.",
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
                description: "GPT-2 XL (1.5B parameters): strong open decoder model. Not suitable for edge devices; requires high-end GPU.",
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
                description: "TODO",
                size_mb: 0,
                params_millions: 1000,
            },
            ModelType::Llama3_8B => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3-8B/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/meta-llama/Llama-3-8B/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/meta-llama/Llama-3-8B/resolve/main/config.json",
                },
                description: "TODO",
                size_mb: 0,
                params_millions: 1000,
            },
            ModelType::Llama3_8B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Decoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3-8B-Instruct/resolve/main/model.safetensors.index.json",
                    tokenizer_url: "https://huggingface.co/meta-llama/Llama-3-8B-Instruct/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/meta-llama/Llama-3-8B-Instruct/resolve/main/config.json",
                },
                description: "TODO",
                size_mb: 0,
                params_millions: 1000,
            },

            // === ENCODER-DECODER ===
            ModelType::BartLargeCnn => ModelInfo {
                architecture: ModelArchitecture::EncoderDecoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/config.json",
                },
                description: "Large BART model fine-tuned for news summarization. Excellent quality, requires GPU.",
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
                description: "Distilled BART-CNN. 60% faster and smaller, great for summarization on limited hardware.",
                size_mb: 1000,
                params_millions: 306,
            },

            ModelType::T5Small => ModelInfo {
                architecture: ModelArchitecture::EncoderDecoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/t5-small/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/t5-small/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/t5-small/resolve/main/config.json",
                },
                description: "Small T5 model for general-purpose seq2seq tasks. Runs efficiently on CPUs and edge devices.",
                size_mb: 240,
                params_millions: 60,
            },

            ModelType::MarianEnIs => ModelInfo {
                architecture: ModelArchitecture::EncoderDecoder,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/Helsinki-NLP/opus-mt-en-is/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/Helsinki-NLP/opus-mt-en-is/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/Helsinki-NLP/opus-mt-en-is/resolve/main/config.json",
                },
                description: "MarianMT model for English → Icelandic translation. Compact and accurate.",
                size_mb: 300,
                params_millions: 74,
            },
        }
    }

    /// Get the architecture type
    pub fn architecture(&self) -> ModelArchitecture {
        self.info().architecture
    }
    //weights_url: "https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/model.safetensors",
    /// Get the repo ID from the URL
    pub fn repo_id(&self) -> String {
        let info = self.info();
        // Extract "org/model" from "https://huggingface.co/org/model/resolve/main/..."
        let url = info.paths.weights_url;
        let parts: Vec<&str> = url.split('/').collect();
        if parts.len() >= 5 {
            format!("{}/{}", parts[3], parts[4])
        } else {
            "unknown".to_string()
        }
    }
}

/// Download model files (weights, tokenizer, config) to a local directory
///
/// If files already exist, they are not re-downloaded.
///
/// # Arguments
/// * `model_dir` - Local directory to store the model files
/// * `paths` - URLs for model weights, tokenizer, and config
///
/// # Example
/// ```no_run
/// use edgetransformers::models::{ModelType, download_model_files};
/// use std::path::Path;
///
/// # async fn example() -> anyhow::Result<()> {
/// let model_type = ModelType::MiniLML6V2;
/// let model_info = model_type.info();
/// let cache_dir = Path::new("./models/mini-lm");
///
/// download_model_files(cache_dir, &model_info.paths).await?;
/// # Ok(())
/// # }
/// ```
/// Download model files with sharded model support
pub async fn download_model_files(
    model_dir: &Path,
    paths: &crate::models::ModelPaths,
) -> Result<()> {
    tokio::fs::create_dir_all(model_dir).await?;

    // Download tokenizer and config first
    download_file(model_dir, "tokenizer.json", paths.tokenizer_url).await?;
    download_file(model_dir, "config.json", paths.config_url).await?;

    // Check if this is a sharded model
    if is_sharded_model(paths.weights_url) {
        download_sharded_weights(model_dir, paths.weights_url).await?;
    } else {
        download_file(model_dir, "model.safetensors", paths.weights_url).await?;
    }

    Ok(())
}

/// Check if the weights URL points to a sharded model (index.json)
fn is_sharded_model(url: &str) -> bool {
    url.ends_with(".index.json") || url.contains("model.safetensors.index.json")
}

/// Download a single file if it doesn't exist
async fn download_file(model_dir: &Path, filename: &str, url: &str) -> Result<()> {
    let local_path = model_dir.join(filename);

    if local_path.exists() {
        println!("✓ {} already exists, skipping download", filename);
        return Ok(());
    }

    println!("Downloading {}...", filename);
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
    println!("✓ Downloaded {}", filename);

    Ok(())
}

/// Download sharded model weights
async fn download_sharded_weights(model_dir: &Path, index_url: &str) -> Result<()> {
    // 1. Download the index file
    let index_filename = "model.safetensors.index.json";
    let index_path = model_dir.join(index_filename);

    if !index_path.exists() {
        println!("Downloading {}...", index_filename);
        let response = reqwest::get(index_url).await?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Failed to download index: HTTP {}",
                response.status()
            ));
        }

        let bytes = response.bytes().await?;
        tokio::fs::write(&index_path, &bytes).await?;
        println!("✓ Downloaded {}", index_filename);
    } else {
        println!("✓ {} already exists, skipping download", index_filename);
    }

    // 2. Parse the index to get shard filenames
    let index_content = tokio::fs::read_to_string(&index_path).await?;
    let index: serde_json::Value = serde_json::from_str(&index_content)?;

    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow!("Invalid index.json: missing weight_map"))?;

    // Get unique shard filenames
    let mut shard_files: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str())
        .map(|s| s.to_string())
        .collect();
    shard_files.sort();
    shard_files.dedup();

    println!("Model has {} shards", shard_files.len());

    // 3. Derive base URL from index URL
    let base_url = index_url
        .rsplit_once('/')
        .map(|(base, _)| base)
        .ok_or_else(|| anyhow!("Invalid index URL"))?;

    // 4. Download each shard
    for (i, shard_filename) in shard_files.iter().enumerate() {
        let shard_path = model_dir.join(shard_filename);

        if shard_path.exists() {
            println!(
                "✓ [{}/{}] {} already exists, skipping",
                i + 1,
                shard_files.len(),
                shard_filename
            );
            continue;
        }

        let shard_url = format!("{}/{}", base_url, shard_filename);
        println!(
            "Downloading [{}/{}] {}...",
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
        println!(
            "✓ [{}/{}] Downloaded {} ({:.2} GB)",
            i + 1,
            shard_files.len(),
            shard_filename,
            bytes.len() as f64 / 1_073_741_824.0
        );
    }

    Ok(())
}