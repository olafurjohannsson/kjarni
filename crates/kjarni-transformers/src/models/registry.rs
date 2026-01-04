//! Pretrained model registry with metadata and download utilities.
//!
//! This module acts as the "App Store" for the Kjarni inference engine, defining
//! all supported model architectures, their download URLs, task types, and metadata.
//! It provides a curated list of pretrained models optimized for edge deployment,
//! along with utilities for downloading and caching model files.
//!
//! # Overview
//!
//! The registry system consists of several key components:
//!
//! - [`ModelType`] — Enum of all supported pretrained models
//! - [`ModelArchitecture`] — The structural family (Llama, BERT, T5, etc.)
//! - [`ModelTask`] — Primary use case (Chat, Embedding, Seq2Seq, etc.)
//! - [`ModelInfo`] — Complete metadata including URLs and descriptions
//! - [`ModelPaths`] — Download URLs for model files
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::models::registry::{ModelType, get_default_cache_dir};
//!
//! // Get model metadata
//! let info = ModelType::Llama3_2_1B_Instruct.info();
//! println!("Model: {}", info.description);
//! println!("Size: {}", format_size(info.size_mb));
//!
//! // Download model files
//! let cache_dir = get_default_cache_dir();
//! let model_dir = ModelType::Llama3_2_1B_Instruct.cache_dir(&cache_dir);
//! download_model_files(&model_dir, &info.paths, WeightsFormat::GGUF).await?;
//! ```
//!
//! # Model Categories
//!
//! Models are organized by task:
//!
//! - **Embeddings** — Vector embeddings for RAG and semantic search
//! - **Chat** — Interactive conversation and instruction following
//! - **Reasoning** — Deep logical reasoning and chain-of-thought
//! - **Seq2Seq** — Translation, summarization, and text-to-text
//! - **Classification** — Sentiment analysis and zero-shot classification
//!
//! # See Also
//!
//! - [`crate::models::base::LanguageModel`] — Core model trait
//! - [`crate::weights::ModelWeights`] — Low-level weight loading

use crate::utils::levenshtein;
use anyhow::{Result, anyhow};
use tokenizers::Model;
use std::path::{Path, PathBuf};
use strum_macros::EnumIter;

/// Model weight storage format for loading and inference.
///
/// Kjarni supports two primary weight formats with different trade-offs between
/// precision, file size, and loading speed.
///
/// # Variants
///
/// * [`SafeTensors`](WeightsFormat::SafeTensors) — High-precision, standard format.
/// * [`GGUF`](WeightsFormat::GGUF) — Quantized, optimized format for edge devices.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::registry::WeightsFormat;
///
/// // Use GGUF for reduced memory footprint
/// let format = WeightsFormat::GGUF;
///
/// // Use SafeTensors for maximum precision
/// let format = WeightsFormat::SafeTensors;
/// ```
pub enum WeightsFormat {
    /// SafeTensors format with high precision (default).
    ///
    /// Standard format using full precision (F32, BF16) weights. Provides
    /// maximum accuracy but larger file sizes. Compatible with Hugging Face.
    SafeTensors,

    /// GGUF format with quantization optimizations.
    ///
    /// Optimized format supporting aggressive quantization (Q4_K, Q8_0).
    /// Reduces memory usage by 4-8x with minimal quality loss.
    GGUF,
}

/// Defines the specific structural family of a model architecture.
///
/// Each architecture variant represents a distinct transformer design with
/// specific positional encoding, normalization, and attention mechanisms.
/// This determines which config struct and implementation will be used to
/// load and run the model.
///
/// # Architecture Categories
///
/// - **Decoders** — Autoregressive LLMs for text generation
/// - **Encoders** — Bidirectional models for embeddings and classification
/// - **Seq2Seq** — Encoder-decoder models for translation and summarization
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::registry::ModelArchitecture;
///
/// let arch = ModelArchitecture::Llama;
/// assert_eq!(arch.category(), "decoder");
/// println!("{}", arch.display_name());
/// ```
///
/// # See Also
///
/// - [`ModelType`] — Specific pretrained model instances
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    // === Decoders (LLMs) ===

    /// Standard Llama architecture with RoPE, SwiGLU, and RMSNorm.
    ///
    /// Used by: Llama 2/3, Alpaca, Vicuna, TinyLlama. The most widely-adopted
    /// open source architecture for decoder-only language models.
    Llama,

    /// Qwen2 family with bias terms in attention projections.
    ///
    /// Similar to Llama but adds bias to QKV and output projections.
    /// Used by Qwen/Qwen2.5 models.
    Qwen2,

    /// Mistral family with sliding window attention.
    ///
    /// Extends Llama with efficient sliding window attention for long contexts.
    /// Used by Mistral-7B and derived models.
    Mistral,

    /// Phi-3 family with LongRoPE scaling.
    ///
    /// Microsoft's compact architecture using advanced RoPE scaling for
    /// extended context lengths. Used by Phi-3 Mini/Small/Medium.
    Phi3,

    // === Encoders ===

    /// BERT family with absolute positional embeddings.
    ///
    /// The classic bidirectional encoder. Used for embeddings, classification,
    /// and named entity recognition.
    Bert,

    /// NomicBERT with rotary position embeddings.
    ///
    /// Modern encoder using RoPE instead of learned positional embeddings.
    /// Provides better long-context performance for embedding tasks.
    NomicBert,

    // === Seq2Seq (Encoder-Decoder) ===

    /// T5/FLAN family with relative positional buckets.
    ///
    /// Text-to-text encoder-decoder using relative position biases.
    /// Excels at instruction following and translation.
    T5,

    /// BART family with learned positional embeddings.
    ///
    /// Facebook's encoder-decoder with absolute learned positions.
    /// Strong performance on summarization and generation tasks.
    Bart,

    /// GPT family (legacy).
    ///
    /// Original GPT-2 architecture. Included for compatibility but largely
    /// superseded by Llama-based models.
    GPT,
}

impl ModelArchitecture {
    /// Returns a human-readable display name for the architecture.
    ///
    /// # Returns
    ///
    /// A static string describing the architecture with key features.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let arch = ModelArchitecture::Llama;
    /// assert_eq!(arch.display_name(), "Llama (Standard)");
    /// ```
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Llama => "Llama (Standard)",
            Self::Qwen2 => "Qwen2 (Biased)",
            Self::Mistral => "Mistral (SWA)",
            Self::Phi3 => "Phi-3 (LongRoPE)",
            Self::Bert => "BERT",
            Self::NomicBert => "Nomic-BERT",
            Self::T5 => "T5",
            Self::Bart => "BART",
            Self::GPT => "GPT",
        }
    }

    /// Returns the broad category of the architecture.
    ///
    /// Architectures are categorized as decoders (LLMs), encoders (embeddings),
    /// or encoder-decoders (seq2seq). Useful for filtering and display grouping.
    ///
    /// # Returns
    ///
    /// One of: `"decoder"`, `"encoder"`, or `"encoder-decoder"`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// assert_eq!(ModelArchitecture::Llama.category(), "decoder");
    /// assert_eq!(ModelArchitecture::Bert.category(), "encoder");
    /// assert_eq!(ModelArchitecture::T5.category(), "encoder-decoder");
    /// ```
    pub fn category(&self) -> &'static str {
        match self {
            // Decoders (LLMs)
            Self::Llama | Self::Qwen2 | Self::Mistral | Self::Phi3 | Self::GPT => "decoder",

            // Encoders (Embeddings/Classifiers)
            Self::Bert | Self::NomicBert => "encoder",

            // Seq2Seq
            Self::T5 | Self::Bart => "encoder-decoder",
        }
    }
}

/// Defines the primary intended use case for a model.
///
/// Each model in the registry is tagged with its primary task, which determines
/// the optimal input format, output interpretation, and performance characteristics.
/// This allows users to quickly filter models by their intended application.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::registry::{ModelType, ModelTask};
///
/// let info = ModelType::NomicEmbedText.info();
/// assert_eq!(info.task, ModelTask::Embedding);
///
/// let info = ModelType::Llama3_2_1B_Instruct.info();
/// assert_eq!(info.task, ModelTask::Chat);
/// ```
///
/// # See Also
///
/// - [`ModelType`] — Specific pretrained models
/// - [`ModelArchitecture`] — Structural model families
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelTask {
    /// Vector embeddings for retrieval-augmented generation (RAG) and semantic search.
    ///
    /// These models convert text into dense vector representations that capture
    /// semantic meaning. Used for similarity search, clustering, and retrieval.
    Embedding,

    /// Reranking search results for improved relevance.
    ///
    /// Cross-encoder models that score query-document pairs. Used as a second-stage
    /// reranker after initial retrieval to improve result quality.
    ReRanking,

    /// Interactive chat and instruction following.
    ///
    /// General-purpose conversational models optimized for following instructions,
    /// answering questions, and engaging in dialogue. Balanced for speed and quality.
    Chat,

    /// Deep reasoning and logical inference.
    ///
    /// Models optimized for chain-of-thought reasoning, mathematical problem solving,
    /// and complex logical tasks. Often slower but more accurate on reasoning tasks.
    Reasoning,

    /// Detecting positive/negative sentiment in text.
    ///
    /// Classification models that predict emotional tone or polarity (positive,
    /// negative, neutral). Fast and specialized for sentiment analysis.
    SentimentAnalysis,

    /// Classifying text into arbitrary user-defined labels.
    ///
    /// Models that can classify text into any set of labels provided at inference time,
    /// without requiring task-specific fine-tuning. Extremely flexible.
    ZeroShotClassification,

    /// Translation, summarization, and text-to-text generation.
    ///
    /// Encoder-decoder models for transforming input text to output text.
    /// Includes translation, summarization, and instruction-following tasks.
    Seq2Seq,

    /// General text generation (legacy).
    ///
    /// Basic autoregressive generation without instruction tuning.
    /// Mostly superseded by Chat and Reasoning models.
    Generation,
}

/// The curated list of pretrained models supported by Kjarni.
///
/// This enum defines all models that can be loaded and run in the Kjarni
/// inference engine. Each variant represents a specific pretrained model with
/// known-good weights, tokenizer, and configuration hosted on Hugging Face.
///
/// # Model Selection Guide
///
/// **For Embeddings (RAG/Search):**
/// - [`NomicEmbedText`](ModelType::NomicEmbedText) — Modern standard, 8K context
/// - [`MiniLML6V2`](ModelType::MiniLML6V2) — Fastest, legacy option
/// - [`BgeM3`](ModelType::BgeM3) — Multilingual, state-of-the-art
///
/// **For Chat (< 4GB VRAM):**
/// - [`Llama3_2_1B_Instruct`](ModelType::Llama3_2_1B_Instruct) — Fast, balanced
/// - [`Llama3_2_3B_Instruct`](ModelType::Llama3_2_3B_Instruct) — Better quality
/// - [`Phi3_5_Mini_Instruct`](ModelType::Phi3_5_Mini_Instruct) — Best reasoning
///
/// **For Chat (8GB+ VRAM):**
/// - [`Llama3_1_8B_Instruct`](ModelType::Llama3_1_8B_Instruct) — Production standard
/// - [`DeepSeek_R1_Distill_Llama_8B`](ModelType::DeepSeek_R1_Distill_Llama_8B) — SOTA reasoning
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::registry::ModelType;
///
/// // Access model metadata
/// let model = ModelType::Llama3_2_1B_Instruct;
/// let info = model.info();
///
/// println!("Name: {}", model.cli_name());
/// println!("Description: {}", info.description);
/// println!("Size: {} MB", info.size_mb);
/// println!("Parameters: {} M", info.params_millions);
///
/// // Check if model is downloaded
/// let cache_dir = get_default_cache_dir();
/// if model.is_downloaded(&cache_dir) {
///     println!("Model already cached!");
/// }
/// ```
///
/// # See Also
///
/// - [`ModelInfo`] — Complete metadata for a model
/// - [`download_model_files`] — Download model weights and config
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumIter)]
pub enum ModelType {
    // === Embeddings & Reranking ===
    MiniLML6V2,     // Fast, Legacy
    NomicEmbedText, // Modern Standard (8k context)
    BgeM3,          // Dense/Multilingual
    MiniLML6V2CrossEncoder,
    MpnetBaseV2,
    DistilBertBaseCased, // TODO: remove?

    // === Classifiers ===
    DistilBertSST2, // Sentiment
    BartLargeMNLI,  // Zero-Shot Classifier

    // === The "Edge" Kings (< 4GB VRAM/RAM) ===
    Qwen2_5_0_5B_Instruct, // Tiny Logic
    Qwen2_5_1_5B_Instruct,
    Llama3_2_1B_Instruct,  // Fast Chat
    Llama3_2_3B_Instruct,  // Balanced Chat
    Phi3_5_Mini_Instruct,  // Logic Powerhouse (3.8B)

    // === The "Workhorse" Tier (8GB+ RAM) ===
    Mistral7B_v0_3_Instruct,      // 7B Standard
    Llama3_1_8B_Instruct,         // 8B Standard
    DeepSeek_R1_Distill_Llama_8B, // SOTA Reasoning (Distilled)

    // === Seq2Seq ===
    FlanT5Base,
    FlanT5Large,
    DistilBartCnn,
    BartLargeCnn,

    // Legacy
    DistilGpt2,
    Gpt2,
}

/// Download URLs for all required model files.
///
/// Contains URLs for weights, tokenizer, and configuration files. Models can
/// be available in multiple formats (SafeTensors, GGUF), and this struct tracks
/// URLs for both variants when available.
///
/// # Fields
///
/// * `weights_url` — URL to model.safetensors or model.safetensors.index.json.
/// * `tokenizer_url` — URL to tokenizer.json (always required).
/// * `config_url` — URL to config.json (always required).
/// * `gguf_url` — Optional URL to quantized .gguf file (e.g., Q4_K_M variant).
///
/// # Example
///
/// ```ignore
/// let info = ModelType::Llama3_2_1B_Instruct.info();
/// let paths = &info.paths;
///
/// println!("SafeTensors: {}", paths.weights_url);
/// if let Some(gguf) = paths.gguf_url {
///     println!("GGUF: {}", gguf);
/// }
/// ```
///
/// # See Also
///
/// - [`download_model_files`] — Download utility using these paths
/// - [`WeightsFormat`] — Format selection for downloading
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// URL to SafeTensors weights file or index.
    ///
    /// Points to either a single `model.safetensors` file or a
    /// `model.safetensors.index.json` for sharded models.
    pub weights_url: &'static str,

    /// URL to tokenizer configuration.
    ///
    /// Always required. Contains vocabulary and tokenization rules.
    pub tokenizer_url: &'static str,

    /// URL to model configuration.
    ///
    /// Always required. Contains hyperparameters like hidden size, number of layers, etc.
    pub config_url: &'static str,

    /// Optional URL to quantized GGUF file.
    ///
    /// When available, points to a recommended quantized variant (typically Q4_K_M).
    /// Provides 4-8x memory savings with minimal quality loss.
    pub gguf_url: Option<&'static str>,
}

/// Complete metadata for a pretrained model.
///
/// Contains all information needed to understand, download, and load a model,
/// including architecture type, task specialization, file URLs, and size metrics.
///
/// # Example
///
/// ```ignore
/// let info = ModelType::Llama3_2_1B_Instruct.info();
///
/// println!("Architecture: {:?}", info.architecture);
/// println!("Task: {:?}", info.task);
/// println!("Description: {}", info.description);
/// println!("Size: {} MB ({} M params)", info.size_mb, info.params_millions);
/// ```
///
/// # See Also
///
/// - [`ModelType::info`] — Get info for a specific model
/// - [`format_size`] — Format size_mb for display
/// - [`format_params`] — Format params_millions for display
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// The model's structural architecture family.
    pub architecture: ModelArchitecture,
    /// The model's primary intended use case.
    pub task: ModelTask,
    /// Download URLs for all model files.
    pub paths: ModelPaths,
    /// Human-readable description of the model's capabilities.
    pub description: &'static str,
    /// Approximate disk size in megabytes (SafeTensors format).
    pub size_mb: usize,
    /// Number of parameters in millions.
    pub params_millions: usize,
}

impl ModelType {
    pub fn architecture(&self) -> ModelArchitecture {
        self.info().architecture
    }
    pub fn is_llama_model(&self) -> bool {
        matches!(self.architecture(), ModelArchitecture::Llama)
    }
    pub fn is_gpt2_model(&self) -> bool {
        matches!(self, Self::DistilGpt2 | Self::Gpt2)
    }
    pub fn is_qwen_model(&self) -> bool {
        matches!(self.architecture(), ModelArchitecture::Qwen2)
    }
    /// Get the CLI-friendly slug (e.g., "llama3.2-1b")
    pub fn cli_name(&self) -> &'static str {
        match self {
            // Embeddings
            Self::MiniLML6V2 => "minilm-l6-v2",
            Self::NomicEmbedText => "nomic-embed-text",
            Self::BgeM3 => "bge-m3",

            // Classifiers
            Self::DistilBertSST2 => "sentiment-distilbert",
            Self::BartLargeMNLI => "zeroshot-bart",

            // Edge LLMs
            Self::Qwen2_5_0_5B_Instruct => "qwen2.5-0.5b",
            Self::Qwen2_5_1_5B_Instruct => "wen2.5-1.5b",
            Self::Llama3_2_1B_Instruct => "llama3.2-1b",
            Self::Llama3_2_3B_Instruct => "llama3.2-3b",
            Self::Phi3_5_Mini_Instruct => "phi3.5-mini",

            // Workhorse LLMs
            Self::Mistral7B_v0_3_Instruct => "mistral-v0.3",
            Self::Llama3_1_8B_Instruct => "llama3.1-8b",
            Self::DeepSeek_R1_Distill_Llama_8B => "deepseek-r1-8b",

            // Seq2Seq
            Self::FlanT5Base => "flan-t5-base",
            Self::FlanT5Large => "flan-t5-large",


            Self::DistilGpt2 => "distilgpt2",
            Self::Gpt2 => "gpt2",

            Self::DistilBartCnn => "distilbartcnn",
            Self::BartLargeCnn => "bart-large-cnn",
            Self::MiniLML6V2CrossEncoder => "",
            Self::DistilBertBaseCased => "",
            Self::MpnetBaseV2 => "",
        }
    }

    pub fn display_group(&self) -> &'static str {
        match self.info().task {
            // Group 1: Generation
            ModelTask::Chat | ModelTask::Reasoning => "LLM (Decoder)",

            // Group 2: Translation/Summary
            ModelTask::Seq2Seq => "Seq2Seq",

            // Group 3: Vector Search
            ModelTask::Embedding => "Embedding",

            // Group 4: Reranking (Distinct from Embeddings!)
            ModelTask::ReRanking => "Re-Ranker",

            // Group 5: Classification
            ModelTask::SentimentAnalysis | ModelTask::ZeroShotClassification => "Classifier",

            ModelTask::Generation => "Generation (Decoder)"
        }
    }

    pub fn is_instruct_model(&self) -> bool {
        matches!(
            self.info().task,
            ModelTask::Chat | ModelTask::Reasoning | ModelTask::Seq2Seq
        )
    }

    pub fn info(&self) -> ModelInfo {
        match self {
            // ================================================================
            // Embeddings
            // ================================================================
            Self::MiniLML6V2 => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Embedding,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "Fastest sentence embedding model. Ideal for basic RAG.",
                size_mb: 90,
                params_millions: 22,
            },

            Self::MiniLML6V2CrossEncoder => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::ReRanking,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "Compact cross-encoder for passage reranking.",
                size_mb: 90,
                params_millions: 22,
            },

            Self::MpnetBaseV2 => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Embedding,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "High-quality sentence embedding model.",
                size_mb: 420,
                params_millions: 110,
            },

            Self::DistilBertBaseCased => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Embedding,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "Lightweight DistilBERT for question answering.",
                size_mb: 260,
                params_millions: 66,
            },

            Self::NomicEmbedText => ModelInfo {
                architecture: ModelArchitecture::NomicBert,
                task: ModelTask::Embedding,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "Modern standard for RAG. 8192 context length, matryoshka embeddings.",
                size_mb: 550,
                params_millions: 137,
            },
            Self::BgeM3 => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Embedding,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/BAAI/bge-m3/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/BAAI/bge-m3/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "Massive multilingual embedding model. State of the art for diverse languages.",
                size_mb: 2200,
                params_millions: 567,
            },

            // ================================================================
            // Classifiers
            // ================================================================
            Self::DistilBertSST2 => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::SentimentAnalysis,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "Ultra-fast sentiment analysis (Positive/Negative).",
                size_mb: 268,
                params_millions: 66,
            },
            Self::BartLargeMNLI => ModelInfo {
                architecture: ModelArchitecture::Bart,
                task: ModelTask::ZeroShotClassification,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/facebook/bart-large-mnli/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/facebook/bart-large-mnli/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/facebook/bart-large-mnli/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "Magical Zero-Shot classifier. Classify text into ANY labels you provide at runtime.",
                size_mb: 1630,
                params_millions: 406,
            },

            // ================================================================
            // Edge LLMs
            // ================================================================
            Self::Qwen2_5_0_5B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Qwen2,
                task: ModelTask::Chat,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                    ),
                },
                description: "Tiny logic engine. Perfect for structured output and sanity checks.",
                size_mb: 990,
                params_millions: 490,
            },
            Self::Qwen2_5_1_5B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Qwen2,
                task: ModelTask::Chat,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
                    ),
                },
                description: "",
                size_mb: 0,
                params_millions: 0,
            },
            Self::Llama3_2_1B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Llama,
                task: ModelTask::Chat,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                    ),
                },
                description: "Official Meta edge model. Very fast, good general chat.",
                size_mb: 2500,
                params_millions: 1230,
            },
            Self::Llama3_2_3B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Llama,
                task: ModelTask::Chat,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/model.safetensors.index.json",
                    tokenizer_url: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                    ),
                },
                description: "The 3B standard. Excellent balance of speed and coherence.",
                size_mb: 6500,
                params_millions: 3210,
            },
            Self::Phi3_5_Mini_Instruct => ModelInfo {
                architecture: ModelArchitecture::Phi3,
                task: ModelTask::Reasoning,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/model.safetensors.index.json",
                    tokenizer_url: "https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
                    ),
                },
                description: "Microsoft's 3.8B reasoning champion. Punches way above its weight.",
                size_mb: 7500,
                params_millions: 3800,
            },

            // ================================================================
            // Workhorse LLMs
            // ================================================================
            Self::Mistral7B_v0_3_Instruct => ModelInfo {
                architecture: ModelArchitecture::Mistral,
                task: ModelTask::Chat,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/model.safetensors.index.json",
                    tokenizer_url: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                    ),
                },
                description: "Mistral v0.3. Extremely reliable 7B model for all tasks.",
                size_mb: 14500,
                params_millions: 7240,
            },
            Self::Llama3_1_8B_Instruct => ModelInfo {
                architecture: ModelArchitecture::Llama,
                task: ModelTask::Chat,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/model.safetensors.index.json",
                    tokenizer_url: "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                    ),
                },
                description: "The open source standard. Robust, smart, and safe.",
                size_mb: 16000,
                params_millions: 8030,
            },
            Self::DeepSeek_R1_Distill_Llama_8B => ModelInfo {
                architecture: ModelArchitecture::Llama, // It's just Llama architecture!
                task: ModelTask::Reasoning,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/resolve/main/model.safetensors.index.json",
                    tokenizer_url: "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/resolve/main/config.json",
                    gguf_url: Some(
                        "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
                    ),
                },
                description: "State-of-the-Art reasoning distilled from DeepSeek R1.",
                size_mb: 16000,
                params_millions: 8030,
            },

            // ================================================================
            // Seq2Seq
            // ================================================================
            Self::FlanT5Base => ModelInfo {
                architecture: ModelArchitecture::T5,
                task: ModelTask::Seq2Seq,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/google/flan-t5-base/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/google/flan-t5-base/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/google/flan-t5-base/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "General purpose instruction follower (Text-to-Text).",
                size_mb: 990,
                params_millions: 250,
            },
            Self::FlanT5Large => ModelInfo {
                architecture: ModelArchitecture::T5,
                task: ModelTask::Seq2Seq,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/google/flan-t5-large/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/google/flan-t5-large/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/google/flan-t5-large/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "Powerful instruction follower. Great for translation and summarization.",
                size_mb: 3000,
                params_millions: 780,
            },

            Self::BartLargeCnn => ModelInfo {
                architecture: ModelArchitecture::Bart,
                task: ModelTask::Seq2Seq,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/facebook/bart-large-cnn/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "BART large fine-tuned for summarization.",
                size_mb: 1600,
                params_millions: 406,
            },

            Self::DistilBartCnn => ModelInfo {
                architecture: ModelArchitecture::Bart,
                task: ModelTask::Seq2Seq,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/olafuraron/distilbart-cnn-12-6/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/olafuraron/distilbart-cnn-12-6/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/olafuraron/distilbart-cnn-12-6/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "Distilled BART for fast summarization.",
                size_mb: 1000,
                params_millions: 306,
            },

            Self::DistilGpt2 => ModelInfo {
                architecture: ModelArchitecture::GPT,
                task: ModelTask::Generation,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/distilgpt2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/distilgpt2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/distilgpt2/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "Distilled GPT-2 for lightweight text generation.",
                size_mb: 319,
                params_millions: 82,
            },

            Self::Gpt2 => ModelInfo {
                architecture: ModelArchitecture::GPT,
                task: ModelTask::Generation,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/gpt2/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/gpt2/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/gpt2/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "GPT-2 small: general-purpose text generator.",
                size_mb: 548,
                params_millions: 117,
            },
        }
    }

    // --- Utility Methods (Keep these short) ---

    pub fn from_cli_name(name: &str) -> Option<ModelType> {
        use strum::IntoEnumIterator;
        let normalized = name.to_lowercase();
        ModelType::iter().find(|m| m.cli_name() == normalized)
    }

    pub fn all() -> impl Iterator<Item = ModelType> {
        use strum::IntoEnumIterator;
        ModelType::iter()
    }

    pub fn find_similar(query: &str) -> Vec<(String, f32)> {
        let all_names: Vec<&str> = ModelType::all().map(|m| m.cli_name()).collect();
        levenshtein::find_similar(query, &all_names, 3, 0.4)
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

    pub fn search(query: &str) -> Vec<(ModelType, f32)> {
        let query_lower = query.to_lowercase();
        let mut matches: Vec<(ModelType, f32)> = ModelType::all()
            .filter_map(|m| {
                let name = m.cli_name().to_lowercase();
                let desc = m.info().description.to_lowercase();
                let name_sim = levenshtein::similarity(&query_lower, &name);
                // Boost score if substring match found
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

    pub fn repo_id(&self) -> String {
        let url = self.info().paths.weights_url;
        let parts: Vec<&str> = url.split('/').collect();
        // Heuristic: https://huggingface.co/{ORG}/{REPO}/resolve/...
        if parts.len() >= 5 {
            format!("{}/{}", parts[3], parts[4])
        } else {
            "unknown/unknown".to_string()
        }
    }
}

// =============================================================================
// Download Utilities
// =============================================================================

/// Downloads all required files for a model to the specified directory.
///
/// Fetches weights, tokenizer, and config files from Hugging Face Hub. Supports
/// both SafeTensors (high precision) and GGUF (quantized) formats. Files are
/// cached locally and reused on subsequent calls.
///
/// # Arguments
///
/// * `model_dir` - Local directory to store downloaded files.
/// * `paths` - Model file URLs from [`ModelInfo`].
/// * `format` - Desired weights format (SafeTensors or GGUF).
///
/// # Returns
///
/// Path to the downloaded weights file (either .safetensors or .gguf).
///
/// # Errors
///
/// Returns an error if:
/// - Network request fails.
/// - File writing fails.
/// - GGUF format requested but not available for this model.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::registry::{
///     ModelType, WeightsFormat, download_model_files, get_default_cache_dir
/// };
///
/// let model = ModelType::Llama3_2_1B_Instruct;
/// let info = model.info();
/// let cache_dir = get_default_cache_dir();
/// let model_dir = model.cache_dir(&cache_dir);
///
/// // Download with GGUF quantization
/// let weights_path = download_model_files(&model_dir, &info.paths, WeightsFormat::GGUF).await?;
/// println!("Downloaded to: {:?}", weights_path);
/// ```
///
/// # Environment Variables
///
/// - `HF_TOKEN` - Optional Hugging Face access token for private models.
///
/// # See Also
///
/// - [`get_default_cache_dir`] — Get system cache directory
/// - [`ModelType::cache_dir`] — Get model-specific cache directory
pub async fn download_model_files(
    model_dir: &Path,
    paths: &ModelPaths,
    format: WeightsFormat,
) -> Result<PathBuf> {
    tokio::fs::create_dir_all(model_dir).await?;

    // 1. ALWAYS download metadata from the Base Repo (Reliable)
    //    This solves the "Missing tokenizer.json in GGUF repo" issue.
    download_file(model_dir, "tokenizer.json", paths.tokenizer_url).await?;
    download_file(model_dir, "config.json", paths.config_url).await?;

    // 2. Decide which weights to download
    let use_gguf = matches!(format, WeightsFormat::GGUF) && paths.gguf_url.is_some();

    if use_gguf {
        println!("  → Selecting GGUF format (Optimized)");
        let url = paths.gguf_url.unwrap();
        let filename = "model.gguf";

        download_file(model_dir, filename, url).await?;
        Ok(model_dir.join(filename))
    } else {
        println!("  → Selecting SafeTensors format (High Precision)");

        // Fallback to SafeTensors if GGUF requested but not available
        if matches!(format, WeightsFormat::GGUF) {
            println!("  ! GGUF not available for this model, falling back to SafeTensors.");
        }

        if paths.weights_url.ends_with(".index.json") {
            download_sharded_weights(model_dir, paths.weights_url).await?;
            Ok(model_dir.join("model.safetensors.index.json"))
        } else {
            download_file(model_dir, "model.safetensors", paths.weights_url).await?;
            Ok(model_dir.join("model.safetensors"))
        }
    }
}

async fn download_file(model_dir: &Path, filename: &str, url: &str) -> Result<()> {
    let local_path = model_dir.join(filename);
    if local_path.exists() {
        println!("  ✓ {} (cached)", filename);
        return Ok(());
    }
    println!("  ↓ {}...", filename);

    let client = reqwest::Client::new();
    let mut req = client.get(url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        req = req.header("Authorization", format!("Bearer {}", token));
    }

    let response = req.send().await?;
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
    // 1. Download index
    download_file(model_dir, "model.safetensors.index.json", index_url).await?;

    // 2. Parse index
    let index_path = model_dir.join("model.safetensors.index.json");
    let content = tokio::fs::read_to_string(index_path).await?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let weight_map = json["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow!("Invalid index.json"))?;

    let mut shards: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    shards.sort();
    shards.dedup();

    // 3. Download shards
    let base_url = index_url.rsplit_once('/').unwrap().0;

    for (i, shard) in shards.iter().enumerate() {
        let url = format!("{}/{}", base_url, shard);
        println!("  Processing shard {}/{}...", i + 1, shards.len());
        download_file(model_dir, shard, &url).await?;
    }

    Ok(())
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Returns the default cache directory for Kjarni models.
///
/// Determines the appropriate system cache location following platform conventions.
/// Can be overridden with the `KJARNI_CACHE_DIR` environment variable.
///
/// # Platform Paths
///
/// - **Linux:** `~/.cache/kjarni`
/// - **macOS:** `~/Library/Caches/kjarni`
/// - **Windows:** `{FOLDERID_LocalAppData}/kjarni`
///
/// # Returns
///
/// Path to the cache directory.
///
/// # Panics
///
/// Panics if the system does not provide a standard cache directory location.
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::registry::get_default_cache_dir;
///
/// let cache = get_default_cache_dir();
/// println!("Models cached at: {:?}", cache);
/// ```
///
/// # Environment Variables
///
/// - `KJARNI_CACHE_DIR` - Override default cache location.
///
/// # See Also
///
/// - [`ModelType::cache_dir`] — Get model-specific subdirectory
/// - [`ModelType::is_downloaded`] — Check if model exists in cache
pub fn get_default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("KJARNI_CACHE_DIR") {
        PathBuf::from(dir)
    } else {
        // Uses the 'dirs' crate to find the system standard cache directory
        // Linux: ~/.cache/kjarni
        // Mac: ~/Library/Caches/kjarni
        // Windows: {FOLDERID_LocalAppData}/kjarni
        dirs::cache_dir()
            .expect("No cache directory found on system")
            .join("kjarni")
    }
}

/// Formats parameter count in human-readable form.
///
/// Converts millions of parameters to billions (B) or millions (M) with
/// appropriate precision for display.
///
/// # Arguments
///
/// * `millions` - Parameter count in millions.
///
/// # Returns
///
/// Formatted string like "1.5B" or "345M".
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::registry::format_params;
///
/// assert_eq!(format_params(1500), "1.5B");
/// assert_eq!(format_params(345), "345M");
/// assert_eq!(format_params(8030), "8.0B");
/// ```
pub fn format_params(millions: usize) -> String {
    if millions >= 1000 {
        format!("{:.1}B", millions as f64 / 1000.0)
    } else {
        format!("{}M", millions)
    }
}

/// Formats file size in human-readable form.
///
/// Converts megabytes to gigabytes (GB) or megabytes (MB) with appropriate
/// precision for display.
///
/// # Arguments
///
/// * `mb` - File size in megabytes.
///
/// # Returns
///
/// Formatted string like "1.6 GB" or "420 MB".
///
/// # Example
///
/// ```ignore
/// use kjarni_transformers::models::registry::format_size;
///
/// assert_eq!(format_size(1600), "1.6 GB");
/// assert_eq!(format_size(420), "420 MB");
/// assert_eq!(format_size(2500), "2.5 GB");
/// ```
pub fn format_size(mb: usize) -> String {
    if mb >= 1000 {
        format!("{:.1} GB", mb as f64 / 1000.0)
    } else {
        format!("{} MB", mb)
    }
}
