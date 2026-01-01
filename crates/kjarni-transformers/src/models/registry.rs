//! Pretrained model registry with metadata.
//!
//! This module acts as the "App Store" for the inference engine, defining
//! supported architectures, canonical URLs, and task types.

use crate::utils::levenshtein;
use anyhow::{Result, anyhow};
use tokenizers::Model;
use std::path::{Path, PathBuf};
use strum_macros::EnumIter;

pub enum WeightsFormat {
    SafeTensors, // Default
    GGUF,        // Optimized
}

/// Defines the specific structural family of the model.
///
/// This determines which `Config` struct and `Decoder` implementation
/// will be used to load the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    // === Decoders ===
    /// Standard Llama architecture (RoPE, SwiGLU, RMSNorm).
    /// Used by: Llama 2/3, Alpaca, Vicuna, TinyLlama.
    Llama,
    /// Qwen2 family (Llama-like but adds Bias to QKV/Output).
    Qwen2,
    /// Mistral family (Llama-like but adds Sliding Window Attention).
    Mistral,
    /// Phi-3 family (Llama-like but uses LongRoPE scaling).
    Phi3,

    // === Encoders ===
    /// BERT family (Absolute Positional Embeddings).
    Bert,
    /// NomicBERT (Uses RoPE in an Encoder).
    NomicBert,

    // === Seq2Seq ===
    /// T5/FLAN family (Relative Positional Buckets).
    T5,
    /// BART family (Learned Positional Embeddings).
    Bart,

    // Legacy
    GPT,
}

impl ModelArchitecture {
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
    /// Returns the broad category of the architecture for filtering/display.
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

/// Defines the primary intended use case for the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelTask {
    /// Vector embeddings for RAG/Search.
    Embedding,
    /// Reranking search results.
    ReRanking,
    /// Interactive chat and instruction following.
    Chat,
    /// Deep reasoning and logic (Chain of Thought).
    Reasoning,
    /// Detecting positive/negative sentiment.
    SentimentAnalysis,
    /// Classifying text into arbitrary labels.
    ZeroShotClassification,
    /// Translation or Text-to-Text generation.
    Seq2Seq,

    Generation,
}

/// The curated list of supported models.
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

#[derive(Debug, Clone)]
pub struct ModelPaths {
    // === Standard / Base Model (SafeTensors) ===
    /// URL to model.safetensors or model.safetensors.index.json
    pub weights_url: &'static str,
    /// URL to tokenizer.json (Always used)
    pub tokenizer_url: &'static str,
    /// URL to config.json (Always used)
    pub config_url: &'static str,

    // === GGUF Optimization (Optional) ===
    /// Direct URL to the recommended .gguf file (e.g. Q4_K_M)
    pub gguf_url: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub architecture: ModelArchitecture,
    pub task: ModelTask,
    pub paths: ModelPaths,
    pub description: &'static str,
    pub size_mb: usize,
    pub params_millions: usize,
}

impl ModelType {
    pub fn architecture(&self) -> ModelArchitecture {
        self.info().architecture
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

// --- Download Helpers (Standardized) ---

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

/// Get the default cache directory, respecting KJARNI_CACHE_DIR env var.
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

/// Format parameter count in a human-readable way (e.g., "1.5B", "345M").
pub fn format_params(millions: usize) -> String {
    if millions >= 1000 {
        format!("{:.1}B", millions as f64 / 1000.0)
    } else {
        format!("{}M", millions)
    }
}

/// Format file size in a human-readable way (e.g., "1.6 GB", "420 MB").
pub fn format_size(mb: usize) -> String {
    if mb >= 1000 {
        format!("{:.1} GB", mb as f64 / 1000.0)
    } else {
        format!("{} MB", mb)
    }
}
