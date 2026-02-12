//! Pretrained model registry with metadata and download utilities.

use crate::utils::levenshtein;
use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use strum_macros::EnumIter;

/// Model weight storage format for loading and inference.
pub enum WeightsFormat {
    /// SafeTensors format
    SafeTensors,

    /// GGUF format with quantization
    GGUF,
}

/// Defines the model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    /// Standard Llama architecture
    Llama,

    /// Qwen2 family with bias terms in attention projections.
    Qwen2,

    /// Mistral family with sliding window attention.
    Mistral,

    /// Phi-3 family with LongRoPE scaling.
    Phi3,

    /// BERT family with absolute positional embeddings.
    Bert,

    /// MPNet with relative attention bias.
    Mpnet,

    /// NomicBERT with rotary position embeddings.
    NomicBert,

    /// T5/FLAN family with relative positional buckets.
    T5,

    /// BART family with learned positional embeddings.
    Bart,

    /// GPT family (legacy).
    GPT,

    /// Whisper family for speech-to-text.
    Whisper,
}

impl ModelArchitecture {
    /// Returns a human-readable display name for the architecture.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Llama => "Llama (Standard)",
            Self::Qwen2 => "Qwen2 (Biased)",
            Self::Mistral => "Mistral (SWA)",
            Self::Phi3 => "Phi-3 (LongRoPE)",
            Self::Bert => "BERT",
            Self::Mpnet => "Mpnet",
            Self::NomicBert => "Nomic-BERT",
            Self::T5 => "T5",
            Self::Bart => "BART",
            Self::GPT => "GPT",
            Self::Whisper => "Whisper (ASR)",
        }
    }

    /// Returns architecture.
    pub fn category(&self) -> &'static str {
        match self {
            // Decoders (LLMs)
            Self::Llama | Self::Qwen2 | Self::Mistral | Self::Phi3 | Self::GPT => "decoder",

            // Encoders (Embeddings/Classifiers)
            Self::Bert | Self::NomicBert | Self::Mpnet => "encoder",

            // Seq2Seq
            Self::T5 | Self::Bart | Self::Whisper => "encoder-decoder",
        }
    }
}

/// Defines the primary intended use case for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelTask {
    /// Vector embedding
    Embedding,

    /// Reranking
    ReRanking,

    /// Text -> Class label
    Classification,

    /// Interactive chat and instruction following
    Chat,

    /// Deep reasoning and logical inference.
    Reasoning,

    /// Detecting positive/negative sentiment in text
    SentimentAnalysis,

    /// Classifying text into arbitrary user-defined labels
    ZeroShotClassification,

    /// Translation, summarization, and text-to-text generation
    Seq2Seq,

    /// General text generation
    Generation,

    /// Long text -> Concise summary.
    Summarization,

    /// Text in language A -> Text in language B.
    Translation,

    /// Speech-to-text transcription.
    SpeechToText,

    /// General text-to-text transformation.
    TextToText,
}

/// The curated list of pretrained models supported by Kjarni.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, EnumIter)]
pub enum ModelType {
    MiniLML6V2,
    NomicEmbedText,
    BgeM3,
    MiniLML6V2CrossEncoder,
    MpnetBaseV2,
    DistilBertBaseCased,

    DistilBertSST2,
    TwitterRobertaSentiment,
    BertMultilingualSentiment,
    ToxicBertMultilingual,
    RobertaGoEmotions,
    DistilRobertaEmotion,

    Qwen2_5_0_5B_Instruct,
    Qwen2_5_1_5B_Instruct,
    Llama3_2_1B_Instruct,
    Llama3_2_3B_Instruct,
    Phi3_5_Mini_Instruct,
    Mistral7B_v0_3_Instruct,
    Llama3_1_8B_Instruct,
    DeepSeek_R1_Distill_Llama_8B,
    FlanT5Base,
    FlanT5Large,
    DistilBartCnn,
    BartLargeCnn,
    WhisperSmall,
    WhisperLargeV3,
    DistilGpt2,
    Gpt2,
}

/// Download URLs for all required model files.
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
    pub gguf_url: Option<&'static str>,
}

/// Complete metadata for a pretrained model.
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

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
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
            Self::MpnetBaseV2 => "mpnet-base-v2",
            Self::DistilBertBaseCased => "distilbert-base",

            // Reranker
            Self::MiniLML6V2CrossEncoder => "minilm-l6-v2-cross-encoder",

            // Classifiers

            // Sentiment
            Self::DistilBertSST2 => "distilbert-sentiment",
            Self::TwitterRobertaSentiment => "roberta-sentiment",
            Self::BertMultilingualSentiment => "bert-sentiment-multilingual",

            // Emotion
            Self::RobertaGoEmotions => "roberta-emotions",
            Self::DistilRobertaEmotion => "distilroberta-emotion",

            // Toxicity
            Self::ToxicBertMultilingual => "toxic-bert",

            // Edge LLMs
            Self::Qwen2_5_0_5B_Instruct => "qwen2.5-0.5b-instruct",
            Self::Qwen2_5_1_5B_Instruct => "qwen2.5-1.5b",
            Self::Llama3_2_1B_Instruct => "llama3.2-1b-instruct",
            Self::Llama3_2_3B_Instruct => "llama3.2-3b-instruct",
            Self::Phi3_5_Mini_Instruct => "phi3.5-mini",

            // Workhorse LLMs
            Self::Mistral7B_v0_3_Instruct => "mistral-7b",
            Self::Llama3_1_8B_Instruct => "llama3.1-8b-instruct",
            Self::DeepSeek_R1_Distill_Llama_8B => "deepseek-r1-8b",

            // Seq2Seq
            Self::FlanT5Base => "flan-t5-base",
            Self::FlanT5Large => "flan-t5-large",
            Self::DistilBartCnn => "distilbart-cnn",
            Self::BartLargeCnn => "bart-large-cnn",
            Self::WhisperSmall => "whisper-small",
            Self::WhisperLargeV3 => "whisper-large-v3",

            // Legacy
            Self::DistilGpt2 => "distilgpt2",
            Self::Gpt2 => "gpt2",
        }
    }

    pub fn display_group(&self) -> &'static str {
        match self.info().task {
            // Generation
            ModelTask::Chat | ModelTask::Reasoning => "LLM (Decoder)",

            //Translation/Summary
            ModelTask::Seq2Seq
            | ModelTask::Summarization
            | ModelTask::Translation
            | ModelTask::TextToText
            | ModelTask::SpeechToText => "Seq2Seq",

            // Vector embedding
            ModelTask::Embedding => "Embedding",

            //  Reranking
            ModelTask::ReRanking => "Re-Ranker",

            // Classification
            ModelTask::SentimentAnalysis
            | ModelTask::ZeroShotClassification
            | ModelTask::Classification => "Classifier",

            ModelTask::Generation => "Generation (Decoder)",
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
                architecture: ModelArchitecture::Mpnet,
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
            Self::DistilBertSST2 => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::SentimentAnalysis,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/onnx/tokenizer.json",
                    config_url: "https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "Fast binary sentiment (positive/negative). Best for simple yes/no sentiment.",
                size_mb: 268,
                params_millions: 66,
            },
            Self::TwitterRobertaSentiment => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::SentimentAnalysis,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/olafuraron/twitter-roberta-base-sentiment-latest-safetensors/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/olafuraron/twitter-roberta-base-sentiment-latest-safetensors/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/olafuraron/twitter-roberta-base-sentiment-latest-safetensors/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "3-class sentiment (negative/neutral/positive). Optimized for social media text.",
                size_mb: 499,
                params_millions: 125,
            },
            Self::BertMultilingualSentiment => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::SentimentAnalysis,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/olafuraron/bert-base-multilingual-uncased-sentiment-safetensors/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/olafuraron/bert-base-multilingual-uncased-sentiment-safetensors/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/olafuraron/bert-base-multilingual-uncased-sentiment-safetensors/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "5-star sentiment (1-5). Multilingual: EN, DE, FR, ES, IT, NL.",
                size_mb: 681,
                params_millions: 168,
            },
            Self::RobertaGoEmotions => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Classification,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/SamLowe/roberta-base-go_emotions/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/SamLowe/roberta-base-go_emotions/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/SamLowe/roberta-base-go_emotions/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "28 emotion labels (multi-label). Detects nuanced emotions like admiration, amusement, anger, etc.",
                size_mb: 499,
                params_millions: 125,
            },
            Self::DistilRobertaEmotion => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Classification,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/olafuraron/emotion-english-distilroberta-base-safetensors/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/olafuraron/emotion-english-distilroberta-base-safetensors/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/olafuraron/emotion-english-distilroberta-base-safetensors/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise.",
                size_mb: 329,
                params_millions: 82,
            },
            Self::ToxicBertMultilingual => ModelInfo {
                architecture: ModelArchitecture::Bert,
                task: ModelTask::Classification,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/olafuraron/toxic-bert-safetensors/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/olafuraron/toxic-bert-safetensors/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/olafuraron/toxic-bert-safetensors/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "Toxic comment classifier. Detects: toxic, severe_toxic, obscene, threat, insult, identity_hate.",
                size_mb: 438,
                params_millions: 110,
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
                description: "Cross-encoder for passage reranking. Use for search result reordering, NOT sentiment.",
                size_mb: 90,
                params_millions: 22,
            },
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
                description: "Balanced edge model. Good reasoning in a small package.",
                size_mb: 3100,
                params_millions: 1540,
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
                architecture: ModelArchitecture::Llama,
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

            Self::WhisperSmall => ModelInfo {
                architecture: ModelArchitecture::Whisper,
                task: ModelTask::SpeechToText,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/openai/whisper-small/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/openai/whisper-small/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "OpenAI Whisper small for speech-to-text transcription.",
                size_mb: 1500,
                params_millions: 244,
            },

            Self::WhisperLargeV3 => ModelInfo {
                architecture: ModelArchitecture::Whisper,
                task: ModelTask::SpeechToText,
                paths: ModelPaths {
                    weights_url: "https://huggingface.co/openai/whisper-large-v3/resolve/main/model.safetensors",
                    tokenizer_url: "https://huggingface.co/openai/whisper-large-v3/resolve/main/tokenizer.json",
                    config_url: "https://huggingface.co/openai/whisper-large-v3/resolve/main/config.json",
                    gguf_url: None,
                },
                description: "OpenAI Whisper large v3 for high-accuracy speech-to-text transcription.",
                size_mb: 7700,
                params_millions: 1550,
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

    pub fn resolve(name: &str) -> Result<ModelType, String> {
        if let Some(m) = Self::from_cli_name(name) {
            return Ok(m);
        }

        // Try substring match first
        let all_names: Vec<&str> = ModelType::all().map(|m| m.cli_name()).collect();
        let substring_matches: Vec<&str> = all_names
            .iter()
            .filter(|n| n.contains(&name.to_lowercase()))
            .copied()
            .collect();

        if !substring_matches.is_empty() {
            return Err(format!(
                "Unknown model '{name}'. Did you mean: {}?",
                substring_matches.join(", ")
            ));
        }

        // Fall back to Levenshtein
        let suggestions = Self::find_similar(name);
        if suggestions.is_empty() {
            Err(format!("Unknown model '{name}'"))
        } else {
            let names: Vec<&str> = suggestions.iter().map(|(n, _)| n.as_str()).collect();
            Err(format!(
                "Unknown model '{name}'. Did you mean: {}?",
                names.join(", ")
            ))
        }
    }

    pub fn from_cli_name(name: &str) -> Option<ModelType> {
        use strum::IntoEnumIterator;
        let normalized = name.to_lowercase();

        // Try CLI names first
        if let Some(m) = ModelType::iter().find(|m| m.cli_name() == normalized) {
            return Some(m);
        }

        // Try HuggingFace aliases
        match normalized.as_str() {
            "all-minilm-l6-v2" | "sentence-transformers/all-minilm-l6-v2" => Some(Self::MiniLML6V2),
            "all-mpnet-base-v2" | "sentence-transformers/all-mpnet-base-v2" => {
                Some(Self::MpnetBaseV2)
            }
            "ms-marco-minilm-l-6-v2" | "cross-encoder/ms-marco-minilm-l-6-v2" => {
                Some(Self::MiniLML6V2CrossEncoder)
            }
            "nomic-embed-text-v1.5" | "nomic-ai/nomic-embed-text-v1.5" => {
                Some(Self::NomicEmbedText)
            }
            "bge-m3" | "baai/bge-m3" => Some(Self::BgeM3),
            "distilbert-base-uncased-finetuned-sst-2-english" => Some(Self::DistilBertSST2),
            "twitter-roberta-base-sentiment-latest" => Some(Self::TwitterRobertaSentiment),
            "bert-base-multilingual-uncased-sentiment"
            | "bert-base-multilingual-uncased-sentiment-safetensors" => {
                Some(Self::BertMultilingualSentiment)
            }
            "toxic-bert" | "toxic-bert-safetensors" | "unitary/toxic-bert" => Some(Self::ToxicBertMultilingual),
            "roberta-base-go_emotions" | "samlowe/roberta-base-go_emotions" => {
                Some(Self::RobertaGoEmotions)
            }
            "emotion-english-distilroberta-base" => Some(Self::DistilRobertaEmotion),

            "distilbart-cnn" | "olafuraron/distilbart-cnn-12-6" | "distilbart-cnn-12-6" => Some(Self::DistilBartCnn),
            "bart-large-cnn" | "facebook/bart-large-cnn" => Some(Self::BartLargeCnn),
            "whisper-small" | "openai/whisper-small" => Some(Self::WhisperSmall),
            "whisper-large-v3" | "openai/whisper-large-v3" => Some(Self::WhisperLargeV3),
            "distilgpt2" | "distilgpt2/resolve/main/model.safetensors" => Some(Self::DistilGpt2),
            "gpt2" | "gpt2/resolve/main/model.safetensors" => Some(Self::Gpt2),
            
            _ => None,
        }
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

// Download Utilities

/// Downloads all required files for a model to the specified directory.
pub async fn download_model_files(
    model_dir: &Path,
    paths: &ModelPaths,
    format: WeightsFormat,
    quiet: bool,
) -> Result<PathBuf> {
    tokio::fs::create_dir_all(model_dir).await?;

    download_file(model_dir, "tokenizer.json", paths.tokenizer_url, quiet).await?;
    download_file(model_dir, "config.json", paths.config_url, quiet).await?;

    let use_gguf = matches!(format, WeightsFormat::GGUF) && paths.gguf_url.is_some();

    if use_gguf {
        let url = paths.gguf_url.unwrap();
        download_file(model_dir, "model.gguf", url, quiet).await?;
        Ok(model_dir.join("model.gguf"))
    } else {
        if matches!(format, WeightsFormat::GGUF) {
            eprintln!("  GGUF not available, falling back to SafeTensors.");
        }

        if paths.weights_url.ends_with(".index.json") {
            download_sharded_weights(model_dir, paths.weights_url, quiet).await?;
            Ok(model_dir.join("model.safetensors.index.json"))
        } else {
            download_file(model_dir, "model.safetensors", paths.weights_url, quiet).await?;
            Ok(model_dir.join("model.safetensors"))
        }
    }
}

async fn download_file(model_dir: &Path, filename: &str, url: &str, quiet: bool) -> Result<()> {
    let local_path = model_dir.join(filename);
    if local_path.exists() {
        return Ok(());
    }

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

async fn download_sharded_weights(model_dir: &Path, index_url: &str, quiet: bool) -> Result<()> {
    download_file(model_dir, "model.safetensors.index.json", index_url, quiet).await?;

    // Parse index
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

    // Download shards
    let base_url = index_url.rsplit_once('/').unwrap().0;

    for (i, shard) in shards.iter().enumerate() {
        let url = format!("{}/{}", base_url, shard);
        if !quiet {
            eprintln!("  Processing shard {}/{}...", i + 1, shards.len());
        }
        download_file(model_dir, shard, &url, quiet).await?;
    }

    Ok(())
}

/// Returns the default cache directory for Kjarni models.
pub fn get_default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("KJARNI_CACHE_DIR") {
        PathBuf::from(dir)
    } else {
        dirs::cache_dir()
            .expect("No cache directory found on system")
            .join("kjarni")
    }
}

/// Formats parameter count in human-readable form.
pub fn format_params(millions: usize) -> String {
    if millions >= 1000 {
        format!("{:.1}B", millions as f64 / 1000.0)
    } else {
        format!("{}M", millions)
    }
}

/// Formats file size in human-readable form.
pub fn format_size(mb: usize) -> String {
    if mb >= 1000 {
        format!("{:.1} GB", mb as f64 / 1000.0)
    } else {
        format!("{} MB", mb)
    }
}
