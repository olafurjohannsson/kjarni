// kjarni/src/config/types.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Top-level Kjarni configuration.
/// 
/// Loaded from kjarni.toml, provides defaults for all operations.
/// CLI flags always override these settings.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct KjarniConfig {
    /// Default model names per task
    #[serde(default)]
    pub defaults: DefaultModels,

    /// Chat task configuration
    #[serde(default)]
    pub chat: ChatTaskConfig,

    /// Generate task configuration  
    #[serde(default)]
    pub generate: GenerateTaskConfig,

    /// Summarize task configuration
    #[serde(default)]
    pub summarize: SummarizeTaskConfig,

    /// Translate task configuration
    #[serde(default)]
    pub translate: TranslateTaskConfig,

    /// Classify task configuration
    #[serde(default)]
    pub classify: ClassifyTaskConfig,

    /// Embed task configuration
    #[serde(default)]
    pub embed: EmbedTaskConfig,

    /// Rerank task configuration
    #[serde(default)]
    pub rerank: RerankTaskConfig,

    /// Index task configuration
    #[serde(default)]
    pub index: IndexTaskConfig,

    /// Search task configuration
    #[serde(default)]
    pub search: SearchTaskConfig,

    /// Transcribe task configuration
    #[serde(default)]
    pub transcribe: TranscribeTaskConfig,

    /// Per-model overrides
    #[serde(default)]
    pub models: HashMap<String, ModelOverride>,

    /// Load/memory configuration
    #[serde(default)]
    pub load: LoadTaskConfig,

    /// Cache configuration
    #[serde(default)]
    pub cache: CacheConfig,

    /// Hardware configuration
    #[serde(default)]
    pub hardware: HardwareConfig,

    /// Output configuration
    #[serde(default)]
    pub output: OutputConfig,
}

// =============================================================================
// Default Models
// =============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DefaultModels {
    #[serde(default = "default_chat_model")]
    pub chat: String,
    #[serde(default = "default_generate_model")]
    pub generate: String,
    #[serde(default = "default_embed_model")]
    pub embed: String,
    #[serde(default = "default_classify_model")]
    pub classify: String,
    #[serde(default = "default_rerank_model")]
    pub rerank: String,
    #[serde(default = "default_summarize_model")]
    pub summarize: String,
    #[serde(default = "default_translate_model")]
    pub translate: String,
    #[serde(default = "default_transcribe_model")]
    pub transcribe: String,
}

impl Default for DefaultModels {
    fn default() -> Self {
        Self {
            chat: default_chat_model(),
            generate: default_generate_model(),
            embed: default_embed_model(),
            classify: default_classify_model(),
            rerank: default_rerank_model(),
            summarize: default_summarize_model(),
            translate: default_translate_model(),
            transcribe: default_transcribe_model(),
        }
    }
}

fn default_chat_model() -> String { "llama3.2-1b-instruct".into() }
fn default_generate_model() -> String { "llama3.2-1b-instruct".into() }
fn default_embed_model() -> String { "minilm-l6-v2".into() }
fn default_classify_model() -> String { "distilbert-sentiment".into() }
fn default_rerank_model() -> String { "minilm-reranker".into() }
fn default_summarize_model() -> String { "distilbart-cnn".into() }
fn default_translate_model() -> String { "flan-t5-base".into() }
fn default_transcribe_model() -> String { "whisper-tiny".into() }

// =============================================================================
// Task Configs - Generation (Chat, Generate)
// =============================================================================

/// Generation parameters shared between Chat and Generate.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerationParams {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: Some(0.05),
            repetition_penalty: default_repetition_penalty(),
        }
    }
}

fn default_temperature() -> f32 { 0.7 }
fn default_max_tokens() -> usize { 512 }
fn default_repetition_penalty() -> f32 { 1.1 }

/// Chat-specific configuration.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ChatTaskConfig {
    #[serde(flatten)]
    pub generation: GenerationParams,
    
    /// Default system prompt
    #[serde(default)]
    pub system_prompt: Option<String>,
    
    /// Mode-specific overrides
    #[serde(default)]
    pub modes: ChatModeOverrides,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ChatModeOverrides {
    #[serde(default)]
    pub default: Option<GenerationParams>,
    #[serde(default)]
    pub creative: Option<GenerationParams>,
    #[serde(default)]
    pub reasoning: Option<GenerationParams>,
}

/// Generate (raw) configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerateTaskConfig {
    #[serde(flatten)]
    pub generation: GenerationParams,
}

impl Default for GenerateTaskConfig {
    fn default() -> Self {
        Self {
            generation: GenerationParams {
                temperature: 0.5,  // Lower for raw generation
                max_tokens: 256,
                ..Default::default()
            }
        }
    }
}

// =============================================================================
// Task Configs - Seq2Seq (Summarize, Translate)
// =============================================================================

/// Seq2Seq generation parameters.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Seq2SeqParams {
    #[serde(default = "default_seq2seq_max_length")]
    pub max_length: usize,
    #[serde(default)]
    pub min_length: Option<usize>,
    #[serde(default = "default_num_beams")]
    pub num_beams: usize,
    #[serde(default = "default_length_penalty")]
    pub length_penalty: f32,
    #[serde(default)]
    pub no_repeat_ngram_size: usize,
    #[serde(default = "default_true")]
    pub early_stopping: bool,
}

impl Default for Seq2SeqParams {
    fn default() -> Self {
        Self {
            max_length: default_seq2seq_max_length(),
            min_length: None,
            num_beams: default_num_beams(),
            length_penalty: default_length_penalty(),
            no_repeat_ngram_size: 3,
            early_stopping: true,
        }
    }
}

fn default_seq2seq_max_length() -> usize { 200 }
fn default_num_beams() -> usize { 4 }
fn default_length_penalty() -> f32 { 2.0 }
fn default_true() -> bool { true }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SummarizeTaskConfig {
    #[serde(flatten)]
    pub seq2seq: Seq2SeqParams,
    
    #[serde(default = "default_summarize_min")]
    pub min_length: usize,
}

impl Default for SummarizeTaskConfig {
    fn default() -> Self {
        Self {
            seq2seq: Seq2SeqParams {
                max_length: 200,
                min_length: Some(50),
                length_penalty: 2.0,
                ..Default::default()
            },
            min_length: default_summarize_min(),
        }
    }
}

fn default_summarize_min() -> usize { 50 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranslateTaskConfig {
    #[serde(flatten)]
    pub seq2seq: Seq2SeqParams,
    
    /// Default source language
    #[serde(default)]
    pub src: Option<String>,
    
    /// Default target language
    #[serde(default)]
    pub dst: Option<String>,
}

impl Default for TranslateTaskConfig {
    fn default() -> Self {
        Self {
            seq2seq: Seq2SeqParams {
                max_length: 512,
                length_penalty: 1.0,
                no_repeat_ngram_size: 0,
                ..Default::default()
            },
            src: None,
            dst: None,
        }
    }
}

// =============================================================================
// Task Configs - Encoder (Classify, Embed, Rerank)
// =============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClassifyTaskConfig {
    #[serde(default = "default_classify_top_k")]
    pub top_k: usize,
    #[serde(default)]
    pub threshold: f32,
    #[serde(default)]
    pub multi_label: bool,
    #[serde(default = "default_max_length")]
    pub max_length: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

impl Default for ClassifyTaskConfig {
    fn default() -> Self {
        Self {
            top_k: default_classify_top_k(),
            threshold: 0.0,
            multi_label: false,
            max_length: default_max_length(),
            batch_size: default_batch_size(),
        }
    }
}

fn default_classify_top_k() -> usize { 5 }
fn default_max_length() -> usize { 512 }
fn default_batch_size() -> usize { 8 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbedTaskConfig {
    #[serde(default = "default_true")]
    pub normalize: bool,
    #[serde(default = "default_pooling")]
    pub pooling: String,
}

impl Default for EmbedTaskConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            pooling: default_pooling(),
        }
    }
}

fn default_pooling() -> String { "mean".into() }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RerankTaskConfig {
    #[serde(default = "default_rerank_top_k")]
    pub top_k: usize,
    #[serde(default = "default_true")]
    pub return_scores: bool,
}

impl Default for RerankTaskConfig {
    fn default() -> Self {
        Self {
            top_k: default_rerank_top_k(),
            return_scores: true,
        }
    }
}

fn default_rerank_top_k() -> usize { 10 }

// =============================================================================
// Task Configs - Index/Search
// =============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IndexTaskConfig {
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    #[serde(default = "default_max_docs")]
    pub max_docs_per_segment: usize,
}

impl Default for IndexTaskConfig {
    fn default() -> Self {
        Self {
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
            max_docs_per_segment: default_max_docs(),
        }
    }
}

fn default_chunk_size() -> usize { 512 }
fn default_chunk_overlap() -> usize { 50 }
fn default_max_docs() -> usize { 10000 }

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchTaskConfig {
    #[serde(default = "default_search_top_k")]
    pub top_k: usize,
    #[serde(default = "default_search_mode")]
    pub mode: String,
    #[serde(default = "default_hybrid_alpha")]
    pub hybrid_alpha: f32,
}

impl Default for SearchTaskConfig {
    fn default() -> Self {
        Self {
            top_k: default_search_top_k(),
            mode: default_search_mode(),
            hybrid_alpha: default_hybrid_alpha(),
        }
    }
}

fn default_search_top_k() -> usize { 10 }
fn default_search_mode() -> String { "semantic".into() }
fn default_hybrid_alpha() -> f32 { 0.5 }

// =============================================================================
// Task Configs - Transcribe
// =============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranscribeTaskConfig {
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub timestamps: bool,
}

impl Default for TranscribeTaskConfig {
    fn default() -> Self {
        Self {
            language: None,  // Auto-detect
            timestamps: false,
        }
    }
}

// =============================================================================
// Per-Model Overrides
// =============================================================================

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ModelOverride {
    /// Load configuration overrides
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub offload_embeddings: Option<bool>,
    #[serde(default)]
    pub offload_lm_head: Option<bool>,
    #[serde(default)]
    pub quantize_lm_head: Option<String>,
    
    /// Generation overrides (for decoder models)
    #[serde(default)]
    pub generation: Option<GenerationParams>,
    
    /// Encoding overrides (for encoder models)
    #[serde(default)]
    pub encoding: Option<EmbedTaskConfig>,
    
    /// Summarize overrides (for seq2seq models)
    #[serde(default)]
    pub summarize: Option<Seq2SeqParams>,
    
    /// Translate overrides (for seq2seq models)
    #[serde(default)]
    pub translate: Option<Seq2SeqParams>,
}

// =============================================================================
// Load Configuration
// =============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoadTaskConfig {
    #[serde(default = "default_dtype")]
    pub dtype: String,
    #[serde(default)]
    pub offload_embeddings: bool,
    #[serde(default)]
    pub offload_lm_head: bool,
    #[serde(default)]
    pub quantize_lm_head: Option<String>,
    #[serde(default)]
    pub prefer_gguf: bool,
    #[serde(default)]
    pub max_batch_size: Option<usize>,
    #[serde(default)]
    pub max_sequence_length: Option<usize>,
}

impl Default for LoadTaskConfig {
    fn default() -> Self {
        Self {
            dtype: default_dtype(),
            offload_embeddings: false,
            offload_lm_head: false,
            quantize_lm_head: None,
            prefer_gguf: false,
            max_batch_size: None,
            max_sequence_length: None,
        }
    }
}

fn default_dtype() -> String { "f32".into() }

// =============================================================================
// Cache Configuration
// =============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CacheConfig {
    #[serde(default)]
    pub dir: Option<PathBuf>,
    #[serde(default = "default_true")]
    pub auto_download: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            dir: None,  // Use system default
            auto_download: true,
        }
    }
}

// =============================================================================
// Hardware Configuration
// =============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HardwareConfig {
    #[serde(default = "default_device")]
    pub device: String,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            device: default_device(),
        }
    }
}

fn default_device() -> String { "auto".into() }

// =============================================================================
// Output Configuration
// =============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputConfig {
    #[serde(default = "default_format")]
    pub format: String,
    #[serde(default)]
    pub quiet: bool,
    #[serde(default = "default_true")]
    pub color: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: default_format(),
            quiet: false,
            color: true,
        }
    }
}

fn default_format() -> String { "text".into() }