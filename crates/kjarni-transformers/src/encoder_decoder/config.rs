use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(non_snake_case)] // To allow serde to match the camelCase keys
pub struct SummarizationParams {
    pub early_stopping: bool,
    #[serde(default = "default_length_penalty")]
    pub length_penalty: f32,
    pub max_length: usize,
    #[serde(default)]
    pub min_length: usize,
    #[serde(default = "default_no_repeat_ngram")]
    pub no_repeat_ngram_size: usize,
    #[serde(default = "default_num_beams")]
    pub num_beams: usize,
    #[serde(default)]
    pub prefix: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(non_snake_case)]
pub struct TranslationParams {
    pub early_stopping: bool,
    pub max_length: usize,
    #[serde(default = "default_num_beams")]
    pub num_beams: usize,
    pub prefix: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(non_snake_case)]
pub struct TaskSpecificParams {
    pub summarization: Option<SummarizationParams>,
    #[serde(alias = "translation_en_to_de")]
    pub translation_en_to_de: Option<TranslationParams>,
    #[serde(alias = "translation_en_to_fr")]
    pub translation_en_to_fr: Option<TranslationParams>,
    #[serde(alias = "translation_en_to_ro")]
    pub translation_en_to_ro: Option<TranslationParams>,
}

impl TaskSpecificParams {
    /// Get summarization params if available
    pub fn summarization(&self) -> Option<&SummarizationParams> {
        self.summarization.as_ref()
    }
}

fn default_length_penalty() -> f32 { 2.0 }
fn default_no_repeat_ngram() -> usize { 3 }
fn default_num_beams() -> usize { 4 }

/// Position encoding strategy for seq2seq models.
#[derive(Debug, Clone)]
pub enum PositionEncodingType {
    /// Learned absolute positions (BART, GPT)
    Learned { offset: usize },
    /// Relative position bias in attention (T5)
    RelativeBias { num_buckets: usize, max_distance: usize },
    /// Sinusoidal positions (Whisper, original Transformer)
    Sinusoidal,
    /// No position encoding (handled elsewhere, e.g., RoPE)
    None,
}

/// Extended configuration for Seq2Seq encoder.
///
/// Supplements `ModelConfig` with seq2seq-specific settings.
#[derive(Debug, Clone)]
pub struct Seq2SeqEncoderConfig {
    /// How positions are encoded
    pub position_encoding: PositionEncodingType,
    /// Apply layer norm after embedding lookup (BART, Whisper)
    pub normalize_embeddings: bool, // TODO this possibly comes from config.json, look at T5, WHisper, BART
    /// Apply final layer norm after all layers (T5, Whisper)
    pub final_layer_norm: bool,
}

impl Seq2SeqEncoderConfig {
    /// BART-style configuration
    pub fn bart() -> Self {
        Self {
            position_encoding: PositionEncodingType::Learned { offset: 2 },
            normalize_embeddings: true,
            final_layer_norm: false,
        }
    }

    /// T5-style configuration
    pub fn t5() -> Self {
        Self {
            position_encoding: PositionEncodingType::RelativeBias {
                num_buckets: 32,
                max_distance: 128,
            },
            normalize_embeddings: false,
            final_layer_norm: true,
        }
    }

    /// Whisper-style configuration
    pub fn whisper() -> Self {
        Self {
            position_encoding: PositionEncodingType::Sinusoidal,
            normalize_embeddings: true,
            final_layer_norm: true,
        }
    }
}


/// Extended configuration for Seq2Seq decoder.
///
/// Supplements `ModelConfig` with seq2seq decoder-specific settings.
#[derive(Debug, Clone)]
pub struct Seq2SeqDecoderConfig {
    /// How positions are encoded
    pub position_encoding: PositionEncodingType,
    /// Apply layer norm after embedding lookup
    pub normalize_embeddings: bool,
    /// Apply final layer norm after all layers
    pub final_layer_norm: bool,
    /// Use pre-norm (T5, Whisper) vs post-norm (BART)
    pub pre_norm: bool,
}

impl Seq2SeqDecoderConfig {
    /// BART-style configuration
    pub fn bart() -> Self {
        Self {
            position_encoding: PositionEncodingType::Learned { offset: 2 },
            normalize_embeddings: true,
            final_layer_norm: false,
            pre_norm: false,
        }
    }

    /// T5-style configuration
    pub fn t5() -> Self {
        Self {
            position_encoding: PositionEncodingType::RelativeBias {
                num_buckets: 32,
                max_distance: 128,
            },
            normalize_embeddings: false,
            final_layer_norm: true,
            pre_norm: true,
        }
    }

    /// Whisper-style configuration
    pub fn whisper() -> Self {
        Self {
            position_encoding: PositionEncodingType::Learned { offset: 0 },
            normalize_embeddings: true,
            final_layer_norm: true,
            pre_norm: true,
        }
    }
}