use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Copy, Serialize)]
#[allow(non_snake_case)] // To allow serde to match the camelCase keys
pub struct SummarizationParams {
    pub early_stopping: bool,
    pub length_penalty: f32,
    pub max_length: usize,
    pub min_length: usize,
    pub no_repeat_ngram_size: usize,
    pub num_beams: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(non_snake_case)]
pub struct TaskSpecificParams {
    pub summarization: SummarizationParams,
}

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