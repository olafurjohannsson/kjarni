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
            position_encoding: PositionEncodingType::None,
            normalize_embeddings: false,
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


#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ========================================================================
    //  TaskParams Parsing Tests
    // ========================================================================

    #[test]
    fn test_parse_task_specific_params() {
        let json = json!({
            "summarization": {
                "early_stopping": true,
                "max_length": 128,
                "prefix": "summarize: "
            },
            "translation_en_to_de": {
                "early_stopping": false,
                "max_length": 200,
                "prefix": "translate English to German: "
            }
        });

        let params: TaskSpecificParams = serde_json::from_value(json).unwrap();

        // 1. Check Summarization
        let sum = params.summarization().unwrap();
        assert!(sum.early_stopping);
        assert_eq!(sum.max_length, 128);
        assert_eq!(sum.prefix.as_deref(), Some("summarize: "));
        // Defaults
        assert_eq!(sum.length_penalty, 2.0); // default_length_penalty
        assert_eq!(sum.num_beams, 4);        // default_num_beams

        // 2. Check Translation
        let trans = params.translation_en_to_de.as_ref().unwrap();
        assert!(!trans.early_stopping);
        assert_eq!(trans.prefix, "translate English to German: ");
    }

    #[test]
    fn test_parse_defaults() {
        let json = json!({
            "summarization": {
                "early_stopping": true,
                "max_length": 10
            }
        });
        let params: TaskSpecificParams = serde_json::from_value(json).unwrap();
        let sum = params.summarization().unwrap();

        assert_eq!(sum.min_length, 0); // Default usize
        assert_eq!(sum.no_repeat_ngram_size, 3); // default fn
        assert!(sum.prefix.is_none()); // Option default
    }

    // ========================================================================
    //  Architecture Config Tests
    // ========================================================================

    #[test]
    fn test_encoder_configs() {
        // BART
        let bart = Seq2SeqEncoderConfig::bart();
        assert!(matches!(bart.position_encoding, PositionEncodingType::Learned { offset: 2 }));
        assert!(bart.normalize_embeddings);
        assert!(!bart.final_layer_norm);

        // T5
        let t5 = Seq2SeqEncoderConfig::t5();
        assert!(matches!(t5.position_encoding, PositionEncodingType::RelativeBias { num_buckets: 32, .. }));
        assert!(t5.final_layer_norm);

        // Whisper
        let whisper = Seq2SeqEncoderConfig::whisper();
        assert!(matches!(whisper.position_encoding, PositionEncodingType::Sinusoidal));
    }

    #[test]
    fn test_decoder_configs() {
        // BART
        let bart = Seq2SeqDecoderConfig::bart();
        assert!(matches!(bart.position_encoding, PositionEncodingType::Learned { offset: 2 }));
        assert!(!bart.pre_norm);

        // T5
        let t5 = Seq2SeqDecoderConfig::t5();
        assert!(t5.pre_norm);
        assert!(t5.final_layer_norm);

        // Whisper
        let whisper = Seq2SeqDecoderConfig::whisper();
        assert!(whisper.pre_norm);
        assert!(matches!(whisper.position_encoding, PositionEncodingType::Learned { offset: 0 }));
    }
}