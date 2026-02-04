//! Generation config resolution for seq2seq models.
//!
//! Simple approach: start with model defaults, apply only what the user explicitly set.

use kjarni_transformers::common::{BeamSearchParams, DecodingStrategy, GenerationConfig};

use super::types::Seq2SeqOverrides;

/// Apply user overrides to a generation config.
///
/// Only modifies values that the user explicitly set (Some(...)). 
/// Model defaults are preserved for everything else.
pub fn apply_overrides(config: &mut GenerationConfig, overrides: &Seq2SeqOverrides) {
    // Length control
    if let Some(v) = overrides.min_length {
        config.min_length = v;
    }
    if let Some(v) = overrides.max_length {
        config.max_length = v;
    }

    // Repetition control
    if let Some(v) = overrides.no_repeat_ngram_size {
        config.no_repeat_ngram_size = v;
    }
    if let Some(v) = overrides.repetition_penalty {
        config.repetition_penalty = v;
    }

    // Handle beam search / greedy
    if let Some(num_beams) = overrides.num_beams {
        if num_beams <= 1 {
            config.strategy = DecodingStrategy::Greedy;
        } else {
            // Get existing beam params or use defaults
            let (length_penalty, early_stopping) = match &config.strategy {
                DecodingStrategy::BeamSearch(params) => {
                    (params.length_penalty, params.early_stopping)
                }
                _ => (1.0, true),
            };

            config.strategy = DecodingStrategy::BeamSearch(BeamSearchParams {
                num_beams,
                length_penalty: overrides.length_penalty.unwrap_or(length_penalty),
                early_stopping: overrides.early_stopping.unwrap_or(early_stopping),
            });
        }
    } else {
        // No num_beams override, but maybe other beam params
        if overrides.length_penalty.is_some() || overrides.early_stopping.is_some() {
            if let DecodingStrategy::BeamSearch(ref mut params) = config.strategy {
                if let Some(v) = overrides.length_penalty {
                    params.length_penalty = v;
                }
                if let Some(v) = overrides.early_stopping {
                    params.early_stopping = v;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_default_config() -> GenerationConfig {
        GenerationConfig {
            max_length: 128,
            min_length: 10,
            no_repeat_ngram_size: 3,
            repetition_penalty: 1.0,
            max_new_tokens: None,
            add_bos_token: false,
            strategy: DecodingStrategy::BeamSearch(BeamSearchParams {
                num_beams: 4,
                length_penalty: 1.0,
                early_stopping: true,
            }),
            speculation: None,
        }
    }

    #[test]
    fn test_no_overrides_preserves_defaults() {
        let mut config = make_default_config();
        apply_overrides(&mut config, &Seq2SeqOverrides::default());

        assert_eq!(config.max_length, 128);
        assert_eq!(config.min_length, 10);
        assert_eq!(config.no_repeat_ngram_size, 3);
    }

    #[test]
    fn test_only_overrides_what_user_sets() {
        let mut config = make_default_config();
        let overrides = Seq2SeqOverrides {
            max_length: Some(256),
            ..Default::default()
        };
        apply_overrides(&mut config, &overrides);

        assert_eq!(config.max_length, 256); // Changed
        assert_eq!(config.min_length, 10);  // Preserved
        assert_eq!(config.no_repeat_ngram_size, 3); // Preserved
    }

    #[test]
    fn test_greedy_decoding() {
        let mut config = make_default_config();
        apply_overrides(&mut config, &Seq2SeqOverrides::greedy());

        assert!(matches!(config.strategy, DecodingStrategy::Greedy));
        // Other values preserved
        assert_eq!(config.no_repeat_ngram_size, 3);
    }
}