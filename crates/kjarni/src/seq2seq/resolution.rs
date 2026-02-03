//! Generation config resolution for seq2seq models.

use kjarni_transformers::common::{BeamSearchParams, DecodingStrategy, GenerationConfig};

use super::types::{Seq2SeqOverrides, Seq2SeqTask};

/// Resolved seq2seq generation config (all values concrete).
///
/// This is the final, fully-resolved configuration used for generation.
/// Created by merging model defaults with user overrides.
#[derive(Debug, Clone)]
pub struct ResolvedSeq2SeqConfig {
    pub(crate) inner: GenerationConfig,
}

impl ResolvedSeq2SeqConfig {
    /// Get a reference to the inner GenerationConfig.
    pub fn as_ref(&self) -> &GenerationConfig {
        &self.inner
    }

    /// Consume and return the inner GenerationConfig.
    pub fn into_inner(self) -> GenerationConfig {
        self.inner
    }

    // Accessors for common fields
    pub fn max_length(&self) -> usize {
        self.inner.max_length
    }

    pub fn min_length(&self) -> usize {
        self.inner.min_length
    }

    pub fn num_beams(&self) -> usize {
        match &self.inner.strategy {
            DecodingStrategy::BeamSearch(params) => params.num_beams,
            DecodingStrategy::Greedy => 1,
            _ => 1,
        }
    }

    pub fn is_beam_search(&self) -> bool {
        matches!(self.inner.strategy, DecodingStrategy::BeamSearch(_))
    }
}

/// Resolve seq2seq generation config from multiple sources.
///
/// Priority (highest to lowest):
/// 1. Runtime overrides (per-call)
/// 2. User overrides (from builder)
/// 3. Task defaults (translation vs summarization)
/// 4. Model defaults (from generation_config.json / task_specific_params)
///
/// # Arguments
///
/// * `model_defaults` - Base config from the model
/// * `task` - Optional task hint for task-specific defaults
/// * `user_overrides` - Overrides set at build time
/// * `runtime_overrides` - Overrides for this specific call
pub fn resolve_seq2seq_config(
    model_defaults: GenerationConfig,
    task: Option<Seq2SeqTask>,
    user_overrides: &Seq2SeqOverrides,
    runtime_overrides: &Seq2SeqOverrides,
) -> ResolvedSeq2SeqConfig {
    let mut config = model_defaults;

    // Apply task-specific defaults
    if let Some(task) = task {
        apply_task_defaults(&mut config, task);
    }

    // Apply user overrides (from builder)
    apply_overrides(&mut config, user_overrides);

    // Apply runtime overrides (per-call) - highest priority
    apply_overrides(&mut config, runtime_overrides);

    ResolvedSeq2SeqConfig { inner: config }
}

/// Apply task-specific defaults to the config.
fn apply_task_defaults(config: &mut GenerationConfig, task: Seq2SeqTask) {
    match task {
        Seq2SeqTask::Translation => {
            // Translation shouldn't force min_length or block n-grams
            // Only set if not already configured appropriately
            if config.min_length > 0 {
                config.min_length = 0;
            }
            if config.no_repeat_ngram_size > 0 {
                config.no_repeat_ngram_size = 0;
            }
        }
        Seq2SeqTask::Summarization => {
            // Summarization benefits from n-gram blocking
            if config.no_repeat_ngram_size == 0 {
                config.no_repeat_ngram_size = 3;
            }
        }
        Seq2SeqTask::General => {
            // Use model defaults as-is
        }
    }
}

/// Apply user overrides to the config.
fn apply_overrides(config: &mut GenerationConfig, overrides: &Seq2SeqOverrides) {
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

    // Handle beam search params
    let has_beam_overrides = overrides.num_beams.is_some()
        || overrides.length_penalty.is_some()
        || overrides.early_stopping.is_some();

    if has_beam_overrides {
        // Get current beam params or create defaults
        let (mut num_beams, mut length_penalty, mut early_stopping) = match &config.strategy {
            DecodingStrategy::BeamSearch(params) => {
                (params.num_beams, params.length_penalty, params.early_stopping)
            }
            _ => (4, 1.0, true), // defaults if not already beam search
        };

        // Apply overrides
        if let Some(v) = overrides.num_beams {
            num_beams = v;
        }
        if let Some(v) = overrides.length_penalty {
            length_penalty = v;
        }
        if let Some(v) = overrides.early_stopping {
            early_stopping = v;
        }

        // num_beams = 1 means greedy decoding
        if num_beams <= 1 {
            config.strategy = DecodingStrategy::Greedy;
        } else {
            config.strategy = DecodingStrategy::BeamSearch(BeamSearchParams {
                num_beams,
                length_penalty,
                early_stopping,
            });
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
    fn test_resolve_no_overrides() {
        let config = make_default_config();
        let resolved = resolve_seq2seq_config(
            config.clone(),
            None,
            &Seq2SeqOverrides::default(),
            &Seq2SeqOverrides::default(),
        );

        assert_eq!(resolved.max_length(), 128);
        assert_eq!(resolved.min_length(), 10);
        assert_eq!(resolved.num_beams(), 4);
    }

    #[test]
    fn test_resolve_user_overrides() {
        let config = make_default_config();
        let user = Seq2SeqOverrides {
            max_length: Some(256),
            num_beams: Some(6),
            ..Default::default()
        };

        let resolved = resolve_seq2seq_config(
            config,
            None,
            &user,
            &Seq2SeqOverrides::default(),
        );

        assert_eq!(resolved.max_length(), 256);
        assert_eq!(resolved.num_beams(), 6);
    }

    #[test]
    fn test_resolve_runtime_overrides_take_precedence() {
        let config = make_default_config();
        let user = Seq2SeqOverrides {
            max_length: Some(256),
            ..Default::default()
        };
        let runtime = Seq2SeqOverrides {
            max_length: Some(512),
            ..Default::default()
        };

        let resolved = resolve_seq2seq_config(config, None, &user, &runtime);

        assert_eq!(resolved.max_length(), 512); // runtime wins
    }

    #[test]
    fn test_resolve_greedy_decoding() {
        let config = make_default_config();
        let overrides = Seq2SeqOverrides::greedy();

        let resolved = resolve_seq2seq_config(
            config,
            None,
            &overrides,
            &Seq2SeqOverrides::default(),
        );

        assert!(!resolved.is_beam_search());
        assert_eq!(resolved.num_beams(), 1);
    }

    #[test]
    fn test_resolve_translation_task() {
        let mut config = make_default_config();
        config.min_length = 50;
        config.no_repeat_ngram_size = 3;

        let resolved = resolve_seq2seq_config(
            config,
            Some(Seq2SeqTask::Translation),
            &Seq2SeqOverrides::default(),
            &Seq2SeqOverrides::default(),
        );

        assert_eq!(resolved.min_length(), 0); // Translation clears min_length
        assert_eq!(resolved.inner.no_repeat_ngram_size, 0); // Translation clears n-gram blocking
    }

    #[test]
    fn test_resolve_summarization_task() {
        let mut config = make_default_config();
        config.no_repeat_ngram_size = 0;

        let resolved = resolve_seq2seq_config(
            config,
            Some(Seq2SeqTask::Summarization),
            &Seq2SeqOverrides::default(),
            &Seq2SeqOverrides::default(),
        );

        assert_eq!(resolved.inner.no_repeat_ngram_size, 3); // Summarization sets n-gram blocking
    }
}