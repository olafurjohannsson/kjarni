//! Generation configuration resolution.
//!
//! Merges model defaults, user overrides, and runtime overrides
//! into a fully resolved configuration.

use super::overrides::GenerationOverrides;
use super::resolved::ResolvedGenerationConfig;
use kjarni_transformers::common::{
    BeamSearchParams, DecodingStrategy, GenerationConfig, SamplingParams,
};

/// Merge model defaults, user overrides, and runtime overrides into a
/// fully resolved generation configuration.
///
/// # Precedence (highest to lowest)
///
/// 1. Runtime overrides
/// 2. User overrides
/// 3. Model defaults
///
/// # Strategy Resolution
///
/// The decoding strategy is resolved based on explicit flags:
/// - `num_beams > 1` → BeamSearch
/// - `do_sample = false` → Greedy
/// - `do_sample = true` → Sample
/// - Otherwise → Keep model default
///
/// # Example
///
/// ```ignore
/// let model_defaults = model.get_default_generation_config();
/// let user = GenerationOverrides { temperature: Some(0.8), ..Default::default() };
/// let runtime = GenerationOverrides::default();
///
/// let resolved = resolve_generation_config(model_defaults, &user, &runtime);
/// ```
pub fn resolve_generation_config(
    model_defaults: GenerationConfig,
    user: &GenerationOverrides,
    runtime: &GenerationOverrides,
) -> ResolvedGenerationConfig {
    let mut config = model_defaults;

    // =========================================================================
    // Step 1: Resolve decoding strategy
    // =========================================================================

    let force_beams = runtime
        .num_beams
        .or(user.num_beams)
        .map(|b| b > 1)
        .unwrap_or(false);

    let force_greedy = runtime.do_sample.or(user.do_sample) == Some(false);
    let force_sampling = runtime.do_sample.or(user.do_sample) == Some(true);

    config.strategy = if force_beams {
        // Get beam search params, defaulting if not already beam search
        let base_params = match &config.strategy {
            DecodingStrategy::BeamSearch(p) => p.clone(),
            _ => BeamSearchParams::default(),
        };
        DecodingStrategy::BeamSearch(base_params)
    } else if force_greedy {
        DecodingStrategy::Greedy
    } else if force_sampling {
        // Get sample params, defaulting if not already sampling
        let base_params = match &config.strategy {
            DecodingStrategy::Sample(p) => p.clone(),
            _ => SamplingParams::default(),
        };
        DecodingStrategy::Sample(base_params)
    } else {
        config.strategy
    };

    // =========================================================================
    // Step 2: Apply common scalar overrides
    // =========================================================================

    // max_new_tokens is Option<usize>
    if let Some(v) = runtime.max_new_tokens.or(user.max_new_tokens) {
        config.max_new_tokens = Some(v);
    }

    // repetition_penalty is f32 (not Option)
    if let Some(v) = runtime.repetition_penalty {
        config.repetition_penalty = v;
    } else if let Some(v) = user.repetition_penalty {
        config.repetition_penalty = v;
    }

    // no_repeat_ngram_size is usize (not Option)
    if let Some(v) = runtime.no_repeat_ngram_size {
        config.no_repeat_ngram_size = v;
    } else if let Some(v) = user.no_repeat_ngram_size {
        config.no_repeat_ngram_size = v;
    }

    // =========================================================================
    // Step 3: Apply strategy-specific overrides
    // =========================================================================

    match &mut config.strategy {
        DecodingStrategy::Sample(params) => {
            if let Some(v) = runtime.temperature.or(user.temperature) {
                params.temperature = v;
            }
            if let Some(v) = runtime.top_k.or(user.top_k) {
                params.top_k = Some(v);
            }
            if let Some(v) = runtime.top_p.or(user.top_p) {
                params.top_p = Some(v);
            }
            if let Some(v) = runtime.min_p.or(user.min_p) {
                params.min_p = Some(v);
            }
        }

        DecodingStrategy::BeamSearch(params) => {
            if let Some(v) = runtime.num_beams.or(user.num_beams) {
                params.num_beams = v;
            }
            if let Some(v) = runtime.length_penalty.or(user.length_penalty) {
                params.length_penalty = v;
            }
        }

        DecodingStrategy::Greedy => {
            // No parameters to set
        }
    }

    ResolvedGenerationConfig { inner: config }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_default_config() -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: Some(256),
            max_length: 2048,
            min_length: 0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
            add_bos_token: true,
            strategy: DecodingStrategy::Sample(SamplingParams {
                temperature: 0.7,
                top_k: Some(50),
                top_p: Some(0.9),
                min_p: None,
            }),
            speculation: None,
        }
    }

    #[test]
    fn test_no_overrides() {
        let defaults = make_default_config();
        let user = GenerationOverrides::default();
        let runtime = GenerationOverrides::default();

        let resolved = resolve_generation_config(defaults.clone(), &user, &runtime);

        assert_eq!(resolved.inner.max_new_tokens, defaults.max_new_tokens);
    }

    #[test]
    fn test_user_override() {
        let defaults = make_default_config();
        let user = GenerationOverrides {
            temperature: Some(0.5),
            max_new_tokens: Some(512),
            ..Default::default()
        };
        let runtime = GenerationOverrides::default();

        let resolved = resolve_generation_config(defaults, &user, &runtime);

        assert_eq!(resolved.inner.max_new_tokens, Some(512));
        if let DecodingStrategy::Sample(p) = &resolved.inner.strategy {
            assert_eq!(p.temperature, 0.5);
        } else {
            panic!("Expected Sample strategy");
        }
    }

    #[test]
    fn test_runtime_overrides_user() {
        let defaults = make_default_config();
        let user = GenerationOverrides {
            temperature: Some(0.5),
            ..Default::default()
        };
        let runtime = GenerationOverrides {
            temperature: Some(0.9),
            ..Default::default()
        };

        let resolved = resolve_generation_config(defaults, &user, &runtime);

        if let DecodingStrategy::Sample(p) = &resolved.inner.strategy {
            assert_eq!(p.temperature, 0.9); // Runtime wins
        } else {
            panic!("Expected Sample strategy");
        }
    }

    #[test]
    fn test_force_greedy() {
        let defaults = make_default_config();
        let user = GenerationOverrides {
            do_sample: Some(false),
            ..Default::default()
        };
        let runtime = GenerationOverrides::default();

        let resolved = resolve_generation_config(defaults, &user, &runtime);

        assert!(matches!(resolved.inner.strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_force_beam_search() {
        let defaults = make_default_config();
        let user = GenerationOverrides {
            num_beams: Some(4),
            ..Default::default()
        };
        let runtime = GenerationOverrides::default();

        let resolved = resolve_generation_config(defaults, &user, &runtime);

        if let DecodingStrategy::BeamSearch(p) = &resolved.inner.strategy {
            assert_eq!(p.num_beams, 4);
        } else {
            panic!("Expected BeamSearch strategy");
        }
    }
}
