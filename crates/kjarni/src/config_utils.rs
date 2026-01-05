use crate::config::GenerationDefaults;
use kjarni_transformers::common::{GenerationConfig, DecodingStrategy, SamplingParams, BeamSearchParams};

/// Merges Model Defaults (Layer 1) + User Config (Layer 2) + Runtime Overrides (Layer 3)
pub fn merge_generation_config(
    model_default: GenerationConfig,
    user_prefs: &GenerationDefaults,
    runtime_overrides: &GenerationDefaults,
) -> GenerationConfig {
    let mut final_config = model_default;

    // Helper: Runtime > User Prefs > Keep Existing
    macro_rules! apply {
        ($field:ident) => {
            if let Some(val) = runtime_overrides.$field {
                final_config.$field = val;
            } else if let Some(val) = user_prefs.$field {
                final_config.$field = val;
            }
        };
        ($field:ident, $target:ident) => {
             if let Some(val) = runtime_overrides.$field {
                final_config.$target = val;
            } else if let Some(val) = user_prefs.$field {
                final_config.$target = val;
            }
        };
        ($field:ident, $target:ident, Option) => {
             if let Some(val) = runtime_overrides.$field {
                final_config.$target = Some(val);
            } else if let Some(val) = user_prefs.$field {
                final_config.$target = Some(val);
            }
        };
    }

    // 1. Common Parameters
    apply!(max_new_tokens, max_new_tokens, Option);
    apply!(repetition_penalty);
    apply!(no_repeat_ngram_size);
    
    // 2. Strategy Switching logic
    // If user explicitly asks for greedy (do_sample = false) or beams, we might need to switch strategy entirely
    let force_greedy = runtime_overrides.do_sample == Some(false);
    let force_beams = runtime_overrides.num_beams.is_some() && runtime_overrides.num_beams.unwrap() > 1;

    if force_greedy {
        final_config.strategy = DecodingStrategy::Greedy;
    } else if force_beams {
        // Switch to Beam Search if not already
        if !matches!(final_config.strategy, DecodingStrategy::BeamSearch(_)) {
            final_config.strategy = DecodingStrategy::BeamSearch(BeamSearchParams::default());
        }
    }

    // 3. Strategy Specific Parameters
    match &mut final_config.strategy {
        DecodingStrategy::Sample(params) => {
            if let Some(t) = runtime_overrides.temperature.or(user_prefs.temperature) {
                params.temperature = t;
            }
            if let Some(k) = runtime_overrides.top_k.or(user_prefs.top_k) {
                params.top_k = Some(k);
            }
            if let Some(p) = runtime_overrides.top_p.or(user_prefs.top_p) {
                params.top_p = Some(p);
            }
            if let Some(mp) = runtime_overrides.min_p.or(user_prefs.min_p) {
                params.min_p = Some(mp);
            }
        }
        DecodingStrategy::BeamSearch(params) => {
            if let Some(b) = runtime_overrides.num_beams.or(user_prefs.num_beams) {
                params.num_beams = b;
            }
            if let Some(lp) = runtime_overrides.length_penalty.or(user_prefs.length_penalty) {
                params.length_penalty = lp;
            }
        }
        DecodingStrategy::Greedy => {
            // Greedy has no params
        }
    }

    final_config
}