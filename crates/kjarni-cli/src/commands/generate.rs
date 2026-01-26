//! Text generation command using decoder models

use anyhow::{anyhow, Result};
use futures::{StreamExt, pin_mut};

use kjarni::{
    models::{Gpt2Model, LlamaModel}, registry, DecoderGenerator, DecoderLanguageModel, DecodingStrategy,
    Device, GenerationConfig, ModelArchitecture, ModelType,
    SamplingParams,
    TokenType,
};
use std::io::{self, Write};
use std::sync::Arc;

use super::util::{model_not_found_error, resolve_input};

pub async fn run(
    prompt: Option<&str>,
    model: &str,
    model_path: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repetition_penalty: f32,
    greedy: bool,
    gpu: bool,
    no_stream: bool,
    quiet: bool,
) -> Result<()> {
    // 1. Resolve prompt
    let prompt_text = resolve_input(prompt)?;

    // 2. Resolve model
    let device = if gpu { Device::Wgpu } else { Device::Cpu };

    if model_path.is_some() {
        return Err(anyhow!("--model-path not yet implemented."));
    }

    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| anyhow!(model_not_found_error(model, Some("decoder"))))?;

    if !is_supported_decoder_architecture(model_type.architecture()) {
        return Err(anyhow!(
            "Model '{}' is not a decoder. Use a decoder model for generation. Detected architecture: {:?}",
            model, model_type.architecture()
        ));
    }

    // Check if downloaded
    if !registry::is_model_downloaded(model)? {
        if !quiet {
            eprintln!("Model '{}' not found locally. Downloading...", model);
        }
        registry::download_model(model, false, quiet).await?;
        if !quiet {
            eprintln!();
        }
    }

    // 3. Load model
    if !quiet {
        eprintln!("Loading model '{}'...", model);
    }

    let loaded_model: Arc<dyn DecoderLanguageModel> = if model_type.is_llama_model() {
        Arc::new(LlamaModel::from_registry(model_type, None, device, None, None).await?)
    } else if model_type.is_gpt2_model() {
        Arc::new(Gpt2Model::from_registry(model_type, None, device, None, None).await?)
    } else {
        return Err(anyhow!(
            "Model '{}' not yet supported for generation.",
            model
        ));
    };

    let generator = DecoderGenerator::new(loaded_model)?;

    // 4. Configure generation
    let config = build_generation_config(
        max_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        greedy,
    );

    if !quiet {
        eprintln!();
    }

    // 5. Generate
    if no_stream {
        let output = generator.generate(&prompt_text, &config, None).await?;
        println!("{}", output);
    } else {
        let stream = generator
            .generate_stream(&prompt_text, &config, None)
            .await?;
        pin_mut!(stream);

        let mut stdout = io::stdout();
        let mut generated_any = false;

        while let Some(token_result) = stream.next().await {
            let token = token_result?;

            // Skip prompt tokens
            if token.token_type == TokenType::Prompt {
                continue;
            }

            print!("{}", token.text);
            stdout.flush()?;
            generated_any = true;
        }

        if generated_any {
            println!();
        }
    }

    Ok(())
}

/// Check if the architecture is a supported decoder for generation
fn is_supported_decoder_architecture(arch: ModelArchitecture) -> bool {
    matches!(
        arch,
        ModelArchitecture::GPT
            | ModelArchitecture::Llama
            | ModelArchitecture::Mistral
            | ModelArchitecture::Qwen2
    )
}

/// Build the decoding strategy based on parameters
fn build_decoding_strategy(
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    greedy: bool,
) -> DecodingStrategy {
    if greedy || temperature == 0.0 {
        DecodingStrategy::Greedy
    } else {
        DecodingStrategy::Sample(SamplingParams {
            temperature,
            top_k: top_k.or(Some(50)),
            top_p: top_p.or(Some(0.9)),
            min_p: min_p.or(Some(0.1)),
        })
    }
}

/// Build the full generation config
fn build_generation_config(
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repetition_penalty: f32,
    greedy: bool,
) -> GenerationConfig {
    let strategy = build_decoding_strategy(temperature, top_k, top_p, min_p, greedy);

    GenerationConfig {
        max_new_tokens: Some(max_tokens),
        repetition_penalty,
        strategy,
        ..Default::default()
    }
}

/// Get default sampling params with overrides
fn get_sampling_params(
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k: top_k.or(Some(50)),
        top_p: top_p.or(Some(0.9)),
        min_p: min_p.or(Some(0.1)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // is_supported_decoder_architecture tests
    // =========================================================================

    #[test]
    fn test_supported_gpt() {
        assert!(is_supported_decoder_architecture(ModelArchitecture::GPT));
    }

    #[test]
    fn test_supported_llama() {
        assert!(is_supported_decoder_architecture(ModelArchitecture::Llama));
    }

    #[test]
    fn test_supported_mistral() {
        assert!(is_supported_decoder_architecture(ModelArchitecture::Mistral));
    }

    #[test]
    fn test_supported_qwen2() {
        assert!(is_supported_decoder_architecture(ModelArchitecture::Qwen2));
    }

    #[test]
    fn test_unsupported_bert() {
        assert!(!is_supported_decoder_architecture(ModelArchitecture::Bert));
    }

    #[test]
    fn test_unsupported_t5() {
        assert!(!is_supported_decoder_architecture(ModelArchitecture::T5));
    }

    #[test]
    fn test_unsupported_bart() {
        assert!(!is_supported_decoder_architecture(ModelArchitecture::Bart));
    }

    #[test]
    fn test_unsupported_whisper() {
        assert!(!is_supported_decoder_architecture(ModelArchitecture::Whisper));
    }

    #[test]
    fn test_unsupported_nomic_bert() {
        assert!(!is_supported_decoder_architecture(ModelArchitecture::NomicBert));
    }

    #[test]
    fn test_unsupported_phi3() {
        // Phi3 might need to be added to supported list if it should work
        assert!(!is_supported_decoder_architecture(ModelArchitecture::Phi3));
    }

    // =========================================================================
    // build_decoding_strategy tests
    // =========================================================================

    #[test]
    fn test_strategy_greedy_flag() {
        let strategy = build_decoding_strategy(0.7, None, None, None, true);
        assert!(matches!(strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_strategy_zero_temperature() {
        let strategy = build_decoding_strategy(0.0, None, None, None, false);
        assert!(matches!(strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_strategy_sampling_default() {
        let strategy = build_decoding_strategy(0.7, None, None, None, false);
        
        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 0.7);
                assert_eq!(params.top_k, Some(50));
                assert_eq!(params.top_p, Some(0.9));
                assert_eq!(params.min_p, Some(0.1));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_strategy_sampling_custom_top_k() {
        let strategy = build_decoding_strategy(0.7, Some(100), None, None, false);
        
        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.top_k, Some(100));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_strategy_sampling_custom_top_p() {
        let strategy = build_decoding_strategy(0.7, None, Some(0.95), None, false);
        
        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.top_p, Some(0.95));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_strategy_sampling_custom_min_p() {
        let strategy = build_decoding_strategy(0.7, None, None, Some(0.05), false);
        
        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.min_p, Some(0.05));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_strategy_sampling_all_custom() {
        let strategy = build_decoding_strategy(1.0, Some(40), Some(0.8), Some(0.02), false);
        
        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 1.0);
                assert_eq!(params.top_k, Some(40));
                assert_eq!(params.top_p, Some(0.8));
                assert_eq!(params.min_p, Some(0.02));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_strategy_greedy_overrides_temperature() {
        // Even with high temperature, greedy flag should win
        let strategy = build_decoding_strategy(1.5, Some(100), Some(0.99), Some(0.01), true);
        assert!(matches!(strategy, DecodingStrategy::Greedy));
    }

    // =========================================================================
    // get_sampling_params tests
    // =========================================================================

    #[test]
    fn test_sampling_params_defaults() {
        let params = get_sampling_params(0.7, None, None, None);
        
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_k, Some(50));
        assert_eq!(params.top_p, Some(0.9));
        assert_eq!(params.min_p, Some(0.1));
    }

    #[test]
    fn test_sampling_params_custom() {
        let params = get_sampling_params(1.2, Some(100), Some(0.95), Some(0.05));
        
        assert_eq!(params.temperature, 1.2);
        assert_eq!(params.top_k, Some(100));
        assert_eq!(params.top_p, Some(0.95));
        assert_eq!(params.min_p, Some(0.05));
    }

    #[test]
    fn test_sampling_params_partial_override() {
        let params = get_sampling_params(0.5, Some(25), None, None);
        
        assert_eq!(params.temperature, 0.5);
        assert_eq!(params.top_k, Some(25));
        assert_eq!(params.top_p, Some(0.9)); // default
        assert_eq!(params.min_p, Some(0.1)); // default
    }

    #[test]
    fn test_sampling_params_zero_temperature() {
        let params = get_sampling_params(0.0, None, None, None);
        assert_eq!(params.temperature, 0.0);
    }

    #[test]
    fn test_sampling_params_high_temperature() {
        let params = get_sampling_params(2.0, None, None, None);
        assert_eq!(params.temperature, 2.0);
    }

    // =========================================================================
    // build_generation_config tests
    // =========================================================================

    #[test]
    fn test_generation_config_basic() {
        let config = build_generation_config(100, 0.7, None, None, None, 1.1, false);
        
        assert_eq!(config.max_new_tokens, Some(100));
        assert_eq!(config.repetition_penalty, 1.1);
        assert!(matches!(config.strategy, DecodingStrategy::Sample(_)));
    }

    #[test]
    fn test_generation_config_greedy() {
        let config = build_generation_config(50, 0.7, None, None, None, 1.0, true);
        
        assert_eq!(config.max_new_tokens, Some(50));
        assert!(matches!(config.strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_generation_config_zero_temp_greedy() {
        let config = build_generation_config(200, 0.0, None, None, None, 1.2, false);
        
        assert!(matches!(config.strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_generation_config_max_tokens() {
        let config = build_generation_config(1000, 0.7, None, None, None, 1.0, false);
        assert_eq!(config.max_new_tokens, Some(1000));
    }

    #[test]
    fn test_generation_config_repetition_penalty() {
        let config = build_generation_config(100, 0.7, None, None, None, 1.5, false);
        assert_eq!(config.repetition_penalty, 1.5);
    }

    #[test]
    fn test_generation_config_no_repetition_penalty() {
        let config = build_generation_config(100, 0.7, None, None, None, 1.0, false);
        assert_eq!(config.repetition_penalty, 1.0);
    }

    #[test]
    fn test_generation_config_sampling_params_passed_through() {
        let config = build_generation_config(100, 0.9, Some(40), Some(0.85), Some(0.05), 1.1, false);
        
        match config.strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 0.9);
                assert_eq!(params.top_k, Some(40));
                assert_eq!(params.top_p, Some(0.85));
                assert_eq!(params.min_p, Some(0.05));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_very_low_temperature() {
        let strategy = build_decoding_strategy(0.001, None, None, None, false);
        
        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 0.001);
            }
            _ => panic!("Expected Sample strategy (temperature > 0)"),
        }
    }

    #[test]
    fn test_very_high_temperature() {
        let strategy = build_decoding_strategy(5.0, None, None, None, false);
        
        match strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 5.0);
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_top_k_zero() {
        let params = get_sampling_params(0.7, Some(0), None, None);
        assert_eq!(params.top_k, Some(0));
    }

    #[test]
    fn test_top_p_zero() {
        let params = get_sampling_params(0.7, None, Some(0.0), None);
        assert_eq!(params.top_p, Some(0.0));
    }

    #[test]
    fn test_top_p_one() {
        let params = get_sampling_params(0.7, None, Some(1.0), None);
        assert_eq!(params.top_p, Some(1.0));
    }

    #[test]
    fn test_min_p_zero() {
        let params = get_sampling_params(0.7, None, None, Some(0.0));
        assert_eq!(params.min_p, Some(0.0));
    }

    #[test]
    fn test_large_max_tokens() {
        let config = build_generation_config(100_000, 0.7, None, None, None, 1.0, false);
        assert_eq!(config.max_new_tokens, Some(100_000));
    }

    #[test]
    fn test_small_max_tokens() {
        let config = build_generation_config(1, 0.7, None, None, None, 1.0, false);
        assert_eq!(config.max_new_tokens, Some(1));
    }

    #[test]
    fn test_zero_max_tokens() {
        let config = build_generation_config(0, 0.7, None, None, None, 1.0, false);
        assert_eq!(config.max_new_tokens, Some(0));
    }

    #[test]
    fn test_negative_repetition_penalty() {
        // Unusual but should be accepted by the config builder
        let config = build_generation_config(100, 0.7, None, None, None, -0.5, false);
        assert_eq!(config.repetition_penalty, -0.5);
    }

    #[test]
    fn test_high_repetition_penalty() {
        let config = build_generation_config(100, 0.7, None, None, None, 5.0, false);
        assert_eq!(config.repetition_penalty, 5.0);
    }

    // =========================================================================
    // Realistic parameter combinations
    // =========================================================================

    #[test]
    fn test_typical_creative_writing_params() {
        // High temperature, moderate top_p, low min_p
        let config = build_generation_config(512, 1.0, None, Some(0.95), Some(0.05), 1.1, false);
        
        match config.strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 1.0);
                assert_eq!(params.top_p, Some(0.95));
                assert_eq!(params.min_p, Some(0.05));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_typical_code_generation_params() {
        // Lower temperature, stricter sampling
        let config = build_generation_config(256, 0.3, Some(40), Some(0.9), None, 1.0, false);
        
        match config.strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 0.3);
                assert_eq!(params.top_k, Some(40));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }

    #[test]
    fn test_deterministic_generation_params() {
        // Greedy for reproducibility
        let config = build_generation_config(100, 0.0, None, None, None, 1.0, false);
        assert!(matches!(config.strategy, DecodingStrategy::Greedy));
    }

    #[test]
    fn test_chatbot_params() {
        // Balanced settings
        let config = build_generation_config(512, 0.7, Some(50), Some(0.9), Some(0.1), 1.1, false);
        
        assert_eq!(config.max_new_tokens, Some(512));
        assert_eq!(config.repetition_penalty, 1.1);
        
        match config.strategy {
            DecodingStrategy::Sample(params) => {
                assert_eq!(params.temperature, 0.7);
                assert_eq!(params.top_k, Some(50));
                assert_eq!(params.top_p, Some(0.9));
                assert_eq!(params.min_p, Some(0.1));
            }
            _ => panic!("Expected Sample strategy"),
        }
    }
}