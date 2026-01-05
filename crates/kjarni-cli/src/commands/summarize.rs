//! Summarization command using encoder-decoder models

use anyhow::{anyhow, Result};

use kjarni::{models::BartModel, registry, BeamSearchParams, DecodingStrategy, Device, EncoderDecoderGenerator, ModelArchitecture, ModelType};

use super::util::resolve_input;

pub async fn run(
    input: Option<&str>,
    model: &str,
    model_path: Option<&str>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    num_beams: Option<usize>,
    length_penalty: Option<f32>,
    no_repeat_ngram: Option<usize>,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // 1. Resolve input
    let text = resolve_input(input)?;

    if text.trim().is_empty() {
        return Err(anyhow!("Input text is empty. Nothing to summarize."));
    }

    // 2. Resolve model
    let device = if gpu { Device::Wgpu } else { Device::Cpu };

    if model_path.is_some() {
        return Err(anyhow!("--model-path not yet implemented."));
    }

    let model_type = ModelType::from_cli_name(model).ok_or_else(|| {
        let mut msg = format!("Unknown model: '{}'.", model);
        let suggestions = ModelType::find_similar(model);
        if !suggestions.is_empty() {
            msg.push_str("\n\nDid you mean?");
            for (name, _) in suggestions {
                msg.push_str(&format!("\n  - {}", name));
            }
        }
        anyhow!(msg)
    })?;

    // Validate it's an encoder-decoder model
    if model_type.architecture() != ModelArchitecture::Bart {
        return Err(anyhow!(
            "Model '{}' is not an encoder-decoder model. Summarization requires models like distilbart-cnn or bart-large-cnn.",
            model
        ));
    }

    // Check if downloaded
    if !registry::is_model_downloaded(model)? {
        if !quiet {
            eprintln!("Model '{}' not found locally. Downloading...", model);
        }
        registry::download_model(model, false).await?;
        if !quiet {
            eprintln!();
        }
    }

    // 3. Load model
    if !quiet {
        eprintln!("Loading model '{}'...", model);
    }

    let loaded_model = match model_type {
        ModelType::BartLargeCnn | ModelType::DistilBartCnn => {
            BartModel::from_registry(model_type, None, device, None, None).await?
        }
        _ => {
            return Err(anyhow!(
                "Model '{}' not yet supported for summarization. Try: distilbart-cnn or bart-large-cnn",
                model
            ));
        }
    };

    let generator = EncoderDecoderGenerator::new(Box::new(loaded_model))?;

    // 4. Build generation config
    // Start with model's default config (from task_specific_params)
    let mut config = generator.model.get_default_generation_config();

    // Override with CLI arguments if provided
    if let Some(min) = min_length {
        config.min_length = min;
    }
    if let Some(max) = max_length {
        config.max_length = max;
    }
    if let Some(ngram) = no_repeat_ngram {
        config.no_repeat_ngram_size = ngram;
    }

    // Override beam search params if provided
    if num_beams.is_some() || length_penalty.is_some() {
        // Get current beam params or create defaults
        let current_params = match &config.strategy {
            DecodingStrategy::BeamSearch(params) => params.clone(),
            _ => BeamSearchParams::default(),
        };

        config.strategy = DecodingStrategy::BeamSearch(BeamSearchParams {
            num_beams: num_beams.unwrap_or(current_params.num_beams),
            length_penalty: length_penalty.unwrap_or(current_params.length_penalty),
            early_stopping: current_params.early_stopping,
        });
    }

    // 5. Generate summary
    if !quiet {
        eprintln!("Summarizing {} characters...", text.len());
        eprintln!();
    }

    let summary = generator.generate(&text, Some(&config)).await?;

    // 6. Output
    println!("{}", summary.trim());

    Ok(())
}
