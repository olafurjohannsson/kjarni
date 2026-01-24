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

    if model_type.architecture() != ModelArchitecture::GPT
        && model_type.architecture() != ModelArchitecture::Llama
        && model_type.architecture() != ModelArchitecture::Mistral
        && model_type.architecture() != ModelArchitecture::Qwen2
    {
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
    let strategy = if greedy || temperature == 0.0 {
        DecodingStrategy::Greedy
    } else {
        DecodingStrategy::Sample(SamplingParams {
            temperature,
            top_k: top_k.or(Some(50)),
            top_p: top_p.or(Some(0.9)),
            min_p: min_p.or(Some(0.1)),
        })
    };

    let config = GenerationConfig {
        max_new_tokens: Some(max_tokens),
        repetition_penalty,
        strategy,
        ..Default::default()
    };

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
