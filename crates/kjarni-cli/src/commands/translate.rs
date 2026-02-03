//! Translation command using encoder-decoder models

use anyhow::{anyhow, Result};
use futures::{StreamExt, pin_mut};
use std::io::{self, Write};

use kjarni::translator::Translator;

use super::util::resolve_input;

pub async fn run(
    input: Option<&str>,
    model: &str,
    model_path: Option<&str>,
    src: Option<&str>,
    dst: Option<&str>,
    max_length: Option<usize>,
    num_beams: Option<usize>,
    length_penalty: Option<f32>,
    no_repeat_ngram: Option<usize>,
    greedy: bool,
    no_stream: bool,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // 1. Resolve input
    let text = resolve_input(input)?;
    if text.trim().is_empty() {
        return Err(anyhow!("Input text is empty. Nothing to translate."));
    }

    // 2. Validate languages
    let source_lang = src.ok_or_else(|| {
        anyhow!("Source language is required. Use --src <language> (e.g., --src en)")
    })?;
    let target_lang = dst.ok_or_else(|| {
        anyhow!("Target language is required. Use --dst <language> (e.g., --dst de)")
    })?;

    // 3. Build translator
    if model_path.is_some() {
        return Err(anyhow!("--model-path not yet implemented."));
    }

    let mut builder = Translator::builder(model);

    if gpu {
        builder = builder.gpu();
    } else {
        builder = builder.cpu();
    }

    if quiet {
        builder = builder.quiet();
    }

    if greedy {
        builder = builder.greedy();
    }

    if let Some(max) = max_length {
        builder = builder.max_length(max);
    }

    if let Some(beams) = num_beams {
        builder = builder.num_beams(beams);
    }

    if let Some(penalty) = length_penalty {
        builder = builder.length_penalty(penalty);
    }

    if let Some(ngram) = no_repeat_ngram {
        builder = builder.no_repeat_ngram_size(ngram);
    }

    let translator = builder.build().await.map_err(|e| anyhow!("{}", e))?;

    // 4. Show info
    if !quiet {
        eprintln!("Model: {}", translator.model_name());
        eprintln!("Device: {:?}", translator.device());
        eprintln!("Translating: {} -> {}", source_lang, target_lang);
        eprintln!("Input: {} characters", text.len());
        eprintln!();
    }

    // 5. Translate
    if no_stream {
        let translation = translator
            .translate(&text, source_lang, target_lang)
            .await
            .map_err(|e| anyhow!("Translation failed: {}", e))?;
        println!("{}", translation.trim());
    } else {
        let stream = translator
            .stream(&text, source_lang, target_lang)
            .await
            .map_err(|e| anyhow!("Translation failed: {}", e))?;

        pin_mut!(stream);

        let mut stdout = io::stdout();
        let mut generated_any = false;

        while let Some(token_result) = stream.next().await {
            let token = token_result.map_err(|e| anyhow!("Streaming error: {}", e))?;
            print!("{}", token);
            stdout.flush()?;
            generated_any = true;
        }

        if generated_any {
            println!();
        }
    }

    Ok(())
}