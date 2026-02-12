//! Summarization command using encoder-decoder models

use anyhow::{anyhow, Result};
use futures::{StreamExt, pin_mut};
use std::io::{self, Write};

use kjarni::summarizer::Summarizer;

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
    greedy: bool,
    no_stream: bool,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // Resolve input
    let text = resolve_input(input)?;
    if text.trim().is_empty() {
        return Err(anyhow!("Input text is empty. Nothing to summarize."));
    }

    // Build summarizer
    if model_path.is_some() {
        return Err(anyhow!("--model-path not yet implemented."));
    }

    let mut builder = Summarizer::builder(model);

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

    if let Some(min) = min_length {
        builder = builder.min_length(min);
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

    let summarizer = builder.build().await.map_err(|e| anyhow!("{}", e))?;

    // Show info
    if !quiet {
        eprintln!("Model: {}", summarizer.model_name());
        eprintln!("Device: {:?}", summarizer.device());
        eprintln!("Input: {} characters", text.len());
        eprintln!();
    }

    // Summarize
    if no_stream {
        let summary = summarizer
            .summarize(&text)
            .await
            .map_err(|e| anyhow!("Summarization failed: {}", e))?;
        println!("{}", summary.trim());
    } else {
        let stream = summarizer
            .stream(&text)
            .await
            .map_err(|e| anyhow!("Summarization failed: {}", e))?;

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