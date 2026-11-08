//! Transcription command using Whisper models

use anyhow::{Result, anyhow};
use futures::{StreamExt, pin_mut};
use std::io::{self, Write};
use std::path::Path;

use kjarni::transcriber::Transcriber;

/// Run the transcribe command.
pub async fn run(
    file: &str,
    model: &str,
    model_path: Option<&str>,
    language: Option<&str>,
    translate: bool,
    timestamps: bool,
    max_tokens: Option<usize>,
    no_stream: bool,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // Validate input file exists
    let path = Path::new(file);
    if !path.exists() {
        return Err(anyhow!("Audio file not found: {}", file));
    }
    if !path.is_file() {
        return Err(anyhow!("Path is not a file: {}", file));
    }

    // Build transcriber
    let mut builder = Transcriber::builder(model);
    if let Some(path) = model_path {
        builder = builder.model_path(path);
    }

    if gpu {
        builder = builder.gpu();
    } else {
        builder = builder.cpu();
    }

    if quiet {
        builder = builder.quiet();
    }

    if let Some(lang) = language {
        builder = builder.language(lang);
    }

    if translate {
        builder = builder.translate();
    }

    if timestamps {
        builder = builder.timestamps(true);
    }

    if let Some(max) = max_tokens {
        builder = builder.max_tokens(max);
    }

    let transcriber = builder.build().await.map_err(|e| anyhow!("{}", e))?;

    // Show info
    if !quiet {
        eprintln!("Model: {}", transcriber.model_name());
        eprintln!("Device: {:?}", transcriber.device());
        if let Some(lang) = language {
            eprintln!("Language: {}", lang);
        } else {
            eprintln!("Language: auto-detect");
        }
        if translate {
            eprintln!("Task: translate to English");
        } else {
            eprintln!("Task: transcribe");
        }
        eprintln!("Input: {}", file);
        eprintln!();
    }

    if !streams(no_stream, timestamps) {
        // Full result mode
        let result = transcriber
            .transcribe_file(file)
            .map_err(|e| anyhow!("Transcription failed: {}", e))?;

        if timestamps && !result.segments.is_empty() {
            // Print with timestamps
            for segment in &result.segments {
                println!(
                    "[{} --> {}] {}",
                    format_timestamp(segment.start),
                    format_timestamp(segment.end),
                    segment.text.trim()
                );
            }
        } else {
            // Print plain text
            println!("{}", result.text.trim());
        }

        if !quiet {
            eprintln!();
            eprintln!("Duration: {:.2}s", result.duration_secs);
            eprintln!("Language: {}", result.language);
        }
    } else {
        // Streaming mode
        let stream = transcriber
            .stream_file(file)
            .await
            .map_err(|e| anyhow!("Transcription failed: {}", e))?;

        pin_mut!(stream);

        let mut stdout = io::stdout();
        let mut generated_any = false;

        while let Some(token_result) = stream.next().await {
            let token = token_result.map_err(|e| anyhow!("Streaming error: {}", e))?;

            // Skip special tokens in streaming output
            if !token.is_special {
                print!("{}", token.text);
                stdout.flush()?;
                generated_any = true;
            }
        }

        if generated_any {
            println!();
        }
    }

    Ok(())
}

/// Whether to print tokens as they decode. Streamed tokens carry no times, so
/// `--timestamps` waits for the full result, which has them per segment.
fn streams(no_stream: bool, timestamps: bool) -> bool {
    !no_stream && !timestamps
}

/// Format seconds as HH:MM:SS.mmm or MM:SS.mmm
fn format_timestamp(seconds: f32) -> String {
    let total_ms = (seconds * 1000.0) as u64;
    let ms = total_ms % 1000;
    let total_secs = total_ms / 1000;
    let secs = total_secs % 60;
    let total_mins = total_secs / 60;
    let mins = total_mins % 60;
    let hours = total_mins / 60;

    if hours > 0 {
        format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, ms)
    } else {
        format!("{:02}:{:02}.{:03}", mins, secs, ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_turn_off_streaming() {
        assert!(streams(false, false));
        assert!(!streams(false, true));
        assert!(!streams(true, false));
        assert!(!streams(true, true));
    }

    #[test]
    fn timestamps_format_as_minutes_or_hours() {
        assert_eq!(format_timestamp(0.0), "00:00.000");
        assert_eq!(format_timestamp(29.54), "00:29.540");
        assert_eq!(format_timestamp(3725.5), "01:02:05.500");
    }
}
