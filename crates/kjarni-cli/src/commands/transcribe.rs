//! Transcription command using Whisper models

use anyhow::{Result, anyhow};
use futures::{StreamExt, pin_mut};
use std::io::{self, Write};
use std::path::Path;

use kjarni::transcriber::{Transcriber, TranscriptionResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Text,
    Srt,
    Vtt,
}

impl OutputFormat {
    fn parse(format: &str) -> Result<Self> {
        match format.trim().to_ascii_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "srt" => Ok(Self::Srt),
            "vtt" => Ok(Self::Vtt),
            other => Err(anyhow!("Unknown format: '{}'. Use: text, srt, vtt", other)),
        }
    }

    fn needs_timestamps(self) -> bool {
        matches!(self, Self::Srt | Self::Vtt)
    }
}

/// Run the transcribe command.
pub async fn run(
    file: &str,
    model: &str,
    model_path: Option<&str>,
    language: Option<&str>,
    translate: bool,
    timestamps: bool,
    format: &str,
    max_tokens: Option<usize>,
    no_stream: bool,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    let output_format = OutputFormat::parse(format)?;
    let need_timestamps = timestamps || output_format.needs_timestamps();

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

    if need_timestamps {
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

    if !streams(no_stream, need_timestamps) {
        // Full result mode
        let result = transcriber
            .transcribe_file(file)
            .map_err(|e| anyhow!("Transcription failed: {}", e))?;

        print!("{}", render_output(&result, output_format, timestamps));

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
/// `--timestamps` and subtitle formats wait for the full result.
fn streams(no_stream: bool, need_timestamps: bool) -> bool {
    !no_stream && !need_timestamps
}

fn render_output(result: &TranscriptionResult, format: OutputFormat, timestamps: bool) -> String {
    match format {
        OutputFormat::Srt => result.to_srt(),
        OutputFormat::Vtt => result.to_vtt(),
        OutputFormat::Text => {
            if timestamps && !result.segments.is_empty() {
                let mut out = String::new();
                for segment in &result.segments {
                    out.push_str(&format!(
                        "[{} --> {}] {}\n",
                        format_timestamp(segment.start),
                        format_timestamp(segment.end),
                        segment.text.trim()
                    ));
                }
                out
            } else {
                format!("{}\n", result.text.trim())
            }
        }
    }
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
    use kjarni::transcriber::TranscriptionSegment;

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

    #[test]
    fn parses_supported_formats_and_rejects_unknown() {
        assert_eq!(OutputFormat::parse("text").unwrap(), OutputFormat::Text);
        assert_eq!(OutputFormat::parse("srt").unwrap(), OutputFormat::Srt);
        assert_eq!(OutputFormat::parse("VTT").unwrap(), OutputFormat::Vtt);
        assert!(!OutputFormat::Text.needs_timestamps());
        assert!(OutputFormat::Srt.needs_timestamps());
        assert!(OutputFormat::Vtt.needs_timestamps());

        let err = OutputFormat::parse("json").unwrap_err().to_string();
        assert!(err.contains("Unknown format"));
        assert!(err.contains("text, srt, vtt"));
    }

    #[test]
    fn renders_srt_vtt_and_text_outputs() {
        let result = TranscriptionResult {
            text: " Hello world. This is a longer sentence that should wrap across lines cleanly."
                .into(),
            segments: vec![
                TranscriptionSegment {
                    start: 0.0,
                    end: 1.2,
                    text: " Hello world.".into(),
                },
                TranscriptionSegment {
                    start: 1.2,
                    end: 5.0,
                    text: " This is a longer sentence that should wrap across lines cleanly."
                        .into(),
                },
            ],
            language: "en".into(),
            duration_secs: 5.0,
        };

        assert_eq!(
            render_output(&result, OutputFormat::Text, false),
            "Hello world. This is a longer sentence that should wrap across lines cleanly.\n"
        );
        assert_eq!(
            render_output(&result, OutputFormat::Text, true),
            "[00:00.000 --> 00:01.200] Hello world.\n\
             [00:01.200 --> 00:05.000] This is a longer sentence that should wrap across lines cleanly.\n"
        );
        assert_eq!(
            render_output(&result, OutputFormat::Srt, false),
            "1\n00:00:00,000 --> 00:00:01,200\nHello world.\n\n\
             2\n00:00:01,200 --> 00:00:05,000\n\
             This is a longer sentence that should wrap\n\
             across lines cleanly.\n\n"
        );
        assert_eq!(
            render_output(&result, OutputFormat::Vtt, false),
            "WEBVTT\n\n\
             00:00:00.000 --> 00:00:01.200\nHello world.\n\n\
             00:00:01.200 --> 00:00:05.000\n\
             This is a longer sentence that should wrap\n\
             across lines cleanly.\n\n"
        );
    }
}
