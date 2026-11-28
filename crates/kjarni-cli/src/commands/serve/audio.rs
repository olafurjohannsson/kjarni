//! `/v1/audio/transcriptions` and `/v1/audio/translations` in OpenAI's format:
//! a multipart upload in, the text out as JSON, plain text, SRT or WebVTT.

use std::io::Write;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use kjarni::transcriber::{Task, TranscriptionResult};
use serde_json::json;

use super::AppState;
use super::error::ApiError;
use super::models::resolve;

/// The formats the audio loader decodes.
const FORMATS: [&str; 4] = ["wav", "mp3", "flac", "ogg"];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Format {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

impl Format {
    pub fn parse(name: &str) -> Result<Self, ApiError> {
        Ok(match name {
            "json" => Self::Json,
            "text" => Self::Text,
            "verbose_json" => Self::VerboseJson,
            "srt" => Self::Srt,
            "vtt" => Self::Vtt,
            other => {
                return Err(ApiError::invalid(
                    "response_format",
                    format!(
                        "response_format must be json, text, verbose_json, srt or vtt, not '{other}'"
                    ),
                ));
            }
        })
    }

    fn needs_timestamps(self) -> bool {
        matches!(self, Self::VerboseJson | Self::Srt | Self::Vtt)
    }
}

/// The audio format from the upload's file name, or failing that from the
/// first bytes of the file.
pub fn audio_format(file_name: Option<&str>, bytes: &[u8]) -> Result<&'static str, ApiError> {
    let extension = file_name
        .and_then(|n| n.rsplit_once('.'))
        .map(|(_, ext)| ext.to_ascii_lowercase());
    if let Some(ext) = &extension
        && let Some(known) = FORMATS.iter().find(|f| *f == ext)
    {
        return Ok(known);
    }
    let sniffed = if bytes.starts_with(b"RIFF") {
        Some("wav")
    } else if bytes.starts_with(b"fLaC") {
        Some("flac")
    } else if bytes.starts_with(b"OggS") {
        Some("ogg")
    } else if bytes.starts_with(b"ID3")
        || (bytes.len() > 1 && bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0)
    {
        Some("mp3")
    } else {
        None
    };
    sniffed.ok_or_else(|| {
        ApiError::invalid(
            "file",
            format!(
                "unsupported audio format{}: send wav, mp3, flac or ogg",
                extension.map(|e| format!(" '{e}'")).unwrap_or_default()
            ),
        )
    })
}

pub fn srt(result: &TranscriptionResult) -> String {
    result.to_srt()
}

pub fn vtt(result: &TranscriptionResult) -> String {
    result.to_vtt()
}

pub fn render(result: &TranscriptionResult, format: Format, task: Task) -> Response {
    let text = |body: String| {
        ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], body).into_response()
    };
    match format {
        Format::Json => Json(json!({ "text": result.text.trim() })).into_response(),
        Format::Text => text(format!("{}\n", result.text.trim())),
        Format::Srt => text(srt(result)),
        Format::Vtt => text(vtt(result)),
        Format::VerboseJson => {
            let segments: Vec<_> = result
                .segments
                .iter()
                .enumerate()
                .map(|(id, s)| json!({ "id": id, "start": s.start, "end": s.end, "text": s.text }))
                .collect();
            Json(json!({
                "task": task.to_string(),
                "language": result.language,
                "duration": result.duration_secs,
                "text": result.text.trim(),
                "segments": segments,
            }))
            .into_response()
        }
    }
}

pub async fn transcriptions(
    state: State<Arc<AppState>>,
    form: Multipart,
) -> Result<Response, ApiError> {
    run(state, form, Task::Transcribe).await
}

pub async fn translations(
    state: State<Arc<AppState>>,
    form: Multipart,
) -> Result<Response, ApiError> {
    run(state, form, Task::Translate).await
}

async fn run(
    State(app): State<Arc<AppState>>,
    mut form: Multipart,
    task: Task,
) -> Result<Response, ApiError> {
    let bad_form =
        |e: axum::extract::multipart::MultipartError| ApiError::bad_request(e.body_text());
    let mut file: Option<(Option<String>, Vec<u8>)> = None;
    let (mut model, mut language, mut format) = (None, None, Format::Json);
    while let Some(field) = form.next_field().await.map_err(bad_form)? {
        match field.name().unwrap_or_default() {
            "file" => {
                let name = field.file_name().map(str::to_string);
                file = Some((name, field.bytes().await.map_err(bad_form)?.to_vec()));
            }
            "model" => model = Some(field.text().await.map_err(bad_form)?),
            "language" => language = Some(field.text().await.map_err(bad_form)?),
            "response_format" => {
                format = Format::parse(field.text().await.map_err(bad_form)?.trim())?
            }
            // prompt, temperature and timestamp_granularities: accepted and
            // not used. Whisper here decodes greedily without a prompt.
            _ => {}
        }
    }
    let Some((file_name, bytes)) = file else {
        return Err(ApiError::invalid(
            "file",
            "send the audio as a 'file' field",
        ));
    };
    let extension = audio_format(file_name.as_deref(), &bytes)?;

    let model = resolve(model.as_deref(), &app.models.defaults.transcribe)?;
    let transcriber = app.models.transcriber(&model).await?;
    // Translation is always into English; a language names the speech.
    let language = language.as_deref().map(str::trim).filter(|l| !l.is_empty());
    let job = transcriber
        .with_settings(language, task, format.needs_timestamps())
        .map_err(|e| ApiError::invalid("language", e.to_string()))?;

    let result = tokio::task::spawn_blocking(move || {
        let mut temp = tempfile::Builder::new()
            .suffix(&format!(".{extension}"))
            .tempfile()
            .map_err(ApiError::internal)?;
        temp.write_all(&bytes).map_err(ApiError::internal)?;
        job.transcribe_file(temp.path())
            .map_err(|e| ApiError::bad_request(format!("could not transcribe the audio: {e}")))
    })
    .await
    .map_err(ApiError::internal)??;

    Ok(render(&result, format, task))
}
