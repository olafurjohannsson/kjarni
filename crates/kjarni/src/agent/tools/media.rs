//! Transcribing recordings and finding photos by description, with Whisper and
//! CLIP on the CPU. Each tool loads its model on its first call and keeps it.

#[cfg(feature = "image-io")]
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, bail};
use async_trait::async_trait;
use serde_json::{Map, Value as Json, json};
#[cfg(feature = "image-io")]
use tokio::sync::Mutex;
use tokio::sync::OnceCell;

#[cfg(feature = "image-io")]
use crate::agent::tool::usize_arg;
use crate::agent::tool::{Access, Tool, ToolSpec, string_arg};
use crate::agent::tools::files::Workspace;
#[cfg(feature = "image-io")]
use crate::common::{DownloadPolicy, ensure_model_downloaded};
#[cfg(feature = "image-io")]
use crate::folder_search::{folder_key, scan};
use crate::transcriber::{Transcriber, TranscriptionResult};
#[cfg(feature = "image-io")]
use crate::{ImageIndex, ModelType};

/// The Whisper model `transcribe` loads.
pub const TRANSCRIBE_MODEL: &str = "whisper-small";

#[cfg(feature = "image-io")]
/// The CLIP model `search_images` loads, which puts photos and descriptions in
/// one space.
pub const IMAGE_MODEL: ModelType = ModelType::ClipVitBase32;

#[cfg(feature = "image-io")]
/// The folder in the model cache that holds each photo folder's vectors.
const IMAGE_INDEXES: &str = "mcp-image-indexes";

#[cfg(feature = "image-io")]
const DEFAULT_PHOTOS: usize = 5;
#[cfg(feature = "image-io")]
const MAX_PHOTOS: usize = 50;

/// A transcription as the model reads it: the language and length, then one
/// timestamped line per segment.
pub fn transcript_text(result: &TranscriptionResult) -> String {
    let mut text = format!(
        "Language: {}. Duration: {:.0} s.\n\n",
        result.language, result.duration_secs
    );
    for s in &result.segments {
        text.push_str(&format!("[{}] {}\n", timestamp(s.start), s.text.trim()));
    }
    if result.segments.is_empty() {
        text.push_str(result.text.trim());
    }
    text.trim_end().to_string()
}

fn timestamp(secs: f32) -> String {
    let s = secs.max(0.0) as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

#[cfg(feature = "image-io")]
/// The `top_k` photos under `folder` that best match `query`, one per line
/// with its score. The folder's vectors are saved in the model cache and
/// reused until its files change; when they are computed, a line in
/// parentheses says how many photos were indexed and which were skipped.
pub fn search_photos(
    index: &mut ImageIndex,
    folder: &Path,
    query: &str,
    top_k: usize,
) -> anyhow::Result<String> {
    let root = crate::registry::cache_dir().join(IMAGE_INDEXES);
    let saved = root.join(format!("{}.json", folder_key(folder)));
    let stamp_file = saved.with_extension("stamp");
    let stamp = scan(folder, IMAGE_MODEL.cli_name()).stamp;
    let current = std::fs::read_to_string(&stamp_file).ok();

    let mut text = String::new();
    if saved.exists() && current.as_deref() == Some(stamp.as_str()) {
        index.load_entries(&saved)?;
    } else {
        index.clear();
        let report = index.add_directory(folder)?;
        std::fs::create_dir_all(&root)?;
        index.save(&saved)?;
        std::fs::write(&stamp_file, stamp)?;
        text.push_str(&format!(
            "(Indexed {} photos in {} for this search.",
            report.added,
            folder.display()
        ));
        if report.skipped > 0 {
            text.push_str(&format!(" {} could not be read.", report.skipped));
        }
        let unsupported: usize = report.unsupported.values().sum();
        if unsupported > 0 {
            let kinds: Vec<String> = report
                .unsupported
                .iter()
                .map(|(ext, n)| format!("{n} .{ext}"))
                .collect();
            text.push_str(&format!(
                " Skipped, format not supported: {}.",
                kinds.join(", ")
            ));
        }
        text.push_str(")\n\n");
    }

    let hits = index.search(query, top_k)?;
    if hits.is_empty() {
        text.push_str("No JPEG or PNG photos in this folder.");
    }
    for (i, hit) in hits.iter().enumerate() {
        text.push_str(&format!(
            "{}. {}  (score {:.3})\n",
            i + 1,
            hit.path.display(),
            hit.score
        ));
    }
    Ok(text.trim_end().to_string())
}

/// Speech to text of a recording, with timestamps.
pub struct Transcribe {
    workspace: Arc<Workspace>,
    transcriber: OnceCell<Arc<Transcriber>>,
}

impl Transcribe {
    pub fn new(workspace: Arc<Workspace>) -> Self {
        Self {
            workspace,
            transcriber: OnceCell::new(),
        }
    }
}

#[async_trait]
impl Tool for Transcribe {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "transcribe",
            "Turn speech in an audio file (wav, mp3, flac, ogg) into text with timestamps, on this computer. \
             Detects the language. Takes about a tenth of the recording's length.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The audio file."}
                },
                "required": ["path"]
            }),
        )
    }

    fn access(&self) -> Access {
        Access::Read
    }

    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String> {
        let path = self.workspace.resolve(string_arg(args, "path")?);
        if !path.is_file() {
            bail!("{} is not a file", path.display());
        }
        let transcriber = self
            .transcriber
            .get_or_try_init(|| async {
                let t = Transcriber::builder(TRANSCRIBE_MODEL)
                    .quiet()
                    .build()
                    .await
                    .context("loading Whisper")?;
                anyhow::Ok(Arc::new(t))
            })
            .await?
            .clone();
        let result =
            tokio::task::spawn_blocking(move || transcriber.transcribe_file(&path)).await??;
        Ok(transcript_text(&result))
    }
}

#[cfg(feature = "image-io")]
/// Finds photos in a folder by describing what is in them.
pub struct SearchImages {
    workspace: Arc<Workspace>,
    index: Mutex<Option<ImageIndex>>,
}

#[cfg(feature = "image-io")]
impl SearchImages {
    pub fn new(workspace: Arc<Workspace>) -> Self {
        Self {
            workspace,
            index: Mutex::new(None),
        }
    }
}

#[cfg(feature = "image-io")]
#[async_trait]
impl Tool for SearchImages {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "search_images",
            "Find the photos in a folder that best match a description, such as \"a dog on a beach\" or \"a receipt\". \
             JPEG and PNG, subfolders included. The first search of a folder looks at every photo, which takes a while for thousands.",
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What the photo shows."},
                    "path": {"type": "string", "description": "Folder of photos. Default: the working directory."},
                    "limit": {"type": "integer", "description": "Photos to return. Default 5."}
                },
                "required": ["query"]
            }),
        )
    }

    fn access(&self) -> Access {
        Access::Read
    }

    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String> {
        let query = string_arg(args, "query")?.to_string();
        let folder = match args.get("path") {
            Some(Json::String(p)) if !p.trim().is_empty() => self.workspace.resolve(p),
            _ => self.workspace.root().to_path_buf(),
        };
        let folder = folder
            .canonicalize()
            .with_context(|| format!("cannot open {}", folder.display()))?;
        if !folder.is_dir() {
            bail!("{} is not a folder", folder.display());
        }
        let limit = usize_arg(args, "limit")?
            .unwrap_or(DEFAULT_PHOTOS)
            .clamp(1, MAX_PHOTOS);

        let mut slot = self.index.lock().await;
        let index = match slot.take() {
            Some(index) => index,
            None => {
                let dir =
                    ensure_model_downloaded(IMAGE_MODEL, None, DownloadPolicy::IfMissing, true)
                        .await
                        .context("downloading CLIP")?;
                tokio::task::spawn_blocking(move || ImageIndex::load(&dir))
                    .await?
                    .context("loading CLIP")?
            }
        };
        let (index, found) = tokio::task::spawn_blocking(move || {
            let mut index = index;
            let found = search_photos(&mut index, &folder, &query, limit);
            (index, found)
        })
        .await?;
        *slot = Some(index);
        found
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcriber::TranscriptionSegment;

    #[test]
    fn a_transcript_has_the_language_length_and_timed_lines() {
        let segment = |start: f32, text: &str| TranscriptionSegment {
            start,
            end: start + 2.0,
            text: text.into(),
        };
        let result = TranscriptionResult {
            text: "Hello. Again.".into(),
            segments: vec![segment(0.0, " Hello."), segment(3725.4, " Again. ")],
            language: "en".into(),
            duration_secs: 3727.6,
        };
        assert_eq!(
            transcript_text(&result),
            "Language: en. Duration: 3728 s.\n\n[00:00:00] Hello.\n[01:02:05] Again."
        );
    }

    #[test]
    fn a_transcript_without_segments_is_its_text() {
        let result = TranscriptionResult {
            text: " Just text. ".into(),
            segments: Vec::new(),
            language: "is".into(),
            duration_secs: 1.2,
        };
        assert_eq!(
            transcript_text(&result),
            "Language: is. Duration: 1 s.\n\nJust text."
        );
    }
}
