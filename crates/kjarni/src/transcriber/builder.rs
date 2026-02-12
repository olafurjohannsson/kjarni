use std::sync::Arc;

use kjarni_transformers::{Device, ModelType, WgpuContext};
use kjarni_models::models::whisper::WhisperModel;

use super::model::Transcriber;
use super::types::{
    Task, TranscriberError, TranscriberResult, TranscriptionProgressCallback,
};
use super::validation;

pub struct TranscriberBuilder {
    model_id: String,
    model_type: Option<ModelType>,
    device: Device,
    context: Option<Arc<WgpuContext>>,
    language: Option<String>,
    task: Task,
    timestamps: bool,
    max_tokens_per_chunk: usize,
    quiet: bool,
    progress_callback: Option<TranscriptionProgressCallback>,
}

impl TranscriberBuilder {
    /// Create a new builder targeting the given model identifier.
    pub fn new(model: &str) -> Self {
        Self {
            model_id: model.to_string(),
            model_type: None,
            device: Device::Cpu,
            context: None,
            language: None,
            task: Task::Transcribe,
            timestamps: false,
            max_tokens_per_chunk: 448,
            quiet: false,
            progress_callback: None,
        }
    }

    /// Use CPU for inference (default).
    pub fn cpu(mut self) -> Self {
        self.device = Device::Cpu;
        self
    }

    /// Use GPU for inference.
    pub fn gpu(mut self) -> Self {
        self.device = Device::Wgpu;
        self
    }

    /// Provide a pre-existing GPU context (avoids creating a new one).
    pub fn with_context(mut self, ctx: Arc<WgpuContext>) -> Self {
        self.context = Some(ctx);
        self
    }

    /// Force a language
    pub fn language(mut self, lang: &str) -> Self {
        self.language = Some(lang.to_string());
        self
    }

    /// Set the task to translate
    pub fn translate(mut self) -> Self {
        self.task = Task::Translate;
        self
    }

    /// Set the task to **transcribe** (default).
    pub fn transcribe(mut self) -> Self {
        self.task = Task::Transcribe;
        self
    }

    /// Enable or disable timestamp segments.
    pub fn timestamps(mut self, enabled: bool) -> Self {
        self.timestamps = enabled;
        self
    }

    /// Maximum tokens per 30-second chunk (default: 448).
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens_per_chunk = n;
        self
    }

    /// Suppress progress output to stderr.
    pub fn quiet(mut self) -> Self {
        self.quiet = true;
        self
    }

    /// Register a progress callback.
    pub fn on_progress<F>(mut self, callback: F) -> Self
    where
        F: Fn(&super::types::TranscriptionProgress, Option<&str>) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    pub async fn build(self) -> TranscriberResult<Transcriber> {
        let model_type = self
            .model_type
            .or_else(|| resolve_model_type(&self.model_id))
            .ok_or_else(|| {
                TranscriberError::InvalidConfig(format!(
                    "Unknown model: '{}'. Try 'whisper-small' or 'whisper-large-v3'.",
                    self.model_id
                ))
            })?;

        validation::validate_config(self.language.as_deref(), self.max_tokens_per_chunk)?;

        let context = if self.device == Device::Wgpu {
            if let Some(ctx) = self.context {
                Some(ctx)
            } else {
                Some(
                    WgpuContext::new()
                        .await
                        .map_err(|_| TranscriberError::GpuUnavailable)?,
                )
            }
        } else {
            None
        };

        let model = WhisperModel::from_registry(model_type, None, self.device, context, None)
            .await
            .map_err(TranscriberError::ModelLoadFailed)?;

        Ok(Transcriber::new(
            Arc::new(model),
            model_type,
            self.language,
            self.task,
            self.timestamps,
            self.max_tokens_per_chunk,
            self.device,
            self.quiet,
            self.progress_callback,
        ))
    }
}


fn resolve_model_type(id: &str) -> Option<ModelType> {
    match id.to_lowercase().as_str() {
        "whisper-small" | "whisper_small" | "small" => Some(ModelType::WhisperSmall),
        "whisper-large-v3" | "whisper_large_v3" | "large-v3" | "large" => {
            Some(ModelType::WhisperLargeV3)
        }
        // Add more as they become supported
        _ => None,
    }
}