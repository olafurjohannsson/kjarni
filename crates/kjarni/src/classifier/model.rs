//! Core Classifier implementation.

use std::path::Path;
use std::sync::Arc;

use kjarni_transformers::cpu::encoder::CpuEncoderOps;
use kjarni_transformers::{WgpuContext, models::ModelType, traits::Device};

use crate::SequenceClassifier;
use crate::common::{default_cache_dir, ensure_model_downloaded};

use super::builder::ClassifierBuilder;
use super::types::{
    ClassificationMode, ClassificationOverrides, ClassificationResult, ClassifierError,
    ClassifierResult, LabelConfig,
};
use super::validation::validate_for_classification;

/// High-level text classifier.
///
/// Wraps a `SequenceClassifier` with a simpler API and sensible defaults.
pub struct Classifier {
    /// The underlying classifier.
    inner: SequenceClassifier,

    /// Model identifier (registry name or path).
    model_id: String,

    /// Model type (None if loaded from path).
    model_type: Option<ModelType>,

    /// Custom labels (overrides model labels).
    custom_labels: Option<Vec<String>>,

    /// Classification mode.
    mode: ClassificationMode,

    /// Default overrides.
    default_overrides: ClassificationOverrides,

    /// Device.
    device: Device,

    /// GPU context if using GPU.
    #[allow(dead_code)]
    context: Option<Arc<WgpuContext>>,
}

impl Classifier {
    /// Create a Classifier with default settings.
    ///
    /// Uses CPU, downloads model if needed.
    pub async fn new(model: &str) -> ClassifierResult<Self> {
        Self::builder(model).build().await
    }

    /// Load a classifier from a local path.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let classifier = Classifier::from_path("/models/icebert")
    ///     .labels(vec!["neikvætt", "jákvætt"])
    ///     .build()
    ///     .await?;
    /// ```
    pub fn from_path(path: impl Into<std::path::PathBuf>) -> ClassifierBuilder {
        ClassifierBuilder::new("custom").model_path(path)
    }

    /// Internal: construct from builder.
    pub(crate) async fn from_builder(builder: ClassifierBuilder) -> ClassifierResult<Self> {
        let device = builder.device.to_device();

        // Create GPU context if needed
        let context = if device == Device::Wgpu {
            if let Some(ctx) = builder.context {
                Some(ctx)
            } else {
                Some(
                    WgpuContext::new()
                        .await
                        .map_err(|_| ClassifierError::GpuUnavailable)?,
                )
            }
        } else {
            None
        };

        // Build load config with dtype if specified
        let mut load_config = builder.load_config.map(|c| c.into_inner());
        if let Some(dtype) = builder.dtype {
            let config = load_config.get_or_insert_with(Default::default);
            config.target_dtype = Some(dtype);
        }

        // Determine loading strategy
        let (inner, model_type, model_id) = if let Some(model_path) = &builder.model_path {
            // Load from local path
            Self::load_from_path(
                model_path,
                device,
                context.clone(),
                load_config,
                builder.quiet,
            )
            .await?
        } else {
            // Load from registry
            Self::load_from_registry(
                &builder.model,
                builder.cache_dir.as_deref(),
                device,
                context.clone(),
                load_config,
                builder.download_policy,
                builder.quiet,
            )
            .await?
        };

        // Resolve labels
        let custom_labels = builder.label_config.map(|c| c.labels);

        // Validate label count if custom labels provided
        if let Some(ref labels) = custom_labels {
            let model_labels = inner.labels();
            let expected_count = model_labels.map(|l| l.len()).unwrap_or(0);

            // If model has labels, check count matches
            if expected_count > 0 && labels.len() != expected_count {
                return Err(ClassifierError::InvalidLabels(format!(
                    "Model expects {} labels but {} provided",
                    expected_count,
                    labels.len()
                )));
            }
        }

        Ok(Self {
            inner,
            model_id,
            model_type,
            custom_labels,
            mode: builder.mode,
            default_overrides: builder.overrides,
            device,
            context,
        })
    }

    /// Load classifier from registry.
    async fn load_from_registry(
        model: &str,
        cache_dir: Option<&Path>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<kjarni_transformers::models::base::ModelLoadConfig>,
        download_policy: crate::common::DownloadPolicy,
        quiet: bool,
    ) -> ClassifierResult<(SequenceClassifier, Option<ModelType>, String)> {
        let model_type = ModelType::resolve(model).map_err(ClassifierError::UnknownModel)?;

        // Validate for classification
        validate_for_classification(model_type)?;

        // Ensure downloaded
        let cache_dir_path = cache_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(default_cache_dir);

        ensure_model_downloaded(model_type, Some(&cache_dir_path), download_policy, quiet)
            .await
            .map_err(|e| match e {
                crate::common::KjarniError::ModelNotDownloaded(m) => {
                    ClassifierError::ModelNotDownloaded(m)
                }
                crate::common::KjarniError::DownloadFailed { model, source } => {
                    ClassifierError::DownloadFailed { model, source }
                }
                other => ClassifierError::ClassificationFailed(other.into()),
            })?;

        let inner = SequenceClassifier::from_registry(
            model_type,
            Some(cache_dir_path),
            device,
            context,
            load_config,
        )
        .await
        .map_err(|e| ClassifierError::LoadFailed {
            model: model.to_string(),
            source: e,
        })?;

        Ok((inner, Some(model_type), model.to_string()))
    }

    /// Load classifier from local path.
    async fn load_from_path(
        path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<kjarni_transformers::models::base::ModelLoadConfig>,
        quiet: bool,
    ) -> ClassifierResult<(SequenceClassifier, Option<ModelType>, String)> {
        // Validate path exists
        if !path.exists() {
            return Err(ClassifierError::ModelPathNotFound(
                path.display().to_string(),
            ));
        }

        // Check for required files
        let config_path = path.join("config.json");
        if !config_path.exists() {
            return Err(ClassifierError::InvalidConfig(format!(
                "config.json not found in {}",
                path.display()
            )));
        }

        let tokenizer_path = path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(ClassifierError::InvalidConfig(format!(
                "tokenizer.json not found in {}",
                path.display()
            )));
        }

        if !quiet {
            eprintln!("Loading classifier from {}...", path.display());
        }

        // Detect model architecture from config.json
        let model_type = Self::detect_model_type(path)?;

        let inner = SequenceClassifier::from_pretrained(
            path,
            device,
            context,
            load_config,
            Some(model_type),
        )
        .map_err(|e| ClassifierError::LoadFailed {
            model: path.display().to_string(),
            source: e,
        })?;

        let model_id = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "custom".to_string());

        Ok((inner, Some(model_type), model_id))
    }

    /// Detect model type from config.json.
    fn detect_model_type(path: &Path) -> ClassifierResult<ModelType> {
        let config_path = path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            ClassifierError::InvalidConfig(format!("Failed to read config.json: {}", e))
        })?;

        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| ClassifierError::InvalidConfig(format!("Invalid config.json: {}", e)))?;

        // Check model_type or architectures field
        let model_type_str = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .or_else(|| {
                config
                    .get("architectures")
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|v| v.as_str())
            });

        match model_type_str {
            Some(s) if s.to_lowercase().contains("bert") => {
                // Default to a known BERT classifier type
                // This is a simplification - in reality you'd want more sophisticated detection
                Ok(ModelType::MiniLML6V2CrossEncoder)
            }
            Some(s) => Err(ClassifierError::IncompatibleModel {
                model: path.display().to_string(),
                reason: format!("Unknown model type '{}'. Expected BERT-based model.", s),
            }),
            None => Err(ClassifierError::InvalidConfig(
                "config.json missing 'model_type' or 'architectures' field".to_string(),
            )),
        }
    }

    // =========================================================================
    // Classification Methods
    // =========================================================================

    /// Classify a single text.
    pub async fn classify(&self, text: &str) -> ClassifierResult<ClassificationResult> {
        self.classify_with_config(text, &ClassificationOverrides::default())
            .await
    }

    /// Classify with custom overrides.
    pub async fn classify_with_config(
        &self,
        text: &str,
        overrides: &ClassificationOverrides,
    ) -> ClassifierResult<ClassificationResult> {
        let merged = self.merge_overrides(overrides);
        let top_k = merged.top_k.unwrap_or(usize::MAX);

        // // Get raw scores
        // let raw_scores = self
        //     .inner
        //     .classify_scores(text)
        //     .await
        //     .map_err(ClassifierError::ClassificationFailed)?;

        let scores = match self.mode {
            ClassificationMode::SingleLabel => self
                .inner
                .classify_scores(text)
                .await
                .map_err(ClassifierError::ClassificationFailed)?,
            ClassificationMode::MultiLabel => {
                // Get raw logits, apply sigmoid
                let logits = self
                    .inner
                    .predict_logits(&[text])
                    .await
                    .map_err(ClassifierError::ClassificationFailed)?;
                logits
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        ClassifierError::ClassificationFailed(anyhow::anyhow!("No results"))
                    })?
                    .iter()
                    .map(|&x| sigmoid(x))
                    .collect()
            }
        };

        // Get labels
        let labels = self.get_labels()?;

        // Build result with labels
        let result = ClassificationResult::from_scores_with_labels(&scores, &labels)
            .ok_or_else(|| ClassifierError::ClassificationFailed(anyhow::anyhow!("No results")))?;

        // Apply threshold filter and top_k
        let filtered_scores: Vec<(String, f32)> = result
            .all_scores
            .iter()
            .filter(|(_, score)| merged.threshold.map(|t| *score >= t).unwrap_or(true))
            .take(top_k)
            .cloned()
            .collect();

        if filtered_scores.is_empty() {
            return Err(ClassifierError::ClassificationFailed(anyhow::anyhow!(
                "No results after filtering"
            )));
        }

        // Preserve original label_index if top label didn't change
        let label_index = if filtered_scores[0].0 == result.label {
            result.label_index
        } else {
            // Find new index - but we've lost it. Need different approach.
            0
        };

        Ok(ClassificationResult {
            label: filtered_scores[0].0.clone(),
            score: filtered_scores[0].1,
            all_scores: filtered_scores,
            label_index,
        })
    }

    /// Classify multiple texts.
    pub async fn classify_batch(
        &self,
        texts: &[&str],
    ) -> ClassifierResult<Vec<ClassificationResult>> {
        self.classify_batch_with_config(texts, &ClassificationOverrides::default())
            .await
    }

    /// Classify multiple texts with custom overrides.
    pub async fn classify_batch_with_config(
        &self,
        texts: &[&str],
        overrides: &ClassificationOverrides,
    ) -> ClassifierResult<Vec<ClassificationResult>> {
        let merged = self.merge_overrides(overrides);
        let top_k = merged.top_k.unwrap_or(usize::MAX);
        let labels = self.get_labels()?;

        let all_scores = self
            .inner
            .classify_batch(texts, usize::MAX)
            .await
            .map_err(ClassifierError::ClassificationFailed)?;

        let results: Vec<ClassificationResult> = all_scores
            .into_iter()
            .filter_map(|label_scores| {
                // Convert from (label, score) to just scores in label order
                let scores: Vec<f32> = labels
                    .iter()
                    .map(|l| {
                        label_scores
                            .iter()
                            .find(|(name, _)| name == l)
                            .map(|(_, s)| *s)
                            .unwrap_or(0.0)
                    })
                    .collect();

                // Apply mode
                let final_scores = match self.mode {
                    ClassificationMode::SingleLabel => scores,
                    ClassificationMode::MultiLabel => scores.iter().map(|&x| sigmoid(x)).collect(),
                };

                let result = ClassificationResult::from_scores_with_labels(&final_scores, &labels)?;

                // Apply threshold and top_k
                let filtered: Vec<(String, f32)> = result
                    .all_scores
                    .iter()
                    .filter(|(_, score)| merged.threshold.map(|t| *score >= t).unwrap_or(true))
                    .take(top_k)
                    .cloned()
                    .collect();

                ClassificationResult::from_scores(filtered)
            })
            .collect();

        Ok(results)
    }

    /// Get raw classification scores without labels.
    pub async fn classify_scores(&self, text: &str) -> ClassifierResult<Vec<f32>> {
        let scores = self
            .inner
            .classify_scores(text)
            .await
            .map_err(ClassifierError::ClassificationFailed)?;

        Ok(match self.mode {
            ClassificationMode::SingleLabel => scores,
            ClassificationMode::MultiLabel => scores.iter().map(|&x| sigmoid(x)).collect(),
        })
    }

    // =========================================================================
    // Label Management
    // =========================================================================

    /// Get the effective labels (custom or model).
    fn get_labels(&self) -> ClassifierResult<Vec<String>> {
        // Custom labels take precedence
        if let Some(ref labels) = self.custom_labels {
            return Ok(labels.clone());
        }

        // Fall back to model labels
        self.inner
            .labels()
            .map(|l| l.to_vec())
            .ok_or(ClassifierError::NoLabels)
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the model identifier.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the model type (None if loaded from path).
    pub fn model_type(&self) -> Option<ModelType> {
        self.model_type
    }

    /// Get the model's CLI name (if from registry).
    pub fn model_name(&self) -> &str {
        self.model_type
            .map(|t| t.cli_name())
            .unwrap_or(&self.model_id)
    }

    pub fn architecture(&self) -> &str {
        self.inner.config().model_type()
    }

    /// Get the device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get the label names.
    pub fn labels(&self) -> Option<Vec<String>> {
        self.get_labels().ok()
    }

    /// Get the number of labels.
    pub fn num_labels(&self) -> usize {
        self.get_labels().map(|l| l.len()).unwrap_or(0)
    }

    /// Get maximum sequence length.
    pub fn max_seq_length(&self) -> usize {
        self.inner.max_seq_length()
    }

    /// Get classification mode.
    pub fn mode(&self) -> ClassificationMode {
        self.mode
    }

    /// Check if using custom labels.
    pub fn has_custom_labels(&self) -> bool {
        self.custom_labels.is_some()
    }

    // =========================================================================
    // Internal
    // =========================================================================

    fn merge_overrides(&self, runtime: &ClassificationOverrides) -> ClassificationOverrides {
        ClassificationOverrides {
            top_k: runtime.top_k.or(self.default_overrides.top_k),
            threshold: runtime.threshold.or(self.default_overrides.threshold),
            return_logits: runtime.return_logits || self.default_overrides.return_logits,
            max_length: runtime.max_length.or(self.default_overrides.max_length),
            batch_size: runtime.batch_size.or(self.default_overrides.batch_size),
        }
    }
}

/// Sigmoid activation for multi-label classification.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
