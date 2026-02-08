//! Types for the classifier module.

use std::fmt;
use thiserror::Error;

/// Errors specific to classification.
#[derive(Debug, Error)]
pub enum ClassifierError {
    /// Model not found in registry.
    #[error("{0}")]
    UnknownModel(String),

    /// Model cannot perform classification.
    #[error("Model '{model}' cannot be used for classification: {reason}")]
    IncompatibleModel { model: String, reason: String },

    /// Model not downloaded.
    #[error("Model '{0}' not downloaded and download policy is set to Never")]
    ModelNotDownloaded(String),

    /// Model path not found.
    #[error("Model path not found: {0}")]
    ModelPathNotFound(String),

    /// Download failed.
    #[error("Failed to download model '{model}': {source}")]
    DownloadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Load failed.
    #[error("Failed to load model '{model}': {source}")]
    LoadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Classification failed.
    #[error("Classification failed: {0}")]
    ClassificationFailed(#[from] anyhow::Error),

    /// GPU unavailable.
    #[error("GPU requested but WebGPU context could not be created")]
    GpuUnavailable,

    /// No labels configured.
    #[error("Model has no label mapping. Provide labels via --labels or use classify_scores().")]
    NoLabels,

    /// Invalid label configuration.
    #[error("Invalid label configuration: {0}")]
    InvalidLabels(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for classifier operations.
pub type ClassifierResult<T> = Result<T, ClassifierError>;

/// Result of classifying a single text.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// The predicted label (highest score).
    pub label: String,

    /// Confidence score for the predicted label (0.0 - 1.0).
    pub score: f32,

    /// All labels with their scores, sorted by score descending.
    pub all_scores: Vec<(String, f32)>,

    /// Index of the predicted label in the original label list.
    pub label_index: usize,
}

impl ClassificationResult {
    /// Create from a list of (label, score) pairs.
    pub fn from_scores(mut scores: Vec<(String, f32)>) -> Option<Self> {
        if scores.is_empty() {
            return None;
        }

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (label, score) = scores[0].clone();

        // Find original index (before sorting)
        let label_index = 0; // After sorting, top is at 0

        Some(Self {
            label,
            score,
            all_scores: scores,
            label_index,
        })
    }

    /// Create from scores with explicit labels.
    pub fn from_scores_with_labels(scores: &[f32], labels: &[String]) -> Option<Self> {
        if scores.is_empty() || labels.is_empty() || scores.len() != labels.len() {
            return None;
        }

        let mut indexed: Vec<(usize, String, f32)> = labels
            .iter()
            .zip(scores.iter())
            .enumerate()
            .map(|(i, (label, &score))| (i, label.clone(), score))
            .collect();

        // Sort by score descending
        indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let (label_index, label, score) = indexed[0].clone();
        let all_scores: Vec<(String, f32)> =
            indexed.iter().map(|(_, l, s)| (l.clone(), *s)).collect();

        Some(Self {
            label,
            score,
            all_scores,
            label_index,
        })
    }

    /// Get the top K predictions.
    pub fn top_k(&self, k: usize) -> Vec<(String, f32)> {
        self.all_scores.iter().take(k).cloned().collect()
    }

    /// Check if the top prediction exceeds a confidence threshold.
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.score >= threshold
    }

    /// Get predictions above a threshold (for multi-label scenarios).
    pub fn above_threshold(&self, threshold: f32) -> Vec<(String, f32)> {
        self.all_scores
            .iter()
            .filter(|(_, score)| *score >= threshold)
            .cloned()
            .collect()
    }
}

impl fmt::Display for ClassificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({:.1}%)", self.label, self.score * 100.0)
    }
}

/// Classification mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClassificationMode {
    /// Single-label classification (softmax, mutually exclusive).
    #[default]
    SingleLabel,

    /// Multi-label classification (sigmoid, independent labels).
    MultiLabel,
}

/// Overrides for classification behavior.
#[derive(Debug, Clone, Default)]
pub struct ClassificationOverrides {
    /// Return only top K results (default: all).
    pub top_k: Option<usize>,

    /// Minimum confidence threshold (0.0-1.0).
    /// Results below this threshold are filtered out.
    pub threshold: Option<f32>,

    /// Return raw logits instead of probabilities.
    pub return_logits: bool,

    /// Maximum sequence length (truncates input).
    pub max_length: Option<usize>,

    /// Batch size for inference (memory vs speed tradeoff).
    pub batch_size: Option<usize>,
}

impl ClassificationOverrides {
    /// Create overrides that return only the top result.
    pub fn top_1() -> Self {
        Self {
            top_k: Some(1),
            ..Default::default()
        }
    }

    /// Create overrides with a confidence threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            threshold: Some(threshold),
            ..Default::default()
        }
    }

    /// Create overrides for multi-label with threshold.
    pub fn multi_label(threshold: f32) -> Self {
        Self {
            threshold: Some(threshold),
            top_k: None, // Return all above threshold
            ..Default::default()
        }
    }
}

/// Label configuration for custom models.
#[derive(Debug, Clone, Default)]
pub struct LabelConfig {
    /// Ordered list of label names.
    /// Index 0 corresponds to output index 0, etc.
    pub labels: Vec<String>,

    /// Optional mapping from label name to display name.
    pub display_names: Option<std::collections::HashMap<String, String>>,
}

impl LabelConfig {
    /// Create from a list of labels.
    pub fn new(labels: Vec<String>) -> Self {
        Self {
            labels,
            display_names: None,
        }
    }

    /// Create from a comma-separated string.
    pub fn from_str(s: &str) -> Self {
        let labels: Vec<String> = s
            .split(',')
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        Self::new(labels)
    }

    /// Common sentiment labels.
    pub fn sentiment() -> Self {
        Self::new(vec!["negative".to_string(), "positive".to_string()])
    }

    /// Common sentiment labels (3-class).
    pub fn sentiment_3class() -> Self {
        Self::new(vec![
            "negative".to_string(),
            "neutral".to_string(),
            "positive".to_string(),
        ])
    }

    /// Validate the label count matches expected.
    pub fn validate(&self, expected_count: usize) -> Result<(), ClassifierError> {
        if self.labels.len() != expected_count {
            return Err(ClassifierError::InvalidLabels(format!(
                "Expected {} labels but got {}",
                expected_count,
                self.labels.len()
            )));
        }
        Ok(())
    }

    /// Get display name for a label.
    pub fn display_name<'a>(&'a self, label: &'a str) -> &'a str {
        self.display_names
            .as_ref()
            .and_then(|m| m.get(label))
            .map(|s| s.as_str())
            .unwrap_or(label)
    }
}
