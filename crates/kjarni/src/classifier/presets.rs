//! Versioned presets for classification models.

use crate::common::KjarniDevice;

/// A classification model preset.
#[derive(Debug, Clone)]
pub struct ClassifierPreset {
    /// Preset name.
    pub name: &'static str,

    /// CLI model name.
    pub model: &'static str,

    /// Classification task type.
    pub task: ClassificationTask,

    /// Expected label names (if known).
    pub labels: Option<&'static [&'static str]>,

    /// Recommended device.
    pub recommended_device: KjarniDevice,

    /// Approximate memory usage in MB.
    pub memory_mb: usize,

    /// Description.
    pub description: &'static str,
}

/// Type of classification task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationTask {
    Sentiment,
    Topic,
    Intent,
    ZeroShot,
    Custom,
}

// =============================================================================
// V1 Presets
// =============================================================================

/// Sentiment analysis (positive/negative).
pub const SENTIMENT_V1: ClassifierPreset = ClassifierPreset {
    name: "SENTIMENT_V1",
    model: "minilm-l6-v2-cross-encoder",
    task: ClassificationTask::Sentiment,
    labels: Some(&["negative", "positive"]),
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 100,
    description: "Fast sentiment analysis",
};

// Add more presets as models become available

/// All V1 presets.
pub const ALL_V1_PRESETS: &[&ClassifierPreset] = &[
    &SENTIMENT_V1,
];

/// Find a preset by name.
pub fn find_preset(name: &str) -> Option<&'static ClassifierPreset> {
    let name_upper = name.to_uppercase();
    ALL_V1_PRESETS
        .iter()
        .find(|p| p.name == name_upper)
        .copied()
}

/// Tier-based preset selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierTier {
    /// Fast, small model.
    Fast,
    /// Balanced accuracy/speed.
    Balanced,
    /// Most accurate.
    Accurate,
}

impl ClassifierTier {
    /// Resolve tier to default preset.
    pub fn resolve(&self) -> &'static ClassifierPreset {
        match self {
            Self::Fast => &SENTIMENT_V1,
            Self::Balanced => &SENTIMENT_V1,
            Self::Accurate => &SENTIMENT_V1,
        }
    }
}