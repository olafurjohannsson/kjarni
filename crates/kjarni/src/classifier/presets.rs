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
    Emotion,
    ZeroShot,
    Toxicity,
    Topic,
    Intent,
    Custom,
}


// =============================================================================
// V1 Presets - Sentiment
// =============================================================================

/// Binary sentiment (positive/negative) - fastest
pub const SENTIMENT_BINARY_V1: ClassifierPreset = ClassifierPreset {
    name: "SENTIMENT_BINARY_V1",
    model: "distilbert-sentiment",
    task: ClassificationTask::Sentiment,
    labels: Some(&["NEGATIVE", "POSITIVE"]),
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 268,
    description: "Fast binary sentiment (positive/negative)",
};

/// 3-class sentiment (negative/neutral/positive)
pub const SENTIMENT_3CLASS_V1: ClassifierPreset = ClassifierPreset {
    name: "SENTIMENT_3CLASS_V1",
    model: "roberta-sentiment",
    task: ClassificationTask::Sentiment,
    labels: Some(&["negative", "neutral", "positive"]),
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 499,
    description: "3-class sentiment, optimized for social media",
};

/// 5-star sentiment (multilingual)
pub const SENTIMENT_5STAR_V1: ClassifierPreset = ClassifierPreset {
    name: "SENTIMENT_5STAR_V1",
    model: "bert-sentiment-multilingual",
    task: ClassificationTask::Sentiment,
    labels: Some(&["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]),
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 681,
    description: "5-star ratings, multilingual (EN/DE/FR/ES/IT/NL)",
};

// =============================================================================
// V1 Presets - Zero-Shot
// =============================================================================

/// Zero-shot classification (large, most capable)
pub const ZEROSHOT_LARGE_V1: ClassifierPreset = ClassifierPreset {
    name: "ZEROSHOT_LARGE_V1",
    model: "bart-zeroshot",
    task: ClassificationTask::ZeroShot,
    labels: None, // User provides at runtime
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 1630,
    description: "Zero-shot classifier, classify into any labels",
};

/// Zero-shot classification (smaller, faster)
pub const ZEROSHOT_BASE_V1: ClassifierPreset = ClassifierPreset {
    name: "ZEROSHOT_BASE_V1",
    model: "deberta-zeroshot",
    task: ClassificationTask::ZeroShot,
    labels: None,
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 738,
    description: "Zero-shot classifier, smaller and faster than BART",
};

// =============================================================================
// V1 Presets - Emotion
// =============================================================================

/// 7 basic emotions
pub const EMOTION_V1: ClassifierPreset = ClassifierPreset {
    name: "EMOTION_V1",
    model: "distilroberta-emotion",
    task: ClassificationTask::Emotion,
    labels: Some(&["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]),
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 329,
    description: "7 basic emotions",
};

/// 28 fine-grained emotions (multi-label)
pub const EMOTION_DETAILED_V1: ClassifierPreset = ClassifierPreset {
    name: "EMOTION_DETAILED_V1",
    model: "roberta-emotions",
    task: ClassificationTask::Emotion,
    labels: Some(&[
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral"
    ]),
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 499,
    description: "28 fine-grained emotions (multi-label)",
};

// =============================================================================
// V1 Presets - Toxicity
// =============================================================================

/// Toxicity detection (multi-label)
pub const TOXICITY_V1: ClassifierPreset = ClassifierPreset {
    name: "TOXICITY_V1",
    model: "toxic-bert",
    task: ClassificationTask::Toxicity,
    labels: Some(&["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]),
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 438,
    description: "Multi-label toxicity detection",
};

// =============================================================================
// Preset Collections
// =============================================================================

/// All V1 presets.
pub const ALL_V1_PRESETS: &[&ClassifierPreset] = &[
    // Sentiment
    &SENTIMENT_BINARY_V1,
    &SENTIMENT_3CLASS_V1,
    &SENTIMENT_5STAR_V1,
    // Zero-Shot
    &ZEROSHOT_LARGE_V1,
    &ZEROSHOT_BASE_V1,
    // Emotion
    &EMOTION_V1,
    &EMOTION_DETAILED_V1,
    // Toxicity
    &TOXICITY_V1,
];

/// Find a preset by name.
pub fn find_preset(name: &str) -> Option<&'static ClassifierPreset> {
    let name_upper = name.to_uppercase();
    ALL_V1_PRESETS
        .iter()
        .find(|p| p.name == name_upper)
        .copied()
}

// =============================================================================
// Tier-Based Selection
// =============================================================================

/// Tier-based preset selection for sentiment analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentimentTier {
    /// Fastest, binary (positive/negative)
    Fast,
    /// Balanced, 3-class (negative/neutral/positive)  
    Balanced,
    /// Most detailed, 5-star multilingual
    Detailed,
}

impl SentimentTier {
    /// Resolve tier to default preset.
    pub fn resolve(&self) -> &'static ClassifierPreset {
        match self {
            Self::Fast => &SENTIMENT_BINARY_V1,
            Self::Balanced => &SENTIMENT_3CLASS_V1,
            Self::Detailed => &SENTIMENT_5STAR_V1,
        }
    }
}

/// Tier-based preset selection for zero-shot classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroShotTier {
    /// Faster, smaller model
    Fast,
    /// More accurate, larger model
    Accurate,
}

impl ZeroShotTier {
    /// Resolve tier to default preset.
    pub fn resolve(&self) -> &'static ClassifierPreset {
        match self {
            Self::Fast => &ZEROSHOT_BASE_V1,
            Self::Accurate => &ZEROSHOT_LARGE_V1,
        }
    }
}

/// Tier-based preset selection for emotion detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmotionTier {
    /// 7 basic emotions
    Basic,
    /// 28 fine-grained emotions
    Detailed,
}

impl EmotionTier {
    /// Resolve tier to default preset.
    pub fn resolve(&self) -> &'static ClassifierPreset {
        match self {
            Self::Basic => &EMOTION_V1,
            Self::Detailed => &EMOTION_DETAILED_V1,
        }
    }
}

/// General classifier tier (for when task is unknown).
/// Defaults to sentiment as the most common use case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierTier {
    /// Fastest model
    Fast,
    /// Balanced speed/accuracy
    Balanced,
    /// Most accurate
    Accurate,
}

impl ClassifierTier {
    /// Resolve tier to default preset (defaults to sentiment).
    pub fn resolve(&self) -> &'static ClassifierPreset {
        match self {
            Self::Fast => &SENTIMENT_BINARY_V1,
            Self::Balanced => &SENTIMENT_3CLASS_V1,
            Self::Accurate => &ZEROSHOT_LARGE_V1,
        }
    }
}
