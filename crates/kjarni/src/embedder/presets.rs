//! Versioned presets for embedding models.

use kjarni_transformers::PoolingStrategy;

use crate::common::KjarniDevice;

/// An embedding model preset.
#[derive(Debug, Clone)]
pub struct EmbedderPreset {
    /// Preset name.
    pub name: &'static str,

    /// CLI model name.
    pub model: &'static str,

    /// Embedding dimension.
    pub dimension: usize,

    /// Default pooling strategy.
    pub default_pooling: PoolingStrategy,

    /// Whether to normalize by default.
    pub normalize_default: bool,

    /// Recommended device.
    pub recommended_device: KjarniDevice,

    /// Approximate memory in MB.
    pub memory_mb: usize,

    /// Description.
    pub description: &'static str,
}


pub const EMBEDDING_SMALL_V1: EmbedderPreset = EmbedderPreset {
    name: "EMBEDDING_SMALL_V1",
    model: "minilm-l6-v2",
    dimension: 384,
    default_pooling: PoolingStrategy::Mean,
    normalize_default: true,
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 90,
    description: "Fast, lightweight embeddings",
};

/// Nomic embedding model - good quality/speed tradeoff.
pub const EMBEDDING_NOMIC_V1: EmbedderPreset = EmbedderPreset {
    name: "EMBEDDING_NOMIC_V1",
    model: "nomic-embed-text",
    dimension: 768,
    default_pooling: PoolingStrategy::Mean,
    normalize_default: true,
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 300,
    description: "High quality general-purpose embeddings",
};

/// All V1 presets.
pub const ALL_V1_PRESETS: &[&EmbedderPreset] = &[
    &EMBEDDING_SMALL_V1,
    &EMBEDDING_NOMIC_V1,
];

/// Find a preset by name.
pub fn find_preset(name: &str) -> Option<&'static EmbedderPreset> {
    let name_upper = name.to_uppercase();
    ALL_V1_PRESETS
        .iter()
        .find(|p| p.name == name_upper)
        .copied()
}

/// Tier-based preset selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedderTier {
    /// Smallest, fastest.
    Small,
    /// Balanced.
    Medium,
    /// Highest quality.
    Large,
}

impl EmbedderTier {
    /// Resolve tier to default preset.
    pub fn resolve(&self) -> &'static EmbedderPreset {
        match self {
            Self::Small => &EMBEDDING_SMALL_V1,
            Self::Medium => &EMBEDDING_NOMIC_V1,
            Self::Large => &EMBEDDING_NOMIC_V1,
        }
    }
}