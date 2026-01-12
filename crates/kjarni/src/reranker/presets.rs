//! Versioned presets for reranking models.

use crate::common::KjarniDevice;

/// A reranking model preset.
#[derive(Debug, Clone)]
pub struct RerankerPreset {
    /// Preset name.
    pub name: &'static str,

    /// CLI model name.
    pub model: &'static str,

    /// Maximum sequence length (query + document).
    pub max_seq_length: usize,

    /// Recommended device.
    pub recommended_device: KjarniDevice,

    /// Approximate memory in MB.
    pub memory_mb: usize,

    /// Description.
    pub description: &'static str,
}

// =============================================================================
// V1 Presets
// =============================================================================

/// MiniLM cross-encoder - small and fast.
pub const RERANKER_MINILM_V1: RerankerPreset = RerankerPreset {
    name: "RERANKER_MINILM_V1",
    model: "minilm-l6-v2-cross-encoder",
    max_seq_length: 512,
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 90,
    description: "Fast, lightweight cross-encoder for reranking",
};

/// MS MARCO MiniLM - optimized for passage retrieval.
pub const RERANKER_MSMARCO_V1: RerankerPreset = RerankerPreset {
    name: "RERANKER_MSMARCO_V1",
    model: "ms-marco-minilm-l-12-v2",
    max_seq_length: 512,
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 130,
    description: "MS MARCO trained cross-encoder for passage reranking",
};

/// All V1 presets.
pub const ALL_V1_PRESETS: &[&RerankerPreset] = &[
    &RERANKER_MINILM_V1,
    &RERANKER_MSMARCO_V1,
];

/// Find a preset by name.
pub fn find_preset(name: &str) -> Option<&'static RerankerPreset> {
    let name_upper = name.to_uppercase();
    ALL_V1_PRESETS
        .iter()
        .find(|p| p.name == name_upper)
        .copied()
}

/// Tier-based preset selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankerTier {
    /// Smallest, fastest.
    Fast,

    /// Balanced quality/speed.
    Balanced,

    /// Highest quality.
    Quality,
}

impl RerankerTier {
    /// Resolve tier to default preset.
    pub fn resolve(&self) -> &'static RerankerPreset {
        match self {
            Self::Fast => &RERANKER_MINILM_V1,
            Self::Balanced => &RERANKER_MSMARCO_V1,
            Self::Quality => &RERANKER_MSMARCO_V1,
        }
    }
}

impl Default for RerankerTier {
    fn default() -> Self {
        Self::Balanced
    }
}