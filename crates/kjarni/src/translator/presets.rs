//! Versioned presets for translation models.

use crate::common::KjarniDevice;

/// A translation model preset.
#[derive(Debug, Clone)]
pub struct TranslatorPreset {
    /// Preset name.
    pub name: &'static str,
    /// CLI model name.
    pub model: &'static str,
    /// Model architecture.
    pub architecture: &'static str,
    /// Languages known to work well.
    pub supported_languages: &'static [&'static str],
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

/// Fast translation with FLAN-T5 Base.
pub const TRANSLATION_FAST_V1: TranslatorPreset = TranslatorPreset {
    name: "TRANSLATION_FAST_V1",
    model: "flan-t5-base",
    architecture: "t5",
    supported_languages: &[
        "English", "German", "French", "Spanish", "Italian", "Portuguese",
    ],
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 990,
    description: "Fast general-purpose translation",
};

/// High-quality translation with FLAN-T5 Large.
pub const TRANSLATION_QUALITY_V1: TranslatorPreset = TranslatorPreset {
    name: "TRANSLATION_QUALITY_V1",
    model: "flan-t5-large",
    architecture: "t5",
    supported_languages: &[
        "English", "German", "French", "Spanish", "Italian", "Portuguese",
        "Russian", "Chinese", "Japanese", "Korean", "Arabic",
    ],
    recommended_device: KjarniDevice::Gpu,
    memory_mb: 3000,
    description: "High-quality translation with more languages",
};

/// All V1 presets.
pub const ALL_V1_PRESETS: &[&TranslatorPreset] = &[
    &TRANSLATION_FAST_V1,
    &TRANSLATION_QUALITY_V1,
];

/// Find a preset by name.
pub fn find_preset(name: &str) -> Option<&'static TranslatorPreset> {
    let name_upper = name.to_uppercase();
    ALL_V1_PRESETS
        .iter()
        .find(|p| p.name == name_upper)
        .copied()
}

// =============================================================================
// Tiers
// =============================================================================

/// Tier-based preset selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslatorTier {
    /// Fast, smaller model.
    Fast,
    /// Higher quality, larger model.
    Quality,
}

impl TranslatorTier {
    /// Resolve tier to default preset.
    pub fn resolve(&self) -> &'static TranslatorPreset {
        match self {
            Self::Fast => &TRANSLATION_FAST_V1,
            Self::Quality => &TRANSLATION_QUALITY_V1,
        }
    }
}