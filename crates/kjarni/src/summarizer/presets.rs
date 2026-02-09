use crate::common::KjarniDevice;

#[derive(Debug, Clone)]
pub struct SummarizerPreset {
    pub name: &'static str,
    pub model: &'static str,
    pub architecture: &'static str,
    pub default_min_length: usize,
    pub default_max_length: usize,
    pub recommended_device: KjarniDevice,
    pub memory_mb: usize,
    pub description: &'static str,
}


// V1 Presets


/// Fast summarization with DistilBART.
pub const SUMMARIZER_FAST_V1: SummarizerPreset = SummarizerPreset {
    name: "SUMMARIZER_FAST_V1",
    model: "distilbart-cnn",
    architecture: "bart",
    default_min_length: 30,
    default_max_length: 130,
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 1000,
    description: "Fast news/article summarization",
};

/// High-quality summarization with BART Large.
pub const SUMMARIZER_QUALITY_V1: SummarizerPreset = SummarizerPreset {
    name: "SUMMARIZER_QUALITY_V1",
    model: "bart-large-cnn",
    architecture: "bart",
    default_min_length: 50,
    default_max_length: 200,
    recommended_device: KjarniDevice::Gpu,
    memory_mb: 1600,
    description: "High-quality summarization",
};

/// General-purpose with FLAN-T5.
pub const SUMMARIZER_T5_V1: SummarizerPreset = SummarizerPreset {
    name: "SUMMARIZER_T5_V1",
    model: "flan-t5-base",
    architecture: "t5",
    default_min_length: 30,
    default_max_length: 150,
    recommended_device: KjarniDevice::Cpu,
    memory_mb: 990,
    description: "Flexible T5-based summarization",
};

pub const ALL_V1_PRESETS: &[&SummarizerPreset] = &[
    &SUMMARIZER_FAST_V1,
    &SUMMARIZER_QUALITY_V1,
    &SUMMARIZER_T5_V1,
];

/// Find a preset by name.
pub fn find_preset(name: &str) -> Option<&'static SummarizerPreset> {
    let name_upper = name.to_uppercase();
    ALL_V1_PRESETS
        .iter()
        .find(|p| p.name == name_upper)
        .copied()
}


// Tiers


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummarizerTier {
    Fast,
    Balanced,
    Quality,
}

impl SummarizerTier {
    pub fn resolve(&self) -> &'static SummarizerPreset {
        match self {
            Self::Fast => &SUMMARIZER_FAST_V1,
            Self::Balanced => &SUMMARIZER_T5_V1,
            Self::Quality => &SUMMARIZER_QUALITY_V1,
        }
    }
}