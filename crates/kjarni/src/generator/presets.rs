
//! Generator presets for common use cases.

use crate::common::KjarniDevice;

/// Preset configuration for generator models.
#[derive(Debug, Clone)]
pub struct GeneratorPreset {
    /// Preset name for lookup.
    pub name: &'static str,
    /// Model name from registry.
    pub model: &'static str,
    /// Model architecture.
    pub architecture: &'static str,
    /// Recommended device.
    pub recommended_device: KjarniDevice,
    /// Recommended temperature.
    pub temperature: Option<f32>,
    /// Default max new tokens.
    pub default_max_tokens: usize,
    /// Approximate memory usage in MB.
    pub memory_mb: usize,
    /// Human-readable description.
    pub description: &'static str,
}


// V1 Presets


/// Fast text generation - smallest model, quick responses.
pub const GENERATOR_FAST_V1: GeneratorPreset = GeneratorPreset {
    name: "GENERATOR_FAST_V1",
    model: "qwen2.5-0.5b",
    architecture: "qwen2",
    recommended_device: KjarniDevice::Cpu,
    temperature: Some(0.7),
    default_max_tokens: 256,
    memory_mb: 500,
    description: "Fast text generation with small model.",
};

/// Balanced text generation - good quality with reasonable speed.
pub const GENERATOR_BALANCED_V1: GeneratorPreset = GeneratorPreset {
    name: "GENERATOR_BALANCED_V1",
    model: "qwen2.5-1.5b",
    architecture: "qwen2",
    recommended_device: KjarniDevice::Cpu,
    temperature: Some(0.7),
    default_max_tokens: 512,
    memory_mb: 1500,
    description: "Balanced speed and quality.",
};

/// Quality text generation - larger model, better outputs.
pub const GENERATOR_QUALITY_V1: GeneratorPreset = GeneratorPreset {
    name: "GENERATOR_QUALITY_V1",
    model: "llama3.2-1b",
    architecture: "llama",
    recommended_device: KjarniDevice::Cpu,
    temperature: Some(0.7),
    default_max_tokens: 1024,
    memory_mb: 1000,
    description: "Higher quality text generation.",
};

/// Creative text generation - higher temperature for variety.
pub const GENERATOR_CREATIVE_V1: GeneratorPreset = GeneratorPreset {
    name: "GENERATOR_CREATIVE_V1",
    model: "llama3.2-1b",
    architecture: "llama",
    recommended_device: KjarniDevice::Cpu,
    temperature: Some(0.9),
    default_max_tokens: 1024,
    memory_mb: 1000,
    description: "Creative generation with higher temperature.",
};

/// Code generation preset.
pub const GENERATOR_CODE_V1: GeneratorPreset = GeneratorPreset {
    name: "GENERATOR_CODE_V1",
    model: "qwen2.5-1.5b",
    architecture: "qwen2",
    recommended_device: KjarniDevice::Cpu,
    temperature: Some(0.2),
    default_max_tokens: 2048,
    memory_mb: 1500,
    description: "Code generation with low temperature for precision.",
};

/// All V1 presets for iteration.
pub const ALL_V1_PRESETS: &[&GeneratorPreset] = &[
    &GENERATOR_FAST_V1,
    &GENERATOR_BALANCED_V1,
    &GENERATOR_QUALITY_V1,
    &GENERATOR_CREATIVE_V1,
    &GENERATOR_CODE_V1,
];

impl GeneratorPreset {
    /// GPT-2 for text completion.
    pub const GPT2: GeneratorPreset = GeneratorPreset {
        name: "GPT2",
        model: "gpt2",
        architecture: "gpt",
        recommended_device: KjarniDevice::Cpu,
        temperature: Some(0.7),
        default_max_tokens: 100,
        memory_mb: 500,
        description: "Classic GPT-2 for text completion.",
    };

    /// Fast text completion (alias for GENERATOR_FAST_V1).
    pub const FAST: GeneratorPreset = GeneratorPreset {
        name: "FAST",
        model: "qwen2.5-0.5b",
        architecture: "qwen2",
        recommended_device: KjarniDevice::Cpu,
        temperature: Some(0.5),
        default_max_tokens: 128,
        memory_mb: 500,
        description: "Fast text generation.",
    };

    /// Quality text completion (alias for GENERATOR_QUALITY_V1).
    pub const QUALITY: GeneratorPreset = GeneratorPreset {
        name: "QUALITY",
        model: "llama3.2-1b",
        architecture: "llama",
        recommended_device: KjarniDevice::Cpu,
        temperature: Some(0.6),
        default_max_tokens: 256,
        memory_mb: 1000,
        description: "Better quality text generation.",
    };
}

/// Generator quality tiers for easy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeneratorTier {
    /// Fastest, smallest model.
    Fast,
    /// Balanced speed and quality.
    #[default]
    Balanced,
    /// Best quality, larger model.
    Quality,
}

impl GeneratorTier {
    /// Get the preset for this tier.
    pub fn resolve(&self) -> &'static GeneratorPreset {
        match self {
            Self::Fast => &GENERATOR_FAST_V1,
            Self::Balanced => &GENERATOR_BALANCED_V1,
            Self::Quality => &GENERATOR_QUALITY_V1,
        }
    }

    /// Get the preset (alias for resolve).
    pub fn preset(&self) -> &'static GeneratorPreset {
        self.resolve()
    }
}

/// Find a preset by name (case-insensitive).
pub fn find_preset(name: &str) -> Option<&'static GeneratorPreset> {
    let name_lower = name.to_lowercase();
    
    ALL_V1_PRESETS
        .iter()
        .find(|p| p.name.to_lowercase() == name_lower)
        .copied()
}

/// List all available preset names.
pub fn list_presets() -> Vec<&'static str> {
    ALL_V1_PRESETS.iter().map(|p| p.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_fast() {
        assert!(!GENERATOR_FAST_V1.model.is_empty());
        assert!(GENERATOR_FAST_V1.memory_mb > 0);
        assert!(GENERATOR_FAST_V1.default_max_tokens > 0);
        assert!(!GENERATOR_FAST_V1.name.is_empty());
        assert!(!GENERATOR_FAST_V1.architecture.is_empty());
    }

    #[test]
    fn test_preset_quality() {
        assert!(!GENERATOR_QUALITY_V1.model.is_empty());
        assert!(GENERATOR_QUALITY_V1.memory_mb >= GENERATOR_FAST_V1.memory_mb);
        assert!(GENERATOR_QUALITY_V1.default_max_tokens >= GENERATOR_FAST_V1.default_max_tokens);
    }

    #[test]
    fn test_preset_balanced() {
        assert!(!GENERATOR_BALANCED_V1.model.is_empty());
        assert!(GENERATOR_BALANCED_V1.memory_mb >= GENERATOR_FAST_V1.memory_mb);
    }

    #[test]
    fn test_preset_creative() {
        assert!(!GENERATOR_CREATIVE_V1.model.is_empty());
        assert!(GENERATOR_CREATIVE_V1.temperature.unwrap() > 0.8);
    }

    #[test]
    fn test_preset_code() {
        assert!(!GENERATOR_CODE_V1.model.is_empty());
        assert!(GENERATOR_CODE_V1.temperature.unwrap() < 0.5);
        assert!(GENERATOR_CODE_V1.default_max_tokens >= 1024);
    }

    #[test]
    fn test_find_preset() {
        assert!(find_preset("GENERATOR_FAST_V1").is_some());
        assert!(find_preset("generator_fast_v1").is_some());
        assert!(find_preset("GENERATOR_QUALITY_V1").is_some());
        assert!(find_preset("generator_balanced_v1").is_some());
        assert!(find_preset("nonexistent").is_none());
    }

    #[test]
    fn test_tier_resolution() {
        let fast = GeneratorTier::Fast.resolve();
        let balanced = GeneratorTier::Balanced.resolve();
        let quality = GeneratorTier::Quality.resolve();
        
        assert!(!fast.model.is_empty());
        assert!(!balanced.model.is_empty());
        assert!(!quality.model.is_empty());
        
        assert_eq!(fast.name, "GENERATOR_FAST_V1");
        assert_eq!(balanced.name, "GENERATOR_BALANCED_V1");
        assert_eq!(quality.name, "GENERATOR_QUALITY_V1");
    }

    #[test]
    fn test_tier_default() {
        assert!(matches!(GeneratorTier::default(), GeneratorTier::Balanced));
    }

    #[test]
    fn test_all_presets_valid() {
        for preset in ALL_V1_PRESETS {
            assert!(!preset.name.is_empty(), "Preset name should not be empty");
            assert!(!preset.model.is_empty(), "Preset model should not be empty");
            assert!(!preset.architecture.is_empty(), "Preset architecture should not be empty");
            assert!(preset.memory_mb > 0, "Preset should have memory requirement");
            assert!(preset.default_max_tokens > 0, "Preset should have max tokens");
            assert!(!preset.description.is_empty(), "Preset should have description");
        }
    }

    #[test]
    fn test_all_presets_unique_names() {
        let names: Vec<_> = ALL_V1_PRESETS.iter().map(|p| p.name).collect();
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(names.len(), unique.len(), "All preset names should be unique");
    }

    #[test]
    fn test_list_presets() {
        let presets = list_presets();
        assert!(!presets.is_empty());
        assert!(presets.contains(&"GENERATOR_FAST_V1"));
        assert!(presets.contains(&"GENERATOR_QUALITY_V1"));
    }

    #[test]
    fn test_legacy_presets() {
        // Legacy presets should still work
        assert!(!GeneratorPreset::GPT2.model.is_empty());
        assert!(!GeneratorPreset::FAST.model.is_empty());
        assert!(!GeneratorPreset::QUALITY.model.is_empty());
    }
}