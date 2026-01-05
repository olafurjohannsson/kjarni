//! Versioned model presets for reproducible deployments.
//!
//! Presets provide sensible defaults that are frozen at compile time.
//! Upgrading kjarni may change which models presets resolve to,
//! but within a version, presets are stable.
//!
//! # Versioning Policy
//!
//! - `V1` presets are frozen and will never change
//! - New versions (V2, V3) may be added with improved models
//! - Deprecated presets will warn but continue to work
//!
//! # Example
//!
//! ```ignore
//! use kjarni::chat::{Chat, presets::CHAT_SMALL_V1};
//!
//! let chat = Chat::from_preset(CHAT_SMALL_V1).build().await?;
//! ```

use super::types::ChatDevice;

/// A versioned model preset.
///
/// Contains all information needed to instantiate a model with
/// known-good defaults. Resolution happens at build time.
#[derive(Debug, Clone)]
pub struct ModelPreset {
    /// Human-readable name for this preset.
    pub name: &'static str,

    /// CLI name of the model (e.g., "llama3.2-1b").
    pub model: &'static str,

    /// Maximum context length to configure.
    pub context_length: usize,

    /// Chat template identifier.
    pub chat_template: &'static str,

    /// Recommended quantization format.
    pub quantization: &'static str,

    /// Recommended device for this model size.
    pub recommended_device: ChatDevice,

    /// Approximate VRAM/RAM required in MB.
    pub memory_mb: usize,

    /// Brief description of the model's strengths.
    pub description: &'static str,
}

// =============================================================================
// V1 Presets - Frozen, will never change
// =============================================================================

/// Smallest chat model. Fast, low memory, basic capability.
///
/// - Model: Llama 3.2 1B Instruct
/// - Memory: ~1.5 GB (Q4)
/// - Best for: Simple queries, quick responses, constrained environments
pub const CHAT_SMALL_V1: ModelPreset = ModelPreset {
    name: "CHAT_SMALL_V1",
    model: "llama3.2-1b",
    context_length: 8192,
    chat_template: "llama3",
    quantization: "q4_k_m",
    recommended_device: ChatDevice::Cpu,
    memory_mb: 1500,
    description: "Fast and lightweight, good for simple tasks",
};

/// Medium chat model. Balanced speed and capability.
///
/// - Model: Llama 3.2 3B Instruct
/// - Memory: ~3.5 GB (Q4)
/// - Best for: General conversation, moderate complexity
pub const CHAT_MEDIUM_V1: ModelPreset = ModelPreset {
    name: "CHAT_MEDIUM_V1",
    model: "llama3.2-3b",
    context_length: 8192,
    chat_template: "llama3",
    quantization: "q4_k_m",
    recommended_device: ChatDevice::Cpu,
    memory_mb: 3500,
    description: "Balanced speed and quality for general use",
};

/// Large chat model. Higher capability, requires more resources.
///
/// - Model: Llama 3.1 8B Instruct
/// - Memory: ~8 GB (Q4)
/// - Best for: Complex reasoning, nuanced responses
pub const CHAT_LARGE_V1: ModelPreset = ModelPreset {
    name: "CHAT_LARGE_V1",
    model: "llama3.1-8b",
    context_length: 8192,
    chat_template: "llama3",
    quantization: "q4_k_m",
    recommended_device: ChatDevice::Gpu,
    memory_mb: 8000,
    description: "High quality responses, best with GPU",
};

/// Reasoning-optimized model. Slower but better at complex tasks.
///
/// - Model: DeepSeek R1 Distill 8B
/// - Memory: ~8 GB (Q4)
/// - Best for: Math, logic, step-by-step reasoning
pub const REASONING_V1: ModelPreset = ModelPreset {
    name: "REASONING_V1",
    model: "deepseek-r1-8b",
    context_length: 8192,
    chat_template: "llama3", // Uses Llama architecture
    quantization: "q4_k_m",
    recommended_device: ChatDevice::Gpu,
    memory_mb: 8000,
    description: "Optimized for reasoning and chain-of-thought",
};

/// Tiny chat model for extreme constraints.
///
/// - Model: Qwen 2.5 0.5B Instruct
/// - Memory: ~500 MB (Q4)
/// - Best for: Embedded, mobile, or very limited environments
pub const CHAT_TINY_V1: ModelPreset = ModelPreset {
    name: "CHAT_TINY_V1",
    model: "qwen2.5-0.5b",
    context_length: 4096,
    chat_template: "chatml",
    quantization: "q4_k_m",
    recommended_device: ChatDevice::Cpu,
    memory_mb: 500,
    description: "Minimal footprint, basic capability",
};

/// Qwen-based small model. Good multilingual support.
///
/// - Model: Qwen 2.5 1.5B Instruct
/// - Memory: ~2 GB (Q4)
/// - Best for: Multilingual tasks, structured output
pub const CHAT_QWEN_SMALL_V1: ModelPreset = ModelPreset {
    name: "CHAT_QWEN_SMALL_V1",
    model: "qwen2.5-1.5b",
    context_length: 8192,
    chat_template: "chatml",
    quantization: "q4_k_m",
    recommended_device: ChatDevice::Cpu,
    memory_mb: 2000,
    description: "Good for multilingual and structured output",
};

// =============================================================================
// Preset Collections
// =============================================================================

/// All available V1 presets.
pub const ALL_V1_PRESETS: &[&ModelPreset] = &[
    &CHAT_TINY_V1,
    &CHAT_SMALL_V1,
    &CHAT_MEDIUM_V1,
    &CHAT_LARGE_V1,
    &CHAT_QWEN_SMALL_V1,
    &REASONING_V1,
];

/// Find a preset by name.
pub fn find_preset(name: &str) -> Option<&'static ModelPreset> {
    let name_upper = name.to_uppercase();
    ALL_V1_PRESETS
        .iter()
        .find(|p| p.name == name_upper)
        .copied()
}

/// Tier-based preset selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTier {
    /// ~500MB, basic capability
    Tiny,
    /// ~1.5GB, fast responses
    Small,
    /// ~3.5GB, balanced
    Medium,
    /// ~8GB, high quality
    Large,
}

impl ChatTier {
    /// Resolve tier to the current default preset for that tier.
    pub fn resolve(&self) -> &'static ModelPreset {
        match self {
            Self::Tiny => &CHAT_TINY_V1,
            Self::Small => &CHAT_SMALL_V1,
            Self::Medium => &CHAT_MEDIUM_V1,
            Self::Large => &CHAT_LARGE_V1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_lookup() {
        assert!(find_preset("CHAT_SMALL_V1").is_some());
        assert!(find_preset("chat_small_v1").is_some()); // case insensitive
        assert!(find_preset("nonexistent").is_none());
    }

    #[test]
    fn test_tier_resolution() {
        assert_eq!(ChatTier::Small.resolve().model, "llama3.2-1b");
        assert_eq!(ChatTier::Large.resolve().model, "llama3.1-8b");
    }
}
