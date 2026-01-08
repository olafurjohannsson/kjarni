// =============================================================================
// kjarni/src/generator/presets.rs
// =============================================================================

//! Generator presets for common use cases.

use crate::common::KjarniDevice;

/// Preset configuration for generator models.
#[derive(Debug, Clone)]
pub struct GeneratorPreset {
    /// Model name from registry.
    pub model: &'static str,
    /// Recommended device.
    pub recommended_device: KjarniDevice,
    /// Recommended temperature.
    pub temperature: Option<f32>,
    /// Recommended max tokens.
    pub max_tokens: Option<usize>,
    /// Human-readable description.
    pub description: &'static str,
}

impl GeneratorPreset {
    /// GPT-2 for text completion.
    pub const GPT2: GeneratorPreset = GeneratorPreset {
        model: "gpt2",
        recommended_device: KjarniDevice::Cpu,
        temperature: Some(0.7),
        max_tokens: Some(100),
        description: "Classic GPT-2 for text completion.",
    };

    /// Fast text completion.
    pub const FAST: GeneratorPreset = GeneratorPreset {
        model: "qwen2.5-0.5b",
        recommended_device: KjarniDevice::Cpu,
        temperature: Some(0.5),
        max_tokens: Some(128),
        description: "Fast text generation.",
    };

    /// Quality text completion.
    pub const QUALITY: GeneratorPreset = GeneratorPreset {
        model: "llama3.2-1b",
        recommended_device: KjarniDevice::Cpu,
        temperature: Some(0.6),
        max_tokens: Some(256),
        description: "Better quality text generation.",
    };
}