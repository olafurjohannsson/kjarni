// =============================================================================
// kjarni/src/chat/presets.rs
// =============================================================================

//! Chat presets for common use cases.

use super::types::ChatMode;
use crate::common::KjarniDevice;

/// Preset configuration for chat models.
#[derive(Debug, Clone)]
pub struct ChatPreset {
    /// Model name from registry.
    pub model: &'static str,
    /// Recommended device.
    pub recommended_device: KjarniDevice,
    /// Default chat mode.
    pub mode: ChatMode,
    /// Default system prompt (if any).
    pub system_prompt: Option<&'static str>,
    /// Recommended temperature.
    pub temperature: Option<f32>,
    /// Recommended max tokens.
    pub max_tokens: Option<usize>,
    /// Human-readable description.
    pub description: &'static str,
}

impl ChatPreset {
    /// Small, fast model for quick responses.
    pub const FAST: ChatPreset = ChatPreset {
        model: "qwen2.5-0.5b-instruct",
        recommended_device: KjarniDevice::Cpu,
        mode: ChatMode::Default,
        system_prompt: None,
        temperature: Some(0.7),
        max_tokens: Some(256),
        description: "Fast responses, lower quality. Good for simple tasks.",
    };

    /// Balanced model for general use.
    pub const BALANCED: ChatPreset = ChatPreset {
        model: "llama3.2-1b-instruct",
        recommended_device: KjarniDevice::Cpu,
        mode: ChatMode::Default,
        system_prompt: None,
        temperature: Some(0.7),
        max_tokens: Some(512),
        description: "Balanced speed and quality. Good for most tasks.",
    };

    /// Larger model for better quality.
    pub const QUALITY: ChatPreset = ChatPreset {
        model: "llama3.2-3b-instruct",
        recommended_device: KjarniDevice::Gpu,
        mode: ChatMode::Default,
        system_prompt: None,
        temperature: Some(0.7),
        max_tokens: Some(1024),
        description: "Higher quality, slower. Best with GPU.",
    };

    /// Optimized for creative writing.
    pub const CREATIVE: ChatPreset = ChatPreset {
        model: "llama3.2-1b-instruct",
        recommended_device: KjarniDevice::Cpu,
        mode: ChatMode::Creative,
        system_prompt: Some("You are a creative writing assistant. Be imaginative and expressive."),
        temperature: Some(0.9),
        max_tokens: Some(1024),
        description: "Higher temperature for creative tasks.",
    };

    /// Optimized for coding assistance.
    pub const CODING: ChatPreset = ChatPreset {
        model: "qwen2.5-1.5b-instruct",
        recommended_device: KjarniDevice::Cpu,
        mode: ChatMode::Reasoning,
        system_prompt: Some("You are a coding assistant. Write clean, well-documented code."),
        temperature: Some(0.3),
        max_tokens: Some(2048),
        description: "Lower temperature for precise code generation.",
    };

    /// Optimized for reasoning and analysis.
    pub const REASONING: ChatPreset = ChatPreset {
        model: "llama3.2-3b-instruct",
        recommended_device: KjarniDevice::Gpu,
        mode: ChatMode::Reasoning,
        system_prompt: Some("You are a logical reasoning assistant. Think step by step."),
        temperature: Some(0.3),
        max_tokens: Some(2048),
        description: "Lower temperature for logical tasks.",
    };
}

// Tier-based presets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTier {
    /// Fastest, smallest model.
    Fast,
    /// Balanced speed and quality.
    #[default]
    Balanced,
    /// Best quality, may need GPU.
    Quality,
}

impl ChatTier {
    /// Get the preset for this tier.
    pub fn preset(&self) -> &'static ChatPreset {
        match self {
            Self::Fast => &ChatPreset::FAST,
            Self::Balanced => &ChatPreset::BALANCED,
            Self::Quality => &ChatPreset::QUALITY,
        }
    }
    pub fn resolve(&self) -> &'static ChatPreset {
        match self {
            Self::Balanced => &ChatPreset::BALANCED,
            Self::Fast => &ChatPreset::FAST,
            Self::Quality => &ChatPreset::QUALITY,
        }
    }
}
