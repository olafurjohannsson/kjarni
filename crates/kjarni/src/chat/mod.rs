//! High-level chat API for conversational AI.
//!
//! This module provides the primary interface for interacting with
//! chat/instruction-tuned language models.
//!
//! # Quick Start
//!
//! ```ignore
//! use kjarni::chat;
//!
//! // Ultra-simple one-liner
//! let response = chat::send("llama3.2-1b", "Hello!").await?;
//!
//! // Or with more control
//! use kjarni::chat::Chat;
//!
//! let chat = Chat::new("llama3.2-1b").await?;
//! let response = chat.send("Hello!").await?;
//! ```
//!
//! # Multi-turn Conversations
//!
//! ```ignore
//! use kjarni::chat::Chat;
//!
//! let chat = Chat::new("llama3.2-1b").await?;
//!
//! // Stateful conversation
//! let mut convo = chat.conversation();
//! convo.send("Hello!").await?;
//! convo.send("What did I just say?").await?;
//!
//! // Or stateless with manual history
//! use kjarni::chat::History;
//!
//! let mut history = History::with_system("You are helpful.");
//! history.push_user("Hello");
//! history.push_assistant("Hi!");
//!
//! let response = chat.send_with_history(&history, "Continue").await?;
//! ```
//!
//! # Using Presets
//!
//! ```ignore
//! use kjarni::chat::{Chat, presets::CHAT_SMALL_V1};
//!
//! let chat = Chat::from_preset(&CHAT_SMALL_V1)
//!     .system_prompt("You are a pirate.")
//!     .build()
//!     .await?;
//! ```
//!
//! # Streaming
//!
//! ```ignore
//! use futures_util::StreamExt;
//! use kjarni::chat::Chat;
//!
//! let chat = Chat::new("llama3.2-1b").await?;
//! let mut stream = chat.stream("Tell me a story.").await?;
//!
//! while let Some(token) = stream.next().await {
//!     print!("{}", token?);
//! }
//! ```

mod builder;
mod conversation;
mod model;
pub mod presets;
mod types;
mod validation;

// Re-exports
pub use builder::ChatBuilder;
pub use conversation::ChatConversation;
pub use model::Chat;
pub use presets::{ChatTier, ModelPreset};
pub use types::{
    ChatDevice, ChatError, ChatMode, ChatResult, ChatWarning,
    DownloadPolicy, History, Message, Role,
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Send a single message and get a response.
///
/// This is the simplest possible API - a one-liner for quick interactions.
/// Creates a Chat instance, sends the message, and returns the response.
///
/// # Example
///
/// ```ignore
/// let response = kjarni::chat::send("llama3.2-1b", "What is Rust?").await?;
/// println!("{}", response);
/// ```
///
/// # Notes
///
/// - Uses CPU by default
/// - Downloads model if not present
/// - No conversation history (stateless)
/// - For repeated interactions, create a `Chat` instance instead
pub async fn send(model: &str, message: &str) -> ChatResult<String> {
    let chat = Chat::new(model).await?;
    chat.send(message).await
}

/// Send a message using a preset.
///
/// # Example
///
/// ```ignore
/// use kjarni::chat::presets::CHAT_SMALL_V1;
///
/// let response = kjarni::chat::send_preset(&CHAT_SMALL_V1, "Hello!").await?;
/// ```
pub async fn send_preset(preset: &ModelPreset, message: &str) -> ChatResult<String> {
    let chat = Chat::from_preset(preset).build().await?;
    chat.send(message).await
}

/// Send a message using a tier (auto-selects best model for that tier).
///
/// # Example
///
/// ```ignore
/// use kjarni::chat::ChatTier;
///
/// let response = kjarni::chat::send_tier(ChatTier::Small, "Hello!").await?;
/// ```
pub async fn send_tier(tier: ChatTier, message: &str) -> ChatResult<String> {
    let preset = tier.resolve();
    send_preset(preset, message).await
}

/// Check if a model is valid for chat.
///
/// Returns Ok(()) if valid, or an error describing why not.
///
/// # Example
///
/// ```ignore
/// if kjarni::chat::is_chat_model("llama3.2-1b").is_ok() {
///     println!("Model can be used for chat");
/// }
/// ```
pub fn is_chat_model(model: &str) -> ChatResult<()> {
    use kjarni_transformers::models::ModelType;

    let model_type = ModelType::from_cli_name(model)
        .ok_or_else(|| ChatError::UnknownModel(model.to_string()))?;

    validation::validate_for_chat(model_type)?;
    Ok(())
}

/// List all available chat-capable models.
///
/// Returns CLI names of models that can be used with Chat.
pub fn available_models() -> Vec<&'static str> {
    use kjarni_transformers::models::ModelType;

    ModelType::all()
        .filter(|m| validation::validate_for_chat(*m).is_ok())
        .map(|m| m.cli_name())
        .collect()
}

/// Get suggested models for chat.
pub fn suggested_models() -> Vec<&'static str> {
    validation::suggest_chat_models()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_chat_model() {
        // Valid chat models
        assert!(is_chat_model("llama3.2-1b").is_ok());
        assert!(is_chat_model("qwen2.5-0.5b").is_ok());

        // Invalid models
        assert!(is_chat_model("minilm-l6-v2").is_err()); // encoder
        assert!(is_chat_model("flan-t5-base").is_err()); // seq2seq
        assert!(is_chat_model("nonexistent").is_err());
    }

    #[test]
    fn test_available_models() {
        let models = available_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"llama3.2-1b"));
    }
}
