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
//! use futures::StreamExt;
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
pub mod conversation;
mod model;
pub mod presets;
mod types;
mod validation;

// Re-exports
pub use crate::common::DownloadPolicy;
pub use builder::ChatBuilder;
pub use conversation::ChatConversation;
pub use model::Chat;
pub use presets::{ChatPreset, ChatTier};
pub use types::{ChatDevice, ChatError, ChatMode, ChatResult, ChatWarning, History, Message, Role};

/// Send a single message and get a response.
///
/// # Example
///
/// ```ignore
/// let response = kjarni::chat::send("llama3.2-1b", "What is Rust?").await?;
/// println!("{}", response);
/// ```
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
pub async fn send_preset(preset: &ChatPreset, message: &str) -> ChatResult<String> {
    let chat = ChatBuilder::from_preset(preset).build().await?;
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
mod tests;
