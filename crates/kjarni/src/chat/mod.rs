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
pub use presets::{ChatTier, ChatPreset};
pub use types::{
    ChatDevice, ChatError, ChatMode, ChatResult, ChatWarning,
    History, Message, Role,
};
pub use crate::common::DownloadPolicy;

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
mod chat_integration_tests {
    use super::*;
    use crate::chat::Chat;
    use crate::chat::presets::ChatPreset;
    use crate::common::DownloadPolicy;
    use crate::chat::types::Role;
    use futures_util::StreamExt;

    /// Helper to load the lightweight Qwen 0.5B model.
    /// This will download the model (~300MB) if not present.
    async fn load_real_model() -> Chat {
        let model_name = ChatPreset::FAST.model; // "qwen2.5-0.5b-instruct"
        
        println!("Loading test model: {}...", model_name);
        
        Chat::builder(model_name)
            .download_policy(DownloadPolicy::IfMissing)
            // Use CPU to ensure tests pass everywhere, even without GPU setup
            .device(crate::common::KjarniDevice::Cpu) 
            .quiet()
            .build()
            .await
            .expect("Failed to load Qwen 0.5B for testing. Do you have internet?")
    }

    #[tokio::test]
    #[ignore = "Requires model download and CPU inference time"]
    async fn test_real_blocking_conversation_flow() {
        let chat = load_real_model().await;
        let mut convo = chat.conversation();

        // 1. First Turn
        let response1 = convo.send("Hello! Answer with exactly one word: 'Hi'.").await.unwrap();
        println!("Response 1: {}", response1);
        
        assert!(!response1.is_empty());
        assert_eq!(convo.len(), 2, "History should contain User + Assistant");
        
        let msgs = convo.history().messages();
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);

        // 2. Second Turn (Context Awareness)
        let response2 = convo.send("What word did you just say?").await.unwrap();
        println!("Response 2: {}", response2);
        
        assert!(!response2.is_empty());
        assert_eq!(convo.len(), 4, "History should contain 4 messages");
        
        // Basic sanity check that the model is actually working contextually
        let response_lower = response2.to_lowercase();
        assert!(response_lower.contains("hi"), "Model should recall previous context");
    }

    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_real_streaming_manual_flow() {
        let chat = load_real_model().await;
        let mut convo = chat.conversation();

        // 1. Manual User Push
        convo.push_user("Count from 1 to 3. Format: '1, 2, 3'.");
        assert_eq!(convo.len(), 1);

        // 2. Stream
        let mut stream = convo.stream_next().await.unwrap();
        let mut full_response = String::new();

        while let Some(token_res) = stream.next().await {
            let token = token_res.unwrap();
            print!("{}", token); // Print to stdout to see progress
            full_response.push_str(&token);
        }
        println!(); // Newline

        assert!(!full_response.is_empty());
        // Streaming does NOT auto-add to history in ChatConversation
        assert_eq!(convo.len(), 1, "History should still only have User message");

        // 3. Manual Assistant Push
        convo.push_assistant(&full_response);
        assert_eq!(convo.len(), 2);
        assert_eq!(convo.history().messages().last().unwrap().content, full_response);
    }

    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_real_system_prompt_adherence() {
        let chat = load_real_model().await;
        
        // Initialize conversation with a specific persona
        let mut convo = ChatConversation::with_system(&chat, "You are a pirate. End every sentence with 'Arrr!'.".to_string());
        
        assert_eq!(convo.len(), 1);
        assert_eq!(convo.history().messages()[0].role, Role::System);

        let response = convo.send("Who are you?").await.unwrap();
        println!("Pirate Response: {}", response);

        assert!(response.to_lowercase().contains("arrr"), "Model should follow system prompt instructions");
    }

    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_clear_history_with_real_model() {
        let chat = load_real_model().await;
        let mut convo = chat.conversation();

        convo.send("My favorite color is Blue.").await.unwrap();
        assert_eq!(convo.len(), 2);

        // Clear history
        convo.clear(false);
        assert!(convo.is_empty());

        // Ask about context - model should hallucinate or say it doesn't know
        let response = convo.send("What is my favorite color?").await.unwrap();
        println!("Memory Wipe Response: {}", response);
        
        let lower = response.to_lowercase();
        assert!(!lower.contains("blue") || lower.contains("don't know") || lower.contains("tell me"), 
            "Model should not recall cleared context");
    }
}