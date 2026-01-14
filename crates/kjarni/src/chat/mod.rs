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
use kjarni_transformers::{ModelArchitecture, ModelType, models::ModelTask};
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
    use crate::generation::GenerationOverrides;
    use futures_util::StreamExt;

    async fn load_real_model() -> Chat {
        let model_name = ChatPreset::FAST.model; // "qwen2.5-0.5b-instruct"
        
        println!("Loading test model: {}...", model_name);
        
        Chat::builder(model_name)
            .download_policy(DownloadPolicy::IfMissing)
            // Use CPU to ensure tests pass everywhere, even without GPU setup
            .device(crate::common::KjarniDevice::Cpu) 
            .generation_config(GenerationOverrides {
                max_new_tokens: Some(50), // Limit tokens for faster tests
                do_sample: Some(false), // Deterministic for tests
                ..GenerationOverrides::default()
            
            })
            .quiet()
            .build()
            .await
            .expect("Failed to load Qwen 0.5B for testing. Do you have internet?")
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

        convo.send("My favorite color is Red.").await.unwrap();
        assert_eq!(convo.len(), 2);

        // Clear history
        convo.clear(false);
        assert!(convo.is_empty());

        // Ask about context - model should hallucinate or say it doesn't know
        let response = convo.send("What is my favorite color?").await.unwrap();
        println!("Memory Wipe Response: {}", response);
        
        let lower = response.to_lowercase();
        assert!(lower.contains("as an ai language model") && lower.contains("don't have personal preferences") && lower.contains("many people find blue to be a calming and soothing color"),
            "Model should not recall cleared context but should respond appropriately.");
    }
    #[tokio::test]
    #[ignore]
    async fn test_real_blocking_conversation_flow() {
        let chat = load_real_model().await;
        let mut convo = chat.conversation();

        // 1. Simpler Prompt for 0.5B model
        // Qwen 0.5B is often chatty. Let's give it a distinctive fact.
        let response1 = convo.send("My name is Olafur.").await.unwrap();
        println!("Response 1: {}", response1);
        
        // 2. Ask for recall
        let response2 = convo.send("What is my name?").await.unwrap();
        println!("Response 2: {}", response2);
        
        assert!(
            response2.contains("Olafur"), 
            "Model failed to recall name from context.\nHistory: {:?}\nResponse: {}", 
            convo.history(), 
            response2
        );
    }

    

}



#[test]
fn test_history_management() {
    let mut history = History::new();
    assert!(history.is_empty());

    // 1. Push sequence
    history.push_user("User 1");
    history.push_assistant("Assistant 1");

    let msgs = history.messages();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].role, Role::User);
    assert_eq!(msgs[0].content, "User 1");
    assert_eq!(msgs[1].role, Role::Assistant);
    assert_eq!(msgs[1].content, "Assistant 1");

    // 2. Clear keeping system
    history.clear(true);
    assert_eq!(history.len(), 1);
    assert_eq!(history.messages()[0].role, Role::System);

    // 3. Full clear
    history.clear(false);
    assert!(history.is_empty());
}

#[test]
fn test_chat_mode_defaults() {
    let creative = ChatMode::Creative;
    assert!(creative.default_temperature() > 0.8);
    assert!(creative.default_max_tokens() >= 1024);

    let reasoning = ChatMode::Reasoning;
    assert!(reasoning.default_temperature() < 0.5);
    
    let default = ChatMode::Default;
    assert_eq!(default.default_temperature(), 0.7);
}

#[test]
fn test_presets_configuration() {
    let fast = ChatPreset::FAST;
    assert_eq!(fast.model, "qwen2.5-0.5b-instruct");
    assert!(fast.temperature.is_some());

    let coding = ChatPreset::CODING;
    assert_eq!(coding.mode, ChatMode::Reasoning);
    assert!(coding.system_prompt.unwrap().to_lowercase().contains("coding"));
}

#[test]
fn test_tier_resolution() {
    assert_eq!(ChatTier::Fast.resolve().model, ChatPreset::FAST.model);
    assert_eq!(ChatTier::Balanced.resolve().model, ChatPreset::BALANCED.model);
    assert_eq!(ChatTier::Quality.resolve().model, ChatPreset::QUALITY.model);
}


// =============================================================================
//  INTEGRATION TESTS (Requires Model Download)
// =============================================================================

mod integration {
    use super::*;
    use crate::common::DownloadPolicy;
    use futures_util::StreamExt;

    async fn load_test_model() -> Chat {
        // Use Qwen 0.5B for fast testing
        Chat::builder("qwen2.5-0.5b-instruct")
            .download_policy(DownloadPolicy::IfMissing)
            .cpu() // Force CPU for CI/reliability
            .quiet()
            .build()
            .await
            .expect("Failed to load Qwen 0.5B. Check internet connection.")
    }

    #[tokio::test]
    async fn test_full_conversation_cycle() {
        let chat = load_test_model().await;
        let mut convo = chat.conversation();

        // 1. Initial Greeting
        let response1 = convo.send("Hello! Reply with 'Alpha'.").await.unwrap();
        println!("Cycle R1: {}", response1);
        assert!(!response1.is_empty());

        // 2. Context Awareness
        let response2 = convo.send("Repeat the word you just said.").await.unwrap();
        println!("Cycle R2: {}", response2);
        assert!(
            response2.to_lowercase().contains("alpha"),
            "Model failed to recall context. History: {:?}",
            convo.history()
        );

        // 3. History Persistence
        assert_eq!(convo.len(), 4); // User, Asst, User, Asst
    }

    #[tokio::test]
    async fn test_system_prompt_adherence() {
        let chat = load_test_model().await;
        
        // Custom system prompt via Builder
        let chat_pirate = Chat::builder("qwen2.5-0.5b-instruct")
            .system("You are a pirate. End sentences with 'Arrr'.")
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();
            
        let response = chat_pirate.send("Who are you?").await.unwrap();
        println!("Pirate: {}", response);
        
        assert!(
            response.to_lowercase().contains("arrr"),
            "System prompt was ignored."
        );
    }

    #[tokio::test]
    async fn test_streaming_and_manual_history() {
        let chat = load_test_model().await;
        let mut convo = chat.conversation();
        
        // 1. Push User Message Manually
        convo.push_user("Count to 3.");
        assert_eq!(convo.len(), 1); // Only User
        
        // 2. Stream Response
        let mut stream = convo.stream_next().await.unwrap();
        let mut full_response = String::new();
        
        while let Some(token_res) = stream.next().await {
            let token = token_res.unwrap();
            full_response.push_str(&token);
        }
        
        println!("Streamed: {}", full_response);
        assert!(!full_response.is_empty());
        
        // 3. History check (Stream shouldn't auto-add)
        assert_eq!(convo.len(), 1); 
        
        // 4. Manual push assistant
        convo.push_assistant(&full_response);
        assert_eq!(convo.len(), 2);
    }

    #[tokio::test]
    async fn test_clearing_history() {
        let chat = load_test_model().await;
        let mut convo = chat.conversation_with_system("System info");
        
        convo.send("My secret number is 42.").await.unwrap();
        assert_eq!(convo.len(), 3); // Sys, User, Asst
        
        // Clear keeping system
        convo.clear(true);
        assert_eq!(convo.len(), 1);
        assert_eq!(convo.history().messages()[0].role, Role::System);
        
        // Test context loss
        let response = convo.send("What is my secret number?").await.unwrap();
        println!("Memory Wipe: {}", response);
        
        assert!(
            !response.contains("42"),
            "Model should have forgotten the number."
        );
    }

    #[tokio::test]
    async fn test_builder_overrides() {
        // Test that setting temperature actually works (by checking config)
        let chat = Chat::builder("qwen2.5-0.5b-instruct")
            .temperature(0.1) // Very deterministic
            .max_tokens(5)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();
            
        // Check internal config if accessible, or run generation
        let response = chat.send("Say Hi").await.unwrap();
        // Just ensuring it runs without panic with overrides
        assert!(!response.is_empty()); 
    }
}
