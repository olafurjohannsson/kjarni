//! Tests for the chat module.
//!
//! Run all tests: `cargo test --package kjarni chat`
//! Run integration tests (requires model): `cargo test --package kjarni chat -- --ignored`

use super::*;
use crate::common::{DownloadPolicy, KjarniDevice};
use crate::generation::GenerationOverrides;

// =============================================================================
// Type Tests
// =============================================================================

mod types_tests {
    use super::*;

    #[test]
    fn test_chat_mode_default() {
        let mode = ChatMode::default();
        assert_eq!(mode, ChatMode::Default);
    }

    #[test]
    fn test_chat_mode_default_temperature() {
        assert!((ChatMode::Default.default_temperature() - 0.7).abs() < 0.01);
        assert!((ChatMode::Reasoning.default_temperature() - 0.3).abs() < 0.01);
        assert!((ChatMode::Creative.default_temperature() - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_chat_mode_default_max_tokens() {
        assert_eq!(ChatMode::Default.default_max_tokens(), 512);
        assert_eq!(ChatMode::Reasoning.default_max_tokens(), 2048);
        assert_eq!(ChatMode::Creative.default_max_tokens(), 1024);
    }

    #[test]
    fn test_download_policy_default() {
        let policy = DownloadPolicy::default();
        assert_eq!(policy, DownloadPolicy::IfMissing);
    }

    #[test]
    fn test_chat_device_default() {
        let device = ChatDevice::default();
        assert_eq!(device, ChatDevice::Cpu);
    }

    #[test]
    fn test_chat_device_resolve() {
        assert_eq!(ChatDevice::Cpu.resolve(), ChatDevice::Cpu);
        assert_eq!(ChatDevice::Gpu.resolve(), ChatDevice::Gpu);
        // Auto should resolve to something
        let resolved = ChatDevice::Auto.resolve();
        assert!(resolved == ChatDevice::Cpu || resolved == ChatDevice::Gpu);
    }

    #[test]
    fn test_role_display() {
        assert_eq!(format!("{}", Role::System), "system");
        assert_eq!(format!("{}", Role::User), "user");
        assert_eq!(format!("{}", Role::Assistant), "assistant");
    }

    #[test]
    fn test_message_constructors() {
        let system = Message::system("You are helpful");
        assert_eq!(system.role, Role::System);
        assert_eq!(system.content, "You are helpful");

        let user = Message::user("Hello");
        assert_eq!(user.role, Role::User);
        assert_eq!(user.content, "Hello");

        let assistant = Message::assistant("Hi there!");
        assert_eq!(assistant.role, Role::Assistant);
        assert_eq!(assistant.content, "Hi there!");
    }

    #[test]
    fn test_history_new() {
        let history = History::new();
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
    }

    #[test]
    fn test_history_with_system() {
        let history = History::with_system("You are helpful");
        assert_eq!(history.len(), 1);
        assert_eq!(history.messages()[0].role, Role::System);
        assert_eq!(history.messages()[0].content, "You are helpful");
    }

    #[test]
    fn test_history_push() {
        let mut history = History::new();
        history.push_user("Hello");
        history.push_assistant("Hi!");

        assert_eq!(history.len(), 2);
        assert_eq!(history.messages()[0].role, Role::User);
        assert_eq!(history.messages()[1].role, Role::Assistant);
    }

    #[test]
    fn test_history_clear() {
        let mut history = History::with_system("System");
        history.push_user("Hello");
        history.push_assistant("Hi!");

        history.clear();
        assert!(history.is_empty());
    }

    #[test]
    fn test_history_clear_keep_system() {
        let mut history = History::with_system("System prompt");
        history.push_user("Hello");
        history.push_assistant("Hi!");

        assert_eq!(history.len(), 3);

        history.clear_keep_system();

        assert_eq!(history.len(), 1);
        assert_eq!(history.messages()[0].role, Role::System);
        assert_eq!(history.messages()[0].content, "System prompt");
    }

    #[test]
    fn test_history_clear_keep_system_no_system() {
        let mut history = History::new();
        history.push_user("Hello");

        history.clear_keep_system();

        assert!(history.is_empty());
    }
}

// =============================================================================
// Builder Tests
// =============================================================================

mod builder_tests {
    use super::*;

    #[test]
    fn test_builder_default_values() {
        let builder = ChatBuilder::new("test-model");

        assert_eq!(builder.model, "test-model");
        assert!(builder.model_path.is_none());
        assert_eq!(builder.device, ChatDevice::Cpu);
        assert!(builder.context.is_none());
        assert!(builder.cache_dir.is_none());
        assert_eq!(builder.mode, ChatMode::Default);
        assert!(builder.system_prompt.is_none());
        assert_eq!(builder.download_policy, DownloadPolicy::IfMissing);
        assert!(!builder.quiet);
    }

    #[test]
    fn test_builder_cpu() {
        let builder = ChatBuilder::new("test-model").cpu();
        assert_eq!(builder.device, ChatDevice::Cpu);
    }

    #[test]
    fn test_builder_gpu() {
        let builder = ChatBuilder::new("test-model").gpu();
        assert_eq!(builder.device, ChatDevice::Gpu);
    }

    #[test]
    fn test_builder_auto_device() {
        let builder = ChatBuilder::new("test-model").auto_device();
        assert_eq!(builder.device, ChatDevice::Auto);
    }

    #[test]
    fn test_builder_cache_dir() {
        let builder = ChatBuilder::new("test-model")
            .cache_dir("/tmp/test-cache");

        assert_eq!(
            builder.cache_dir,
            Some(std::path::PathBuf::from("/tmp/test-cache"))
        );
    }

    #[test]
    fn test_builder_mode() {
        let builder = ChatBuilder::new("test-model").mode(ChatMode::Reasoning);
        assert_eq!(builder.mode, ChatMode::Reasoning);
    }

    #[test]
    fn test_builder_reasoning() {
        let builder = ChatBuilder::new("test-model").reasoning();
        assert_eq!(builder.mode, ChatMode::Reasoning);
    }

    #[test]
    fn test_builder_creative() {
        let builder = ChatBuilder::new("test-model").creative();
        assert_eq!(builder.mode, ChatMode::Creative);
    }

    #[test]
    fn test_builder_system_prompt() {
        let builder = ChatBuilder::new("test-model")
            .system_prompt("You are a pirate");

        assert_eq!(builder.system_prompt, Some("You are a pirate".to_string()));
    }

    #[test]
    fn test_builder_temperature() {
        let builder = ChatBuilder::new("test-model").temperature(0.5);
        assert_eq!(builder.generation_overrides.temperature, Some(0.5));
    }

    #[test]
    fn test_builder_max_tokens() {
        let builder = ChatBuilder::new("test-model").max_tokens(1024);
        assert_eq!(builder.generation_overrides.max_new_tokens, Some(1024));
    }

    #[test]
    fn test_builder_top_p() {
        let builder = ChatBuilder::new("test-model").top_p(0.9);
        assert_eq!(builder.generation_overrides.top_p, Some(0.9));
    }

    #[test]
    fn test_builder_top_k() {
        let builder = ChatBuilder::new("test-model").top_k(50);
        assert_eq!(builder.generation_overrides.top_k, Some(50));
    }

    #[test]
    fn test_builder_repetition_penalty() {
        let builder = ChatBuilder::new("test-model").repetition_penalty(1.1);
        assert_eq!(builder.generation_overrides.repetition_penalty, Some(1.1));
    }

    #[test]
    fn test_builder_offline() {
        let builder = ChatBuilder::new("test-model").offline();
        assert_eq!(builder.download_policy, DownloadPolicy::Never);
    }

    #[test]
    fn test_builder_quiet() {
        let builder = ChatBuilder::new("test-model").quiet(true);
        assert!(builder.quiet);
    }

    #[test]
    fn test_builder_chain() {
        let builder = ChatBuilder::new("test-model")
            .gpu()
            .mode(ChatMode::Creative)
            .system_prompt("Be creative")
            .temperature(0.9)
            .max_tokens(2048)
            .quiet(true)
            .offline();

        assert_eq!(builder.device, ChatDevice::Gpu);
        assert_eq!(builder.mode, ChatMode::Creative);
        assert_eq!(builder.system_prompt, Some("Be creative".to_string()));
        assert_eq!(builder.generation_overrides.temperature, Some(0.9));
        assert_eq!(builder.generation_overrides.max_new_tokens, Some(2048));
        assert!(builder.quiet);
        assert_eq!(builder.download_policy, DownloadPolicy::Never);
    }

    #[test]
    fn test_builder_from_preset() {
        let builder = ChatBuilder::from_preset(&presets::CHAT_SMALL_V1);

        assert_eq!(builder.model, presets::CHAT_SMALL_V1.model);
        assert_eq!(builder.device, presets::CHAT_SMALL_V1.recommended_device);
    }
}

// =============================================================================
// Preset Tests
// =============================================================================

mod preset_tests {
    use super::*;

    #[test]
    fn test_chat_small_v1_preset() {
        let preset = &presets::CHAT_SMALL_V1;

        assert_eq!(preset.name, "CHAT_SMALL_V1");
        assert_eq!(preset.model, "llama3.2-1b");
        assert!(preset.context_length > 0);
        assert!(preset.memory_mb > 0);
    }

    #[test]
    fn test_chat_medium_v1_preset() {
        let preset = &presets::CHAT_MEDIUM_V1;

        assert_eq!(preset.name, "CHAT_MEDIUM_V1");
        assert_eq!(preset.model, "llama3.2-3b");
    }

    #[test]
    fn test_chat_large_v1_preset() {
        let preset = &presets::CHAT_LARGE_V1;

        assert_eq!(preset.name, "CHAT_LARGE_V1");
        assert_eq!(preset.model, "llama3.1-8b");
    }

    #[test]
    fn test_reasoning_v1_preset() {
        let preset = &presets::REASONING_V1;

        assert_eq!(preset.name, "REASONING_V1");
        assert!(preset.description.contains("reasoning"));
    }

    #[test]
    fn test_chat_tiny_v1_preset() {
        let preset = &presets::CHAT_TINY_V1;

        assert_eq!(preset.name, "CHAT_TINY_V1");
        assert!(preset.memory_mb < presets::CHAT_SMALL_V1.memory_mb);
    }

    #[test]
    fn test_find_preset_exists() {
        let preset = presets::find_preset("CHAT_SMALL_V1");
        assert!(preset.is_some());
        assert_eq!(preset.unwrap().name, "CHAT_SMALL_V1");
    }

    #[test]
    fn test_find_preset_case_insensitive() {
        let preset = presets::find_preset("chat_small_v1");
        assert!(preset.is_some());
    }

    #[test]
    fn test_find_preset_not_found() {
        let preset = presets::find_preset("NONEXISTENT");
        assert!(preset.is_none());
    }

    #[test]
    fn test_chat_tier_resolve() {
        let tiny = presets::ChatTier::Tiny.resolve();
        let small = presets::ChatTier::Small.resolve();
        let medium = presets::ChatTier::Medium.resolve();
        let large = presets::ChatTier::Large.resolve();

        assert_eq!(tiny.name, "CHAT_TINY_V1");
        assert_eq!(small.name, "CHAT_SMALL_V1");
        assert_eq!(medium.name, "CHAT_MEDIUM_V1");
        assert_eq!(large.name, "CHAT_LARGE_V1");
    }

    #[test]
    fn test_all_presets_valid() {
        for preset in presets::ALL_V1_PRESETS {
            assert!(!preset.name.is_empty());
            assert!(!preset.model.is_empty());
            assert!(preset.context_length > 0);
            assert!(preset.memory_mb > 0);
            assert!(!preset.description.is_empty());
        }
    }
}

// =============================================================================
// Validation Tests
// =============================================================================

mod validation_tests {
    use super::*;
    use kjarni_transformers::models::ModelType;

    #[test]
    fn test_validate_chat_model() {
        if let Some(model_type) = ModelType::from_cli_name("llama3.2-1b") {
            let result = validation::validate_for_chat(model_type);
            assert!(result.is_ok(), "Llama should be valid for chat");
        }
    }

    #[test]
    fn test_validate_encoder_model_invalid() {
        if let Some(model_type) = ModelType::from_cli_name("minilm-l6-v2") {
            let result = validation::validate_for_chat(model_type);
            assert!(result.is_err(), "Encoder should not be valid for chat");
        }
    }

    #[test]
    fn test_validate_seq2seq_model_invalid() {
        if let Some(model_type) = ModelType::from_cli_name("flan-t5-base") {
            let result = validation::validate_for_chat(model_type);
            assert!(result.is_err(), "T5 should not be valid for chat");
        }
    }

    #[test]
    fn test_is_decoder_architecture() {
        use kjarni_transformers::models::ModelArchitecture;

        assert!(validation::is_decoder_architecture(ModelArchitecture::Llama));
        assert!(validation::is_decoder_architecture(ModelArchitecture::Qwen2));
        assert!(validation::is_decoder_architecture(ModelArchitecture::Mistral));
        assert!(validation::is_decoder_architecture(ModelArchitecture::GPT));
        assert!(!validation::is_decoder_architecture(ModelArchitecture::Bert));
        assert!(!validation::is_decoder_architecture(ModelArchitecture::T5));
    }

    #[test]
    fn test_suggest_chat_models() {
        let suggestions = validation::suggest_chat_models();
        assert!(!suggestions.is_empty());
        assert!(suggestions.contains(&"llama3.2-1b"));
    }
}

// =============================================================================
// Error Tests
// =============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn test_error_display_unknown_model() {
        let err = ChatError::UnknownModel("fake-model".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("fake-model"));
        assert!(msg.contains("Unknown model"));
    }

    #[test]
    fn test_error_display_incompatible_model() {
        let err = ChatError::IncompatibleModel {
            model: "test-model".to_string(),
            reason: "not a decoder".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("test-model"));
        assert!(msg.contains("not a decoder"));
    }

    #[test]
    fn test_error_display_not_downloaded() {
        let err = ChatError::ModelNotDownloaded("test-model".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test-model"));
        assert!(msg.contains("not downloaded"));
    }

    #[test]
    fn test_error_display_gpu_unavailable() {
        let err = ChatError::GpuUnavailable;
        let msg = format!("{}", err);
        assert!(msg.contains("GPU"));
    }

    #[test]
    fn test_chat_warning_display() {
        let warning = ChatWarning {
            message: "Test warning".to_string(),
            suggestion: Some("Try this instead".to_string()),
        };
        let msg = format!("{}", warning);
        assert!(msg.contains("Test warning"));
        assert!(msg.contains("Try this instead"));
    }

    #[test]
    fn test_chat_warning_no_suggestion() {
        let warning = ChatWarning {
            message: "Test warning".to_string(),
            suggestion: None,
        };
        let msg = format!("{}", warning);
        assert!(msg.contains("Test warning"));
    }
}

// =============================================================================
// Convenience Function Tests
// =============================================================================

mod convenience_tests {
    use super::*;

    #[test]
    fn test_is_chat_model_valid() {
        // Llama should be valid
        let result = is_chat_model("llama3.2-1b");
        assert!(result.is_ok());
    }

    #[test]
    fn test_is_chat_model_invalid_encoder() {
        let result = is_chat_model("minilm-l6-v2");
        assert!(result.is_err());
    }

    #[test]
    fn test_is_chat_model_unknown() {
        let result = is_chat_model("nonexistent-model-12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_available_models_returns_list() {
        let models = available_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"llama3.2-1b"));
    }

    #[test]
    fn test_suggested_models_returns_list() {
        let models = suggested_models();
        assert!(!models.is_empty());
    }
}

// =============================================================================
// Conversation Tests
// =============================================================================

mod conversation_tests {
    use super::*;

    // Note: Can't test ChatConversation without a Chat instance
    // These would be integration tests

    #[test]
    fn test_history_iteration() {
        let mut history = History::new();
        history.push_user("Hello");
        history.push_assistant("Hi!");
        history.push_user("How are you?");

        let messages: Vec<_> = history.messages().iter().collect();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(messages[2].role, Role::User);
    }
}

// =============================================================================
// Integration Tests (require model download)
// =============================================================================

mod integration_tests {
    use super::*;

    /// Test that we can create a chat instance.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_chat_new() {
        let chat = Chat::new("llama3.2-1b").await;
        assert!(chat.is_ok(), "Failed to create chat: {:?}", chat.err());
    }

    /// Test single message send.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_send_message() {
        let chat = Chat::new("llama3.2-1b")
            .await
            .expect("Failed to load chat");

        let response = chat.send("Say hello in one word")
            .await
            .expect("Send failed");

        assert!(!response.is_empty());
    }

    /// Test send with system prompt.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_send_with_system() {
        let chat = Chat::builder("llama3.2-1b-instruct")
            .system_prompt("You are a pirate. Always respond like a pirate.")
            .build()
            .await
            .expect("Failed to load chat");

        let response = chat.send("Hello")
            .await
            .expect("Send failed");

        // Response should have pirate-like language (hard to test precisely)
        assert!(!response.is_empty());
    }

    /// Test send with history.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_send_with_history() {
        let chat = Chat::new("llama3.2-1b-instruct")
            .await
            .expect("Failed to load chat");

        let mut history = History::new();
        history.push_user("My name is Alice");
        history.push_assistant("Nice to meet you, Alice!");

        let response = chat.send_with_history(&history, "What is my name?")
            .await
            .expect("Send failed");

        // Should remember the name from history
        assert!(!response.is_empty());
    }

    /// Test send with custom config.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_send_with_config() {
        let chat = Chat::new("llama3.2-1b-instruct")
            .await
            .expect("Failed to load chat");

        let overrides = GenerationOverrides {
            temperature: Some(0.1),
            max_new_tokens: Some(10),
            ..Default::default()
        };

        let response = chat.send_with_config("Say hi", &overrides)
            .await
            .expect("Send failed");

        // Should be short due to max_new_tokens
        assert!(!response.is_empty());
    }

    /// Test conversation state.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_conversation() {
        let chat = Chat::new("llama3.2-1b-instruct")
            .await
            .expect("Failed to load chat");

        let mut convo = chat.conversation();

        let response1 = convo.send("My favorite color is blue")
            .await
            .expect("First send failed");
        assert!(!response1.is_empty());

        // History should have 2 messages now (user + assistant)
        assert_eq!(convo.len(), 2);

        let response2 = convo.send("What is my favorite color?")
            .await
            .expect("Second send failed");
        assert!(!response2.is_empty());

        // Should have 4 messages now
        assert_eq!(convo.len(), 4);
    }

    /// Test conversation with custom system.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_conversation_with_system() {
        let chat = Chat::new("llama3.2-1b-instruct")
            .await
            .expect("Failed to load chat");

        let mut convo = chat.conversation_with_system("Always respond in exactly 3 words");

        let response = convo.send("Hello")
            .await
            .expect("Send failed");
        println!("Response: {}", response);
        
        assert!(!response.is_empty());
        // First message should be system prompt
        assert_eq!(convo.history().messages()[0].role, Role::System);
    }

    /// Test conversation clear.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_conversation_clear() {
        let chat = Chat::builder("llama3.2-1b-instruct")
            .system_prompt("Test system")
            .build()
            .await
            .expect("Failed to load chat");

        let mut convo = chat.conversation();

        convo.send("Hello").await.expect("Send failed");
        assert!(convo.len() > 1);

        convo.clear();

        // Should keep system prompt
        assert_eq!(convo.len(), 1);
        assert_eq!(convo.history().messages()[0].role, Role::System);
    }

    /// Test streaming (basic test, not full stream consumption).
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_stream() {
        use futures_util::StreamExt;

        let chat = Chat::new("llama3.2-1b-instruct")
            .await
            .expect("Failed to load chat");

        let mut stream = chat.stream("Say hello")
            .await
            .expect("Stream creation failed");

        let mut tokens = Vec::new();
        while let Some(result) = stream.next().await {
            match result {
                Ok(text) => tokens.push(text),
                Err(e) => panic!("Stream error: {}", e),
            }
            // Limit iterations for test
            if tokens.len() > 10 {
                break;
            }
        }

        assert!(!tokens.is_empty());
    }

    /// Test GPU chat (if available).
    #[tokio::test]
    #[ignore = "Requires GPU and model download"]
    async fn test_chat_gpu() {
        let chat = Chat::builder("llama3.2-1b-instruct")
            .gpu()
            .build()
            .await
            .expect("Failed to load chat on GPU");

        let response = chat.send("Hello")
            .await
            .expect("GPU send failed");

        assert!(!response.is_empty());
    }

    /// Test chat accessors.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_chat_accessors() {
        let chat = Chat::builder("llama3.2-1b-instruct")
            .system_prompt("Test system")
            .mode(ChatMode::Creative)
            .build()
            .await
            .expect("Failed to load chat");

        assert_eq!(chat.model_name(), "llama3.2-1b");
        assert_eq!(chat.mode(), ChatMode::Creative);
        assert_eq!(chat.system_prompt(), Some("Test system"));
        assert!(chat.context_size() > 0);
    }

    /// Test one-liner send function.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_send_convenience_function() {
        let response = send("llama3.2-1b-instruct", "Say hello")
            .await
            .expect("Send function failed");

        assert!(!response.is_empty());
    }
    #[test]
    fn test_is_chat_model() {
        // Valid chat models
        assert!(is_chat_model("llama3.2-1b-instruct").is_ok());
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
        assert!(models.contains(&"llama3.2-1b-instruct"));
    }
    /// Test unknown model error.
    #[tokio::test]
    async fn test_unknown_model_error() {
        let result = Chat::new("completely-fake-model-that-does-not-exist").await;
        assert!(matches!(result, Err(ChatError::UnknownModel(_))));
    }

    /// Test offline mode with missing model.
    #[tokio::test]
    async fn test_offline_missing_model() {
        let result = Chat::builder("llama3.2-1b-instruct")
            .offline()
            .cache_dir("/tmp/kjarni-test-empty-cache-12345")
            .build()
            .await;

        match result {
            Err(e) => assert!(matches!(e, ChatError::ModelNotDownloaded(_)),
                "Expected ModelNotDownloaded error, got {:?}", e),
            Ok(_) => assert!(false, "Expected error but got Ok"),
        }
        
        // Should fail because model not downloaded and offline mode
        
    }

    /// Test different modes affect generation.
    #[tokio::test]
    #[ignore = "Requires model download"]
    async fn test_modes() {
        // Just test that different modes don't crash
        for mode in [ChatMode::Default, ChatMode::Reasoning, ChatMode::Creative] {
            let chat = Chat::builder("llama3.2-1b")
                .mode(mode)
                .max_tokens(20)
                .build()
                .await
                .expect("Failed to load chat");

            let response = chat.send("Test")
                .await
                .expect("Send failed");

            assert!(!response.is_empty());
        }
    }
}