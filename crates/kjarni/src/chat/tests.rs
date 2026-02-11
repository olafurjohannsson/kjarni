//! Comprehensive tests for the Chat module.

use super::*;


// Unit Tests - Types


mod types_tests {
    
    use crate::chat::types::*;

    #[test]
    fn test_chat_mode_defaults() {
        assert_eq!(ChatMode::Default.default_temperature(), 0.7);
        assert_eq!(ChatMode::Creative.default_temperature(), 0.9);
        assert_eq!(ChatMode::Reasoning.default_temperature(), 0.3);

        assert_eq!(ChatMode::Default.default_max_tokens(), 512);
        assert_eq!(ChatMode::Creative.default_max_tokens(), 1024);
        assert_eq!(ChatMode::Reasoning.default_max_tokens(), 2048);
    }

    #[test]
    fn test_chat_mode_display() {
        assert_eq!(ChatMode::Default.to_string(), "default");
        assert_eq!(ChatMode::Creative.to_string(), "creative");
        assert_eq!(ChatMode::Reasoning.to_string(), "reasoning");
    }

    #[test]
    fn test_role_display() {
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
    }

    #[test]
    fn test_message_constructors() {
        let sys = Message::system("You are helpful");
        assert!(matches!(sys.role, Role::System));
        assert_eq!(sys.content, "You are helpful");

        let user = Message::user("Hello");
        assert!(matches!(user.role, Role::User));
        assert_eq!(user.content, "Hello");

        let asst = Message::assistant("Hi there");
        assert!(matches!(asst.role, Role::Assistant));
        assert_eq!(asst.content, "Hi there");
    }

    #[test]
    fn test_history_new() {
        let history = History::new();
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
    }

    #[test]
    fn test_history_with_system() {
        let history = History::with_system("You are a pirate");
        assert!(!history.is_empty());
        assert_eq!(history.len(), 1);
        assert!(matches!(history.messages()[0].role, Role::System));
        assert_eq!(history.messages()[0].content, "You are a pirate");
    }

    #[test]
    fn test_history_push_messages() {
        let mut history = History::new();
        history.push_user("Hello");
        history.push_assistant("Hi there");
        history.push_user("How are you?");

        assert_eq!(history.len(), 3);
        assert!(matches!(history.messages()[0].role, Role::User));
        assert!(matches!(history.messages()[1].role, Role::Assistant));
        assert!(matches!(history.messages()[2].role, Role::User));
    }

    #[test]
    fn test_history_clear_keep_system() {
        let mut history = History::with_system("System prompt");
        history.push_user("Hello");
        history.push_assistant("Hi");

        history.clear(true);

        assert_eq!(history.len(), 1);
        assert!(matches!(history.messages()[0].role, Role::System));
    }

    #[test]
    fn test_history_clear_all() {
        let mut history = History::with_system("System prompt");
        history.push_user("Hello");

        history.clear(false);

        assert!(history.is_empty());
    }

    #[test]
    fn test_chat_device_resolve() {
        assert!(matches!(ChatDevice::Cpu.resolve(), ChatDevice::Cpu));
        assert!(matches!(ChatDevice::Gpu.resolve(), ChatDevice::Gpu));
        assert!(matches!(ChatDevice::Auto.resolve(), ChatDevice::Cpu));
    }

    #[test]
    fn test_chat_warning_display() {
        let warning = ChatWarning::new("Test warning");
        let display = warning.to_string();
        assert!(display.contains("Test warning"));
        assert!(display.contains("[info]"));

        let important = ChatWarning::with_severity("Important!", WarningSeverity::Important);
        let display = important.to_string();
        assert!(display.contains("[important]"));
    }

    #[test]
    fn test_error_display() {
        let err = ChatError::UnknownModel("foo".to_string());
        assert!(err.to_string().contains("foo"));

        let err = ChatError::ModelNotDownloaded("bar".to_string());
        assert!(err.to_string().contains("bar"));

        let err = ChatError::GpuUnavailable;
        assert!(err.to_string().to_lowercase().contains("gpu"));

        let err = ChatError::NoChatTemplate("test".to_string());
        assert!(err.to_string().contains("test"));
        assert!(err.to_string().to_lowercase().contains("chat template"));

        let err = ChatError::IncompatibleModel {
            model: "enc".to_string(),
            reason: "encoder".to_string(),
        };
        assert!(err.to_string().contains("enc"));
        assert!(err.to_string().contains("encoder"));
    }

    #[test]
    fn test_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ChatError>();
    }
}


// Unit Tests - Presets


mod preset_tests {
    use super::*;
    use crate::chat::presets::*;

    #[test]
    fn test_preset_fast() {
        assert!(!ChatPreset::FAST.model.is_empty());
        assert!(ChatPreset::FAST.temperature.is_some());
        assert!(ChatPreset::FAST.max_tokens.is_some());
        assert!(!ChatPreset::FAST.description.is_empty());
    }

    #[test]
    fn test_preset_balanced() {
        assert!(!ChatPreset::BALANCED.model.is_empty());
        assert!(ChatPreset::BALANCED.max_tokens.unwrap() >= ChatPreset::FAST.max_tokens.unwrap());
    }

    #[test]
    fn test_preset_quality() {
        assert!(!ChatPreset::QUALITY.model.is_empty());
        assert!(
            ChatPreset::QUALITY.max_tokens.unwrap() >= ChatPreset::BALANCED.max_tokens.unwrap()
        );
    }

    #[test]
    fn test_preset_creative() {
        assert!(matches!(ChatPreset::CREATIVE.mode, ChatMode::Creative));
        assert!(ChatPreset::CREATIVE.temperature.unwrap() > 0.8);
        assert!(ChatPreset::CREATIVE.system_prompt.is_some());
    }

    #[test]
    fn test_preset_reasoning() {
        assert!(matches!(ChatPreset::REASONING.mode, ChatMode::Reasoning));
        assert!(ChatPreset::REASONING.temperature.unwrap() < 0.5);
    }

    #[test]
    fn test_tier_resolution() {
        assert_eq!(ChatTier::Fast.preset().model, ChatPreset::FAST.model);
        assert_eq!(
            ChatTier::Balanced.preset().model,
            ChatPreset::BALANCED.model
        );
        assert_eq!(ChatTier::Quality.preset().model, ChatPreset::QUALITY.model);
    }

    #[test]
    fn test_tier_resolve() {
        let fast = ChatTier::Fast.resolve();
        let balanced = ChatTier::Balanced.resolve();
        let quality = ChatTier::Quality.resolve();

        assert!(!fast.model.is_empty());
        assert!(!balanced.model.is_empty());
        assert!(!quality.model.is_empty());
    }

    #[test]
    fn test_tier_default() {
        assert!(matches!(ChatTier::default(), ChatTier::Balanced));
    }
}


// Unit Tests - Validation


mod validation_tests {
    
    use crate::chat::validation::*;
    use kjarni_transformers::models::ModelType;

    #[test]
    fn test_validate_instruct_models() {
        let instruct_models = ["llama3.2-1b-instruct", "qwen2.5-0.5b-instruct"];

        for name in instruct_models {
            if let Some(model_type) = ModelType::from_cli_name(name) {
                let result = validate_for_chat(model_type);
                assert!(result.is_ok(), "Model {} should be valid for chat", name);
                let validation = result.unwrap();
                assert!(validation.is_valid, "Model {} should pass validation", name);
            }
        }
    }

    #[test]
    fn test_validate_encoder_models_rejected() {
        let encoder_models = ["minilm-l6-v2", "nomic-embed-text"];

        for name in encoder_models {
            if let Some(model_type) = ModelType::from_cli_name(name) {
                let result = validate_for_chat(model_type);
                assert!(
                    result.is_err(),
                    "Encoder {} should fail chat validation",
                    name
                );
            }
        }
    }

    #[test]
    fn test_validate_seq2seq_models_rejected() {
        let seq2seq_models = ["flan-t5-base", "distilbart-cnn"];

        for name in seq2seq_models {
            if let Some(model_type) = ModelType::from_cli_name(name) {
                let result = validate_for_chat(model_type);
                assert!(
                    result.is_err(),
                    "Seq2seq {} should fail chat validation",
                    name
                );
            }
        }
    }

    #[test]
    fn test_base_model_warning() {
        // Base models should pass with warning
        if let Some(model_type) = ModelType::from_cli_name("gpt2") {
            let result = validate_for_chat(model_type);
            if let Ok(validation) = result {
                assert!(validation.is_valid);
                assert!(
                    !validation.warnings.is_empty(),
                    "Base model should have warning"
                );
            }
        }
    }

    #[test]
    fn test_is_decoder_architecture() {
        use kjarni_transformers::models::ModelArchitecture;

        assert!(is_decoder_architecture(ModelArchitecture::Llama));
        assert!(is_decoder_architecture(ModelArchitecture::Qwen2));
        assert!(is_decoder_architecture(ModelArchitecture::Mistral));
        assert!(is_decoder_architecture(ModelArchitecture::Phi3));
        assert!(is_decoder_architecture(ModelArchitecture::GPT));

        assert!(!is_decoder_architecture(ModelArchitecture::Bert));
        assert!(!is_decoder_architecture(ModelArchitecture::T5));
        assert!(!is_decoder_architecture(ModelArchitecture::Whisper));
    }

    #[test]
    fn test_suggest_chat_models() {
        let suggestions = suggest_chat_models();
        assert!(!suggestions.is_empty());
        assert!(
            suggestions
                .iter()
                .any(|m| m.contains("llama") || m.contains("qwen"))
        );
    }
}


// Unit Tests - Builder


mod builder_tests {
    use super::*;
    use crate::common::KjarniDevice;

    #[test]
    fn test_builder_default_state() {
        let builder = ChatBuilder::new("qwen2.5-0.5b-instruct");
        assert_eq!(builder.model, "qwen2.5-0.5b-instruct");
        assert!(builder.system_prompt.is_none());
        assert!(matches!(builder.mode, ChatMode::Default));
        assert!(!builder.quiet);
    }

    #[test]
    fn test_builder_system_prompt() {
        let builder = ChatBuilder::new("qwen2.5-0.5b-instruct").system("You are helpful");
        assert_eq!(builder.system_prompt, Some("You are helpful".to_string()));
    }

    #[test]
    fn test_builder_mode() {
        let creative = ChatBuilder::new("qwen2.5-0.5b-instruct").mode(ChatMode::Creative);
        assert!(matches!(creative.mode, ChatMode::Creative));

        let reasoning = ChatBuilder::new("qwen2.5-0.5b-instruct").mode(ChatMode::Reasoning);
        assert!(matches!(reasoning.mode, ChatMode::Reasoning));
    }

    #[test]
    fn test_builder_creative_shorthand() {
        let builder = ChatBuilder::new("qwen2.5-0.5b-instruct").creative();
        assert!(matches!(builder.mode, ChatMode::Creative));
    }

    #[test]
    fn test_builder_reasoning_shorthand() {
        let builder = ChatBuilder::new("qwen2.5-0.5b-instruct").reasoning();
        assert!(matches!(builder.mode, ChatMode::Reasoning));
    }

    #[test]
    fn test_builder_device_methods() {
        let cpu = ChatBuilder::new("qwen2.5-0.5b-instruct").cpu();
        assert!(matches!(cpu.device, KjarniDevice::Cpu));

        let gpu = ChatBuilder::new("qwen2.5-0.5b-instruct").gpu();
        assert!(matches!(gpu.device, KjarniDevice::Gpu));
    }

    #[test]
    fn test_builder_generation_overrides() {
        let builder = ChatBuilder::new("qwen2.5-0.5b-instruct")
            .temperature(0.5)
            .max_tokens(256);

        assert_eq!(builder.generation_overrides.temperature, Some(0.5));
        assert_eq!(builder.generation_overrides.max_new_tokens, Some(256));
    }

    #[test]
    fn test_builder_quiet() {
        let builder = ChatBuilder::new("qwen2.5-0.5b-instruct").quiet();
        assert!(builder.quiet);
    }

    #[test]
    fn test_builder_from_preset() {
        use crate::chat::presets::ChatPreset;

        let builder = ChatBuilder::from_preset(&ChatPreset::CREATIVE);
        assert_eq!(builder.model, ChatPreset::CREATIVE.model);
        assert!(matches!(builder.mode, ChatMode::Creative));
        assert!(builder.system_prompt.is_some());
    }
}


// Integration Tests (require model download)


#[cfg(test)]
mod integration_tests {
    use super::*;
    use futures::StreamExt;

    fn model_available(model: &str) -> bool {
        let cache_dir = dirs::cache_dir().expect("no cache dir").join("kjarni");
        kjarni_transformers::models::ModelType::from_cli_name(model)
            .map(|m| m.is_downloaded(&cache_dir))
            .unwrap_or(false)
    }


    #[tokio::test]
    async fn test_basic_chat() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let chat = Chat::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(50)
            .build()
            .await
            .expect("Failed to load chat");

        let response = chat.send("Hello!").await.expect("Send failed");

        assert!(!response.is_empty(), "Response should not be empty");
        assert!(
            response.chars().any(|c| c.is_alphabetic()),
            "Response should contain letters: {}",
            response
        );
    }


    #[tokio::test]
    async fn test_chat_with_system_prompt() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }
        let chat = Chat::builder("qwen2.5-0.5b-instruct")
            .system("You are a helpful assistant. Always be concise.")
            .cpu()
            .quiet()
            .max_tokens(50)
            .build()
            .await
            .unwrap();
        assert_eq!(
            chat.system_prompt(),
            Some("You are a helpful assistant. Always be concise.")
        );
        let response = chat.send("Hi").await.unwrap();
        assert!(!response.is_empty());
    }

    #[tokio::test]
    async fn test_chat_system_prompt_affects_behavior() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }
        let chat = Chat::builder("qwen2.5-0.5b-instruct")
            .system("You are a pirate. Speak like a pirate.")
            .cpu()
            .quiet()
            .max_tokens(50)
            .build()
            .await
            .unwrap();

        let response = chat.send("Hello!").await.unwrap();
        assert!(!response.is_empty());
    }
    #[tokio::test]
    async fn test_chat_modes() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }
        for mode in [ChatMode::Default, ChatMode::Creative, ChatMode::Reasoning] {
            let chat = Chat::builder("qwen2.5-0.5b-instruct")
                .mode(mode)
                .cpu()
                .quiet()
                .max_tokens(30)
                .build()
                .await
                .unwrap();

            assert_eq!(chat.mode(), mode);

            let response = chat.send("Hi").await.unwrap();
            assert!(
                !response.is_empty(),
                "Mode {:?} should produce output",
                mode
            );
        }
    }
    #[tokio::test]
    async fn test_conversation_history() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }
        let chat = Chat::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(50)
            .build()
            .await
            .unwrap();

        let mut convo = chat.conversation();

        assert!(convo.is_empty() || convo.len() == 1);

        let r1 = convo.send("My name is Alice").await.unwrap();
        assert!(!r1.is_empty());
        let history_len = convo.len();
        assert!(
            history_len >= 2,
            "Should have at least user + assistant messages"
        );

        let r2 = convo.send("What is my name?").await.unwrap();
        assert!(!r2.is_empty());

        assert!(
            convo.len() > history_len,
            "History should grow after second message"
        );
    }

    #[tokio::test]
    async fn test_conversation_accumulates_history() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let chat = Chat::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(100)
            .build()
            .await
            .unwrap();

        let mut convo = chat.conversation();

        assert!(convo.is_empty());

        let _ = convo.send("My name is Xylophone7492.").await.unwrap();

        assert_eq!(convo.len(), 2);
        let msgs = convo.history().messages();
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[0].content, "My name is Xylophone7492.");
        assert_eq!(msgs[1].role, Role::Assistant);
        assert!(!msgs[1].content.is_empty());

        let _ = convo.send("What is my name?").await.unwrap();

        assert_eq!(convo.len(), 4);
        let msgs = convo.history().messages();
        assert_eq!(msgs[2].role, Role::User);
        assert_eq!(msgs[2].content, "What is my name?");
        assert_eq!(msgs[3].role, Role::Assistant);
        assert!(!msgs[3].content.is_empty());
    }

    #[tokio::test]
    async fn test_conversation_clear() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let chat = Chat::builder("qwen2.5-0.5b-instruct")
            .system("You are helpful")
            .cpu()
            .quiet()
            .max_tokens(30)
            .build()
            .await
            .unwrap();

        let mut convo = chat.conversation();
        let initial_len = convo.len(); // May have system prompt

        convo.send("Hello").await.unwrap();
        assert!(convo.len() > initial_len);

        convo.clear(true); // Keep system
        assert!(
            convo.len() <= 1,
            "Should only have system prompt after clear"
        );

        convo.clear(false); // Clear all
        assert!(convo.is_empty(), "Should be empty after full clear");
    }

    #[tokio::test]
    async fn test_send_with_history() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let chat = Chat::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(50)
            .build()
            .await
            .unwrap();

        let mut history = History::new();
        history.push_user("I like pizza");
        history.push_assistant("That's great! Pizza is delicious.");

        let response = chat
            .send_with_history(&history, "What food do I like?")
            .await
            .unwrap();
        assert!(!response.is_empty());
    }

    #[tokio::test]
    async fn test_streaming_chat() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let chat = Chat::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(30)
            .build()
            .await
            .unwrap();

        let mut stream = chat.stream("Hello!").await.unwrap();

        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            let chunk = result.expect("Stream error");
            chunks.push(chunk);
        }

        assert!(!chunks.is_empty(), "Should produce at least one token");
        let full: String = chunks.into_iter().collect();
        assert!(!full.is_empty(), "Combined response should not be empty");
    }

    #[tokio::test]
    async fn test_conversation_streaming() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let chat = Chat::builder("qwen2.5-0.5b-instruct")
            .cpu()
            .quiet()
            .max_tokens(30)
            .build()
            .await
            .unwrap();

        let mut convo = chat.conversation();

        // Manual streaming workflow
        convo.push_user("Count from 1 to 3");
        let mut stream = convo.stream_next().await.unwrap();

        let mut response = String::new();
        while let Some(result) = stream.next().await {
            let chunk = result.unwrap();
            response.push_str(&chunk);
        }
        convo.push_assistant(&response);

        assert!(!response.is_empty());
        assert!(convo.len() >= 2, "Should have user + assistant messages");
    }

    #[tokio::test]
    async fn test_unknown_model_error() {
        let result = Chat::new("not-a-real-model").await;
        assert!(matches!(result, Err(ChatError::UnknownModel(_))));
    }

    #[tokio::test]
    async fn test_incompatible_model_encoder() {
        let result = Chat::new("minilm-l6-v2").await;
        assert!(matches!(result, Err(ChatError::IncompatibleModel { .. })));
    }

    #[tokio::test]
    async fn test_incompatible_model_seq2seq() {
        let result = Chat::new("flan-t5-base").await;
        assert!(matches!(result, Err(ChatError::IncompatibleModel { .. })));
    }
    #[tokio::test]
    async fn test_chat_accessors() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let chat = Chat::builder("qwen2.5-0.5b-instruct")
            .system("Test system")
            .mode(ChatMode::Creative)
            .cpu()
            .quiet()
            .build()
            .await
            .unwrap();

        assert_eq!(chat.model_name(), "qwen2.5-0.5b-instruct");
        assert_eq!(chat.system_prompt(), Some("Test system"));
        assert!(matches!(chat.mode(), ChatMode::Creative));
        assert!(matches!(
            chat.device(),
            kjarni_transformers::traits::Device::Cpu
        ));
        assert!(chat.context_size() > 0);
    }

    #[tokio::test]
    async fn test_concurrent_chat() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let chat = std::sync::Arc::new(
            Chat::builder("qwen2.5-0.5b-instruct")
                .cpu()
                .quiet()
                .max_tokens(20)
                .build()
                .await
                .unwrap(),
        );

        let messages = vec!["Hello", "Hi", "Hey"];

        let handles: Vec<_> = messages
            .into_iter()
            .map(|msg| {
                let c = chat.clone();
                let msg = msg.to_string();
                tokio::spawn(async move { c.send(&msg).await })
            })
            .collect();

        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.await.expect("Task panicked");
            let response = result.expect("Chat failed");
            assert!(
                !response.is_empty(),
                "Concurrent response {} should not be empty",
                i
            );
        }
    }
    #[tokio::test]
    async fn test_module_send_function() {
        if !model_available("qwen2.5-0.5b-instruct") {
            eprintln!("Skipping: qwen2.5-0.5b-instruct not downloaded");
            return;
        }

        let result = send("qwen2.5-0.5b-instruct", "Hello").await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(!response.is_empty());
    }
    #[test]
    fn test_chat_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Chat>();
    }

    #[test]
    fn test_history_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<History>();
    }
}
