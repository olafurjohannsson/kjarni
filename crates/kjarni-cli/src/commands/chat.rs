//! Interactive chat command

use anyhow::{Result, anyhow};
use futures_util::StreamExt;
use std::io::{self, BufRead, Write};

use kjarni::{
    ChatTemplate, Conversation, DecoderGenerator, DecoderLanguageModel, DecodingStrategy, Device,
    GenerationConfig, Llama3ChatTemplate, ModelArchitecture, ModelType, SamplingParams, TokenType,
    models::{LlamaModel, QwenModel},
    registry,
};

pub async fn run(
    model: &str,
    model_path: Option<&str>,
    system_prompt: Option<&str>,
    temperature: f32,
    max_tokens: usize,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // 1. Resolve model
    let device = if gpu { Device::Wgpu } else { Device::Cpu };

    if model_path.is_some() {
        return Err(anyhow!("--model-path not yet implemented."));
    }

    let model_type = ModelType::from_cli_name(model).ok_or_else(|| {
        let mut msg = format!("Unknown model: '{}'.", model);
        let suggestions = ModelType::find_similar(model);
        if !suggestions.is_empty() {
            msg.push_str("\n\nDid you mean?");
            for (name, _) in suggestions {
                msg.push_str(&format!("\n  - {}", name));
            }
        }
        anyhow!(msg)
    })?;

    if model_type.architecture() != ModelArchitecture::Llama
        || model_type.architecture() != ModelArchitecture::Qwen2
        || model_type.architecture() != ModelArchitecture::Mistral
    {
        return Err(anyhow!(
            "Model '{}' is not a decoder. Chat requires a decoder model.",
            model
        ));
    }

    // Warn if using non-instruct model
    if !model_type.is_instruct_model() && !quiet {
        eprintln!(
            "Warning: '{}' is not an instruct-tuned model. Chat quality may be poor.",
            model
        );
        eprintln!("Consider using: llama-3.2-3b-instruct or llama-3-8b-instruct");
        eprintln!();
    }

    // Check if downloaded
    if !registry::is_model_downloaded(model)? {
        if !quiet {
            eprintln!("Model '{}' not found locally. Downloading...", model);
        }
        registry::download_model(model, false).await?;
        if !quiet {
            eprintln!();
        }
    }

    // 2. Load model
    if !quiet {
        eprintln!("Loading model '{}'...", model);
    }

    let loaded_model: Box<dyn DecoderLanguageModel> = if model_type.is_llama_model() {
        Box::new(LlamaModel::from_registry(model_type, None, device, None, None).await?)
    } else if model_type.is_qwen_model() {
        Box::new(QwenModel::from_registry(model_type, None, device, None, None).await?)
    } else {
        return Err(anyhow!(
            "Model '{}' not yet supported for chat. Try: llama-3.2-3b-instruct",
            model
        ));
    };

    let generator = DecoderGenerator::new(loaded_model)?;

    // 3. Setup chat template and conversation
    let template = Llama3ChatTemplate::for_generation();

    let default_system = template
        .default_system_prompt()
        .unwrap_or("You are a helpful, harmless, and honest assistant.");

    let mut conversation = match system_prompt {
        Some(prompt) => Conversation::with_system(prompt),
        None => Conversation::with_system(default_system),
    };

    // 4. Generation config
    let config = GenerationConfig {
        max_new_tokens: Some(max_tokens),
        strategy: DecodingStrategy::Sample(SamplingParams {
            temperature,
            top_k: Some(50),
            top_p: Some(0.9),
            min_p: Some(0.1),
        }),
        add_bos_token: false, // Template adds BOS
        ..Default::default()
    };

    // 5. Interactive loop
    if !quiet {
        eprintln!();
        eprintln!("Chat started. Type '/help' for commands, '/quit' to exit.");
        eprintln!();
    }

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Print prompt
        print!("> ");
        stdout.flush()?;

        // Read user input
        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            // EOF
            break;
        }

        let input = input.trim();

        // Handle empty input
        if input.is_empty() {
            continue;
        }

        // Handle commands
        if input.starts_with('/') {
            match input {
                "/quit" | "/exit" | "/q" => {
                    if !quiet {
                        eprintln!("Goodbye!");
                    }
                    break;
                }
                "/clear" | "/reset" => {
                    conversation.clear(true); // Keep system prompt
                    if !quiet {
                        eprintln!("Conversation cleared.");
                    }
                    continue;
                }
                "/help" | "/?" => {
                    print_help();
                    continue;
                }
                "/system" => {
                    if let Some(prompt) = conversation.system_prompt() {
                        println!("System: {}", prompt);
                    } else {
                        println!("No system prompt set.");
                    }
                    continue;
                }
                "/history" => {
                    print_history(&conversation);
                    continue;
                }
                cmd if cmd.starts_with("/system ") => {
                    let new_system = cmd.strip_prefix("/system ").unwrap();
                    conversation = Conversation::with_system(new_system);
                    if !quiet {
                        eprintln!("System prompt updated. Conversation cleared.");
                    }
                    continue;
                }
                _ => {
                    eprintln!(
                        "Unknown command: {}. Type '/help' for available commands.",
                        input
                    );
                    continue;
                }
            }
        }

        // Add user message
        conversation.push_user(input);

        // Format prompt with template
        let prompt = template.apply(&conversation);

        // Generate response
        let stream = generator.generate_stream(&prompt, &config, None).await?;
        futures_util::pin_mut!(stream);

        let mut response = String::new();
        let mut in_response = false;

        while let Some(token_result) = stream.next().await {
            let token = token_result?;

            // Skip prompt tokens
            if token.token_type == TokenType::Prompt {
                continue;
            }

            // Check for stop sequences
            let text = &token.text;
            if text.contains("<|eot_id|>") || text.contains("<|end_of_text|>") {
                break;
            }

            print!("{}", text);
            stdout.flush()?;
            response.push_str(text);
            in_response = true;
        }

        // Ensure newline after response
        if in_response {
            println!();
            println!();
        }

        // Add assistant response to history
        let response_trimmed = response.trim().to_string();
        if !response_trimmed.is_empty() {
            conversation.push_assistant(response_trimmed);
        }
    }

    Ok(())
}

fn print_help() {
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  /help, /?           Show this help");
    eprintln!("  /quit, /exit, /q    Exit chat");
    eprintln!("  /clear, /reset      Clear conversation history");
    eprintln!("  /system             Show current system prompt");
    eprintln!("  /system <prompt>    Set new system prompt (clears history)");
    eprintln!("  /history            Show conversation history");
    eprintln!();
}

fn print_history(conversation: &Conversation) {
    eprintln!();
    eprintln!("Conversation history ({} messages):", conversation.len());
    eprintln!("{}", "-".repeat(40));

    for msg in conversation.messages() {
        let role = match msg.role {
            kjarni::Role::System => "System",
            kjarni::Role::User => "User",
            kjarni::Role::Assistant => "Assistant",
        };

        // Truncate long messages
        let content = if msg.content.len() > 100 {
            format!("{}...", &msg.content[..100])
        } else {
            msg.content.clone()
        };

        eprintln!("[{}]: {}", role, content);
    }
    eprintln!();
}
