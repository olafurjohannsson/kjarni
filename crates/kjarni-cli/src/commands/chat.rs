// =============================================================================
// kjarni_cli/src/commands/chat.rs
// =============================================================================

//! Interactive chat command using the high-level Chat API.

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use std::io::{self, BufRead, Write};

use kjarni::chat::Chat;

pub async fn run(
    model: &str,
    _model_path: Option<&str>, // TODO: Handle custom model paths
    system_prompt: Option<&str>,
    temperature: f32,
    max_tokens: usize,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // 1. Initialize the Chat instance using the Builder
    let mut builder = Chat::builder(model)
        .temperature(temperature)
        .max_tokens(max_tokens);

    if quiet {
        builder = builder.quiet();
    }

    if gpu {
        builder = builder.gpu();
    } else {
        builder = builder.cpu();
    }

    if let Some(system) = system_prompt {
        builder = builder.system(system);  // Fixed: was system_prompt()
    }

    let chat = builder
        .build()
        .await
        .map_err(|e| anyhow!("Failed to initialize chat: {}", e))?;

    // 2. Setup stateful conversation
    let mut convo = chat.conversation();

    // 3. Welcome message
    if !quiet {
        println!("\nKjarni Chat: {}", chat.model_name());
        println!("Device: {:?}", chat.device());
        println!("Type '/help' for commands, '/quit' to exit.\n");
    }

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break; // EOF
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle Slash Commands
        if input.starts_with('/') {
            match input {
                "/quit" | "/exit" | "/q" => break,
                "/clear" | "/reset" => {
                    convo.clear(true);  // Fixed: keep system prompt
                    if !quiet {
                        println!("Conversation history cleared.");
                    }
                    continue;
                }
                "/clearall" => {
                    convo.clear(false);  // Clear everything including system
                    if !quiet {
                        println!("Conversation history cleared (including system prompt).");
                    }
                    continue;
                }
                "/system" => {
                    if let Some(s) = chat.system_prompt() {
                        println!("System Prompt: {}", s);
                    } else {
                        println!("No system prompt set.");
                    }
                    continue;
                }
                "/history" => {
                    println!("\n--- History ({} messages) ---", convo.len());
                    for msg in convo.history().messages() {
                        println!("[{}]: {}", msg.role, msg.content);
                    }
                    println!("--- End ---\n");
                    continue;
                }
                "/help" => {
                    println!("\nCommands:");
                    println!("  /quit, /exit, /q  - Exit chat");
                    println!("  /clear, /reset    - Reset history (keep system prompt)");
                    println!("  /clearall         - Reset history (remove system prompt too)");
                    println!("  /system           - Show system prompt");
                    println!("  /history          - Show conversation history");
                    println!("  /help             - Show this help\n");
                    continue;
                }
                _ => {
                    println!("Unknown command: {}. Type /help for available commands.", input);
                    continue;
                }
            }
        }

        // 4. Stream the response
        // First, add user message to history
        convo.push_user(input);

        // Get the stream
        let stream_result = convo.stream_next().await;
        
        let mut stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[Error starting stream: {}]", e);
                // Remove the user message we just added since we failed
                // (In a more robust impl, we'd have a rollback mechanism)
                continue;
            }
        };

        // Collect and print tokens
        let mut full_response = String::new();

        while let Some(token_result) = stream.next().await {
            match token_result {
                Ok(text) => {
                    print!("{}", text);
                    stdout.flush()?;
                    full_response.push_str(&text);
                }
                Err(e) => {
                    eprintln!("\n[Error during generation: {}]", e);
                    break;
                }
            }
        }

        // Add the assistant response to history
        if !full_response.is_empty() {
            convo.push_assistant(full_response.trim());
        }

        println!("\n");
    }

    if !quiet {
        println!("Goodbye!");
    }

    Ok(())
}