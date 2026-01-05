//! Interactive chat command using the high-level Chat API.

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use std::io::{self, BufRead, Write};

use kjarni::chat::Chat;

pub async fn run(
    model: &str,
    _model_path: Option<&str>, // Handled by builder if needed
    system_prompt: Option<&str>,
    temperature: f32,
    max_tokens: usize,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    // 1. Initialize the Chat instance using the Builder
    // This handles model resolution, downloading, and loading internally
    let mut builder = Chat::builder(model)
        .temperature(temperature)
        .max_tokens(max_tokens)
        .quiet(quiet);

    if gpu {
        builder = builder.gpu();
    } else {
        builder = builder.cpu();
    }

    if let Some(system) = system_prompt {
        builder = builder.system_prompt(system);
    }

    let chat = builder.build().await.map_err(|e| anyhow!("Failed to initialize chat: {}", e))?;

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
                    convo.clear();
                    if !quiet { println!("Conversation history cleared."); }
                    continue;
                }
                "/system" => {
                    if let Some(s) = chat.system_prompt() {
                        println!("System Prompt: {}", s);
                    }
                    continue;
                }
                "/help" => {
                    println!("\nCommands:\n  /quit  - Exit\n  /clear - Reset history\n  /system - Show system prompt\n");
                    continue;
                }
                _ => {
                    println!("Unknown command: {}", input);
                    continue;
                }
            }
        }

        // 4. Generate Streaming Response
        // We use the high-level stream method which handles the tokio::spawn and Arc cloning
        let mut stream = convo.stream(input).await.map_err(|e| anyhow!("Stream failed: {}", e))?;

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

        // Add the collected response to the stateful conversation history
        convo.add_response(full_response);
        println!("\n");
    }

    if !quiet {
        println!("Goodbye!");
    }

    Ok(())
}