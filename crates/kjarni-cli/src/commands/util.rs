use anyhow::{anyhow, Result};
use std::io::{self, BufRead, Read};
use std::path::Path;

use kjarni::ModelType;

/// Resolve input from: direct text, file path, or stdin
/// 
/// Rules:
/// - If input is None, read from stdin
/// - If input looks like a file path and exists, read the file
/// - Otherwise, treat input as literal text
pub fn resolve_input(input: Option<&str>) -> Result<String> {
    match input {
        Some(text) => {
            // Check if it's a file path
            let path = Path::new(text);
            if path.exists() && path.is_file() {
                std::fs::read_to_string(path)
                    .map_err(|e| anyhow!("Failed to read file '{}': {}", text, e))
            } else {
                Ok(text.to_string())
            }
        }
        None => {
            // Read from stdin
            let stdin = io::stdin();
            let mut buffer = String::new();
            stdin.lock().read_to_string(&mut buffer)?;
            
            if buffer.is_empty() {
                return Err(anyhow!("No input provided. Pass text as argument, a file path, or pipe via stdin."));
            }
            
            Ok(buffer)
        }
    }
}

/// Resolve input as lines (for batch processing)
/// 
/// Useful for: rerank documents, batch encode, etc.
pub fn resolve_input_lines(input: Option<&str>) -> Result<Vec<String>> {
    let text = resolve_input(input)?;
    Ok(text.lines().map(|s| s.to_string()).filter(|s| !s.is_empty()).collect())
}

/// Check if stdin has data (non-blocking check)
pub fn stdin_has_data() -> bool {
    use std::io::IsTerminal;
    !io::stdin().is_terminal()
}



/// Create a helpful error message with "did you mean?" suggestions
pub fn model_not_found_error(name: &str, arch_hint: Option<&str>) -> String {
    let mut msg = format!("Unknown model: '{}'.", name);
    
    if let Some(arch) = arch_hint {
        msg.push_str(&format!(" Run 'kjarni model list --arch {}' to see available models.", arch));
    } else {
        msg.push_str(" Run 'kjarni model list' to see available models.");
    }

    let suggestions = ModelType::find_similar(name);
    if !suggestions.is_empty() {
        msg.push_str("\n\nDid you mean?");
        for (suggestion, _) in suggestions {
            msg.push_str(&format!("\n  - {}", suggestion));
        }
    }

    msg
}