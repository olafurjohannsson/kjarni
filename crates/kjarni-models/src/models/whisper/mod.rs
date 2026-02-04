mod config;
mod model;

pub use config::WhisperConfig;
pub use model::{WhisperModel};

#[derive(Debug, Clone, PartialEq)]
pub enum WhisperTask {
    Transcribe { language: Option<String> },
    Translate { from_language: Option<String> },
}

impl WhisperTask {
    /// Get the task token ID
    pub fn task_token(&self) -> u32 {
        match self {
            WhisperTask::Transcribe { .. } => 50359, // <|transcribe|>
            WhisperTask::Translate { .. } => 50358,  // <|translate|>
        }
    }
    
    /// Get language token ID (e.g., <|en|> = 50259)
    pub fn language_token(&self, language: &str) -> Option<u32> {
        // Map language codes to token IDs
        match language {
            "en" => Some(50259),
            "de" => Some(50261),
            "es" => Some(50262),
            "fr" => Some(50265),
            // ... etc
            _ => None,
        }
    }
}