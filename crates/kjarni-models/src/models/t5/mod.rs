//! T5 and FLAN-T5 model family

pub mod config;
pub mod model;

pub use config::T5Config;

#[derive(Debug, Clone, PartialEq)]
pub enum T5Task {
    Summarization,
    TranslationEnToDe,
    TranslationEnToFr,
    TranslationEnToRo,
    TranslationCustom { from: String, to: String },
    Question,
    Unknown,
}

impl T5Task {
    /// Detect task from input text prefix
    pub fn from_input(input: &str) -> Self {
        let input_lower = input.to_lowercase();
        
        if input_lower.starts_with("summarize:") || input_lower.starts_with("summarize :") {
            T5Task::Summarization
        } else if input_lower.starts_with("translate english to german:") {
            T5Task::TranslationEnToDe
        } else if input_lower.starts_with("translate english to french:") {
            T5Task::TranslationEnToFr
        } else if input_lower.starts_with("translate english to romanian:") {
            T5Task::TranslationEnToRo
        } else if input_lower.starts_with("translate ") {
            // Try to parse "translate X to Y:" pattern
            Self::parse_translation(&input_lower)
        } else if input_lower.contains("?") {
            T5Task::Question
        } else {
            T5Task::Unknown
        }
    }
    
    fn parse_translation(input: &str) -> Self {
        // Parse "translate {from} to {to}:"
        let input = input.strip_prefix("translate ").unwrap_or(input);
        if let Some((from_part, rest)) = input.split_once(" to ") {
            if let Some((to_part, _)) = rest.split_once(':') {
                return T5Task::TranslationCustom {
                    from: from_part.trim().to_string(),
                    to: to_part.trim().to_string(),
                };
            }
        }
        T5Task::Unknown
    }
}