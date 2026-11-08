//! Model validation for translation.

use std::path::Path;

use kjarni_transformers::models::{ModelArchitecture, ModelType};

use super::languages::normalize_language;

use super::types::{TranslatorError, TranslatorResult};

/// Validate that a model can be used for translation.
pub fn validate_for_translation(model_type: ModelType) -> TranslatorResult<()> {
    let info = model_type.info();
    let cli_name = model_type.cli_name();

    // Must be T5 (BART is primarily for summarization)
    match info.architecture {
        ModelArchitecture::T5 | ModelArchitecture::Marian => {
            // T5/FLAN-T5 takes the languages in a prompt; Marian has one direction per model
        }
        ModelArchitecture::Bart => {
            // BART can technically translate but isn't ideal
            return Err(TranslatorError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: "BART models are optimized for summarization, not translation. Use flan-t5-base or flan-t5-large.".to_string(),
            });
        }
        _ => {
            return Err(TranslatorError::IncompatibleModel {
                model: cli_name.to_string(),
                reason: format!(
                    "Architecture '{}' cannot perform translation. Use T5 models (flan-t5-base, flan-t5-large).",
                    info.architecture.display_name()
                ),
            });
        }
    }

    Ok(())
}

/// A language a one-direction model reads or writes: its code in the model's
/// config, and the name [`normalize_language`] gives the code, if any.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelLanguage {
    pub code: String,
    pub name: Option<&'static str>,
}

impl ModelLanguage {
    pub fn new(code: &str) -> Self {
        let base = code.split('_').next().unwrap_or(code);
        Self {
            code: code.to_string(),
            name: normalize_language(code).or_else(|| normalize_language(base)),
        }
    }

    /// Whether `lang`, a normalized name or a code, names this language.
    fn is(&self, lang: &str) -> bool {
        self.name == Some(lang) || self.code.eq_ignore_ascii_case(lang)
    }

    fn label(&self) -> &str {
        self.name.unwrap_or(&self.code)
    }
}

/// The directions a model translates.
#[derive(Debug, Clone, PartialEq)]
pub enum Directions {
    /// T5: any direction, named in a prompt.
    Prompt,
    /// A Marian model: from these languages to these. With `target_token`, the
    /// model writes the language whose `>>code<<` token starts its input.
    Fixed {
        from: Vec<ModelLanguage>,
        to: Vec<ModelLanguage>,
        target_token: bool,
    },
    /// A Marian model whose config names no languages: the text goes in as is.
    Unchecked,
}

impl Directions {
    /// The directions of a registry model.
    pub fn of(model_type: ModelType) -> Self {
        let fixed = |from: &str, to: &str| Self::Fixed {
            from: vec![ModelLanguage::new(from)],
            to: vec![ModelLanguage::new(to)],
            target_token: false,
        };
        match model_type {
            ModelType::OpusMtEnDe => fixed("en", "de"),
            ModelType::OpusMtDeEn => fixed("de", "en"),
            _ => Self::Prompt,
        }
    }

    /// The directions of a Marian model in `dir`, from the `source_lang` and
    /// `target_lang` of its tokenizer_config.json, as `MarianTokenizer` reads
    /// them: codes joined by `+`. A model with more than one target picks it by
    /// the `>>code<<` tokens `has_token` finds.
    pub fn of_marian_folder(dir: &Path, has_token: impl Fn(&str) -> bool) -> Self {
        let config: Option<serde_json::Value> =
            std::fs::read_to_string(dir.join("tokenizer_config.json"))
                .ok()
                .and_then(|text| serde_json::from_str(&text).ok());
        let languages = |key: &str| -> Vec<ModelLanguage> {
            config
                .as_ref()
                .and_then(|c| c[key].as_str())
                .map(|codes| {
                    codes
                        .split('+')
                        .map(str::trim)
                        .filter(|c| !c.is_empty())
                        .map(ModelLanguage::new)
                        .collect()
                })
                .unwrap_or_default()
        };
        let (from, mut to) = (languages("source_lang"), languages("target_lang"));
        if from.is_empty() || to.is_empty() {
            return Self::Unchecked;
        }
        let mut target_token = false;
        if to.len() > 1 {
            let with_token: Vec<ModelLanguage> = to
                .iter()
                .filter(|l| has_token(&format!(">>{}<<", l.code)))
                .cloned()
                .collect();
            if !with_token.is_empty() {
                to = with_token;
                target_token = true;
            }
        }
        Self::Fixed {
            from,
            to,
            target_token,
        }
    }

    /// The language `input` names for this model: a normalized name, or, for
    /// a code [`normalize_language`] does not know, the model's own code.
    pub fn language(&self, input: &str) -> Option<String> {
        if let Some(name) = normalize_language(input) {
            return Some(name.to_string());
        }
        match self {
            Self::Fixed { from, to, .. } => from
                .iter()
                .chain(to)
                .find(|l| l.code.eq_ignore_ascii_case(input))
                .map(|l| l.code.clone()),
            Self::Unchecked => Some(input.to_string()),
            Self::Prompt => None,
        }
    }
}

fn labels(languages: &[ModelLanguage]) -> String {
    let mut seen: Vec<&str> = Vec::new();
    for l in languages {
        if !seen.contains(&l.label()) {
            seen.push(l.label());
        }
    }
    seen.join(", ")
}

/// What the model is given: a T5 prompt, or the text for a Marian model, after
/// its target's `>>code<<` token when it has more than one target. A Marian
/// model refuses a direction it does not translate. Languages are as
/// [`Directions::language`] gives them.
pub fn translation_input(
    directions: &Directions,
    model: &str,
    text: &str,
    from: &str,
    to: &str,
) -> TranslatorResult<String> {
    match directions {
        Directions::Prompt => Ok(format!("translate {} to {}: {}", from, to, text)),
        Directions::Unchecked => Ok(text.to_string()),
        Directions::Fixed {
            from: sources,
            to: targets,
            target_token,
        } => {
            let target = targets.iter().find(|l| l.is(to));
            let (true, Some(target)) = (sources.iter().any(|l| l.is(from)), target) else {
                return Err(TranslatorError::IncompatibleModel {
                    model: model.to_string(),
                    reason: format!(
                        "It only translates {} to {}, not {from} to {to}.",
                        labels(sources),
                        labels(targets)
                    ),
                });
            };
            if *target_token {
                Ok(format!(">>{}<< {}", target.code, text))
            } else {
                Ok(text.to_string())
            }
        }
    }
}

/// Check that a model loaded from a folder can translate: T5 and Marian can,
/// BART summarizes.
pub fn validate_folder_for_translation(
    name: &str,
    architecture: ModelArchitecture,
) -> TranslatorResult<()> {
    match architecture {
        ModelArchitecture::T5 | ModelArchitecture::Marian => Ok(()),
        ModelArchitecture::Bart => Err(TranslatorError::IncompatibleModel {
            model: name.to_string(),
            reason: "BART models are optimized for summarization, not translation. Use a Marian (opus-mt) or T5 model.".to_string(),
        }),
        other => Err(TranslatorError::IncompatibleModel {
            model: name.to_string(),
            reason: format!(
                "Architecture '{}' cannot perform translation. Use Marian or T5 models.",
                other.display_name()
            ),
        }),
    }
}

/// Get all available translation models.
pub fn get_translation_models() -> Vec<&'static str> {
    ModelType::all()
        .filter(|m| validate_for_translation(*m).is_ok())
        .map(|m| m.cli_name())
        .collect()
}
