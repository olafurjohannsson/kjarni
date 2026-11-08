//! Named-entity recognition: people, places, organizations and the like.
//!
//! ```ignore
//! use kjarni::ner::EntityRecognizer;
//!
//! let ner = EntityRecognizer::new("distilbert-ner").await?;
//! for e in ner.recognize("Sarah Connor met Apple's CEO in San Francisco.").await? {
//!     println!("{} {} {:.2}", e.label, e.text, e.score);
//! }
//!
//! // Any *ForTokenClassification model from the Hugging Face Hub:
//! let icelandic = EntityRecognizer::from_path("IceBERT-ner").build().await?;
//! ```
//!
//! Entities come out as transformers' `pipeline("ner")` gives them for the
//! same [`Aggregation`], with the same spans and scores. The default is
//! `First`, which keeps every word whole.

mod builder;
mod model;
mod types;
mod validation;

pub use builder::EntityRecognizerBuilder;
pub use kjarni_models::{Aggregation, Entity, TokenLabel};
pub use model::EntityRecognizer;
pub use types::{NerError, NerResult};

/// The model `EntityRecognizer::new` uses when given none.
pub const DEFAULT_MODEL: &str = "distilbert-ner";

#[cfg(test)]
mod tests;
