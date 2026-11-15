//! Zero-shot classification with a natural-language-inference model.
//!
//! Each candidate label becomes a hypothesis ("This example is sizing.") and
//! the model scores whether the text entails it. The scoring is
//! transformers' `pipeline("zero-shot-classification")`: with one label
//! chosen per text, a softmax of the entailment logits across the labels;
//! with labels judged independently, a softmax of entailment against
//! contradiction for each label.

use super::types::{ClassificationResult, ClassifierError, ClassifierResult};

/// The hypothesis transformers uses when given none.
pub const DEFAULT_HYPOTHESIS_TEMPLATE: &str = "This example is {}.";

/// Options for [`Classifier::zero_shot_with`](super::Classifier::zero_shot_with).
#[derive(Debug, Clone)]
pub struct ZeroShotOptions {
    /// The hypothesis each label is put into, with `{}` where the label goes.
    pub hypothesis_template: String,
    /// Judge each label on its own, so several can score high, instead of
    /// sharing one probability across all labels.
    pub multi_label: bool,
}

impl Default for ZeroShotOptions {
    fn default() -> Self {
        Self {
            hypothesis_template: DEFAULT_HYPOTHESIS_TEMPLATE.to_string(),
            multi_label: false,
        }
    }
}

impl ZeroShotOptions {
    pub fn hypothesis_template(mut self, template: impl Into<String>) -> Self {
        self.hypothesis_template = template.into();
        self
    }

    pub fn multi_label(mut self, multi_label: bool) -> Self {
        self.multi_label = multi_label;
        self
    }
}

/// Where the entailment and contradiction logits sit in the model's output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct NliLabels {
    pub entailment: usize,
    pub contradiction: usize,
}

impl NliLabels {
    /// Finds the label that starts with "entail", as transformers does. The
    /// contradiction logit is the first one, or the last when entailment is
    /// first: `not_entailment` in two-label models.
    pub(crate) fn of(labels: &[String]) -> ClassifierResult<Self> {
        let entailment = labels
            .iter()
            .position(|l| l.to_lowercase().starts_with("entail"))
            .ok_or_else(|| {
                ClassifierError::InvalidConfig(format!(
                    "zero-shot classification needs a natural-language-inference model with an \
                     entailment label; this model's labels are {labels:?}"
                ))
            })?;
        let contradiction = if entailment == 0 { labels.len() - 1 } else { 0 };
        Ok(Self {
            entailment,
            contradiction,
        })
    }
}

/// The hypothesis for each label.
pub(crate) fn hypotheses(template: &str, labels: &[&str]) -> ClassifierResult<Vec<String>> {
    if labels.is_empty() {
        return Err(ClassifierError::InvalidConfig(
            "zero-shot classification needs at least one candidate label".to_string(),
        ));
    }
    if !template.contains("{}") {
        return Err(ClassifierError::InvalidConfig(format!(
            "the hypothesis template {template:?} has no {{}} for the label"
        )));
    }
    Ok(labels
        .iter()
        .map(|l| template.replacen("{}", l, 1))
        .collect())
}

/// One text's result from the logits of its (text, hypothesis) pairs, one
/// row per label in the order given.
pub(crate) fn result(
    rows: &[Vec<f32>],
    labels: &[&str],
    nli: NliLabels,
    multi_label: bool,
) -> ClassificationResult {
    let scores: Vec<f32> = if multi_label || labels.len() == 1 {
        rows.iter()
            .map(|r| {
                let (c, e) = (r[nli.contradiction], r[nli.entailment]);
                let m = c.max(e);
                let (c, e) = ((c - m).exp(), (e - m).exp());
                e / (c + e)
            })
            .collect()
    } else {
        let entail: Vec<f32> = rows.iter().map(|r| r[nli.entailment]).collect();
        let m = entail.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = entail.iter().map(|x| (x - m).exp()).collect();
        let sum: f32 = exp.iter().sum();
        exp.into_iter().map(|x| x / sum).collect()
    };

    let (label_index, &score) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("at least one label");
    let mut all_scores: Vec<(String, f32)> = labels
        .iter()
        .map(|l| l.to_string())
        .zip(scores.iter().copied())
        .collect();
    all_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    ClassificationResult {
        label: labels[label_index].to_string(),
        score,
        all_scores,
        label_index,
    }
}
