use crate::SequenceClassifier;
use anyhow::{Result, anyhow};
use kjarni_transformers::{
    LanguageModel, activations::softmax_inplace, cpu::encoder::traits::EncoderLanguageModel,
};
use ndarray::{Array2, Array3};
use tokenizers::{EncodeInput, Encoding};

/// An engine for performing zero-shot classification using a model
/// trained on Natural Language Inference (NLI).
pub struct ZeroShotClassifier {
    /// The underlying NLI model, loaded as a SequenceClassifier.
    nli_model: SequenceClassifier,
    /// The template used to turn a label into a hypothesis.
    hypothesis_template: String,
    /// The index of the "entailment" logit in the NLI model's output.
    entailment_id: usize,
}

impl ZeroShotClassifier {
    pub fn new(nli_model: SequenceClassifier) -> Result<Self> {
        // We need to know the label-to-id mapping of the NLI model.
        // Find the index of the "entailment" label, CASE-INSENSITIVELY.
        let entailment_id = nli_model
            .labels()
            .and_then(|labels| {
                labels
                    .iter()
                    // Convert both the target string and the model's label to lowercase for comparison.
                    .position(|r| r.to_lowercase() == "entailment") // Removed the "entails" check as "entailment" is canonical.
            })
            .ok_or_else(|| {
                // If it fails, provide more detail about what labels ARE available.
                let available_labels = nli_model.labels().map(|l| l.join(", ")).unwrap_or_else(|| "None found".to_string());
                anyhow!("Loaded NLI model config does not contain an 'entailment' label. Available labels: [{}]", available_labels)
            })?;

        Ok(Self {
            nli_model,
            // This is a common, effective template.
            hypothesis_template: "This example is {}.".to_string(),
            entailment_id,
        })
    }

    /// Performs zero-shot classification on a batch of texts.
    pub async fn classify(
        &self,
        texts: &[&str],
        candidate_labels: &[&str],
    ) -> Result<Vec<Vec<(String, f32)>>> {
        if texts.is_empty() || candidate_labels.is_empty() {
            return Ok(vec![]);
        }

        // 1. THIS IS THE ROBUST WAY TO CREATE SENTENCE PAIRS FOR THE TOKENIZER.
        let mut inputs_to_tokenize: Vec<EncodeInput> =
            Vec::with_capacity(texts.len() * candidate_labels.len());
        for &text in texts {
            for &label in candidate_labels {
                let hypothesis = self.hypothesis_template.replace("{}", label);
                // Create a tuple of (premise, hypothesis) for the tokenizer.
                // The tokenizer knows how to format this correctly (e.g., with [SEP] tokens).
                let pair: EncodeInput = (text.to_string(), hypothesis).into();
                inputs_to_tokenize.push(pair);
            }
        }

        // 2. Run all pairs through the NLI model to get raw logits.
        // We need a new method on SequenceClassifier to handle this input type.
        let all_logits = self
            .nli_model
            .predict_from_pairs(&inputs_to_tokenize)
            .await?;

        // 3. Process the results.
        let mut final_results = Vec::with_capacity(texts.len());
println!("\n[DEBUG] RAW LOGITS FROM RUST (contradiction, neutral, entailment):");
for (i, logits_for_label) in all_logits.iter().enumerate() {
    println!("  - Pair ('{}'): {:?}", candidate_labels[i], logits_for_label);
}
        for text_logits in all_logits.chunks(candidate_labels.len()) {
            // 4. Extract the "entailment" logit for each candidate label.
            let mut entailment_scores: Vec<f32> = text_logits
                .iter()
                .map(|logits_for_label| logits_for_label[self.entailment_id])
                .collect();

            // 5. Apply Softmax to the entailment scores to turn them into probabilities.
            softmax_inplace(&mut entailment_scores);

            // 6. Pair the final probabilities with the labels and sort.
            let mut labeled_scores: Vec<(String, f32)> = candidate_labels
                .iter()
                .map(|&label| label.to_string())
                .zip(entailment_scores)
                .collect();

            labeled_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            final_results.push(labeled_scores);
        }

        Ok(final_results)
    }
}

impl SequenceClassifier {
    pub async fn predict_from_pairs(&self, pairs: &[EncodeInput<'_>]) -> Result<Vec<Vec<f32>>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        // This method is almost identical to get_hidden_states_batch, but for pairs.
        let encodings = self
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(|e| anyhow!("Tokenizer failed: {}", e))?;

        let (hidden_states, attention_mask) =
            self.get_hidden_states_from_encodings(encodings).await?;

        let logits = if let Some(ref head) = self.cpu_head {
            head.forward(&hidden_states, Some(&attention_mask))?
        } else {
            return Err(anyhow!("No classification head available"));
        };

        Ok(logits.outer_iter().map(|row| row.to_vec()).collect())
    }

    /// Takes a batch of pre-tokenized `Encoding` objects and returns the
    /// final hidden states from the encoder. This is a helper method primarily
    /// used by other model engines like the `ZeroShotClassifier`.
    pub(crate) async fn get_hidden_states_from_encodings(
        &self,
        encodings: Vec<Encoding>,
    ) -> Result<(Array3<f32>, Array2<f32>)> {
        if encodings.is_empty() {
            let hidden_states = Array3::zeros((0, 0, self.hidden_size()));
            let attention_mask = Array2::zeros((0, 0));
            return Ok((hidden_states, attention_mask));
        }

        // 1. Extract token IDs and attention masks from the encodings.
        let batch_size = encodings.len();
        let sequence_length = encodings[0].len();
        let input_ids_vec: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_ids())
            .cloned()
            .collect();
        let attention_mask_vec: Vec<u32> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask())
            .cloned()
            .collect();

        // 2. Convert to ndarray Arrays.
        let input_ids = Array2::from_shape_vec((batch_size, sequence_length), input_ids_vec)?;
        let attention_mask =
            Array2::from_shape_vec((batch_size, sequence_length), attention_mask_vec)?;

        // 3. Call the NEW high-level trait method.
        //    `self` (the SequenceClassifier) implements EncoderLanguageModel.
        self.get_hidden_states_batch_from_ids(&input_ids, &attention_mask)
            .await
    }
}
