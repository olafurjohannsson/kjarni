use crate::SequenceClassifier;
use anyhow::{Result, anyhow};
use kjarni_transformers::{
    LanguageModel, activations::softmax_inplace, cpu::encoder::traits::EncoderLanguageModel,
};
use ndarray::{Array2, Array3, s};
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

        println!("\n--- [DEBUG] INSIDE ZeroShotClassifier::classify ---");
        println!(
            "  - Received {} texts and {} candidate labels.",
            texts.len(),
            candidate_labels.len()
        );

        // 1. Create sentence pairs
        let mut inputs_to_tokenize: Vec<EncodeInput> =
            Vec::with_capacity(texts.len() * candidate_labels.len());

        for &text in texts {
            for &label in candidate_labels {
                let hypothesis = self.hypothesis_template.replace("{}", label);
                let pair: EncodeInput = (text.to_string(), hypothesis).into();
                inputs_to_tokenize.push(pair);
            }
        }
        println!(
            "  - Created {} sentence pairs to tokenize.",
            inputs_to_tokenize.len()
        );

        // =========================================================================
        // --- [DEBUG START] CHECK TOKENIZATION IDs ---
        // =========================================================================
        if let Some(first_pair) = inputs_to_tokenize.first() {
            // We manually encode the first pair here just to see the IDs
            let tokenizer = &self.nli_model.tokenizer;
            if let Ok(encoding) = tokenizer.encode(first_pair.clone(), true) {
                println!("\n[DEBUG] Rust Input IDs for first pair:");
                println!("{:?}", encoding.get_ids());
                println!("[DEBUG] Tokens:");
                println!("{:?}", encoding.get_tokens());
            }
        }
        // =========================================================================
        // --- [DEBUG END] ---

        // 2. Run all pairs through the NLI model to get raw logits.
        let all_logits = self
            .nli_model
            .predict_from_pairs(&inputs_to_tokenize)
            .await?;

        println!(
            "  - Received `all_logits` from model with length: {}",
            all_logits.len()
        );
        if let Some(first_logit) = all_logits.first() {
            println!(
                "  - Logit vector length (e.g., for first pair): {}",
                first_logit.len()
            );
        }

        // --- Detailed Log of Raw Logits ---
        println!("\n  --- [DEBUG] RAW LOGITS FROM RUST (contradiction, neutral, entailment) ---");
        let mut pair_index = 0;
        for i in 0..texts.len() {
            println!("    -- For Text {}: '{}' --", i, texts[i]);
            for j in 0..candidate_labels.len() {
                if pair_index < all_logits.len() {
                    println!(
                        "      - Pair ('{}'): {:?}",
                        candidate_labels[j], all_logits[pair_index]
                    );
                } else {
                    println!(
                        "      - !! Ran out of logits at pair_index {} !!",
                        pair_index
                    );
                }
                pair_index += 1;
            }
        }
        println!("  --- [DEBUG] END OF RAW LOGITS ---");

        // 3. Process the results.
        let mut final_results = Vec::with_capacity(texts.len());

        println!(
            "\n  - Processing logits in chunks of {}",
            candidate_labels.len()
        );
        for (i, text_logits) in all_logits.chunks(candidate_labels.len()).enumerate() {
            println!("\n    -- Processing Chunk {} (for Text {}) --", i, i);
            println!("      - Chunk length: {}", text_logits.len());

            // 4. Extract the "entailment" logit for each candidate label.
            let mut entailment_scores: Vec<f32> = text_logits
                .iter()
                .map(|logits_for_label| logits_for_label[self.entailment_id])
                .collect();
            println!(
                "      - Extracted entailment scores (pre-softmax): {:?}",
                entailment_scores
            );

            // 5. Apply Softmax to the entailment scores.
            softmax_inplace(&mut entailment_scores);
            println!(
                "      - Final probabilities (post-softmax): {:?}",
                entailment_scores
            );

            // 6. Pair the final probabilities with the labels and sort.
            let mut labeled_scores: Vec<(String, f32)> = candidate_labels
                .iter()
                .map(|&label| label.to_string())
                .zip(entailment_scores)
                .collect();

            labeled_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            println!("      - Final sorted labeled scores: {:?}", labeled_scores);
            final_results.push(labeled_scores);
        }

        println!(
            "\n  - Finished processing. Final results vec has length: {}",
            final_results.len()
        );
        println!("--- [DEBUG] END OF ZeroShotClassifier::classify ---");
        Ok(final_results)
    }
}

impl SequenceClassifier {
    pub async fn predict_from_pairs(&self, pairs: &[EncodeInput<'_>]) -> Result<Vec<Vec<f32>>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(pairs.to_vec(), true)
            .map_err(|e| anyhow!("Tokenizer failed: {}", e))?;

        // 1. Run Encoder
        let (encoder_hidden_states, attention_mask) =
            self.get_hidden_states_from_encodings(encodings).await?;

        // 2. Prepare features for Head (Decoder Logic)
        let features_for_head = if let Some(decoder) = &self.cpu_decoder {
            // === BART PATH ===
            let batch_size = encoder_hidden_states.shape()[0];
            let eos_id = self.config.eos_token_id().unwrap_or(2);
            let decoder_input_ids = Array2::from_elem((batch_size, 1), eos_id as u32);
            println!(
                "[DEBUG] Cross-attention mask shape: {:?}",
                attention_mask.dim()
            );
            println!(
                "[DEBUG] Cross-attention mask sample: {:?}",
                attention_mask.slice(s![0, ..])
            );
            let decoder_output = decoder.forward(
                &decoder_input_ids,
                &encoder_hidden_states,
                None,                  // No self-attn mask needed
                Some(&attention_mask), // Mask padding in encoder
                None,                  // No cache
                None,                  // No cross cache
                0,
            )?;
            // --- DEBUG PRINT START ---
            println!("\n--- RUST DEBUG VALUES ---");
            println!("[2] Decoder Last Hidden (Input to Head) - First 10 values:");
            // We expect shape [Batch, 1, Hidden], so we slice [0, 0, ..10]
            let slice = decoder_output.last_hidden_state.slice(s![0, 0, ..10]);
            println!("{:?}", slice);
            // --- DEBUG PRINT END ---
            decoder_output.last_hidden_state
        } else {
            // === BERT PATH ===
            encoder_hidden_states
        };

        // 3. Run Head
        // Note: For BART, we pass None for the mask to the head because the seq_len is 1 (EOS only).
        // For BERT, we pass the attention_mask.
        let head_mask = if self.cpu_decoder.is_some() {
            None
        } else {
            Some(&attention_mask)
        };

        let logits = if let Some(ref head) = self.cpu_head {
            head.forward(&features_for_head, head_mask)?
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
