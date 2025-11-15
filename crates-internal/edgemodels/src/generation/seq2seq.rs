use crate::generation2::{apply_no_repeat_ngram, apply_repetition_penalty};
use anyhow::{Result, anyhow};
use edgetransformers::encoder_decoder::TransformerEncoderDecoder;
use edgetransformers::models::base::EncoderDecoderLanguageModel; // Or wherever you put the trait
use edgetransformers::models::base::{BeamHypothesis, DecodingStrategy, GenerationConfig};
use edgetransformers::prelude::*;
use edgetransformers::traits::EncoderOutput;
use ndarray::{Array1, Array2, s};

/// Selects the top `k` tokens and their log probabilities from a log probability distribution.
///
/// This is a core component of beam search, used to find the most likely next
/// tokens to expand upon.
///
/// # Arguments
/// * `log_probs`: A 1D array of log probabilities for the entire vocabulary.
/// * `k`: The number of top tokens to select.
///
/// # Returns
/// A `Vec` containing tuples of `(token_id, log_probability)`, sorted from highest
/// log probability to lowest.
pub fn get_top_k_from_log_probs(log_probs: &Array1<f32>, k: usize) -> Vec<(u32, f32)> {
    // 1. Create a vector of (index, value) pairs from the log probabilities array.
    let mut indexed_log_probs: Vec<(usize, f32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &lp)| (i, lp))
        .collect();

    // 2. Sort the vector in descending order based on the log probability (the f32 value).
    //    We use `partial_cmp` and reverse the comparison to sort from highest to lowest.
    //    `unwrap()` is safe here because f32 log_probs from a softmax will not be NaN.
    // indexed_log_probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed_log_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    // 3. Take the top `k` elements from the sorted vector.
    indexed_log_probs.truncate(k);

    // 4. Map the result to the final `(u32, f32)` format for token IDs.
    indexed_log_probs
        .into_iter()
        .map(|(i, lp)| (i as u32, lp))
        .collect()
}

pub struct Seq2SeqGenerator {
    pub model: Box<dyn EncoderDecoderLanguageModel>,
}

impl Seq2SeqGenerator {
    pub fn new(model: Box<dyn EncoderDecoderLanguageModel>) -> Self {
        Self { model }
    }

    pub async fn generate(&self, input_text: &str, config: &GenerationConfig) -> Result<String> {
        let encoding = self
            .model
            .tokenizer()
            .encode(input_text, true)
            .map_err(|e| anyhow!(e))?;
        let attention_mask = Array2::ones((1, encoding.len()));
        let encoder_output = self.encode_input(input_text).await?;
        self.generate_from_encoding(&encoder_output, &attention_mask, config)
            .await
    }

    async fn encode_input(&self, text: &str) -> Result<EncoderOutput> {
        let encoding = self
            .model
            .tokenizer()
            .encode(text, true)
            .map_err(|e| anyhow!(e))?;

        let input_ids = Array2::from_shape_vec(
            (1, encoding.len()),
            encoding.get_ids().iter().map(|&id| id).collect(),
        )?;

        let attention_mask = Array2::ones(input_ids.dim());

        let encoder_output: EncoderOutput = self
            .model
            .encoder()
            .forward(&input_ids, &attention_mask, None)
            .await?;

        Ok(encoder_output)
    }
    pub async fn generate_from_encoding(
        &self,
        encoder_output: &EncoderOutput,
        encoder_attention_mask: &Array2<f32>,
        config: &GenerationConfig,
    ) -> Result<String> {
        let beam_params = match &config.strategy {
            DecodingStrategy::BeamSearch(params) => params,
            // If the user passes a Greedy or Sample config, we return a helpful error.
            _ => {
                return Err(anyhow!(
                    "Seq2SeqGenerator only supports the BeamSearch strategy."
                ));
            }
        };
        let batch_size = encoder_output.last_hidden_state.shape()[0];
        let eos_token_id = 2;
        let decoder_start_token_id = 2;

        // Initialize Beams with an empty cache for each.
        let initial_cache = self.model.new_cache(batch_size, config.max_length)?;
        let mut beams = vec![BeamHypothesis {
            tokens: vec![decoder_start_token_id],
            score: 0.0,
            cache: initial_cache,
        }];
        let mut completed_beams: Vec<BeamHypothesis> = Vec::new();

        // Main Generation Loop
        for step in 0..config.max_length {
            if beams.is_empty() {
                break;
            }
            let mut all_new_candidates: Vec<BeamHypothesis> = Vec::new();

            // for hypo in beams.drain(..) {
            for hypo in &beams {
                let last_token = *hypo.tokens.last().unwrap();
                let decoder_input_ids = Array2::from_elem((1, 1), last_token);
                let decoder_attention_mask = Array2::ones((batch_size, hypo.tokens.len()));

                // 1. Clone the cache *before* it gets mutated.
                let mut current_cache = hypo.cache.clone_box();

                // 2. Pass the mutable clone to the forward pass.
                let decoder_output = self
                    .model
                    .decoder()
                    .forward(
                        &decoder_input_ids,
                        &encoder_output.last_hidden_state,
                        Some(&encoder_attention_mask),
                        Some(&decoder_attention_mask),
                        Some(current_cache.as_mut()),
                    )
                    .await?;


                // 3. Project to logits (this now correctly includes the bias).
                // let last_hidden_state = decoder_output.last_hidden_state.slice(s![0, -1, ..]);
                // let mut logits: Array1<f32> = self.lm_head.dot(&last_hidden_state);
                // if let Some(bias) = &self.final_logits_bias {
                //     logits += bias;
                // }

                // let logits_3d = self
                //     .model
                //     .project_to_logits(&decoder_output.last_hidden_state)?;
                // let mut logits = logits_3d.slice(s![0, 0, ..]).to_owned();

                // --- REVERTED LOGITS LOGIC START ---
                // This is the logic from your OLD, working code.

                // 1. Get the hidden state for the very last token. This is a 1D view.
                let last_hidden_state = decoder_output.last_hidden_state.slice(s![0, -1, ..]);

                // 2. Perform matrix-vector multiplication: [vocab, hidden] @ [hidden] -> [vocab]
                let mut logits: Array1<f32> = self.model.lm_head().dot(&last_hidden_state);

                // 3. Add the bias if it exists.
                if let Some(bias) = self.model.final_logits_bias() {
                    logits += bias;
                }

                logits = apply_repetition_penalty(logits, &hypo.tokens, config.repetition_penalty);
                logits = apply_no_repeat_ngram(logits, &hypo.tokens, config.no_repeat_ngram_size);

                let log_probs = log_softmax_1d(&logits);
                // let top_candidates = get_top_k_from_log_probs(&log_probs, config.num_beams);
                let mut top_candidates: Vec<(u32, f32)> = log_probs
                    .iter()
                    .enumerate()
                    .map(|(id, &lp)| (id as u32, lp))
                    .collect();
                top_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                top_candidates.truncate(beam_params.num_beams);

                // 5. Create new hypotheses, each getting a clone of the *updated* cache.
                for (token_id, token_log_prob) in top_candidates {
                    let mut new_tokens = hypo.tokens.clone();
                    new_tokens.push(token_id);

                    all_new_candidates.push(BeamHypothesis {
                        tokens: new_tokens,
                        score: hypo.score + token_log_prob,
                        cache: current_cache.clone_box(),
                    });
                }
            }

            // --- 3. PRUNE AND MANAGE BEAMS (Identical to your working code) ---
            all_new_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            beams.clear();

            for candidate in all_new_candidates {
                if candidate.tokens.last() == Some(&eos_token_id) {
                    if candidate.tokens.len() >= config.min_length {
                        completed_beams.push(candidate);
                    }
                    // Don't add to active beams if it's too short
                } else {
                    beams.push(candidate);
                }
                if beams.len() == beam_params.num_beams {
                    break;
                }
            }
            if beam_params.early_stopping && completed_beams.len() >= beam_params.num_beams {
                break;
            }
        }

        let final_hypotheses = if completed_beams.is_empty() {
            // If no beams reached EOS, use the active (unfinished) beams as candidates.
            beams
        } else {
            // Otherwise, only consider the beams that properly finished.
            completed_beams
        };

        if final_hypotheses.is_empty() {
            return Err(anyhow!("No hypothesis was generated."));
        }

        let best_hypo = final_hypotheses
            .iter()
            .max_by(|a, b| {
                let score_a = a.score / (a.tokens.len() as f32).powf(beam_params.length_penalty);
                let score_b = b.score / (b.tokens.len() as f32).powf(beam_params.length_penalty);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let mut tokens_to_decode = &best_hypo.tokens[..];
        if tokens_to_decode.first() == Some(&decoder_start_token_id) {
            tokens_to_decode = &tokens_to_decode[1..];
        }
        if tokens_to_decode.last() == Some(&eos_token_id) {
            tokens_to_decode = &tokens_to_decode[..tokens_to_decode.len() - 1];
        }

        self.model
            .tokenizer()
            .decode(tokens_to_decode, true)
            .map_err(|e| anyhow!(e))
    }
}

// fn find_best_beams_and_get_indices(
//     logits_2d: Array2<f32>, // Shape: [num_beams, vocab_size]
//     current_beams: &[BeamHypothesis],
//     config: &GenerationConfig,
// ) -> (Vec<u32>, Vec<usize>, Vec<BeamHypothesis>) {
//     let num_beams = config.num_beams;
//     let mut candidates: Vec<(f32, usize, u32)> = Vec::with_capacity(num_beams * num_beams);

//     // 1. Gather all possible next hypotheses
//     for (beam_idx, (beam, logits_for_beam)) in
//         current_beams.iter().zip(logits_2d.rows()).enumerate()
//     {
//         let log_probs = log_softmax_1d(&logits_for_beam.to_owned());
//         let top_k = get_top_k_from_log_probs(&log_probs, num_beams);

//         for (token_id, token_log_prob) in top_k {
//             let new_score = beam.score + token_log_prob;
//             candidates.push((new_score, beam_idx, token_id));
//         }
//     }

//     // 2. Sort all candidates by score
//     candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

//     // 3. Select the top `num_beams` candidates to form the new beams
//     let top_candidates = &candidates[..num_beams];

//     let mut next_tokens = Vec::with_capacity(num_beams);
//     let mut reorder_indices = Vec::with_capacity(num_beams);
//     let mut updated_beams = Vec::with_capacity(num_beams);

//     for &(new_score, source_beam_idx, new_token_id) in top_candidates {
//         let mut new_tokens = current_beams[source_beam_idx].tokens.clone();
//         new_tokens.push(new_token_id);

//         next_tokens.push(new_token_id);
//         reorder_indices.push(source_beam_idx);

//         // The cache is no longer stored in the hypothesis for this strategy
//         updated_beams.push(BeamHypothesis {
//             tokens: new_tokens,
//             score: new_score,
//             cache: None,
//         });
//     }

//     (next_tokens, reorder_indices, updated_beams)
// }

// pub fn log_softmax_1d(array: &Array1<f32>) -> Array1<f32> {
//     let max_val = array.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
//     let exp_logits = (array - max_val).mapv(f32::exp);
//     let sum_exp = exp_logits.sum();
//     exp_logits.mapv(f32::ln) - sum_exp.ln()
// }
pub fn log_softmax_1d(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let scaled_logits = logits - max_val;
    let exp_sum = scaled_logits.mapv(f32::exp).sum();
    scaled_logits - exp_sum.ln()
}
