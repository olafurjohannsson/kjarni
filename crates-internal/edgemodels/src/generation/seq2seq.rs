use anyhow::{Result, anyhow};
use edgetransformers::cache::GpuBeamKVCache;
use edgetransformers::gpu_ops::GpuTensor;
use edgetransformers::models::base::EncoderDecoderLanguageModel;
// Or wherever you put the trait
use edgetransformers::models::base::{BeamHypothesis, DecodingStrategy, GenerationConfig};
use edgetransformers::prelude::*;
use edgetransformers::traits::EncoderOutput;
use ndarray::s;
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

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
        println!("encoder: {:?}", encoder_output.last_hidden_state);

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
            _ => {
                return Err(anyhow!(
                    "Seq2SeqGenerator only supports the BeamSearch strategy."
                ));
            }
        };

        let batch_size = encoder_output.last_hidden_state.shape()[0];
        assert_eq!(
            batch_size, 1,
            "Beam search is currently only supported for batch_size=1"
        );

        let eos_token_id = 2;
        let decoder_start_token_id = 2;
        let context = self.model.context().unwrap();

        let (mut cache, num_beams) = {
            match &config.strategy {
                DecodingStrategy::BeamSearch(params) => (
                    self.model
                        .new_cache(batch_size, config.max_length, params.num_beams)?,
                    params.num_beams,
                ),
                DecodingStrategy::Greedy => {
                    (self.model.new_cache(batch_size, config.max_length, 0)?, 0)
                }
                DecodingStrategy::Sample(params) => {
                    (self.model.new_cache(batch_size, config.max_length, 0)?, 0)
                }
            }
        };

        let gpu_cache = cache.as_any_mut().downcast_mut::<GpuBeamKVCache>().unwrap();
        let mut current_tokens = Array2::from_elem((num_beams, 1), decoder_start_token_id);

        let mut beams: Vec<BeamHypothesis> = (0..num_beams)
            .map(|i| BeamHypothesis {
                tokens: vec![decoder_start_token_id],
                score: if i == 0 { 0.0 } else { f32::NEG_INFINITY },
            })
            .collect();

        // Expand the encoder output to match the number of beams
        let expanded_encoder_output = encoder_output
            .last_hidden_state
            .broadcast((
                num_beams,
                encoder_output.last_hidden_state.shape()[1],
                encoder_output.last_hidden_state.shape()[2],
            ))
            .unwrap()
            .to_owned();
        let expanded_encoder_mask = encoder_attention_mask
            .broadcast((num_beams, encoder_attention_mask.shape()[1]))
            .unwrap()
            .to_owned();

        let mut completed_beams: Vec<BeamHypothesis> = Vec::new();
        let mut current_tokens_gpu = GpuTensor::from_ndarray(&context, &current_tokens)?;
        let expanded_encoder_output_gpu =
            GpuTensor::from_ndarray(&context, &expanded_encoder_output)?;
        let expanded_encoder_mask_gpu = GpuTensor::from_ndarray(&context, &expanded_encoder_mask)?;

        for step in 0..config.max_length {
            let decoder_attention_mask_cpu: Array2<f32> =
                Array2::ones((num_beams, gpu_cache.get_seq_length() + 1));
            let decoder_attention_mask_gpu =
                GpuTensor::from_ndarray(&context, &decoder_attention_mask_cpu)?;

            let decoder_output = self
                .model
                .gpu_decoder()
                .forward(
                    &current_tokens_gpu,
                    &expanded_encoder_output_gpu,
                    Some(&expanded_encoder_mask_gpu),
                    Some(&decoder_attention_mask_gpu),
                    Some(gpu_cache),
                )
                .await?;

            // --- GATHER AND SCORE ---
            let last_hidden_state = decoder_output.last_hidden_state.slice(s![.., -1, ..]); // Shape: [num_beams, hidden_size]

            // Apply LM head and penalties per beam
            let mut logits_2d = Array2::zeros((num_beams, self.model.config().vocab_size()));
            for i in 0..num_beams {
                let mut logits_row = self.model.lm_head().dot(&last_hidden_state.row(i));
                if let Some(bias) = self.model.final_logits_bias() {
                    logits_row += bias;
                }

                // Apply repetition penalty
                logits_row = apply_repetition_penalty(
                    logits_row,
                    &beams[i].tokens,
                    config.repetition_penalty,
                );

                // Apply n-gram blocking
                logits_row = apply_no_repeat_ngram(
                    logits_row,
                    &beams[i].tokens,
                    config.no_repeat_ngram_size,
                );

                logits_2d.row_mut(i).assign(&logits_row);
            }

            let (next_tokens_vec, reorder_indices_vec, updated_beams) =
                find_best_beams_and_get_indices(logits_2d, &beams, config, num_beams);

            // First reorder with CURRENT seq_length
            let reorder_indices_gpu = GpuTensor::from_ndarray(
                &context,
                &Array1::from(
                    reorder_indices_vec
                        .iter()
                        .map(|&i| i as u32)
                        .collect::<Vec<_>>(),
                ),
            )?;

            let mut encoder = context.device.create_command_encoder(&Default::default());
            gpu_cache.reorder(&mut encoder, &reorder_indices_gpu);
            context.queue.submit(Some(encoder.finish()));

            // THEN increment the cache length AFTER reordering
            gpu_cache.increment_len(1);

            // After filtering completed beams:
            println!("Beam scores: {:?}", beams.iter().map(|b| b.score).collect::<Vec<_>>());
            beams.clear();
            for candidate in updated_beams {
                if candidate.tokens.last() == Some(&eos_token_id) {
                    if candidate.tokens.len() >= config.min_length {
                        completed_beams.push(candidate);
                    }
                } else {
                    beams.push(candidate);
                }

                if beams.len() + completed_beams.len() >= num_beams {
                    break;
                }
            }

            // Pad beams if we have fewer than num_beams
            while beams.len() < num_beams {
                // Add dummy beams with very low scores
                beams.push(BeamHypothesis {
                    tokens: vec![decoder_start_token_id],
                    score: f32::NEG_INFINITY,
                });
            }

            // Now safe to create the tensor
            let active_tokens: Vec<u32> = beams
                .iter()
                .map(|b| *b.tokens.last().unwrap_or(&decoder_start_token_id))
                .collect();
            let next_tokens_cpu = Array2::from_shape_vec((num_beams, 1), active_tokens)?;
            current_tokens_gpu = GpuTensor::from_ndarray(&context, &next_tokens_cpu)?;

            println!(
                "Step {}: cache_len={}, beams={:?}",
                step,
                gpu_cache.get_seq_length(),
                beams.iter().map(|b| &b.tokens).collect::<Vec<_>>()
            );
        }

        let final_hypotheses = if completed_beams.is_empty() {
            beams
        } else {
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

fn find_best_beams_and_get_indices(
    logits_2d: Array2<f32>, // Shape: [num_beams, vocab_size]
    current_beams: &[BeamHypothesis],
    config: &GenerationConfig,
    num_beams: usize,
) -> (Vec<u32>, Vec<usize>, Vec<BeamHypothesis>) {
    let mut candidates: Vec<(f32, usize, u32)> = Vec::with_capacity(num_beams * num_beams);

    // 1. Gather all possible next hypotheses
    for (beam_idx, (beam, logits_for_beam)) in
        current_beams.iter().zip(logits_2d.rows()).enumerate()
    {
        let log_probs = log_softmax_1d(&logits_for_beam.to_owned());
        let top_k = get_top_k_from_log_probs(&log_probs, num_beams);

        for (token_id, token_log_prob) in top_k {
            let new_score = beam.score + token_log_prob;
            candidates.push((new_score, beam_idx, token_id));
        }
    }

    // 2. Sort all candidates by score
    candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // 3. Select the top `num_beams` candidates to form the new beams
    let top_candidates = &candidates[..num_beams];

    let mut next_tokens = Vec::with_capacity(num_beams);
    let mut reorder_indices = Vec::with_capacity(num_beams);
    let mut updated_beams = Vec::with_capacity(num_beams);

    for &(new_score, source_beam_idx, new_token_id) in top_candidates {
        let mut new_tokens = current_beams[source_beam_idx].tokens.clone();
        new_tokens.push(new_token_id);

        next_tokens.push(new_token_id);
        reorder_indices.push(source_beam_idx);

        // The cache is no longer stored in the hypothesis for this strategy
        updated_beams.push(BeamHypothesis {
            tokens: new_tokens,
            score: new_score,
            // cache: None,
        });
    }
    
    println!("Top candidates: {:?}", candidates.iter().take(10).collect::<Vec<_>>());

    (next_tokens, reorder_indices, updated_beams)
}

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

pub fn apply_repetition_penalty(
    mut logits: Array1<f32>,
    generated_ids: &[u32],
    penalty: f32,
) -> Array1<f32> {
    if penalty == 1.0 {
        return logits;
    }
    for &id in generated_ids {
        let idx = id as usize;
        if logits[idx] < 0.0 {
            logits[idx] *= penalty;
        } else {
            logits[idx] /= penalty;
        }
    }
    logits
}

/// Efficient no-repeat n-gram blocking for generation.
///
/// This prevents generating any n-gram that has already appeared in the sequence.
/// Adapted from HuggingFace’s implementation logic.
///
/// # Arguments
/// * `logits` — the next-token logits (modified in place)
/// * `tokens` — sequence of already generated tokens
/// * `ngram_size` — size of the n-gram window
///
/// # Returns
/// Modified logits with -inf applied to blocked tokens.
pub fn apply_no_repeat_ngram(
    mut logits: Array1<f32>,
    tokens: &[u32],
    ngram_size: usize,
) -> Array1<f32> {
    if ngram_size == 0 || tokens.len() < ngram_size {
        return logits;
    }

    // Map from ngram_prefix → set of next tokens that follow that prefix.
    let mut ngram_map: HashMap<Vec<u32>, HashSet<u32>> = HashMap::new();

    // Build the prefix map efficiently.
    for window in tokens.windows(ngram_size) {
        let prefix = &window[..ngram_size - 1];
        let next_token = window[ngram_size - 1];
        ngram_map
            .entry(prefix.to_vec())
            .or_default()
            .insert(next_token);
    }

    // The prefix of the last (n-1) tokens
    let current_prefix = &tokens[tokens.len() - (ngram_size - 1)..];

    // If we’ve seen this prefix before, block all tokens that previously followed it
    if let Some(blocked_tokens) = ngram_map.get(current_prefix) {
        for &t in blocked_tokens {
            if (t as usize) < logits.len() {
                logits[t as usize] = f32::NEG_INFINITY;
            }
        }
    }

    logits
}
