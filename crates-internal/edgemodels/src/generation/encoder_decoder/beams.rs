use crate::generation::common::{
    apply_no_repeat_ngram, apply_repetition_penalty_mut as apply_repetition_penalty,
    get_top_k_from_log_probs, log_softmax_1d,
};
use crate::generation::encoder_decoder::{GenerationBackend, HasShape, StepInput};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use bytemuck;
use edgetransformers::cache::{Cache, GpuBeamKVCache};
use edgetransformers::gpu_context::WgpuContext;
use edgetransformers::gpu_ops::{GpuTensor, GpuTensorPool};
use edgetransformers::models::base::{
    DecodingStrategy, EncoderDecoderLanguageModel, GenerationConfig, LanguageModel,
};
use edgetransformers::prelude::*;
use edgetransformers::traits::EncoderOutput;
use ndarray::{Array1, Array2, Array3, s};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct BeamHypothesis {
    pub tokens: Vec<u32>,
    pub score: f32,
    // pub cache: Box<dyn Cache>,
}

pub async fn run_beam_search<B: GenerationBackend>(
    model: &dyn EncoderDecoderLanguageModel,
    backend: B,
    encoder_output: &EncoderOutput,
    config: &GenerationConfig,
) -> Result<String> {
    let (num_beams, length_penalty) = match &config.strategy {
        DecodingStrategy::BeamSearch(params) => (params.num_beams, params.length_penalty),
        DecodingStrategy::Greedy => (1, 1.0),
        _ => return Err(anyhow!("Only BeamSearch and Greedy are supported.")),
    };

    let eos_token_id = 2;
    let decoder_start_token_id = 2;
    let forced_bos_token_id = Some(0);

    // --- Initialize State ---
    let mut cache = model.new_cache(1, config.max_length, num_beams)?;
    let native_encoder_state = backend.prepare_encoder_state(encoder_output)?;
    println!(
        "Encoder output shape: {:?}",
        encoder_output.last_hidden_state.shape()
    );
    println!(
        "Encoder output (first 5 values): {:?}",
        &encoder_output.last_hidden_state.as_slice().unwrap()[..5]
    );
    let initial_tokens = vec![decoder_start_token_id; num_beams];
    let mut current_tokens_tensor = backend.create_token_tensor(&initial_tokens, num_beams)?;

    let mut beams = (0..num_beams)
        .map(|i| BeamHypothesis {
            tokens: vec![decoder_start_token_id],
            score: if i == 0 { 0.0 } else { f32::NEG_INFINITY },
        })
        .collect::<Vec<_>>();

    let mut completed_beams: Vec<BeamHypothesis> = Vec::new();

    // --- Main Loop ---
    for step in 0..config.max_length {
        let attention_mask =
            backend.prepare_attention_mask(cache.get_seq_length() + 1, num_beams)?;

        let step_input = StepInput {
            tokens: &current_tokens_tensor,
            encoder_state: Some(&native_encoder_state),
            attention_mask: &attention_mask,
        };

        // 1. Forward Pass (Writes new KV to cache at index `seq_len`)
        let outputs_3d = backend.forward(model, step_input, cache.as_mut()).await?;
        let last_hidden_state = outputs_3d.slice(s![.., 0, ..]);
        if step <= 1 {
            // Only log first 2 steps
            println!(
                "Step {}, Beam 0 hidden state (first 5 values): {:?}",
                step,
                &last_hidden_state.row(0).as_slice().unwrap()[..5]
            );
        }

        // 2. Project to Logits & Apply Penalties
        let vocab_size = model.config().vocab_size();
        let mut logits_2d = Array2::<f32>::zeros((num_beams, vocab_size));

        for i in 0..num_beams {
            let hidden_row = last_hidden_state.row(i);
            let mut logits_row = model.lm_head().dot(&hidden_row);

            if let Some(bias) = model.final_logits_bias() {
                logits_row += bias;
            }
            // ADD THIS:
            if step <= 1 && i == 0 {
                println!(
                    "Step {}, Beam 0 raw logits for token 23083: {}",
                    step, logits_row[23083]
                );

                // After penalties
                let mut test_logits = logits_row.clone();
                // ... apply penalties ...

                let log_probs = log_softmax_1d(&test_logits);
                println!(
                    "Step {}, Beam 0 log_prob for token 23083: {}",
                    step, log_probs[23083]
                );
            }

            // 3. Check the Cache Length (Crucial for CPU inference)
            println!("Cache Seq Len: {}", cache.get_seq_length());

            if step == 0 {
                if let Some(forced_bos_id) = forced_bos_token_id {
                    let forced_id_usize = forced_bos_id as usize;
                    if forced_id_usize < vocab_size {
                        // Set all scores to -inf
                        logits_row.fill(f32::NEG_INFINITY);
                        // Set the forced token to a high value (0.0 in log space is max probability 1.0)
                        logits_row[forced_id_usize] = 0.0;
                    }
                }
            }

            // Apply Penalties (only if not forcing BOS)
            let apply_penalties = step > 0 || forced_bos_token_id.is_none();
            if apply_penalties {
                if config.repetition_penalty != 1.0 {
                    apply_repetition_penalty(
                        &mut logits_row,
                        &beams[i].tokens,
                        config.repetition_penalty,
                    );
                }
                if config.no_repeat_ngram_size > 0 {
                    apply_no_repeat_ngram(
                        &mut logits_row,
                        &beams[i].tokens,
                        config.no_repeat_ngram_size,
                    );
                }
            }

            logits_2d.row_mut(i).assign(&logits_row);
        }

        // 3. Select Next Tokens
        let (next_tokens_vec, reorder_indices_vec, updated_beams) =
            find_best_beams_and_get_indices(logits_2d, &beams, config, num_beams);
        println!("Decisions:");
        for i in 0..num_beams.min(3) {
            // Show top 3 beams
            println!(
                "  Beam {}: selected token {} (from old beam {}), new score {:.4}",
                i, next_tokens_vec[i], reorder_indices_vec[i], updated_beams[i].score
            );
        }
        // 4. Update State
        cache.increment_len(1);

        beams = updated_beams;
        backend.update_token_tensor(&mut current_tokens_tensor, &next_tokens_vec)?;
        //current_tokens_tensor = backend.create_token_tensor(&next_tokens_vec, num_beams)?;

        if num_beams > 1 {
            backend.reorder_cache(cache.as_mut(), &reorder_indices_vec)?;
        }

        if beams
            .iter()
            .all(|b| *b.tokens.last().unwrap() == eos_token_id)
        {
            break;
        }
    }

    // --- Finalize ---
    let final_hypotheses = if completed_beams.is_empty() {
        beams
    } else {
        completed_beams
    };
    if final_hypotheses.is_empty() {
        return Err(anyhow!("No hypothesis generated."));
    }

    let best_hypo = final_hypotheses
        .iter()
        .max_by(|a, b| {
            let score_a = a.score / (a.tokens.len() as f32).powf(length_penalty);
            let score_b = b.score / (b.tokens.len() as f32).powf(length_penalty);
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

    model
        .tokenizer()
        .decode(tokens_to_decode, true)
        .map_err(|e| anyhow!(e))
}

pub fn find_best_beams_and_get_indices(
    logits_2d: Array2<f32>,
    current_beams: &[BeamHypothesis],
    config: &GenerationConfig,
    num_beams: usize,
) -> (Vec<u32>, Vec<usize>, Vec<BeamHypothesis>) {
    let mut candidates: Vec<(f32, usize, u32)> = Vec::with_capacity(num_beams * num_beams);

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

    candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    candidates.truncate(num_beams);

    let mut next_tokens = Vec::with_capacity(num_beams);
    let mut reorder_indices = Vec::with_capacity(num_beams);
    let mut updated_beams = Vec::with_capacity(num_beams);

    for &(new_score, source_beam_idx, new_token_id) in &candidates {
        let mut new_tokens = current_beams[source_beam_idx].tokens.clone();
        new_tokens.push(new_token_id);
        next_tokens.push(new_token_id);
        reorder_indices.push(source_beam_idx);
        updated_beams.push(BeamHypothesis {
            tokens: new_tokens,
            score: new_score,
        });
    }

    (next_tokens, reorder_indices, updated_beams)
}
