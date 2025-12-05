use crate::generation::common::{
    apply_no_repeat_ngram, apply_repetition_penalty_mut as apply_repetition_penalty,
    get_top_k_from_log_probs, log_softmax_1d, StreamedToken, TokenType,
};
use crate::generation::encoder_decoder::{GenerationBackend, HasShape, StepInput};
use anyhow::{anyhow, Result};
use async_stream::try_stream;
use edgetransformers::cache::Cache;
use edgetransformers::models::base::{
    DecodingStrategy, EncoderDecoderLanguageModel, GenerationConfig, LanguageModel,
};
use edgetransformers::traits::EncoderOutput;
use futures_core::Stream;
use ndarray::{s, Array2};
use rayon::prelude::*;
pub struct BeamHypothesis {
    pub tokens: Vec<u32>,
    pub score: f32,
}

/// Non-streaming beam search (existing)
pub async fn run_beam_search<B: GenerationBackend>(
    model: &dyn EncoderDecoderLanguageModel,
    backend: &B,
    encoder_output: &EncoderOutput,
    config: &GenerationConfig,
) -> Result<String> {
    let t_search = std::time::Instant::now();
    let (tokens, _) = run_beam_search_inner(model, backend, encoder_output, config, false).await?;
    log::info!("[Seq2Seq] Beam Search Loop: {:?}", t_search.elapsed());

    let eos_token_id = 2u32;
    let decoder_start_token_id = 2u32;

    let mut tokens_to_decode = &tokens[..];
    if tokens_to_decode.first() == Some(&decoder_start_token_id) {
        tokens_to_decode = &tokens_to_decode[1..];
    }
    if tokens_to_decode.last() == Some(&eos_token_id) {
        tokens_to_decode = &tokens_to_decode[..tokens_to_decode.len() - 1];
    }

    let t_decode = std::time::Instant::now();
    let res = model.tokenizer().decode(tokens_to_decode, true).map_err(|e| anyhow!(e));
    log::info!("[Seq2Seq] Tokenizer Decode: {:?}", t_decode.elapsed());

    res
}

/// Streaming beam search - yields tokens as they're generated
pub fn run_beam_search_stream<'a, B: GenerationBackend + 'a>(
    model: &'a dyn EncoderDecoderLanguageModel,
    backend: &'a B,
    encoder_output: &'a EncoderOutput,
    config: &'a GenerationConfig,
) -> impl Stream<Item=Result<StreamedToken>> + 'a {
    try_stream! {
        let (num_beams, length_penalty) = match &config.strategy {
            DecodingStrategy::BeamSearch(params) => (params.num_beams, params.length_penalty),
            DecodingStrategy::Greedy => (1, 1.0),
            _ => {
                Err(anyhow!("Only BeamSearch and Greedy are supported."))?;
                unreachable!()
            }
        };

        if num_beams > 1 {
            log::warn!(
                "Streaming with beam search (num_beams={}): output may change as beams are reordered. \
                For stable streaming, use DecodingStrategy::Greedy.",
                num_beams
            );
        }

        let eos_token_id = 2u32;
        let decoder_start_token_id = 2u32;
        let forced_bos_token_id = Some(0u32);
        let vocab_size = model.config().vocab_size();

        // Initialize
        let mut cache = model.new_cache(1, config.max_length, num_beams)?;
        let native_encoder_state = backend.prepare_encoder_state(model, encoder_output)?;

        let initial_tokens = vec![decoder_start_token_id; num_beams];
        let mut current_tokens_tensor = backend.create_token_tensor(&initial_tokens, num_beams)?;

        let mut beams: Vec<BeamHypothesis> = (0..num_beams)
            .map(|i| BeamHypothesis {
                tokens: vec![decoder_start_token_id],
                score: if i == 0 { 0.0 } else { f32::NEG_INFINITY },
            })
            .collect();

        // Main loop
        for step in 0..config.max_length {
            let attention_mask = backend.prepare_attention_mask(cache.get_seq_length() + 1, num_beams)?;

            let step_input = StepInput {
                tokens: &current_tokens_tensor,
                encoder_state: Some(&native_encoder_state),
                attention_mask: &attention_mask,
            };

            // Forward pass
            let outputs_3d = backend.forward(model, step_input, cache.as_mut()).await?;
            let last_hidden_state = outputs_3d.slice(s![.., 0, ..]);

            // Compute logits
            let mut logits_2d = Array2::<f32>::zeros((num_beams, vocab_size));
            for i in 0..num_beams {
                let hidden_row = last_hidden_state.row(i);



                let mut logits_row = model.lm_head().dot(&hidden_row);

                if let Some(bias) = model.final_logits_bias() {
                    logits_row += bias;
                }

                if step == 0 {
                    if let Some(forced_bos_id) = forced_bos_token_id {
                        logits_row.fill(f32::NEG_INFINITY);
                        logits_row[forced_bos_id as usize] = 0.0;
                    }
                } else {
                    if config.repetition_penalty != 1.0 {
                        apply_repetition_penalty(&mut logits_row, &beams[i].tokens, config.repetition_penalty);
                    }
                    if config.no_repeat_ngram_size > 0 {
                        apply_no_repeat_ngram(&mut logits_row, &beams[i].tokens, config.no_repeat_ngram_size);
                    }
                }

                logits_2d.row_mut(i).assign(&logits_row);
            }

            // Select next tokens
            let (next_tokens_vec, reorder_indices_vec, updated_beams) =
                find_best_beams_and_get_indices(logits_2d, &beams, config, num_beams);

            cache.increment_len(1);
            beams = updated_beams;
            backend.update_token_tensor(&mut current_tokens_tensor, &next_tokens_vec)?;

            if num_beams > 1 {
                backend.reorder_cache(cache.as_mut(), &reorder_indices_vec)?;
            }

            // === STREAM BEST BEAM'S TOKEN ===
            let best_beam = &beams[0];
            let new_token = *best_beam.tokens.last().unwrap();

            if new_token != eos_token_id && new_token != decoder_start_token_id {
                let decoded = model.tokenizer()
                    .decode(&[new_token], true)
                    .map_err(|e| anyhow!(e))?;

                yield StreamedToken {
                    text: decoded,
                    id: new_token,
                    token_type: TokenType::Generated,
                };
            }

            // Check completion
            if beams.iter().all(|b| *b.tokens.last().unwrap() == eos_token_id) {
                break;
            }
        }
    }
}

/// Inner implementation shared by both streaming and non-streaming
async fn run_beam_search_inner<B: GenerationBackend>(
    model: &dyn EncoderDecoderLanguageModel,
    backend: &B,
    encoder_output: &EncoderOutput,
    config: &GenerationConfig,
    _is_streaming: bool,
) -> Result<(Vec<u32>, f32)> {
    let (num_beams, length_penalty) = match &config.strategy {
        DecodingStrategy::BeamSearch(params) => (params.num_beams, params.length_penalty),
        DecodingStrategy::Greedy => (1, 1.0),
        _ => return Err(anyhow!("Only BeamSearch and Greedy are supported.")),
    };

    let eos_token_id = 2u32;
    let decoder_start_token_id = 2u32;
    let forced_bos_token_id = Some(0u32);
    let vocab_size = model.config().vocab_size();

    let mut cache = model.new_cache(1, config.max_length, num_beams)?;
    let native_encoder_state = backend.prepare_encoder_state(model, encoder_output)?;

    let initial_tokens = vec![decoder_start_token_id; num_beams];
    let mut current_tokens_tensor = backend.create_token_tensor(&initial_tokens, num_beams)?;

    let mut beams: Vec<BeamHypothesis> = (0..num_beams)
        .map(|i| BeamHypothesis {
            tokens: vec![decoder_start_token_id],
            score: if i == 0 { 0.0 } else { f32::NEG_INFINITY },
        })
        .collect();

    for step in 0..config.max_length {
        let t_step_start = std::time::Instant::now();

        let attention_mask = backend.prepare_attention_mask(cache.get_seq_length() + 1, num_beams)?;

        let step_input = StepInput {
            tokens: &current_tokens_tensor,
            encoder_state: Some(&native_encoder_state),
            attention_mask: &attention_mask,
        };

        // --- FORWARD PASS TIMING ---
        let t_fwd_start = std::time::Instant::now();
        let outputs_3d = backend.forward(model, step_input, cache.as_mut()).await?;
        let t_fwd = t_fwd_start.elapsed();

        // --- PROJECTION / LOGITS TIMING ---
        let t_proj_start = std::time::Instant::now();
        let last_hidden_state = outputs_3d.slice(s![.., 0, ..]);

        // let mut logits_2d = Array2::<f32>::zeros((num_beams, vocab_size));

        let mut logits_2d = model.lm_head_layer().matmul(&last_hidden_state.view());

        if let Some(bias) = model.final_logits_bias() {
            logits_2d = logits_2d + bias;
        }
        logits_2d.outer_iter_mut().enumerate().for_each(|(i, mut logits_row)| {

            // A. Forced BOS logic (Step 0)
            if step == 0 {
                if let Some(forced_bos_id) = forced_bos_token_id {
                    logits_row.fill(f32::NEG_INFINITY);
                    logits_row[forced_bos_id as usize] = 0.0;
                }
            } else {
                // B. Repetition Penalty
                if config.repetition_penalty != 1.0 {
                    apply_repetition_penalty(
                        &mut logits_row.to_owned(),
                        &beams[i].tokens,
                        config.repetition_penalty,
                    );
                }

                // C. N-Gram Blocker
                if config.no_repeat_ngram_size > 0 {
                    apply_no_repeat_ngram(
                        &mut logits_row.to_owned(),
                        &beams[i].tokens,
                        config.no_repeat_ngram_size,
                    );
                }
            }
        });

        let t_proj = t_proj_start.elapsed();

        // --- BEAM SELECTION TIMING ---
        let t_sel_start = std::time::Instant::now();
        let (next_tokens_vec, reorder_indices_vec, updated_beams) =
            find_best_beams_and_get_indices(logits_2d, &beams, config, num_beams);
        let t_sel = t_sel_start.elapsed();

        cache.increment_len(1);
        beams = updated_beams;

        // --- UPDATE TIMING ---
        let t_update_start = std::time::Instant::now();
        backend.update_token_tensor(&mut current_tokens_tensor, &next_tokens_vec)?;

        if num_beams > 1 {
            backend.reorder_cache(cache.as_mut(), &reorder_indices_vec)?;
        }
        let t_update = t_update_start.elapsed();

        let t_total = t_step_start.elapsed();

        log::info!(
            "Step {}: Fwd: {:?}, Proj: {:?}, Select: {:?}, Update: {:?} | Total: {:?} ({:.2} t/s)", 
            step, t_fwd, t_proj, t_sel, t_update, t_total, 1.0 / t_total.as_secs_f32()
        );

        if beams.iter().all(|b| *b.tokens.last().unwrap() == eos_token_id) {
            log::info!("All beams finished at step {}", step);
            break;
        }
    }

    let best_beam = beams
        .into_iter()
        .max_by(|a, b| {
            let score_a = a.score / (a.tokens.len() as f32).powf(length_penalty);
            let score_b = b.score / (b.tokens.len() as f32).powf(length_penalty);
            score_a.partial_cmp(&score_b).unwrap()
        })
        .ok_or_else(|| anyhow!("No beams"))?;

    Ok((best_beam.tokens, best_beam.score))
}

pub fn find_best_beams_and_get_indices(
    logits_2d: Array2<f32>,
    current_beams: &[BeamHypothesis],
    _config: &GenerationConfig,
    num_beams: usize,
) -> (Vec<u32>, Vec<usize>, Vec<BeamHypothesis>) {
    let mut candidates: Vec<(f32, usize, u32)> = Vec::with_capacity(num_beams * num_beams);

    for (beam_idx, (beam, logits_for_beam)) in current_beams.iter().zip(logits_2d.rows()).enumerate() {
        let log_probs = log_softmax_1d(&logits_for_beam.to_owned());
        let top_k = get_top_k_from_log_probs(&log_probs, num_beams);

        for (token_id, token_log_prob) in top_k {
            candidates.push((beam.score + token_log_prob, beam_idx, token_id));
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
        updated_beams.push(BeamHypothesis { tokens: new_tokens, score: new_score });
    }

    (next_tokens, reorder_indices, updated_beams)
}