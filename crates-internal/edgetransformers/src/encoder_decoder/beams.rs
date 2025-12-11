use crate::cache::Cache;
use crate::common::{
    apply_no_repeat_ngram, apply_no_repeat_ngram_inplace, apply_repetition_penalty_inplace, apply_repetition_penalty_mut as apply_repetition_penalty,
    get_top_k_from_log_probs, log_softmax_1d,
    StreamedToken, TokenType,
};
use crate::encoder_decoder::traits::{
    EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel, StepInput,
};
use crate::models::base::{DecodingStrategy, GenerationConfig};
use crate::traits::EncoderOutput;
use anyhow::{anyhow, Result};
use async_stream::try_stream;
use futures_core::Stream;
use ndarray::{s, Array2};

#[derive(Clone)]
pub struct BeamHypothesis {
    pub tokens: Vec<u32>,
    pub score: f32,
}

struct BeamContext<'a, B: EncoderDecoderGenerationBackend> {
    model: &'a dyn EncoderDecoderLanguageModel,
    backend: &'a B,
    config: &'a GenerationConfig,

    // Mutable state
    cache: Box<dyn Cache>,
    current_tokens_tensor: B::Tensor,
    encoder_state: B::Tensor,
    beams: Vec<BeamHypothesis>,

    // Constants
    num_beams: usize,
    eos_token_id: u32,
    decoder_start_token_id: u32,
}

impl<'a, B: EncoderDecoderGenerationBackend> BeamContext<'a, B> {
    // This `new` method needs to be async because `backend.encode` is async
    async fn new(
        model: &'a dyn EncoderDecoderLanguageModel,
        backend: &'a B,
        // UPDATED: We start from the raw text/tokens, not a pre-encoded state
        input_text: &'a str,
        config: &'a GenerationConfig,
    ) -> Result<Self> {
        let num_beams = match &config.strategy {
            DecodingStrategy::BeamSearch(params) => params.num_beams,
            DecodingStrategy::Greedy => 1,
            _ => return Err(anyhow!("Unsupported strategy")),
        };

        // Tokenize the input text
        let encoding = model
            .tokenizer()
            .encode(input_text, true)
            .map_err(|e| anyhow!(e))?;
        let ids = encoding.get_ids();
        log::error!("CHECKPOINT 1 - Input IDs: {:?}", &ids[..10.min(ids.len())]);
        let encoder_tokens = encoding.get_ids();

        // NEW: The backend is now responsible for the entire encoding pass.
        let encoder_state = backend.encode(model, encoder_tokens, num_beams).await?;

        // Downcast the Box<dyn Cache> to the concrete type B::Cache
        let cache = model.new_cache(1, config.max_length, num_beams)?;

        let decoder_start_token_id = model.decoder_start_token_id();
        let current_tokens_tensor =
            backend.create_token_tensor(&vec![decoder_start_token_id; num_beams], num_beams)?;

        let beams = (0..num_beams)
            .map(|i| BeamHypothesis {
                tokens: vec![decoder_start_token_id],
                score: if i == 0 { 0.0 } else { f32::NEG_INFINITY },
            })
            .collect();

        Ok(Self {
            model,
            backend,
            config,
            cache,
            current_tokens_tensor,
            encoder_state,
            beams,
            num_beams,
            eos_token_id: model.config().eos_token_id().unwrap_or(2),
            decoder_start_token_id,
        })
    }
}

async fn beam_step<B: EncoderDecoderGenerationBackend>(
    ctx: &mut BeamContext<'_, B>,
    step: usize,
) -> Result<()> {
    let logits_3d = ctx
        .backend
        .decode_step(
            ctx.model,
            &ctx.current_tokens_tensor,
            &ctx.encoder_state,
            ctx.cache.as_mut(),
        )
        .await?;

    let mut logits_2d = logits_3d.slice(s![.., -1, ..]).to_owned();
    // if let Some(bias) = ctx.model.final_logits_bias() {
    //     logits_2d += bias;
    // }
    let forced_bos_token_id: Option<u32> = Some(0); // ← ADD THIS
    let t = logits_2d.clone();
    logits_2d
        .outer_iter_mut()
        .enumerate()
        .for_each(|(i, mut logits_row)| {
            if step == 0 {
                let slice = t.slice(s![0, 0..10]);
                log::error!("CHECKPOINT 3 - Logits Step 0: {:?}", slice);
                // log::error!("Rust Logits Step 0 [0, 0..10]: {:?}", slice);
                // Use forced_bos_token_id, not bos_token_id
                if let Some(forced_bos_id) = forced_bos_token_id {
                    logits_row.fill(f32::NEG_INFINITY);
                    logits_row[forced_bos_id as usize] = 0.0;
                }
                // if let Some(bos) = ctx.model.config().bos_token_id() {
                //     // Optional: Force BOS if model requires it explicitly via logits
                //     logits_row.fill(f32::NEG_INFINITY);
                //     logits_row[bos as usize] = 0.0;
                // }
            } else {
                if ctx.config.repetition_penalty != 1.0 {
                    apply_repetition_penalty_inplace(
                        &mut logits_row, // ✅ Pass the view directly
                        &ctx.beams[i].tokens,
                        ctx.config.repetition_penalty,
                    );
                }
                if ctx.config.no_repeat_ngram_size > 0 {
                    // Debug: count how many tokens are -inf BEFORE
                    let blocked_before = logits_row
                        .iter()
                        .filter(|&&x| x == f32::NEG_INFINITY)
                        .count();

                    apply_no_repeat_ngram_inplace(
                        &mut logits_row,
                        &ctx.beams[i].tokens,
                        ctx.config.no_repeat_ngram_size,
                    );

                    // Debug: count how many tokens are -inf AFTER
                    let blocked_after = logits_row
                        .iter()
                        .filter(|&&x| x == f32::NEG_INFINITY)
                        .count();
                }
            }
        });

    // 5. Select Candidates
    let (next_tokens, reorder_indices, updated_beams) =
        find_best_beams_and_get_indices(logits_2d, &ctx.beams, ctx.config, ctx.num_beams);

    // 6. Update State
    ctx.cache.increment_len(1);
    ctx.beams = updated_beams;
    ctx.backend
        .update_token_tensor(&mut ctx.current_tokens_tensor, &next_tokens)?;

    if ctx.num_beams > 1 {
        ctx.backend
            .reorder_cache(ctx.cache.as_mut(), &reorder_indices)?;
    }

    Ok(())
}
// ============================================================================
//  Executes the full search, then decodes ONLY the final winner.
// ============================================================================
// pub async fn run_beam_search<B: EncoderDecoderGenerationBackend>(
//     model: &dyn EncoderDecoderLanguageModel,
//     backend: &B,
//     encoder_output: &EncoderOutput,
//     config: &GenerationConfig,
// ) -> Result<String> {
//     // 1. Initialize Context
//     let mut ctx = BeamContext::new(model, backend, encoder_output, config)?;

//     // 2. Loop until finished
//     for step in 0..config.max_length {
//         beam_step(&mut ctx, step).await?;

//         // Stop if ALL beams have generated EOS
//         if ctx
//             .beams
//             .iter()
//             .all(|b| *b.tokens.last().unwrap() == ctx.eos_token_id)
//         {
//             break;
//         }
//     }

//     // 3. Select Winner & Decode ONCE
//     // The beams are already sorted by score in `find_best_beams_and_get_indices`
//     // so beams[0] is the global winner.
//     let best_beam = &ctx.beams[0];

//     // Filter out special tokens
//     let tokens = &best_beam.tokens;
//     let start = if tokens.first() == Some(&ctx.decoder_start_token_id) {
//         1
//     } else {
//         0
//     };
//     let end = if tokens.last() == Some(&ctx.eos_token_id) {
//         tokens.len() - 1
//     } else {
//         tokens.len()
//     };

//     let clean_tokens = if start < end {
//         &tokens[start..end]
//     } else {
//         &[]
//     };

//     let text = model
//         .tokenizer()
//         .decode(clean_tokens, true)
//         .map_err(|e| anyhow!(e))?;
//     Ok(text)
// }

pub async fn run_beam_search<B: EncoderDecoderGenerationBackend>(
    model: &dyn EncoderDecoderLanguageModel,
    backend: &B,
    input_text: &str, // Takes raw text now
    config: &GenerationConfig,
) -> Result<String> {
    // 1. Initialize Context
    let mut ctx = BeamContext::new(model, backend, input_text, config).await?;

    // 2. Loop until finished
    for step in 0..config.max_length {
        beam_step(&mut ctx, step).await?;

        // Stop if ALL beams have generated EOS
        if ctx
            .beams
            .iter()
            .all(|b| *b.tokens.last().unwrap() == ctx.eos_token_id)
        {
            break;
        }
    }

    // 3. Select Winner & Decode ONCE
    // The beams are already sorted by score in `find_best_beams_and_get_indices`
    // so beams[0] is the global winner.
    let best_beam = &ctx.beams[0];

    // Filter out special tokens
    let tokens = &best_beam.tokens;
    let start = if tokens.first() == Some(&ctx.decoder_start_token_id) {
        1
    } else {
        0
    };
    let end = if tokens.last() == Some(&ctx.eos_token_id) {
        tokens.len() - 1
    } else {
        tokens.len()
    };

    let clean_tokens = if start < end {
        &tokens[start..end]
    } else {
        &[]
    };

    let text = model
        .tokenizer()
        .decode(clean_tokens, true)
        .map_err(|e| anyhow!(e))?;
    Ok(text)
}
// ============================================================================
//  Streaming Implementation
// ============================================================================
pub fn run_beam_search_stream<'a, B: EncoderDecoderGenerationBackend + 'a>(
    model: &'a dyn EncoderDecoderLanguageModel,
    backend: &'a B,
    input_text: &'a str, // Takes raw text now
    config: &'a GenerationConfig,
) -> impl Stream<Item=Result<StreamedToken>> + 'a {
    try_stream! {
        let mut ctx = BeamContext::new(model, backend, input_text, config).await?;

        if ctx.num_beams > 1 {
            log::warn!("Streaming Beam Search is unstable (tokens may change). Use Greedy for stability.");
        }

        for step in 0..config.max_length {
            beam_step(&mut ctx, step).await?;

            // Stream the *current* best token
            let best_beam = &ctx.beams[0];
            let new_token = *best_beam.tokens.last().unwrap();

            if new_token != ctx.eos_token_id && new_token != ctx.decoder_start_token_id {
                let text = model.tokenizer().decode(&[new_token], true).unwrap_or_default();
                yield StreamedToken {
                    text,
                    id: new_token,
                    token_type: TokenType::Generated,
                };
            }

            if ctx.beams.iter().all(|b| *b.tokens.last().unwrap() == ctx.eos_token_id) {
                break;
            }
        }
    }
}

pub fn find_best_beams_and_get_indices(
    logits_2d: Array2<f32>,
    current_beams: &[BeamHypothesis],
    _config: &GenerationConfig,
    num_beams: usize,
) -> (Vec<u32>, Vec<usize>, Vec<BeamHypothesis>) {
    let mut candidates: Vec<(f32, usize, u32)> = Vec::with_capacity(num_beams * num_beams);

    for (beam_idx, (beam, logits_for_beam)) in
        current_beams.iter().zip(logits_2d.rows()).enumerate()
    {
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
        updated_beams.push(BeamHypothesis {
            tokens: new_tokens,
            score: new_score,
        });
    }

    (next_tokens, reorder_indices, updated_beams)
}
