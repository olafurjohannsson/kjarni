//! Speculative decoding: something cheap proposes several tokens, and the target model
//! checks all of them in one pass over its weights.
//!
//! The output is what plain decoding gives. With greedy decoding it is the same text.
//! With sampling it follows the same distribution, using speculative sampling
//! (Leviathan et al. 2023, Chen et al. 2023).

use crate::{
    common::{
        CancellationToken, DecodingStrategy, GenerationConfig, SpeculationParams, StreamedToken,
        TokenType, apply_no_repeat_ngram, apply_repetition_penalty_mut, sample_token,
        sampling_probs,
    },
    decoder::{
        backend::AnyDecoderBackend,
        traits::{DecoderGenerationBackend, DecoderLanguageModel},
    },
    stats::GenerationStats,
};

use crate::Cache;
use anyhow::{Result, anyhow};
use ndarray::{Array1, Array2, s};
use rand::Rng;
use std::sync::Arc;

/// Cached draft model context for speculative decoding.
pub struct DraftModelContext {
    pub model: Arc<dyn DecoderLanguageModel + Send + Sync>,
    pub backend: AnyDecoderBackend,
}

impl DraftModelContext {
    /// The draft runs on the device it was loaded for.
    pub fn load(model: Arc<dyn DecoderLanguageModel + Send + Sync>) -> Result<Self> {
        let backend = AnyDecoderBackend::for_model(model.as_ref())?;

        log::info!(
            "Draft model loaded: device={:?}, vocab_size={}, context_size={}",
            model.device(),
            model.vocab_size(),
            model.context_size()
        );

        Ok(Self { model, backend })
    }
}

/// How a speculative run went.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculationStats {
    /// Passes over the target's weights after the prompt.
    pub passes: usize,
    /// Tokens proposed for checking.
    pub proposed: usize,
    /// Proposed tokens the target kept.
    pub accepted: usize,
    /// Tokens produced.
    pub generated: usize,
}

impl SpeculationStats {
    /// Tokens produced by each pass over the target. Plain decoding gets 1.0.
    pub fn tokens_per_pass(&self) -> f64 {
        self.generated as f64 / self.passes.max(1) as f64
    }
}

/// What the proposed tokens come from.
enum Proposer {
    Model(DraftState),
    PromptLookup { ngram: usize },
}

/// A draft model whose cache holds every token accepted so far.
struct DraftState {
    model: Arc<dyn DecoderLanguageModel + Send + Sync>,
    backend: AnyDecoderBackend,
    cache: Box<dyn Cache>,
    token: <AnyDecoderBackend as DecoderGenerationBackend>::DecodeToken,
    /// The draft's prediction for the token after those in the cache.
    logits: Array1<f32>,
    /// Proposals of the current round that are in the cache.
    fed: usize,
}

impl DraftState {
    /// Runs `token`, the one at `position`, through the draft.
    async fn feed(&mut self, token: u32, position: usize) -> Result<()> {
        self.backend.update_decode_token(&mut self.token, token)?;
        self.logits = self
            .backend
            .decode_one(
                self.model.as_ref(),
                &self.token,
                position + 1,
                self.cache.as_mut(),
            )
            .await?;
        Ok(())
    }
}

/// The proposed tokens, and for sampling the distribution each was drawn from.
#[derive(Default)]
struct Proposal {
    tokens: Vec<u32>,
    distributions: Vec<Array1<f32>>,
}

impl Proposer {
    /// Proposes up to `count` tokens to follow `history`.
    async fn propose(
        &mut self,
        history: &[u32],
        count: usize,
        config: &GenerationConfig,
        stop_tokens: &std::collections::HashSet<u32>,
        vocab_size: usize,
    ) -> Result<Proposal> {
        let mut proposal = Proposal::default();
        match self {
            Proposer::Model(draft) => {
                draft.fed = 0;
                let mut context = history.to_vec();
                for _ in 0..count {
                    let mut logits = draft.logits.clone();
                    apply_penalties(&mut logits, &context, config);
                    let token = match &config.strategy {
                        DecodingStrategy::Sample(params) => {
                            let distribution = sampling_probs(logits, params);
                            let token =
                                sample_from_distribution(&distribution, &mut rand::thread_rng());
                            proposal.distributions.push(distribution);
                            token
                        }
                        strategy => sample_token(logits, strategy)?,
                    };
                    proposal.tokens.push(token);
                    if stop_tokens.contains(&token) {
                        break;
                    }
                    draft.feed(token, context.len()).await?;
                    draft.fed += 1;
                    context.push(token);
                }
            }
            Proposer::PromptLookup { ngram } => {
                proposal.tokens = prompt_lookup(history, *ngram, count);
                if matches!(config.strategy, DecodingStrategy::Sample(_)) {
                    proposal.distributions = proposal
                        .tokens
                        .iter()
                        .map(|&token| {
                            let mut certain = Array1::zeros(vocab_size);
                            certain[token as usize] = 1.0;
                            certain
                        })
                        .collect();
                }
            }
        }
        Ok(proposal)
    }

    /// Brings the proposer up to date with `tokens`, of which the proposed `kept` follow
    /// the `previous_len` tokens it already had.
    async fn sync(&mut self, tokens: &[u32], previous_len: usize, kept: usize) -> Result<()> {
        if let Proposer::Model(draft) = self {
            let shared = previous_len + kept.min(draft.fed);
            draft.cache.set_seq_length(shared);
            for (position, &token) in tokens.iter().enumerate().skip(shared) {
                draft.feed(token, position).await?;
            }
        }
        Ok(())
    }
}

/// The tokens that followed the latest earlier occurrence of the end of `history`, trying
/// the longest match first. Nothing is proposed when the end has not occurred before.
pub(crate) fn prompt_lookup(history: &[u32], ngram: usize, count: usize) -> Vec<u32> {
    for length in (1..=ngram.min(history.len().saturating_sub(1))).rev() {
        let tail = &history[history.len() - length..];
        let found = history[..history.len() - 1]
            .windows(length)
            .rposition(|window| window == tail);
        if let Some(start) = found {
            let from = start + length;
            let to = (from + count).min(history.len());
            return history[from..to].to_vec();
        }
    }
    Vec::new()
}

/// Runs speculative decoding with a draft model proposing the tokens.
pub async fn run_speculative_generation_loop(
    target: Arc<dyn DecoderLanguageModel + Send + Sync>,
    target_backend: AnyDecoderBackend,
    draft: Arc<dyn DecoderLanguageModel + Send + Sync>,
    draft_backend: AnyDecoderBackend,
    input_tokens: Vec<u32>,
    config: GenerationConfig,
    spec_params: &SpeculationParams,
    tx: tokio::sync::mpsc::Sender<Result<StreamedToken>>,
    cancellation: Option<CancellationToken>,
) -> Result<SpeculationStats> {
    if input_tokens.is_empty() {
        return Err(anyhow!("Empty prompt"));
    }
    let capacity = cache_capacity(&input_tokens, &config, spec_params);
    let mut cache = draft.new_cache(1, capacity, 1)?;
    let tokens = Array2::from_shape_vec((1, input_tokens.len()), input_tokens.clone())?;
    let logits = draft_backend
        .prefill(draft.as_ref(), &tokens, cache.as_mut())
        .await?;
    let token = draft_backend.new_decode_token()?;
    let proposer = Proposer::Model(DraftState {
        model: draft,
        backend: draft_backend,
        cache,
        token,
        logits,
        fed: 0,
    });
    run_loop(
        target,
        target_backend,
        proposer,
        input_tokens,
        config,
        spec_params,
        tx,
        cancellation,
    )
    .await
}

/// Runs speculative decoding with the tokens proposed from earlier in the text, which
/// needs no second model. It helps where the output repeats its input: editing, rewriting
/// and extraction.
pub async fn run_prompt_lookup_generation_loop(
    target: Arc<dyn DecoderLanguageModel + Send + Sync>,
    target_backend: AnyDecoderBackend,
    input_tokens: Vec<u32>,
    config: GenerationConfig,
    spec_params: &SpeculationParams,
    ngram: usize,
    tx: tokio::sync::mpsc::Sender<Result<StreamedToken>>,
    cancellation: Option<CancellationToken>,
) -> Result<SpeculationStats> {
    run_loop(
        target,
        target_backend,
        Proposer::PromptLookup {
            ngram: ngram.max(1),
        },
        input_tokens,
        config,
        spec_params,
        tx,
        cancellation,
    )
    .await
}

fn generation_limit(prompt_len: usize, config: &GenerationConfig) -> usize {
    config
        .max_new_tokens
        .map_or(config.max_length, |n| prompt_len + n)
}

fn cache_capacity(
    input_tokens: &[u32],
    config: &GenerationConfig,
    spec_params: &SpeculationParams,
) -> usize {
    generation_limit(input_tokens.len(), config) + spec_params.num_tokens + 1
}

async fn run_loop(
    target: Arc<dyn DecoderLanguageModel + Send + Sync>,
    target_backend: AnyDecoderBackend,
    mut proposer: Proposer,
    input_tokens: Vec<u32>,
    config: GenerationConfig,
    spec_params: &SpeculationParams,
    tx: tokio::sync::mpsc::Sender<Result<StreamedToken>>,
    cancellation: Option<CancellationToken>,
) -> Result<SpeculationStats> {
    let prompt_len = input_tokens.len();
    if prompt_len == 0 {
        return Err(anyhow!("Empty prompt"));
    }
    if matches!(config.strategy, DecodingStrategy::BeamSearch(_)) {
        return Err(anyhow!(
            "speculative decoding needs greedy decoding or sampling, not beam search"
        ));
    }

    let max_len = generation_limit(prompt_len, &config);
    let context_limit = target.context_size();
    let mut speculation = SpeculationStats::default();

    log::info!(
        "Speculative decoding: prompt={}, max={}, depth={}",
        prompt_len,
        max_len,
        spec_params.num_tokens
    );

    let capacity = cache_capacity(&input_tokens, &config, spec_params);
    let mut target_cache = target.new_cache(1, capacity, 1)?;

    let mut all_tokens = input_tokens.clone();
    let mut stats = GenerationStats::new();

    stats.start_prefill(prompt_len);
    let tokens_array = Array2::from_shape_vec((1, prompt_len), input_tokens.clone())?;
    target_backend
        .prefill(target.as_ref(), &tokens_array, target_cache.as_mut())
        .await?;
    stats.end_prefill();

    let tokenizer = target.tokenizer();
    for &token_id in &input_tokens {
        if Some(token_id) == target.bos_token_id() {
            continue;
        }

        // Check cancellation
        if let Some(ref c) = cancellation
            && c.is_cancelled()
        {
            log::debug!("Cancelled during prompt emission");
            return Ok(speculation);
        }

        let text = tokenizer
            .decode(&[token_id], false)
            .map_err(|e| anyhow!("Decode error: {}", e))?;

        if tx
            .send(Ok(StreamedToken {
                text,
                id: token_id,
                token_type: TokenType::Prompt,
            }))
            .await
            .is_err()
        {
            return Ok(speculation);
        }
    }

    let stop_tokens = target.stop_token_ids();
    let vocab_size = target.vocab_size();

    'outer: while all_tokens.len() < max_len && all_tokens.len() < context_limit {
        // Check cancellation
        if let Some(ref c) = cancellation
            && c.is_cancelled()
        {
            log::debug!("Cancelled in speculation loop");
            break;
        }

        // Every token checked needs a position in the target, and the pass also yields
        // one more token, so a proposal past the limit would be work thrown away.
        let room = max_len.min(context_limit) - all_tokens.len() - 1;
        let proposal = proposer
            .propose(
                &all_tokens,
                spec_params.num_tokens.min(room),
                &config,
                &stop_tokens,
                vocab_size,
            )
            .await?;
        let draft_tokens = &proposal.tokens;

        log::trace!("Proposed {} tokens: {:?}", draft_tokens.len(), draft_tokens);

        // What verifies a draft token is the target's prediction from the state
        // before it, so the last accepted token goes first: row i is the prediction
        // that should have produced d_i, and the final row is the bonus token that
        // comes free when every draft token is accepted.
        let last_accepted = *all_tokens
            .last()
            .ok_or_else(|| anyhow!("speculation with an empty context"))?;
        let mut verify_input = Vec::with_capacity(draft_tokens.len() + 1);
        verify_input.push(last_accepted);
        verify_input.extend_from_slice(draft_tokens);

        // Re-scoring that last token means rewinding one position, or it would be
        // written into the cache twice.
        target_cache.set_seq_length(all_tokens.len() - 1);

        let mut verify_logits = target_backend
            .forward_batch(target.as_ref(), &verify_input, target_cache.as_mut())
            .await?;
        speculation.passes += 1;
        speculation.proposed += draft_tokens.len();

        // Each row gets the penalties plain decoding would apply at that position,
        // from a history that includes the proposed tokens before it.
        let mut context = all_tokens.clone();
        for i in 0..verify_logits.shape()[0] {
            let mut row = verify_logits.slice(s![i, ..]).to_owned();
            apply_penalties(&mut row, &context, &config);
            verify_logits.slice_mut(s![i, ..]).assign(&row);
            if let Some(&token) = draft_tokens.get(i) {
                context.push(token);
            }
        }

        let accepted_tokens = match &config.strategy {
            DecodingStrategy::Sample(params) => accept_sampled(
                draft_tokens,
                &proposal.distributions,
                &verify_logits,
                &|logits| sampling_probs(logits, params),
                &mut rand::thread_rng(),
            ),
            _ => accept_greedy(draft_tokens, &verify_logits)?,
        };

        let kept = accepted_tokens.len() - 1;
        speculation.accepted += kept;
        log::trace!("Accepted {}/{} proposed tokens", kept, draft_tokens.len());

        let previous_len = all_tokens.len();
        for &token in &accepted_tokens {
            // Before emitting, not after. The plain loop stops on a stop token
            // without sending it.
            if stop_tokens.contains(&token) {
                all_tokens.push(token);
                break 'outer;
            }

            all_tokens.push(token);
            stats.record_token();
            speculation.generated += 1;

            let text = tokenizer
                .decode(&[token], false)
                .map_err(|e| anyhow!("Decode error: {}", e))?;

            if tx
                .send(Ok(StreamedToken {
                    text,
                    id: token,
                    token_type: TokenType::Generated,
                }))
                .await
                .is_err()
            {
                break 'outer;
            }
        }

        target_cache.set_seq_length(all_tokens.len());
        proposer.sync(&all_tokens, previous_len, kept).await?;
    }

    stats.print_summary();
    log::info!(
        "Speculation: {:.2} tokens per pass, {}/{} proposed tokens accepted",
        speculation.tokens_per_pass(),
        speculation.accepted,
        speculation.proposed
    );
    Ok(speculation)
}

/// Accepts proposed tokens while they are what the target would choose, then adds the
/// target's own next token.
fn accept_greedy(draft_tokens: &[u32], verify_logits: &Array2<f32>) -> Result<Vec<u32>> {
    let mut accepted = Vec::with_capacity(draft_tokens.len() + 1);
    for (i, &draft_token) in draft_tokens.iter().enumerate() {
        let logits = verify_logits.slice(s![i, ..]).to_owned();
        let target_token = sample_token(logits, &DecodingStrategy::Greedy)?;
        accepted.push(target_token);
        if target_token != draft_token {
            return Ok(accepted);
        }
    }
    let last = verify_logits.slice(s![draft_tokens.len(), ..]).to_owned();
    accepted.push(sample_token(last, &DecodingStrategy::Greedy)?);
    Ok(accepted)
}

/// Speculative sampling. A token drawn from q is kept with probability min(1, p/q), and a
/// rejection draws from the normalized max(0, p - q), which together give exactly p.
fn accept_sampled<R: Rng>(
    draft_tokens: &[u32],
    draft_distributions: &[Array1<f32>],
    verify_logits: &Array2<f32>,
    distribution: &dyn Fn(Array1<f32>) -> Array1<f32>,
    rng: &mut R,
) -> Vec<u32> {
    let mut accepted = Vec::with_capacity(draft_tokens.len() + 1);
    for (i, &draft_token) in draft_tokens.iter().enumerate() {
        let target = distribution(verify_logits.slice(s![i, ..]).to_owned());
        let p_target = target[draft_token as usize];
        let p_draft = draft_distributions[i][draft_token as usize];

        if rng.r#gen::<f32>() * p_draft < p_target {
            accepted.push(draft_token);
        } else {
            let residual = compute_residual(&target, &draft_distributions[i]);
            accepted.push(sample_from_distribution(&residual, rng));
            return accepted;
        }
    }
    let last = distribution(verify_logits.slice(s![draft_tokens.len(), ..]).to_owned());
    accepted.push(sample_from_distribution(&last, rng));
    accepted
}

/// Computes residual distribution: normalize(max(0, p_target - p_draft))
fn compute_residual(target: &Array1<f32>, draft: &Array1<f32>) -> Array1<f32> {
    let residual: Array1<f32> = target
        .iter()
        .zip(draft.iter())
        .map(|(&t, &d)| (t - d).max(0.0))
        .collect();

    let sum: f32 = residual.sum();
    if sum > 1e-10 {
        residual / sum
    } else {
        target.clone()
    }
}

#[inline]
fn sample_from_distribution<R: Rng>(probs: &Array1<f32>, rng: &mut R) -> u32 {
    let r: f32 = rng.r#gen();

    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }

    (probs.len() - 1) as u32
}

/// Applies the same logit penalties the plain decoding loop applies.
fn apply_penalties(logits: &mut Array1<f32>, history: &[u32], config: &GenerationConfig) {
    if config.repetition_penalty != 1.0 {
        apply_repetition_penalty_mut(logits, history, config.repetition_penalty);
    }
    if config.no_repeat_ngram_size > 0 {
        apply_no_repeat_ngram(logits, history, config.no_repeat_ngram_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_residual() {
        let target = Array1::from_vec(vec![0.5, 0.3, 0.2]);
        let draft = Array1::from_vec(vec![0.3, 0.5, 0.2]);

        let residual = compute_residual(&target, &draft);

        assert!((residual.sum() - 1.0).abs() < 1e-5);
        assert!(residual[0] > 0.9); // Should be ~1.0 after normalization
    }

    #[test]
    fn test_greedy_accept_all() {
        let draft = vec![1, 2, 3];
        let mut logits = Array2::zeros((4, 10));
        logits[[0, 1]] = 10.0;
        logits[[1, 2]] = 10.0;
        logits[[2, 3]] = 10.0;
        logits[[3, 7]] = 10.0;

        assert_eq!(accept_greedy(&draft, &logits).unwrap(), vec![1, 2, 3, 7]);
    }

    #[test]
    fn test_greedy_reject_middle() {
        let draft = vec![1, 2, 3];
        let mut logits = Array2::zeros((4, 10));
        logits[[0, 1]] = 10.0; // Matches draft[0]
        logits[[1, 5]] = 10.0; // Doesn't match draft[1]=2
        logits[[2, 3]] = 10.0;

        // Accept first, reject second with correction
        assert_eq!(accept_greedy(&draft, &logits).unwrap(), vec![1, 5]);
    }

    #[test]
    fn nothing_proposed_still_gives_the_next_token() {
        let mut logits = Array2::zeros((1, 10));
        logits[[0, 4]] = 10.0;
        assert_eq!(accept_greedy(&[], &logits).unwrap(), vec![4]);
    }

    /// Frequencies of the first token out of speculative sampling, for a draft drawn from
    /// `draft` and a target whose distribution is `target`.
    fn first_token_frequencies(target: [f32; 3], draft: [f32; 3], rounds: usize) -> [f32; 3] {
        let mut rng = StdRng::seed_from_u64(7);
        let draft = Array1::from_vec(draft.to_vec());
        let row = Array1::from_vec(target.to_vec());
        let mut logits = Array2::zeros((2, 3));
        logits.slice_mut(s![0, ..]).assign(&row);
        logits.slice_mut(s![1, ..]).assign(&row);

        let mut counts = [0usize; 3];
        for _ in 0..rounds {
            let token = sample_from_distribution(&draft, &mut rng);
            let accepted = accept_sampled(
                &[token],
                std::slice::from_ref(&draft),
                &logits,
                &|row| row,
                &mut rng,
            );
            counts[accepted[0] as usize] += 1;
        }
        counts.map(|n| n as f32 / rounds as f32)
    }

    fn assert_close(got: [f32; 3], want: [f32; 3]) {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 0.01, "got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn sampling_follows_the_target_whatever_the_draft() {
        let target = [0.5, 0.3, 0.2];
        for draft in [[0.3, 0.5, 0.2], [0.8, 0.1, 0.1], [0.5, 0.3, 0.2]] {
            assert_close(first_token_frequencies(target, draft, 100_000), target);
        }
    }

    /// A prompt lookup proposal is certain, which is a draft distribution of one token.
    #[test]
    fn sampling_follows_the_target_with_a_certain_proposal() {
        let target = [0.5, 0.3, 0.2];
        assert_close(
            first_token_frequencies(target, [0.0, 1.0, 0.0], 100_000),
            target,
        );
    }

    #[test]
    fn prompt_lookup_copies_what_followed_the_match() {
        assert_eq!(prompt_lookup(&[1, 2, 3, 4, 5, 1, 2], 3, 2), vec![3, 4]);
    }

    #[test]
    fn prompt_lookup_prefers_the_longest_then_the_latest_match() {
        assert_eq!(
            prompt_lookup(&[7, 8, 1, 9, 2, 8, 1, 3, 2, 8, 1], 2, 1),
            vec![3]
        );
        assert_eq!(prompt_lookup(&[5, 1, 6, 2, 7, 1], 2, 1), vec![6]);
    }

    #[test]
    fn prompt_lookup_can_run_into_the_tail_itself() {
        assert_eq!(prompt_lookup(&[4, 4, 4], 1, 5), vec![4]);
    }

    #[test]
    fn prompt_lookup_proposes_nothing_without_a_match() {
        assert!(prompt_lookup(&[1, 2, 3], 3, 4).is_empty());
        assert!(prompt_lookup(&[1], 3, 4).is_empty());
        assert!(prompt_lookup(&[], 3, 4).is_empty());
    }
}
