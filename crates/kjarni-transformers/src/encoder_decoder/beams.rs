use anyhow::{Result, anyhow};
use async_stream::try_stream;
use futures::stream::Stream;
use ndarray::{Array2, s};

use crate::cache::Cache;
use crate::common::{
    DecodingStrategy, GenerationConfig, StreamedToken, TokenType,
    apply_no_repeat_ngram_inplace, apply_repetition_penalty_inplace,
    get_top_k_from_log_probs, log_softmax_1d,
};
use crate::encoder_decoder::traits::{
    EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel,
};

#[derive(Clone, Debug)]
pub struct BeamHypothesis {
    pub tokens: Vec<u32>,
    pub score: f32,
}

impl BeamHypothesis {
    pub fn normalized_score(&self, length_penalty: f32, prompt_len: f32) -> f32 {
        let len = (self.tokens.len() as f32) - prompt_len;
        let lp = if len > 0.0 {
            len.powf(length_penalty)
        } else {
            1.0
        };
        self.score / lp
    }
}

struct FinishedHypotheses {
    hypotheses: Vec<BeamHypothesis>,
    length_penalty: f32,
    prompt_len: f32,
    num_beams: usize,
    worst_score: f32,
}

impl FinishedHypotheses {
    fn new(num_beams: usize, length_penalty: f32, prompt_len: usize) -> Self {
        Self {
            hypotheses: Vec::with_capacity(num_beams),
            length_penalty,
            prompt_len: prompt_len as f32,
            num_beams,
            worst_score: f32::NEG_INFINITY,
        }
    }

    fn add(&mut self, hypothesis: BeamHypothesis) {
        if hypothesis.score == f32::NEG_INFINITY {
            return;
        }

        let score = hypothesis.normalized_score(self.length_penalty, self.prompt_len);

        if self.len() < self.num_beams || score > self.worst_score {
            self.hypotheses.push(hypothesis);

            self.hypotheses.sort_by(|a, b| {
                b.normalized_score(self.length_penalty, self.prompt_len)
                    .partial_cmp(&a.normalized_score(self.length_penalty, self.prompt_len))
                    .unwrap()
            });

            if self.hypotheses.len() > self.num_beams {
                self.hypotheses.truncate(self.num_beams);
            }

            self.worst_score = self
                .hypotheses
                .last()
                .map(|h| h.normalized_score(self.length_penalty, self.prompt_len))
                .unwrap_or(f32::NEG_INFINITY);
        }
    }

    fn len(&self) -> usize {
        self.hypotheses.len()
    }

    fn is_done(&self, early_stopping: bool, best_sum_logprobs: f32, cur_len: usize) -> bool {
        if self.len() < self.num_beams {
            return false;
        }

        if early_stopping {
            return true;
        }

        let best_possible_len = (cur_len as f32) - self.prompt_len;
        let lp = if best_possible_len > 0.0 {
            best_possible_len.powf(self.length_penalty)
        } else {
            1.0
        };

        let highest_attainable_score = best_sum_logprobs / lp;
        self.worst_score >= highest_attainable_score
    }

    fn best(&self) -> Option<&BeamHypothesis> {
        self.hypotheses.first()
    }
}

struct BeamContext<'a, B: EncoderDecoderGenerationBackend> {
    model: &'a dyn EncoderDecoderLanguageModel,
    backend: &'a B,
    config: &'a GenerationConfig,

    cache: Box<dyn Cache>,
    current_tokens_tensor: B::Tensor,
    encoder_state: B::Tensor,
    beams: Vec<BeamHypothesis>,
    finished: FinishedHypotheses,

    num_beams: usize,
    eos_token_id: u32,
    decoder_start_token_id: u32,
    forced_bos_token_id: Option<u32>,
    forced_eos_token_id: Option<u32>,
    early_stopping: bool,
}

impl<'a, B: EncoderDecoderGenerationBackend> BeamContext<'a, B> {
    async fn new(
        model: &'a dyn EncoderDecoderLanguageModel,
        backend: &'a B,
        input_text: &'a str,
        config: &'a GenerationConfig,
    ) -> Result<Self> {
        let (num_beams, length_penalty, early_stopping) = match &config.strategy {
            DecodingStrategy::BeamSearch(params) => (
                params.num_beams,
                params.length_penalty,
                params.early_stopping,
            ),
            DecodingStrategy::Greedy | DecodingStrategy::Sample(_) => (1, 1.0, true),
        };

        let encoding = model
            .tokenizer()
            .encode(input_text, true)
            .map_err(|e| anyhow!(e))?;
        let encoder_tokens = encoding.get_ids();

        let encoder_state = backend.encode(model, encoder_tokens, num_beams).await?;
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

        let forced_bos_token_id = model.forced_bos_token_id();

        // prompt_len: 1 for decoder_start, +1 if forced_bos
        let prompt_len = if forced_bos_token_id.is_some() { 2 } else { 1 };

        let finished = FinishedHypotheses::new(num_beams, length_penalty, prompt_len);

        Ok(Self {
            model,
            backend,
            config,
            cache,
            current_tokens_tensor,
            encoder_state,
            beams,
            finished,
            num_beams,
            eos_token_id: model.eos_token_id().unwrap_or(2),
            decoder_start_token_id,
            forced_bos_token_id,
            early_stopping,
            forced_eos_token_id: model.forced_eos_token_id(),
        })
    }
}

fn find_best_beams_and_get_candidates(
    logits_2d: Array2<f32>,
    current_beams: &[BeamHypothesis],
    num_beams: usize,
) -> Vec<(BeamHypothesis, u32, usize, usize)> {
    let mut candidates: Vec<(f32, usize, u32, usize)> = Vec::with_capacity(num_beams * num_beams);

    for (beam_idx, (beam, logits_for_beam)) in
        current_beams.iter().zip(logits_2d.rows()).enumerate()
    {
        if beam.score == f32::NEG_INFINITY {
            continue;
        }

        let log_probs = log_softmax_1d(&logits_for_beam.to_owned());
        let top_k_len = num_beams * 2;
        let top_k = get_top_k_from_log_probs(&log_probs, top_k_len);

        for (rank, (token_id, token_log_prob)) in top_k.into_iter().enumerate() {
            candidates.push((beam.score + token_log_prob, beam_idx, token_id, rank));
        }
    }

    candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    candidates.truncate(num_beams * 2);

    candidates
        .into_iter()
        .map(|(new_score, source_beam_idx, new_token_id, rank)| {
            let mut new_tokens = current_beams[source_beam_idx].tokens.clone();
            new_tokens.push(new_token_id);
            (
                BeamHypothesis {
                    tokens: new_tokens,
                    score: new_score,
                },
                new_token_id,
                source_beam_idx,
                rank,
            )
        })
        .collect()
}

async fn beam_step<B: EncoderDecoderGenerationBackend>(
    ctx: &mut BeamContext<'_, B>,
    step: usize,
) -> Result<bool> {
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

    // step 0 generates token 2, so current_len = step + 2
    let current_len = step + 2;

    logits_2d
        .outer_iter_mut()
        .enumerate()
        .for_each(|(i, mut logits_row)| {
            if ctx.beams[i].score == f32::NEG_INFINITY {
                return;
            }

            if step == 0 {
                if let Some(forced_bos_id) = ctx.forced_bos_token_id {
                    logits_row.fill(f32::NEG_INFINITY);
                    logits_row[forced_bos_id as usize] = 0.0;
                    return;
                }
            }

            if current_len < ctx.config.min_length {
                logits_row[ctx.eos_token_id as usize] = f32::NEG_INFINITY;
            }

            if ctx.config.repetition_penalty != 1.0 {
                apply_repetition_penalty_inplace(
                    &mut logits_row,
                    &ctx.beams[i].tokens,
                    ctx.config.repetition_penalty,
                );
            }
            if ctx.config.no_repeat_ngram_size > 0 {
                apply_no_repeat_ngram_inplace(
                    &mut logits_row,
                    &ctx.beams[i].tokens,
                    ctx.config.no_repeat_ngram_size,
                );
            }

            if current_len >= ctx.config.max_length - 1 {
                if let Some(forced_eos_id) = ctx.forced_eos_token_id {
                    logits_row.fill(f32::NEG_INFINITY);
                    logits_row[forced_eos_id as usize] = 0.0;
                }
            }
        });

    let all_candidates = find_best_beams_and_get_candidates(logits_2d, &ctx.beams, ctx.num_beams);

    let mut active_beams = Vec::with_capacity(ctx.num_beams);
    let mut active_tokens = Vec::with_capacity(ctx.num_beams);
    let mut active_reorder = Vec::with_capacity(ctx.num_beams);

    for (global_rank, (beam, next_token, source_idx, _beam_token_rank)) in
        all_candidates.into_iter().enumerate()
    {
        if next_token == ctx.eos_token_id {
            if global_rank >= ctx.num_beams {
                continue;
            }
            ctx.finished.add(beam);
        } else {
            active_beams.push(beam);
            active_tokens.push(next_token);
            active_reorder.push(source_idx);

            if active_beams.len() == ctx.num_beams {
                break;
            }
        }
    }

    let best_unfinished_score = active_beams
        .first()
        .map(|b| b.score)
        .unwrap_or(f32::NEG_INFINITY);

    if ctx
        .finished
        .is_done(ctx.early_stopping, best_unfinished_score, current_len)
    {
        return Ok(true);
    }

    if active_beams.is_empty() {
        return Ok(true);
    }

    // pad if beams died
    while active_beams.len() < ctx.num_beams {
        active_beams.push(BeamHypothesis {
            tokens: active_beams.last().unwrap().tokens.clone(),
            score: f32::NEG_INFINITY,
        });
        active_tokens.push(active_tokens.last().copied().unwrap_or(ctx.eos_token_id));
        active_reorder.push(active_reorder.last().copied().unwrap_or(0));
    }

    ctx.cache.increment_len(1);
    ctx.beams = active_beams;
    ctx.backend
        .update_token_tensor(&mut ctx.current_tokens_tensor, &active_tokens)?;

    if ctx.num_beams > 1 {
        ctx.backend
            .reorder_cache(ctx.cache.as_mut(), &active_reorder)?;
    }

    Ok(false)
}

pub async fn run_beam_search<B: EncoderDecoderGenerationBackend>(
    model: &dyn EncoderDecoderLanguageModel,
    backend: &B,
    input_text: &str,
    config: &GenerationConfig,
) -> Result<String> {
    let mut ctx = BeamContext::new(model, backend, input_text, config).await?;
    let mut stopped_early = false;

    for step in 0..config.max_length {
        let should_stop = beam_step(&mut ctx, step).await?;
        if should_stop {
            stopped_early = true;
            break;
        }
        if ctx.beams.iter().all(|b| b.score == f32::NEG_INFINITY) {
            break;
        }
    }

    // only add active beams if we didn't early-stop
    if !stopped_early {
        for beam in ctx.beams {
            if beam.score != f32::NEG_INFINITY {
                ctx.finished.add(beam);
            }
        }
    }

    let best_beam = ctx
        .finished
        .best()
        .ok_or_else(|| anyhow!("no hypotheses found"))?;

    let tokens = &best_beam.tokens;

    let mut start = 0;
    if let Some(first) = tokens.first() {
        if *first == ctx.decoder_start_token_id {
            start += 1;
        }
    }

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

pub fn run_beam_search_stream<'a, B: EncoderDecoderGenerationBackend + 'a>(
    model: &'a dyn EncoderDecoderLanguageModel,
    backend: &'a B,
    input_text: &'a str,
    config: &'a GenerationConfig,
) -> impl Stream<Item = Result<StreamedToken>> + 'a {
    try_stream! {
        let mut ctx = BeamContext::new(model, backend, input_text, config).await?;

        if ctx.num_beams > 1 {
            log::warn!("streaming beam search is unstable, tokens may change. use greedy for stability.");
        }

        for step in 0..config.max_length {
            let should_stop = beam_step(&mut ctx, step).await?;

            if let Some(best_beam) = ctx.beams.iter().find(|b| b.score != f32::NEG_INFINITY) {
                let new_token = *best_beam.tokens.last().unwrap();

                if new_token != ctx.eos_token_id && new_token != ctx.decoder_start_token_id {
                    let text = ctx.model.tokenizer().decode(&[new_token], true).unwrap_or_default();
                    yield StreamedToken {
                        text,
                        id: new_token,
                        token_type: TokenType::Generated,
                    };
                }
            }

            if should_stop {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{any::Any, collections::HashSet, sync::Arc};

    use async_trait::async_trait;
    use ndarray::{Array2, Array3};
    use tokenizers::Tokenizer;

    use super::*;
    use crate::{
        Device, LanguageModel, WgpuContext,
        cpu::encoder::{CpuEncoderOps, GpuEncoderOps, traits::EncoderLanguageModel},
        encoder_decoder::traits::{
            CpuEncoderDecoderOps, EncoderDecoderGenerationBackend, GpuEncoderDecoderOps,
        },
        traits::InferenceModel,
    };

    #[test]
    fn test_length_penalty_formula() {
        let beam = BeamHypothesis {
            tokens: vec![0; 6],
            score: -5.0,
        };
        let prompt_len = 1.0;

        let score_linear = beam.normalized_score(1.0, prompt_len);
        assert!((score_linear - (-1.0)).abs() < 1e-5);

        let score_square = beam.normalized_score(2.0, prompt_len);
        assert!((score_square - (-0.2)).abs() < 1e-5);
    }

    #[test]
    fn test_finished_hypotheses_queue_logic() {
        let mut finished = FinishedHypotheses::new(2, 1.0, 0);

        finished.add(BeamHypothesis {
            tokens: vec![0; 5],
            score: -10.0,
        });
        assert_eq!(finished.len(), 1);
        assert_eq!(finished.worst_score, -2.0);

        finished.add(BeamHypothesis {
            tokens: vec![0; 5],
            score: -5.0,
        });
        assert_eq!(finished.len(), 2);
        assert_eq!(finished.best().unwrap().score, -5.0);
        assert_eq!(finished.worst_score, -2.0);

        finished.add(BeamHypothesis {
            tokens: vec![0; 5],
            score: f32::NEG_INFINITY,
        });
        assert_eq!(finished.len(), 2);

        finished.add(BeamHypothesis {
            tokens: vec![0; 2],
            score: -2.0,
        });
        assert_eq!(finished.len(), 2);
        assert!(finished.worst_score > -2.0);
    }

    #[test]
    fn test_is_done_heuristic() {
        let mut finished = FinishedHypotheses::new(2, 1.0, 0);

        finished.add(BeamHypothesis {
            tokens: vec![0; 5],
            score: -10.0,
        });
        assert!(!finished.is_done(false, -5.0, 10));

        finished.add(BeamHypothesis {
            tokens: vec![0; 5],
            score: -10.0,
        });
        assert!(finished.is_done(true, -5.0, 20));

        assert!(!finished.is_done(false, -5.0, 10));
        assert!(finished.is_done(false, -50.0, 10));
    }

    #[test]
    fn test_find_best_beams_candidates_sorting() {
        let logits = Array2::from_shape_vec(
            (2, 3),
            vec![-1.0, -2.0, -3.0, -0.5, -4.0, -5.0],
        )
        .unwrap();

        let beams = vec![
            BeamHypothesis {
                tokens: vec![0],
                score: 0.0,
            },
            BeamHypothesis {
                tokens: vec![0],
                score: 0.0,
            },
        ];

        let candidates = find_best_beams_and_get_candidates(logits, &beams, 2);

        let best = &candidates[0];
        assert_eq!(best.1, 0);
        assert_eq!(best.2, 1);
        assert!((best.0.score - (-0.0405)).abs() < 0.001);

        let second = &candidates[1];
        assert_eq!(second.1, 0);
        assert_eq!(second.2, 0);
        assert!((second.0.score - (-0.4076)).abs() < 0.001);
    }

    #[test]
    fn test_find_best_beams_ignores_dead_beams() {
        let logits = Array2::zeros((2, 5));
        let beams = vec![
            BeamHypothesis {
                tokens: vec![],
                score: 0.0,
            },
            BeamHypothesis {
                tokens: vec![],
                score: f32::NEG_INFINITY,
            },
        ];

        let candidates = find_best_beams_and_get_candidates(logits, &beams, 2);

        for (_, _, source_beam, _) in candidates {
            assert_eq!(source_beam, 0);
        }
    }

    #[test]
    fn test_find_best_beams_small_vocab_handling() {
        let logits = Array2::zeros((2, 2));
        let beams = vec![
            BeamHypothesis {
                tokens: vec![],
                score: 0.0,
            },
            BeamHypothesis {
                tokens: vec![],
                score: 0.0,
            },
        ];

        let candidates = find_best_beams_and_get_candidates(logits, &beams, 4);
        assert_eq!(candidates.len(), 4);
    }

    #[test]
    fn test_finished_hypotheses_ordering() {
        let mut finished = FinishedHypotheses::new(2, 1.0, 0);

        finished.add(BeamHypothesis {
            tokens: vec![0],
            score: -10.0,
        });
        finished.add(BeamHypothesis {
            tokens: vec![0],
            score: -5.0,
        });
        finished.add(BeamHypothesis {
            tokens: vec![0],
            score: -1.0,
        });

        assert_eq!(finished.len(), 2);
        assert_eq!(finished.best().unwrap().score, -1.0);
        assert_eq!(finished.worst_score, -5.0);
    }

    #[test]
    fn test_finished_hypotheses_length_penalty_sorting() {
        let mut finished = FinishedHypotheses::new(2, 2.0, 0);

        finished.add(BeamHypothesis {
            tokens: vec![0; 10],
            score: -10.0,
        });
        finished.add(BeamHypothesis {
            tokens: vec![0; 2],
            score: -2.0,
        });

        let best = finished.best().unwrap();
        assert_eq!(best.tokens.len(), 10);
    }

    struct MockModel {
        vocab: usize,
        eos: u32,
        decoder_start: u32,
    }

    impl InferenceModel for MockModel {
        fn device(&self) -> Device {
            Device::Cpu
        }
        fn context(&self) -> Option<Arc<WgpuContext>> {
            None
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl LanguageModel for MockModel {
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn hidden_size(&self) -> usize {
            10
        }
        fn num_layers(&self) -> usize {
            1
        }
        fn num_heads(&self) -> usize {
            1
        }
        fn context_size(&self) -> usize {
            128
        }
        fn tokenizer(&self) -> &Tokenizer {
            unimplemented!()
        }
        fn eos_token_id(&self) -> Option<u32> {
            Some(self.eos)
        }
        fn bos_token_id(&self) -> Option<u32> {
            Some(0)
        }
        fn forced_bos_token_id(&self) -> Option<u32> {
            None
        }
        fn forced_eos_token_id(&self) -> Option<u32> {
            None
        }
        fn pad_token_id(&self) -> Option<u32> {
            None
        }
        fn stop_token_ids(&self) -> HashSet<u32> {
            HashSet::from([self.eos])
        }
        fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn crate::cache::Cache>> {
            Ok(Box::new(MockCache { len: 0 }))
        }
    }

    #[async_trait]
    impl EncoderLanguageModel for MockModel {
        fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
            None
        }
        fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
            None
        }
    }

    #[async_trait]
    impl EncoderDecoderLanguageModel for MockModel {
        fn encoder_decoder_cpu_ops(&self) -> Option<&dyn CpuEncoderDecoderOps> {
            None
        }
        fn encoder_decoder_gpu_ops(&self) -> Option<&dyn GpuEncoderDecoderOps> {
            None
        }
        fn decoder_start_token_id(&self) -> u32 {
            self.decoder_start
        }
        fn get_default_generation_config(&self) -> GenerationConfig {
            GenerationConfig::default()
        }
    }

    #[derive(Clone)]
    struct MockCache {
        len: usize,
    }

    impl Cache for MockCache {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn get_seq_length(&self) -> usize {
            self.len
        }
        fn set_seq_length(&mut self, len: usize) {
            self.len = len;
        }
        fn clear(&mut self) {
            self.len = 0;
        }
        fn increment_len(&mut self, new_tokens: usize) {
            self.len += new_tokens;
        }
        fn clone_box(&self) -> Box<dyn Cache> {
            Box::new(self.clone())
        }
    }

    struct MockStepBackend;

    #[async_trait]
    impl EncoderDecoderGenerationBackend for MockStepBackend {
        type Tensor = Array2<u32>;

        async fn encode(
            &self,
            _: &dyn EncoderDecoderLanguageModel,
            _: &[u32],
            _: usize,
        ) -> Result<Self::Tensor> {
            Ok(Array2::zeros((1, 1)))
        }

        fn create_token_tensor(&self, _: &[u32], _: usize) -> Result<Self::Tensor> {
            Ok(Array2::zeros((1, 1)))
        }

        fn update_token_tensor(&self, _: &mut Self::Tensor, _: &[u32]) -> Result<()> {
            Ok(())
        }

        fn reorder_cache(&self, _: &mut dyn Cache, _: &[usize]) -> Result<()> {
            Ok(())
        }

        async fn decode_step(
            &self,
            _: &dyn EncoderDecoderLanguageModel,
            _: &Self::Tensor,
            _: &Self::Tensor,
            _: &mut dyn Cache,
        ) -> Result<Array3<f32>> {
            let mut logits = Array3::<f32>::zeros((2, 1, 10));
            logits[[0, 0, 5]] = 100.0;
            logits[[1, 0, 1]] = 100.0;
            Ok(logits)
        }
    }

    #[tokio::test]
    async fn test_beam_step_logic() {
        let num_beams = 2;
        let eos_id = 1;
        let model = MockModel {
            vocab: 10,
            eos: eos_id,
            decoder_start: 0,
        };
        let backend = MockStepBackend;
        let config = GenerationConfig::default();

        let cache = Box::new(MockCache { len: 0 });
        let finished = FinishedHypotheses::new(num_beams, 1.0, 0);

        let beams = vec![
            BeamHypothesis {
                tokens: vec![0],
                score: 0.0,
            },
            BeamHypothesis {
                tokens: vec![0],
                score: 0.0,
            },
        ];

        let mut ctx = BeamContext {
            model: &model,
            backend: &backend,
            config: &config,
            cache,
            current_tokens_tensor: Array2::zeros((1, 1)),
            encoder_state: Array2::zeros((1, 1)),
            beams,
            finished,
            num_beams,
            eos_token_id: eos_id,
            decoder_start_token_id: 0,
            forced_bos_token_id: None,
            forced_eos_token_id: None,
            early_stopping: false,
        };

        let done = beam_step(&mut ctx, 0).await.unwrap();

        assert_eq!(ctx.finished.len(), 1);
        assert_eq!(ctx.finished.best().unwrap().tokens.last(), Some(&eos_id));
        assert_eq!(ctx.beams.len(), 2);
        assert_eq!(ctx.beams[0].tokens.last(), Some(&5));
        assert!(!done);
    }
}