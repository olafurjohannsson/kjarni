use anyhow::{anyhow, Result};
use async_trait::async_trait;
use bytemuck;
use edgetransformers::cache::{Cache, GpuBeamKVCache};
use edgetransformers::gpu_context::WgpuContext;
use edgetransformers::gpu_ops::{GpuTensor, GpuTensorPool};
use edgetransformers::models::base::{
    BeamHypothesis, DecodingStrategy, EncoderDecoderLanguageModel, GenerationConfig, LanguageModel,
};
use edgetransformers::prelude::*;
use edgetransformers::traits::EncoderOutput;
use ndarray::{s, Array1, Array2, Array3};
use std::sync::Arc;
use tokio::sync::Mutex;

//=========================================================================================
// 1. PUBLIC API
//=========================================================================================

pub struct Seq2SeqGenerator {
    pub model: Box<dyn EncoderDecoderLanguageModel>,
}

impl Seq2SeqGenerator {
    pub fn new(model: Box<dyn EncoderDecoderLanguageModel>) -> Self {
        Self { model }
    }

    pub async fn generate(&self, input_text: &str, config: &GenerationConfig) -> Result<String> {
        let encoder_output = self.encode_input(input_text).await?;
        self.generate_from_encoding(&encoder_output, config).await
    }

    async fn encode_input(&self, text: &str) -> Result<EncoderOutput> {
        let encoding = self
            .model
            .tokenizer()
            .encode(text, true)
            .map_err(|e| anyhow!(e))?;
        let input_ids = Array2::from_shape_vec((1, encoding.len()), encoding.get_ids().to_vec())?;
        let attention_mask = Array2::ones(input_ids.dim());
        self.model
            .encoder()
            .forward(&input_ids, &attention_mask, None)
            .await
    }

    pub async fn generate_from_encoding(
        &self,
        encoder_output: &EncoderOutput,
        config: &GenerationConfig,
    ) -> Result<String> {
        match self.model.device() {
            Device::Cpu => {
                todo!("CPU backend for beam search not yet implemented");
            }
            Device::Wgpu => {
                let context = self.model.context().unwrap();
                let backend = GpuBackend {
                    context: context.clone(),
                    pool: Arc::new(Mutex::new(GpuTensorPool::new(context))),
                };

                // 1. Determine Beam Count
                let num_beams = match &config.strategy {
                    DecodingStrategy::BeamSearch(params) => params.num_beams,
                    DecodingStrategy::Greedy => 1,
                    _ => return Err(anyhow!("Only BeamSearch and Greedy are supported.")),
                };

                let original_encoder_state_cpu = &encoder_output.last_hidden_state;

                // 2. Expand encoder state to match num_beams
                // (Even for greedy/1 beam, this ensures dimensions are consistent)
                let expanded_encoder_state_cpu = original_encoder_state_cpu
                    .broadcast((
                        num_beams,
                        original_encoder_state_cpu.shape()[1],
                        original_encoder_state_cpu.shape()[2],
                    ))
                    .unwrap()
                    .to_owned();

                let expanded_encoder_output = EncoderOutput {
                    last_hidden_state: expanded_encoder_state_cpu,
                };

                // 3. Run the generation loop
                run_beam_search(
                    self.model.as_ref(),
                    backend,
                    &expanded_encoder_output,
                    config,
                )
                    .await
            }
        }
    }
}

//=========================================================================================
// 2. CORE GENERATION LOOP (Backend Agnostic)
//=========================================================================================

async fn run_beam_search<B: GenerationBackend>(
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
        // ========================================================================
        // --- DEBUG: PRINT STATE AT THE START OF THE LOOP ---
        println!("\n==================== STEP {} ====================", step);
        println!("[STATE] Cache seq_length: {}", cache.get_seq_length());
        println!("[STATE] Input Tokens Tensor Shape: {:?}", current_tokens_tensor.shape()); // Assuming GpuTensor has a shape method
        for (i, beam) in beams.iter().enumerate() {
            println!("[STATE] Beam {}: Score: {:.4}, Tokens: {:?}", i, beam.score, beam.tokens);
        }
        // ========================================================================
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

        // 2. Project to Logits & Apply Penalties
        let vocab_size = model.config().vocab_size();
        let mut logits_2d = Array2::<f32>::zeros((num_beams, vocab_size));

        for i in 0..num_beams {
            let hidden_row = last_hidden_state.row(i);
            let mut logits_row = model.lm_head().dot(&hidden_row);

            if let Some(bias) = model.final_logits_bias() {
                logits_row += bias;
            }

            // --- FIX: Force BOS Token at Step 0 ---
            // If the config specifies a forced BOS token (like BART's 0), we must enforce it here.
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
            if i == 0 {
                let log_probs = log_softmax_1d(&logits_row);
                let top_k = get_top_k_from_log_probs(&log_probs, 5);
                println!("\n[DEBUG] Step {} Beam 0 Top Candidates:", step);
                for (id, score) in top_k {
                    let token = model
                        .tokenizer()
                        .decode(&[id], true)
                        .unwrap_or_else(|_| "???".to_string());
                    println!("  Token: {:<15} ID: {:<6} Score: {:.4}", token, id, score);
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

        // 4. Update State

        println!("[UPDATE] Next tokens chosen: {:?}", next_tokens_vec);
        println!("[UPDATE] Parent beam indices for reorder: {:?}", reorder_indices_vec);
        // We must increment the cache length NOW.
        // The forward pass wrote data to the current position.
        // We need to tell the cache "you now have 1 more valid token" BEFORE we try to reorder it.
        cache.increment_len(1);
        // --- CRITICAL FIX END ---

        beams = updated_beams;
        backend.update_token_tensor(&mut current_tokens_tensor, &next_tokens_vec)?;

        // Now it is safe to reorder, because seq_len > 0
        if num_beams > 1 {
            backend.reorder_cache(cache.as_mut(), &reorder_indices_vec)?;
        }

        // 5. Check Completion
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
    println!("DEBUG: Raw Best Beam Tokens: {:?}", best_hypo.tokens);
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

//=========================================================================================
// 3. BACKEND IMPLEMENTATION (WGPU)
//=========================================================================================

pub struct StepInput<'a, T> {
    pub tokens: &'a T,
    pub encoder_state: Option<&'a T>,
    pub attention_mask: &'a T,
}
pub trait HasShape {
    fn shape(&self) -> &[usize];
}
impl HasShape for GpuTensor {
    fn shape(&self) -> &[usize] {
        self.shape()
    }
}
impl<S: ndarray::Data, D: ndarray::Dimension> HasShape for ndarray::ArrayBase<S, D> {
    fn shape(&self) -> &[usize] {
        self.shape()
    }
}
#[async_trait(?Send)]
pub trait GenerationBackend: Send + Sync {
    type Cache: Cache;
    type Tensor: Send + Sync + HasShape;

    async fn forward<'a>(
        &'a self,
        model: &'a dyn EncoderDecoderLanguageModel,
        inputs: StepInput<'a, Self::Tensor>,
        cache: &'a mut dyn Cache,
    ) -> Result<Array3<f32>>;

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor>;
    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()>;
    fn prepare_encoder_state(&self, encoder_output: &EncoderOutput) -> Result<Self::Tensor>;
    fn prepare_attention_mask(&self, seq_len: usize, num_beams: usize) -> Result<Self::Tensor>;
    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()>;
}

struct GpuBackend {
    context: Arc<WgpuContext>,
    pool: Arc<Mutex<GpuTensorPool>>,
}

#[async_trait(?Send)]
impl GenerationBackend for GpuBackend {
    type Cache = GpuBeamKVCache;
    type Tensor = GpuTensor;

    async fn forward<'a>(
        &'a self,
        model: &'a dyn EncoderDecoderLanguageModel,
        inputs: StepInput<'a, Self::Tensor>,
        cache: &'a mut dyn Cache,
    ) -> Result<Array3<f32>> {
        // let mut pool_guard: MutexGuard<GpuTensorPool> = self.pool.lock().await;
        // let mut frame: GpuFrameContext = GpuFrameContext::new(&self.context, pool_guard);
        //
        // /// Extraxt Command Encoder and encapsulated pool
        // let (encoder, pool) = frame.resources();

        let gpu_decoder = model.gpu_decoder();
        let decoder_output = gpu_decoder
            .forward(
                inputs.tokens,
                inputs.encoder_state.unwrap(),
                None,
                Some(inputs.attention_mask),
                Some(cache),
            )
            .await?;

        // frame.finish();
        Ok(decoder_output.last_hidden_state)
    }

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor> {
        let tokens_ndarray =
            Array2::from_shape_vec((num_beams, tokens.len() / num_beams), tokens.to_vec())?;
        GpuTensor::from_ndarray(&self.context, &tokens_ndarray)
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()> {
        let new_tokens_bytes: &[u8] = bytemuck::cast_slice(new_tokens);
        self.context
            .queue
            .write_buffer(tensor.buffer(), 0, new_tokens_bytes);
        Ok(())
    }

    fn prepare_encoder_state(&self, encoder_output: &EncoderOutput) -> Result<Self::Tensor> {
        GpuTensor::from_ndarray(&self.context, &encoder_output.last_hidden_state)
    }

    fn prepare_attention_mask(&self, seq_len: usize, num_beams: usize) -> Result<Self::Tensor> {
        let mask_cpu: Array2<f32> = Array2::ones((num_beams, seq_len));
        GpuTensor::from_ndarray(&self.context, &mask_cpu)
    }

    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()> {
        let gpu_cache = cache.as_any_mut().downcast_mut::<GpuBeamKVCache>().unwrap();
        let indices_ndarray = Array1::from_vec(indices.iter().map(|&i| i as u32).collect());
        let indices_gpu = GpuTensor::from_ndarray(&self.context, &indices_ndarray)?;

        let mut encoder = self
            .context
            .device
            .create_command_encoder(&Default::default());
        gpu_cache.reorder(&mut encoder, &indices_gpu);
        self.context.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

//=========================================================================================
// 4. HELPER FUNCTIONS
//=========================================================================================

fn find_best_beams_and_get_indices(
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

pub fn get_top_k_from_log_probs(log_probs: &Array1<f32>, k: usize) -> Vec<(u32, f32)> {
    let mut indexed_log_probs: Vec<(usize, f32)> = log_probs
        .iter()
        .enumerate()
        .map(|(i, &lp)| (i, lp))
        .collect();
    indexed_log_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed_log_probs.truncate(k);
    indexed_log_probs
        .into_iter()
        .map(|(i, lp)| (i as u32, lp))
        .collect()
}

pub fn log_softmax_1d(logits: &Array1<f32>) -> Array1<f32> {
    let max_val = logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let scaled_logits = logits - max_val;
    let exp_sum = scaled_logits.mapv(f32::exp).sum();
    scaled_logits - exp_sum.ln()
}

pub fn apply_repetition_penalty(logits: &mut Array1<f32>, tokens: &[u32], penalty: f32) {
    for &token in tokens {
        let idx = token as usize;
        if idx < logits.len() {
            let score = logits[idx];
            if score < 0.0 {
                logits[idx] = score * penalty;
            } else {
                logits[idx] = score / penalty;
            }
        }
    }
}

pub fn apply_no_repeat_ngram(logits: &mut Array1<f32>, tokens: &[u32], ngram_size: usize) {
    let n = ngram_size;
    // We can't form a prefix of length n-1, so we can't complete an n-gram.
    if tokens.len() < n - 1 {
        return;
    }

    // The sequence of tokens that would form the start of a new n-gram.
    let current_prefix = &tokens[tokens.len() - (n - 1)..];

    // Iterate through all historical n-grams in the generated sequence.
    for window in tokens.windows(n) {
        // Check if a historical n-gram starts with the same prefix.
        if &window[..n - 1] == current_prefix {
            // If it does, ban the token that completed that n-gram.
            let banned_token = window[n - 1] as usize;
            if banned_token < logits.len() {
                logits[banned_token] = f32::NEG_INFINITY;
            }
        }
    }
}
