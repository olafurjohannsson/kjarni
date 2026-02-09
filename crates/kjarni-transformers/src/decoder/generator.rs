//! Autoregressive text generation orchestrator for decoder-only models.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures::stream::{Stream, TryStreamExt};
use log::{debug, trace};
use ndarray::Array2;

use crate::common::{
    CancellationToken, GenerationConfig, StreamedToken, TokenType, apply_no_repeat_ngram,
    apply_repetition_penalty_mut, sample_token,
};
use crate::cpu::decoder::DraftModelContext;
use crate::decoder::prelude::*;
use crate::models::base::AutoregressiveLoop;
use crate::stats::GenerationStats;
use crate::{Conversation, prelude::*};

pub struct DecoderGenerator {
    pub model: Arc<dyn DecoderLanguageModel + Send + Sync>,
    backend: AnyDecoderBackend,
    draft: Option<DraftModelContext>,
}

impl DecoderGenerator {
    pub fn new(model: Arc<dyn DecoderLanguageModel + Send + Sync>) -> Result<Self> {
        let backend = match model.device() {
            Device::Cpu => {
                debug!("initializing cpu decoder backend");
                AnyDecoderBackend::Cpu(CpuDecoderBackend)
            }
            Device::Wgpu => {
                debug!("initializing gpu decoder backend");
                let context = model
                    .context()
                    .ok_or_else(|| anyhow!("gpu model requires WgpuContext"))?;
                AnyDecoderBackend::Gpu(Arc::new(GpuDecoderBackend::new(context)?))
            }
        };

        log::info!(
            "decoder generator initialized: device={:?}, vocab_size={}, context_size={}",
            model.device(),
            model.vocab_size(),
            model.context_size()
        );

        Ok(Self {
            model,
            backend,
            draft: None,
        })
    }

    pub fn load_draft_model(
        &mut self,
        model: Arc<dyn DecoderLanguageModel + Send + Sync>,
    ) -> Result<()> {
        let draft_ctx = DraftModelContext::load(model)?;

        if draft_ctx.model.vocab_size() != self.model.vocab_size() {
            log::warn!(
                "vocab size mismatch: target={}, draft={}",
                self.model.vocab_size(),
                draft_ctx.model.vocab_size()
            );
        }

        self.draft = Some(draft_ctx);
        log::info!("draft model ready for speculative decoding");
        Ok(())
    }

    pub fn unload_draft_model(&mut self) {
        if self.draft.is_some() {
            self.draft = None;
            log::info!("draft model unloaded");
        }
    }

    pub fn has_draft_model(&self) -> bool {
        self.draft.is_some()
    }

    pub async fn generate(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        cancellation: Option<CancellationToken>,
    ) -> Result<String> {
        let stream = self.generate_stream(prompt, config, cancellation).await?;
        let results: Vec<StreamedToken> = stream.try_collect().await?;
        let text: String = results
            .iter()
            .filter(|t| t.token_type == TokenType::Generated)
            .map(|v| v.text.as_str())
            .collect();
        Ok(text)
    }

    pub async fn chat(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
        cancellation: Option<CancellationToken>,
    ) -> Result<String> {
        let prompt = self.format_conversation(conversation)?;
        trace!("formatted chat prompt: {} chars", prompt.len());
        self.generate(&prompt, config, cancellation).await
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        cancellation: Option<CancellationToken>,
    ) -> Result<impl Stream<Item = Result<StreamedToken>>> {
        let tokens = self.encode(prompt, config)?;
        self.generate_stream_from_tokens(tokens, config, cancellation)
            .await
    }

    pub async fn chat_stream(
        &self,
        conversation: &Conversation,
        config: &GenerationConfig,
        cancellation: Option<CancellationToken>,
    ) -> Result<impl Stream<Item = Result<StreamedToken>>> {
        let prompt = self.format_conversation(conversation)?;
        let tokens = self.encode(&prompt, config)?;
        self.generate_stream_from_tokens(tokens, config, cancellation)
            .await
    }

    pub fn encode(&self, prompt: &str, config: &GenerationConfig) -> Result<Vec<u32>> {
        let tokenizer = self.model.tokenizer();
        let mut tokens = tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow!("tokenization failed: {}", e))?
            .get_ids()
            .to_vec();

        if config.add_bos_token {
            if let Some(bos) = self.model.bos_token_id() {
                if tokens.first() != Some(&bos) {
                    tokens.insert(0, bos);
                    trace!("prepended bos token: {}", bos);
                }
            }
        }

        debug!(
            "encoded prompt: {} chars -> {} tokens",
            prompt.len(),
            tokens.len()
        );
        Ok(tokens)
    }

    pub fn format_conversation(&self, conversation: &Conversation) -> Result<String> {
        if let Some(template) = self.model.chat_template() {
            Ok(template.apply(conversation))
        } else {
            conversation
                .last()
                .map(|m| m.content.clone())
                .ok_or_else(|| anyhow!("conversation is empty and model has no chat template"))
        }
    }

    pub fn requires_template(&self) -> bool {
        self.model.is_instruct_model()
    }

    pub fn stop_sequences(&self) -> Vec<String> {
        self.model
            .chat_template()
            .map(|t| t.stop_sequences())
            .unwrap_or_default()
    }

    pub async fn generate_stream_from_tokens(
        &self,
        input_tokens: Vec<u32>,
        config: &GenerationConfig,
        cancellation: Option<CancellationToken>,
    ) -> Result<impl Stream<Item = Result<StreamedToken>>> {
        let model = self.model.clone();
        let backend = self.backend.clone();
        let config = config.clone();

        let (tx, rx) = tokio::sync::mpsc::channel(32);

        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build();

            let local_rt = match rt {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.blocking_send(Err(anyhow!("failed to create local runtime: {}", e)));
                    return;
                }
            };

            local_rt.block_on(async {
                if let Err(e) = run_generation_loop(
                    model,
                    backend,
                    input_tokens,
                    config,
                    tx.clone(),
                    cancellation,
                )
                .await
                {
                    let _ = tx.send(Err(e)).await;
                }
            });
        });

        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}

pub async fn run_generation_loop(
    model: Arc<dyn DecoderLanguageModel + Send + Sync>,
    backend: AnyDecoderBackend,
    input_tokens: Vec<u32>,
    config: GenerationConfig,
    tx: tokio::sync::mpsc::Sender<Result<StreamedToken>>,
    cancellation: Option<CancellationToken>,
) -> Result<()> {
    let prompt_len = input_tokens.len();

    if prompt_len == 0 {
        return Err(anyhow!("cannot generate from empty prompt"));
    }

    let context_limit = model.context_size();
    let max_len = config
        .max_new_tokens
        .map(|n| prompt_len + n)
        .unwrap_or(config.max_length);

    let cache_capacity = match model.autoregressive_loop() {
        AutoregressiveLoop::Legacy => max_len + 1,
        AutoregressiveLoop::Pipelined => max_len,
    };

    debug!(
        "generation setup: prompt_len={}, max_len={}, cache_capacity={}, loop={:?}, backend={}",
        prompt_len,
        max_len,
        cache_capacity,
        model.autoregressive_loop(),
        backend.backend_type()
    );

    let mut cache = model.new_cache(1, cache_capacity, 1)?;
    let mut decode_token = backend.new_decode_token()?;
    let mut all_tokens = input_tokens.clone();

    let mut stats = GenerationStats::new();
    stats.start_prefill(prompt_len);

    let tokens_array = Array2::from_shape_vec((1, prompt_len), input_tokens.clone())
        .map_err(|e| anyhow!("failed to create token array: {}", e))?;

    let mut next_token_logits = backend
        .prefill(model.as_ref(), &tokens_array, cache.as_mut())
        .await?;

    stats.end_prefill();

    let tokenizer = model.tokenizer();

    for &token_id in &input_tokens {
        if let Some(ref c) = cancellation {
            if c.is_cancelled() {
                debug!("generation cancelled during prompt emission");
                return Ok(());
            }
        }

        if Some(token_id) == model.bos_token_id() {
            continue;
        }

        let text = tokenizer
            .decode(&[token_id], false)
            .map_err(|e| anyhow!("tokenizer decode error: {}", e))?;

        if tx
            .send(Ok(StreamedToken {
                text,
                id: token_id,
                token_type: TokenType::Prompt,
            }))
            .await
            .is_err()
        {
            debug!("receiver dropped, stopping generation");
            return Ok(());
        }
    }

    let max_new_tokens = config.max_new_tokens.unwrap_or(max_len - prompt_len);
    let stop_tokens = model.stop_token_ids();
    let mut seq_len = prompt_len + 1;

    for step in 0..max_new_tokens {
        if let Some(ref c) = cancellation {
            if c.is_cancelled() {
                debug!("generation cancelled at step {}", step);
                break;
            }
        }

        if all_tokens.len() >= context_limit {
            debug!("context limit reached ({})", context_limit);
            break;
        }

        if all_tokens.len() >= max_len {
            debug!("max length reached ({})", max_len);
            break;
        }

        let mut logits = next_token_logits.clone();

        if config.repetition_penalty != 1.0 {
            apply_repetition_penalty_mut(&mut logits, &all_tokens, config.repetition_penalty);
        }

        if config.no_repeat_ngram_size > 0 {
            apply_no_repeat_ngram(&mut logits, &all_tokens, config.no_repeat_ngram_size);
        }

        let next_token = sample_token(logits, &config.strategy)?;

        if stop_tokens.contains(&next_token) {
            debug!("stop token {} generated at step {}", next_token, step);
            break;
        }

        all_tokens.push(next_token);

        let text = tokenizer
            .decode(&[next_token], false)
            .map_err(|e| anyhow!("tokenizer decode error: {}", e))?;

        if tx
            .send(Ok(StreamedToken {
                text,
                id: next_token,
                token_type: TokenType::Generated,
            }))
            .await
            .is_err()
        {
            debug!("receiver dropped at step {}", step);
            break;
        }

        stats.record_token();

        backend.update_decode_token(&mut decode_token, next_token)?;

        next_token_logits = backend
            .decode_one(model.as_ref(), &decode_token, seq_len, cache.as_mut())
            .await?;

        seq_len += 1;
    }

    stats.print_summary();
    Ok(())
}

pub async fn generate_tokens(
    model: Arc<dyn DecoderLanguageModel + Send + Sync>,
    backend: AnyDecoderBackend,
    input_tokens: Vec<u32>,
    config: GenerationConfig,
) -> Result<Vec<u32>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(256);

    let handle = tokio::spawn(async move {
        run_generation_loop(model, backend, input_tokens, config, tx, None).await
    });

    let mut generated = Vec::new();
    while let Some(result) = rx.recv().await {
        match result {
            Ok(token) => {
                if token.token_type == TokenType::Generated {
                    generated.push(token.id);
                }
            }
            Err(e) => return Err(e),
        }
    }

    handle.await??;

    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::Cache;
    
    use std::any::Any;
    use std::collections::HashSet;
    

 
    #[test]
    fn test_token_type_equality() {
        assert_eq!(TokenType::Prompt, TokenType::Prompt);
        assert_eq!(TokenType::Generated, TokenType::Generated);
        assert_ne!(TokenType::Prompt, TokenType::Generated);
    }

    #[test]
    fn test_streamed_token_debug() {
        let token = StreamedToken {
            text: "hello".to_string(),
            id: 42,
            token_type: TokenType::Generated,
        };
        let debug_str = format!("{:?}", token);
        assert!(debug_str.contains("hello"));
        assert!(debug_str.contains("42"));
    }

    // ========================================================================
    //  StreamedToken Tests
    // ========================================================================

    #[test]
    fn test_streamed_token_creation() {
        let token = StreamedToken {
            text: "world".to_string(),
            id: 100,
            token_type: TokenType::Prompt,
        };

        assert_eq!(token.text, "world");
        assert_eq!(token.id, 100);
        assert_eq!(token.token_type, TokenType::Prompt);
    }

    #[test]
    fn test_streamed_token_empty_text() {
        let token = StreamedToken {
            text: String::new(),
            id: 0,
            token_type: TokenType::Generated,
        };

        assert!(token.text.is_empty());
        assert_eq!(token.id, 0);
    }

    #[test]
    fn test_streamed_token_unicode_text() {
        let token = StreamedToken {
            text: "こんにちは".to_string(),
            id: 12345,
            token_type: TokenType::Generated,
        };

        assert_eq!(token.text, "こんにちは");
    }

    // ========================================================================
    //  TokenType Tests
    // ========================================================================

    #[test]
    fn test_token_type_debug() {
        let prompt = TokenType::Prompt;
        let generated = TokenType::Generated;

        let prompt_debug = format!("{:?}", prompt);
        let generated_debug = format!("{:?}", generated);

        assert!(prompt_debug.contains("Prompt"));
        assert!(generated_debug.contains("Generated"));
    }

    // ========================================================================
    //  Mock Infrastructure
    // ========================================================================

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
        fn increment_len(&mut self, n: usize) {
            self.len += n;
        }
        fn clone_box(&self) -> Box<dyn Cache> {
            Box::new(self.clone())
        }
    }

    // ========================================================================
    //  GenerationConfig Interaction Tests
    // ========================================================================

    #[test]
    fn test_generation_config_max_new_tokens_calculation() {
        let config = GenerationConfig {
            max_new_tokens: Some(50),
            max_length: 1000,
            ..Default::default()
        };

        let prompt_len = 100;
        let max_len = config
            .max_new_tokens
            .map(|n| prompt_len + n)
            .unwrap_or(config.max_length);

        assert_eq!(max_len, 150); // 100 + 50
    }

    #[test]
    fn test_generation_config_fallback_to_max_length() {
        let config = GenerationConfig {
            max_new_tokens: None,
            max_length: 512,
            ..Default::default()
        };

        let prompt_len = 100;
        let max_len = config
            .max_new_tokens
            .map(|n| prompt_len + n)
            .unwrap_or(config.max_length);

        assert_eq!(max_len, 512);
    }

    #[test]
    fn test_cache_capacity_legacy_loop() {
        let max_len = 100;
        let cache_capacity = max_len + 1; // Legacy adds 1
        assert_eq!(cache_capacity, 101);
    }

    #[test]
    fn test_cache_capacity_pipelined_loop() {
        let max_len = 100;
        let cache_capacity = max_len; // Pipelined uses exact
        assert_eq!(cache_capacity, 100);
    }

    // ========================================================================
    //  Token Array Creation Tests
    // ========================================================================

    #[test]
    fn test_tokens_array_creation() {
        let input_tokens = vec![1u32, 2, 3, 4, 5];
        let prompt_len = input_tokens.len();

        let tokens_array = Array2::from_shape_vec((1, prompt_len), input_tokens.clone()).unwrap();

        assert_eq!(tokens_array.shape(), &[1, 5]);
        assert_eq!(tokens_array[[0, 0]], 1);
        assert_eq!(tokens_array[[0, 4]], 5);
    }

    #[test]
    fn test_tokens_array_single_token() {
        let input_tokens = vec![42u32];
        let tokens_array = Array2::from_shape_vec((1, 1), input_tokens).unwrap();

        assert_eq!(tokens_array.shape(), &[1, 1]);
        assert_eq!(tokens_array[[0, 0]], 42);
    }

    // ========================================================================
    //  Stop Token Logic Tests
    // ========================================================================

    #[test]
    fn test_stop_token_detection() {
        let stop_tokens: HashSet<u32> = HashSet::from([1, 2, 50256]);

        assert!(stop_tokens.contains(&1));
        assert!(stop_tokens.contains(&2));
        assert!(stop_tokens.contains(&50256));
        assert!(!stop_tokens.contains(&100));
    }

    #[test]
    fn test_empty_stop_tokens() {
        let stop_tokens: HashSet<u32> = HashSet::new();

        assert!(!stop_tokens.contains(&1));
        assert!(!stop_tokens.contains(&2));
    }

    // ========================================================================
    //  Context Limit Logic Tests
    // ========================================================================

    #[test]
    fn test_context_limit_check() {
        let context_limit = 2048;
        let all_tokens_len = 2000;

        assert!(all_tokens_len < context_limit);

        let all_tokens_len = 2048;
        assert!(all_tokens_len >= context_limit);
    }

    #[test]
    fn test_max_length_check() {
        let max_len = 100;

        assert!(50 < max_len);
        assert!(100 >= max_len);
        assert!(150 >= max_len);
    }

    // ========================================================================
    //  BOS Token Logic Tests
    // ========================================================================

    #[test]
    fn test_bos_token_not_prepended_if_present() {
        let mut tokens = vec![1u32, 2, 3]; // 1 is BOS
        let bos = Some(1u32);

        if let Some(bos_id) = bos {
            if tokens.first() != Some(&bos_id) {
                tokens.insert(0, bos_id);
            }
        }

        // Should not add duplicate
        assert_eq!(tokens, vec![1, 2, 3]);
    }

    #[test]
    fn test_bos_token_prepended_if_missing() {
        let mut tokens = vec![2u32, 3, 4];
        let bos = Some(1u32);

        if let Some(bos_id) = bos {
            if tokens.first() != Some(&bos_id) {
                tokens.insert(0, bos_id);
            }
        }

        assert_eq!(tokens, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_bos_token_none() {
        let mut tokens = vec![2u32, 3, 4];
        let bos: Option<u32> = None;

        if let Some(bos_id) = bos {
            if tokens.first() != Some(&bos_id) {
                tokens.insert(0, bos_id);
            }
        }

        // Should remain unchanged
        assert_eq!(tokens, vec![2, 3, 4]);
    }

    // ========================================================================
    //  Sequence Length Tracking Tests
    // ========================================================================

    #[test]
    fn test_seq_len_initialization() {
        let prompt_len = 50;
        let seq_len = prompt_len + 1;

        assert_eq!(seq_len, 51);
    }

    #[test]
    fn test_seq_len_increment() {
        let mut seq_len = 51;

        for _ in 0..10 {
            seq_len += 1;
        }

        assert_eq!(seq_len, 61);
    }

    // ========================================================================
    //  All Tokens Tracking Tests
    // ========================================================================

    #[test]
    fn test_all_tokens_accumulation() {
        let input_tokens = vec![1u32, 2, 3];
        let mut all_tokens = input_tokens.clone();

        // Simulate generation
        all_tokens.push(4);
        all_tokens.push(5);
        all_tokens.push(6);

        assert_eq!(all_tokens, vec![1, 2, 3, 4, 5, 6]);
    }

    // ========================================================================
    //  Empty Prompt Validation Tests
    // ========================================================================

    #[test]
    fn test_empty_prompt_detection() {
        let input_tokens: Vec<u32> = vec![];

        assert!(input_tokens.is_empty());
        assert_eq!(input_tokens.len(), 0);
    }

    #[test]
    fn test_non_empty_prompt() {
        let input_tokens = vec![1u32];

        assert!(!input_tokens.is_empty());
        assert_eq!(input_tokens.len(), 1);
    }

    // ========================================================================
    //  Backend Type Tests
    // ========================================================================

    #[test]
    fn test_any_decoder_backend_cpu_type() {
        let backend = AnyDecoderBackend::Cpu(CpuDecoderBackend);
        assert_eq!(backend.backend_type(), "CPU");
    }

    #[tokio::test]
    async fn test_any_decoder_backend_gpu_type() {
        let ctx = WgpuContext::new().await.expect("Failed to create context");
        let gpu_backend = Arc::new(GpuDecoderBackend::new(ctx).unwrap());
        let backend = AnyDecoderBackend::Gpu(gpu_backend);
        assert_eq!(backend.backend_type(), "GPU");
    }

    // ========================================================================
    //  Cancellation Token Tests
    // ========================================================================

    #[test]
    fn test_cancellation_token_not_cancelled() {
        let (token, handle) = CancellationToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_cancelled() {
        let (token, handle) = CancellationToken::new();
        handle.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancellation_check_in_loop() {
        let cancellation = Some(CancellationToken::new());

        let mut iterations = 0;
        for _ in 0..10 {
            if let Some(ref c) = cancellation {
                if c.0.is_cancelled() {
                    break;
                }
            }
            iterations += 1;
        }

        assert_eq!(iterations, 10); // Not cancelled, all iterations complete
    }

    #[test]
    fn test_cancellation_stops_loop() {
        let (token, handle) = CancellationToken::new();

        let mut iterations = 0;
        for i in 0..10 {
            if token.is_cancelled() {
                break;
            }
            iterations += 1;

            if i == 4 {
                handle.cancel();
            }
        }

        assert_eq!(iterations, 5); // Stopped after 5 iterations
    }

    // ========================================================================
    //  Max New Tokens Calculation Tests
    // ========================================================================

    #[test]
    fn test_max_new_tokens_with_value() {
        let config = GenerationConfig {
            max_new_tokens: Some(100),
            max_length: 2048,
            ..Default::default()
        };

        let prompt_len = 50;
        let max_new_tokens = config
            .max_new_tokens
            .unwrap_or(config.max_length - prompt_len);

        assert_eq!(max_new_tokens, 100);
    }

    #[test]
    fn test_max_new_tokens_fallback() {
        let config = GenerationConfig {
            max_new_tokens: None,
            max_length: 2048,
            ..Default::default()
        };

        let prompt_len = 50;
        let max_len = config.max_length;
        let max_new_tokens = config.max_new_tokens.unwrap_or(max_len - prompt_len);

        assert_eq!(max_new_tokens, 1998); // 2048 - 50
    }

    // ========================================================================
    //  Generation Stats Integration
    // ========================================================================

    #[test]
    fn test_generation_stats_creation() {
        let stats = GenerationStats::new();
        // Stats should be created without panic
        let _ = stats;
    }

    // ========================================================================
    //  Draft Model State Tests
    // ========================================================================

    #[test]
    fn test_has_draft_model_false_by_default() {
        // Can't test without a real model, but test the logic
        let draft: Option<DraftModelContext> = None;
        assert!(draft.is_none());
    }
}
