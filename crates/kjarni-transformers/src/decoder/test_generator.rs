use crate::activations::Activation;
use crate::cache::Cache;
use crate::common::{CancellationToken, GenerationConfig, StreamedToken, TokenType};
use crate::decoder::prelude::*;
use crate::models::base::{AutoregressiveLoop, ModelInput};
use crate::traits::{
    AttentionLayout, DecoderLayerLayout, DecoderLayout, FeedForwardLayout, InferenceModel,
    ModelConfig, ModelLayout, ModelMetadata, NormalizationStrategy,
};
use crate::{ChatTemplate, Conversation, Device, LanguageModel, WgpuContext};
use anyhow::Result;
use futures::{StreamExt, TryStreamExt};
use ndarray::{Array2, Array3};
use std::any::Any;
use std::collections::HashSet;
use std::sync::Arc;
use tokenizers::Tokenizer;

// =========================================================================
//  1. Mock Configuration
// =========================================================================

#[derive(Debug, Clone)]
struct MockConfig {
    vocab_size: usize,
    hidden_size: usize,
}

impl ModelConfig for MockConfig {
    fn model_type(&self) -> &str {
        "mock"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            decoder_layers: None,
            hidden_size: self.hidden_size,
            num_layers: 1,
            num_attention_heads: 1,
            num_kv_heads: 1,
            head_dim: self.hidden_size,
            vocab_size: self.vocab_size,
            max_seq_len: 1024,
            norm_eps: 1e-5,
            activation: Activation::Gelu,
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: false,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: true,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            normalization_strategy: NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        // Return dummy layout paths since we won't be loading weights
        ModelLayout {
            token_embedding: "token_emb".to_string(),
            lm_head: "lm_head".to_string(),
            encoder: None,
            decoder: Some(DecoderLayout {
                position_embedding: None,
                token_type_embedding: None,
                embedding_norm_weight: None,
                embedding_norm_bias: None,
                final_norm_weight: Some("ln_f".to_string()),
                final_norm_bias: None,
                layer: DecoderLayerLayout {
                    self_attn: AttentionLayout {
                        q_weight: "q".into(),
                        k_weight: "k".into(),
                        v_weight: "v".into(),
                        o_weight: "o".into(),
                        q_bias: None,
                        k_bias: None,
                        v_bias: None,
                        o_bias: None,
                        norm_weight: "ln1".into(),
                        norm_bias: None,
                    },
                    cross_attn: None,
                    ffn: FeedForwardLayout {
                        up_weight: "up".into(),
                        down_weight: "down".into(),
                        gate_weight: None,
                        gate_bias: None,
                        up_bias: None,
                        down_bias: None,
                        norm_weight: "ln2".into(),
                        norm_bias: None,
                    },
                },
            }),
        }
    }
}

// =========================================================================
//  2. Mock Model
// =========================================================================

struct MockDecoderModel {
    vocab_size: usize,
    context_size: usize,
    device: Device,
    tokenizer: Tokenizer,
    pub context_size_override: Option<usize>,

    decoder: MockCpuDecoder,
    config: MockConfig,
}

impl MockDecoderModel {
    fn new() -> Self {
        // Minimal JSON tokenizer
        let tokenizer_json = r#"{
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [
            { "id": 0, "content": "[BOS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true },
            { "id": 1, "content": "[EOS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true }
          ],
          "normalizer": null,
          "pre_tokenizer": null,
          "post_processor": null,
          "decoder": null,
          "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "vocab": {
              "[BOS]": 0, "[EOS]": 1, "hello": 2, "world": 3
            },
            "merges": []
          }
        }"#;

        let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes())
            .expect("Failed to create mock tokenizer");

        let config = MockConfig {
            vocab_size: 100,
            hidden_size: 64,
        };

        Self {
            vocab_size: 100,
            context_size: 1024,
            device: Device::Cpu,
            tokenizer,
            context_size_override: None,
            decoder: MockCpuDecoder {
                hidden_size: 64,
                num_layers: 1,
            },
            config,
        }
    }
}

// 1. Implement LanguageModel
impl LanguageModel for MockDecoderModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn hidden_size(&self) -> usize {
        64
    }
    fn num_layers(&self) -> usize {
        1
    }
    fn num_heads(&self) -> usize {
        1
    }
    fn context_size(&self) -> usize {
        self.context_size_override.unwrap_or(self.context_size)
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn eos_token_id(&self) -> Option<u32> {
        Some(1)
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
        let mut set = HashSet::new();
        set.insert(1);
        set
    }

    fn new_cache(
        &self,
        _batch: usize,
        _capacity: usize,
        _num_beams: usize,
    ) -> Result<Box<dyn Cache>> {
        Ok(Box::new(MockCache { len: 0 }))
    }
}

// 2. Implement InferenceModel
impl InferenceModel for MockDecoderModel {
    fn device(&self) -> Device {
        self.device
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        None
    }
    // fn config(&self) -> &ModelConfig { unimplemented!() }
    // fn quantization_config(&self) -> Option<&QuantizationConfig> { None }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// 3. Implement DecoderLanguageModel
impl DecoderLanguageModel for MockDecoderModel {
    // Return SELF as the ops provider
    fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps> {
        Some(self)
    }
    fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps> {
        None
    }
    fn autoregressive_loop(&self) -> AutoregressiveLoop {
        AutoregressiveLoop::Pipelined
    }
    fn is_instruct_model(&self) -> bool {
        false
    }
    fn chat_template(&self) -> Option<&dyn ChatTemplate> {
        None
    }
}

// 4. Implement CpuDecoderOps for the Model itself
impl CpuDecoderOps for MockDecoderModel {
    fn decoder(&self) -> &dyn CpuDecoder {
        &self.decoder
    }
    fn embed(&self, tokens: &[u32], pos: usize) -> Result<Array3<f32>> {
        Ok(Array3::zeros((1, 1, self.hidden_size())))
    }
    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, _hidden) = hidden_states.dim();

        // Initialize with a very low value so the chosen token stands out
        let mut logits = Array3::from_elem((batch, seq, self.vocab_size), -100.0);

        // Force prediction of token ID 2 ("hello")
        // This ensures we don't hit EOS (ID 1) prematurely
        use ndarray::s;
        logits.slice_mut(s![.., .., 2]).fill(100.0);

        Ok(logits)
    }

    fn get_attention_mask(&self, seq_len: usize, past_len: usize) -> Result<Array2<f32>> {
        Ok(Array2::ones((1, seq_len + past_len)))
    }
}

// 5. Mock CpuDecoder Component
struct MockCpuDecoder {
    hidden_size: usize,
    num_layers: usize,
}

impl CpuDecoder for MockCpuDecoder {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn embed(&self, input: ModelInput<'_>, _offset: usize) -> Result<Array3<f32>> {
        let (batch, seq) = match input {
            ModelInput::TokensCpu(t) => (t.nrows(), t.ncols()),
            _ => (1, 1),
        };
        Ok(Array3::zeros((batch, seq, self.hidden_size)))
    }

    fn embed_and_normalize(&self, input: ModelInput<'_>, offset: usize) -> Result<Array3<f32>> {
        self.embed(input, offset)
    }

    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        _mask: &Array2<f32>,
        _offset: usize,
        _cache: Option<&mut dyn Cache>,
        _start: usize,
        _end: usize,
    ) -> Result<Array3<f32>> {
        // Just pass through
        Ok(hidden_states.clone())
    }
}

// Mock Cache Implementation
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

// =========================================================================
//  Tests
// =========================================================================

#[tokio::test]
async fn test_generator_setup() {
    let model = Arc::new(MockDecoderModel::new());
    let generator = DecoderGenerator::new(model).expect("Failed to create generator");

    assert_eq!(generator.model.vocab_size(), 100);
}

#[tokio::test]
async fn test_encode_with_bos() {
    let model = Arc::new(MockDecoderModel::new());
    let generator = DecoderGenerator::new(model).unwrap();

    let config = GenerationConfig {
        add_bos_token: true,
        ..Default::default()
    };

    let tokens = generator.encode("Hello", &config).unwrap();
    assert_eq!(tokens[0], 0, "BOS token should be prepended");
}

#[tokio::test]
async fn test_generation_flow_control() {
    // Now this test should actually RUN because we have a valid CpuDecoderOps implementation.
    // It will generate random tokens (since logits are all 0.0), but the loop logic is verified.

    let model = Arc::new(MockDecoderModel::new());
    let generator = DecoderGenerator::new(model).unwrap();

    let config = GenerationConfig {
        max_new_tokens: Some(3),
        strategy: crate::common::DecodingStrategy::Greedy,
        ..Default::default()
    };

    let stream = generator
        .generate_stream("Test", &config, None)
        .await
        .unwrap();
    let tokens: Vec<StreamedToken> = stream.try_collect().await.unwrap();

    // Prompt + 3 generated
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0].text, "hello");
    assert_eq!(tokens[0].id, 2);
    assert_eq!(tokens[0].token_type, TokenType::Generated);

    assert_eq!(tokens[1].text, "hello");
    assert_eq!(tokens[1].id, 2);
    assert_eq!(tokens[1].token_type, TokenType::Generated);

    assert_eq!(tokens[2].text, "hello");
    assert_eq!(tokens[2].id, 2);
    assert_eq!(tokens[2].token_type, TokenType::Generated);
}

#[tokio::test]
async fn test_cancellation_during_stream() {
    let model = Arc::new(MockDecoderModel::new());
    let generator = DecoderGenerator::new(model).unwrap();

    let (cancel_token, handle) = CancellationToken::new();
    handle.cancel();

    let config = GenerationConfig {
        max_new_tokens: Some(10),
        ..Default::default()
    };

    let stream = generator
        .generate_stream("Test", &config, Some(cancel_token))
        .await
        .unwrap();
    let tokens: Vec<StreamedToken> = stream.try_collect().await.unwrap();

    let generated_count = tokens
        .iter()
        .filter(|t| t.token_type == TokenType::Generated)
        .count();
    assert_eq!(generated_count, 0);
}

#[tokio::test]
async fn test_context_limit_check() {
    let mut model = MockDecoderModel::new();
    model.context_size_override = Some(5);
    let generator = DecoderGenerator::new(Arc::new(model)).unwrap();

    let config = GenerationConfig {
        max_new_tokens: Some(10),
        ..Default::default()
    };

    let stream = generator
        .generate_stream("Hello world", &config, None)
        .await
        .unwrap();
    let tokens: Vec<StreamedToken> = stream.try_collect().await.unwrap();
    let expected_len = if generator.model.bos_token_id().is_some() {
        4 // 5 context - 1 hidden BOS
    } else {
        5
    };
    // Total tokens should not exceed context size
    assert_eq!(tokens.len(), expected_len);
}

#[tokio::test]
async fn test_chat_formatting_fallback() {
    let model = Arc::new(MockDecoderModel::new());
    let generator = DecoderGenerator::new(model).unwrap();

    let mut conv = Conversation::new();
    conv.push_user("Hello");

    let prompt = generator.format_conversation(&conv).unwrap();
    assert_eq!(prompt, "Hello");
}
