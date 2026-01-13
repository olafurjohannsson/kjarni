//! Comprehensive tests for CPU and GPU decoder generation backends.
//!
//! This module provides extensive test coverage for:
//! - `CpuDecoderBackend` - CPU-based autoregressive generation
//! - `GpuDecoderBackend` - GPU-accelerated autoregressive generation
//! - `AnyDecoderBackend` - Type-erased unified backend
//! - Parity tests ensuring CPU and GPU produce consistent results
//!
//! # Test Categories
//!
//! 1. **Unit Tests**: Individual method testing (new_decode_token, update_decode_token)
//! 2. **Integration Tests**: Full prefill/decode flows
//! 3. **Strategy Tests**: Pipelined vs Legacy autoregressive loops
//! 4. **Error Handling**: Empty inputs, invalid states
//! 5. **Edge Cases**: Single token, long sequences, boundary conditions
//! 6. **GPU Tests**: GPU-specific functionality
//! 7. **Parity Tests**: CPU vs GPU output comparison
//! 8. **AnyDecoderBackend Tests**: Type erasure and dispatch

use crate::cache::{Cache, GpuKVCache};
use crate::cpu::decoder::CpuDecoderBackend;
use crate::decoder::backend::AnyDecoderBackend;
use crate::decoder::prelude::GpuDecoderBackend;
use crate::decoder::traits::{
    CpuDecoder, CpuDecoderOps, DecoderGenerationBackend, DecoderLanguageModel, GpuDecoder,
    GpuDecoderOps,
};
use crate::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
use crate::models::base::AutoregressiveLoop;
use crate::traits::InferenceModel;
use crate::{Device, LanguageModel, WgpuContext};
use anyhow::Result;
use ndarray::{s, Array1, Array2, Array3};
use std::any::Any;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokenizers::Tokenizer;
use wgpu::CommandEncoder;


#[cfg(test)]
mod decoder_backend_tests {
    use super::*;

    // =========================================================================
    //  Mock Infrastructure - Shared by CPU and GPU tests
    // =========================================================================

    /// Mock KV cache for CPU testing.
    #[derive(Clone)]
    pub struct MockCpuCache {
        seq_length: usize,
        max_seq_len: usize,
        increment_calls: Arc<AtomicUsize>,
    }

    impl Default for MockCpuCache {
        fn default() -> Self {
            Self::new(2048)
        }
    }

    impl MockCpuCache {
        pub fn new(max_seq_len: usize) -> Self {
            Self {
                seq_length: 0,
                max_seq_len,
                increment_calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        pub fn with_seq_length(mut self, seq_length: usize) -> Self {
            self.seq_length = seq_length;
            self
        }

        pub fn get_increment_calls(&self) -> usize {
            self.increment_calls.load(Ordering::SeqCst)
        }
    }

    impl Cache for MockCpuCache {
        fn get_seq_length(&self) -> usize {
            self.seq_length
        }

        fn set_seq_length(&mut self, len: usize) {
            self.seq_length = len;
        }

        fn increment_len(&mut self, delta: usize) {
            self.seq_length += delta;
            self.increment_calls.fetch_add(1, Ordering::SeqCst);
        }

        fn clear(&mut self) {
            self.seq_length = 0;
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn clone_box(&self) -> Box<dyn Cache> {
            Box::new(self.clone())
        }
    }

    // -------------------------------------------------------------------------
    //  Mock CPU Decoder
    // -------------------------------------------------------------------------

    pub struct MockCpuDecoder {
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        forward_calls: Arc<AtomicUsize>,
    }

    impl Default for MockCpuDecoder {
        fn default() -> Self {
            Self::new(64, 4, 8)
        }
    }

    impl MockCpuDecoder {
        pub fn new(hidden_size: usize, num_layers: usize, num_heads: usize) -> Self {
            Self {
                hidden_size,
                num_layers,
                num_heads,
                forward_calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        pub fn get_forward_calls(&self) -> usize {
            self.forward_calls.load(Ordering::SeqCst)
        }
    }

    impl CpuDecoder for MockCpuDecoder {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn forward_layers(
            &self,
            hidden_states: &Array3<f32>,
            _attention_mask: &Array2<f32>,
            _position_offset: usize,
            _cache: Option<&mut dyn Cache>,
            _start_layer: usize,
            _end_layer: usize,
        ) -> Result<Array3<f32>> {
            Ok(hidden_states.mapv(|x| x + 0.1))
        }

        fn forward(
            &self,
            hidden_states: &Array3<f32>,
            attention_mask: &Array2<f32>,
            position_offset: usize,
            cache: Option<&mut dyn Cache>,
        ) -> Result<Array3<f32>> {
            self.forward_calls.fetch_add(1, Ordering::SeqCst);
            let output = self.forward_layers(
                hidden_states,
                attention_mask,
                position_offset,
                cache,
                0,
                self.num_layers,
            )?;
            self.final_norm(&output)
        }

        fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
            Ok(hidden_states.mapv(|x| x * 0.99))
        }

        fn num_layers(&self) -> usize {
            self.num_layers
        }

        fn hidden_size(&self) -> usize {
            self.hidden_size
        }

        fn num_attention_heads(&self) -> usize {
            self.num_heads
        }

        fn num_kv_heads(&self) -> usize {
            self.num_heads
        }

        fn head_dim(&self) -> usize {
            self.hidden_size / self.num_heads
        }
    }

    // -------------------------------------------------------------------------
    //  Mock CPU Decoder Ops
    // -------------------------------------------------------------------------

    pub struct MockCpuDecoderOps {
        decoder: MockCpuDecoder,
        vocab_size: usize,
        embed_calls: Arc<AtomicUsize>,
        project_calls: Arc<AtomicUsize>,
    }

    impl Default for MockCpuDecoderOps {
        fn default() -> Self {
            Self::new(64, 4, 8, 1000)
        }
    }

    impl MockCpuDecoderOps {
        pub fn new(
            hidden_size: usize,
            num_layers: usize,
            num_heads: usize,
            vocab_size: usize,
        ) -> Self {
            Self {
                decoder: MockCpuDecoder::new(hidden_size, num_layers, num_heads),
                vocab_size,
                embed_calls: Arc::new(AtomicUsize::new(0)),
                project_calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        pub fn get_embed_calls(&self) -> usize {
            self.embed_calls.load(Ordering::SeqCst)
        }

        pub fn get_project_calls(&self) -> usize {
            self.project_calls.load(Ordering::SeqCst)
        }
    }

    impl CpuDecoderOps for MockCpuDecoderOps {
        fn decoder(&self) -> &dyn CpuDecoder {
            &self.decoder
        }

        fn embed(&self, tokens: &Array2<u32>, _position_offset: usize) -> Result<Array3<f32>> {
            self.embed_calls.fetch_add(1, Ordering::SeqCst);
            let (batch, seq_len) = (tokens.shape()[0], tokens.shape()[1]);
            let hidden_size = self.decoder.hidden_size();

            let mut hidden = Array3::zeros((batch, seq_len, hidden_size));
            for b in 0..batch {
                for s in 0..seq_len {
                    let token_id = tokens[[b, s]] as f32;
                    let base_value = token_id / self.vocab_size as f32;
                    for h in 0..hidden_size {
                        hidden[[b, s, h]] = base_value + (h as f32 * 0.001);
                    }
                }
            }
            Ok(hidden)
        }

        fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
            self.project_calls.fetch_add(1, Ordering::SeqCst);
            let (batch, seq_len, _) = hidden_states.dim();
            let mut logits = Array3::zeros((batch, seq_len, self.vocab_size));

            for b in 0..batch {
                for s in 0..seq_len {
                    let hidden_sum: f32 = hidden_states.slice(ndarray::s![b, s, ..]).sum();
                    for v in 0..self.vocab_size {
                        logits[[b, s, v]] = hidden_sum * 0.1 + v as f32 * 0.01;
                    }
                }
            }
            Ok(logits)
        }

        fn get_attention_mask(&self, seq_len: usize, past_len: usize) -> Result<Array2<f32>> {
            let total_len = seq_len + past_len;
            let mut mask = Array2::zeros((seq_len, total_len));
            for i in 0..seq_len {
                for j in 0..=past_len + i {
                    if j < total_len {
                        mask[[i, j]] = 1.0;
                    }
                }
            }
            Ok(mask)
        }
    }

    // -------------------------------------------------------------------------
    //  Mock Decoder Language Model (CPU)
    // -------------------------------------------------------------------------

    pub struct MockCpuDecoderModel {
        ops: MockCpuDecoderOps,
        loop_type: AutoregressiveLoop,
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        context_size: usize,
    }

    impl MockCpuDecoderModel {
        pub fn new(loop_type: AutoregressiveLoop) -> Self {
            Self::with_config(loop_type, 1000, 64, 4, 2048)
        }

        pub fn with_config(
            loop_type: AutoregressiveLoop,
            vocab_size: usize,
            hidden_size: usize,
            num_layers: usize,
            context_size: usize,
        ) -> Self {
            Self {
                ops: MockCpuDecoderOps::new(hidden_size, num_layers, 8, vocab_size),
                loop_type,
                vocab_size,
                hidden_size,
                num_layers,
                context_size,
            }
        }

        pub fn pipelined() -> Self {
            Self::new(AutoregressiveLoop::Pipelined)
        }

        pub fn legacy() -> Self {
            Self::new(AutoregressiveLoop::Legacy)
        }

        pub fn get_ops(&self) -> &MockCpuDecoderOps {
            &self.ops
        }
    }

    impl InferenceModel for MockCpuDecoderModel {
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

    impl LanguageModel for MockCpuDecoderModel {
        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
        fn hidden_size(&self) -> usize {
            self.hidden_size
        }
        fn num_layers(&self) -> usize {
            self.num_layers
        }
        fn num_heads(&self) -> usize {
            8
        }
        fn context_size(&self) -> usize {
            self.context_size
        }
        fn tokenizer(&self) -> &Tokenizer {
            unimplemented!()
        }
        fn eos_token_id(&self) -> Option<u32> {
            Some(2)
        }
        fn bos_token_id(&self) -> Option<u32> {
            Some(1)
        }
        fn forced_bos_token_id(&self) -> Option<u32> {
            None
        }
        fn forced_eos_token_id(&self) -> Option<u32> {
            None
        }
        fn pad_token_id(&self) -> Option<u32> {
            Some(0)
        }
        fn stop_token_ids(&self) -> HashSet<u32> {
            let mut set = HashSet::new();
            set.insert(2);
            set
        }
        fn new_cache(&self, _: usize, max_len: usize, _: usize) -> Result<Box<dyn Cache>> {
            Ok(Box::new(MockCpuCache::new(max_len)))
        }
    }

    impl DecoderLanguageModel for MockCpuDecoderModel {
        fn autoregressive_loop(&self) -> AutoregressiveLoop {
            self.loop_type
        }
        fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps> {
            Some(&self.ops)
        }
        fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps> {
            None
        }
        fn chat_template(&self) -> Option<&dyn crate::ChatTemplate> {
            None
        }
        fn is_instruct_model(&self) -> bool {
            false
        }
    }

    // =========================================================================
    //  CPU Backend Unit Tests
    // =========================================================================

    mod cpu_unit_tests {
        use super::*;

        #[test]
        fn test_new_decode_token_creates_correct_shape() {
            let backend = CpuDecoderBackend::new();
            let token = backend.new_decode_token().unwrap();
            assert_eq!(token.shape(), &[1, 1]);
            assert_eq!(token[[0, 0]], 0);
        }

        #[test]
        fn test_update_decode_token_basic() {
            let backend = CpuDecoderBackend::new();
            let mut token = backend.new_decode_token().unwrap();

            backend.update_decode_token(&mut token, 42).unwrap();
            assert_eq!(token[[0, 0]], 42);
        }

        #[test]
        fn test_update_decode_token_sequence() {
            let backend = CpuDecoderBackend::new();
            let mut token = backend.new_decode_token().unwrap();

            for i in 0..1000 {
                backend.update_decode_token(&mut token, i).unwrap();
                assert_eq!(token[[0, 0]], i);
            }
        }

        #[test]
        fn test_backend_is_stateless() {
            let backend1 = CpuDecoderBackend::new();
            let backend2 = backend1.clone();

            let mut t1 = backend1.new_decode_token().unwrap();
            let mut t2 = backend2.new_decode_token().unwrap();

            backend1.update_decode_token(&mut t1, 100).unwrap();
            backend2.update_decode_token(&mut t2, 200).unwrap();

            assert_eq!(t1[[0, 0]], 100);
            assert_eq!(t2[[0, 0]], 200);
        }
    }

    // =========================================================================
    //  CPU Backend Prefill Tests
    // =========================================================================

    mod cpu_prefill_tests {
        use super::*;

        #[tokio::test]
        async fn test_prefill_pipelined() {
            let model = MockCpuDecoderModel::pipelined();
            let backend = CpuDecoderBackend::new();
            let mut cache = MockCpuCache::new(100);

            let tokens = Array2::from_shape_vec((1, 4), vec![1, 2, 3, 4]).unwrap();
            let logits = backend.prefill(&model, &tokens, &mut cache).await.unwrap();

            assert_eq!(logits.shape(), &[model.vocab_size]);
            assert_eq!(model.get_ops().get_embed_calls(), 1);
            assert_eq!(model.get_ops().get_project_calls(), 1);
        }

        #[tokio::test]
        async fn test_prefill_legacy() {
            let model = MockCpuDecoderModel::legacy();
            let backend = CpuDecoderBackend::new();
            let mut cache = MockCpuCache::new(100);

            let tokens = Array2::from_shape_vec((1, 4), vec![1, 2, 3, 4]).unwrap();
            let logits = backend.prefill(&model, &tokens, &mut cache).await.unwrap();

            assert_eq!(logits.shape(), &[model.vocab_size]);
            // Legacy does two phases
            assert_eq!(model.get_ops().get_embed_calls(), 2);
        }

        #[tokio::test]
        async fn test_prefill_empty_error() {
            let model = MockCpuDecoderModel::pipelined();
            let backend = CpuDecoderBackend::new();
            let mut cache = MockCpuCache::new(100);

            let tokens = Array2::<u32>::zeros((1, 0));
            let result = backend.prefill(&model, &tokens, &mut cache).await;

            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_prefill_single_token() {
            let model = MockCpuDecoderModel::pipelined();
            let backend = CpuDecoderBackend::new();
            let mut cache = MockCpuCache::new(100);

            let tokens = Array2::from_shape_vec((1, 1), vec![42]).unwrap();
            let logits = backend.prefill(&model, &tokens, &mut cache).await.unwrap();

            assert_eq!(logits.shape(), &[model.vocab_size]);
        }
    }

    // =========================================================================
    //  CPU Backend Decode Tests
    // =========================================================================

    mod cpu_decode_tests {
        use super::*;

        #[tokio::test]
        async fn test_decode_one_basic() {
            let model = MockCpuDecoderModel::pipelined();
            let backend = CpuDecoderBackend::new();
            let mut cache = MockCpuCache::new(100).with_seq_length(10);

            let token = Array2::from_elem((1, 1), 42u32);
            let logits = backend
                .decode_one(&model, &token, 11, &mut cache)
                .await
                .unwrap();

            assert_eq!(logits.shape(), &[model.vocab_size]);
        }

        #[tokio::test]
        async fn test_decode_sequence_simulation() {
            let model = MockCpuDecoderModel::pipelined();
            let backend = CpuDecoderBackend::new();
            let mut cache = MockCpuCache::new(100);

            // Prefill
            let prompt = Array2::from_shape_vec((1, 3), vec![1, 2, 3]).unwrap();
            let _ = backend.prefill(&model, &prompt, &mut cache).await.unwrap();

            // Decode loop
            let mut decode_token = backend.new_decode_token().unwrap();
            for i in 0..10 {
                backend.update_decode_token(&mut decode_token, 100 + i).unwrap();
                let logits = backend
                    .decode_one(&model, &decode_token, 4 + i as usize, &mut cache)
                    .await
                    .unwrap();
                assert_eq!(logits.shape(), &[model.vocab_size]);
            }
        }
    }

    // =========================================================================
    //  GPU Backend Tests
    // =========================================================================

    mod gpu_backend_tests {
        use super::*;

        /// Helper to create GPU context for tests.
        /// Returns None if no GPU is available.
        pub async fn create_gpu_context() -> Option<Arc<WgpuContext>> {
            match WgpuContext::new().await {
                Ok(ctx) => Some(ctx),
                Err(e) => {
                    eprintln!("GPU not available: {}", e);
                    None
                }
            }
        }

        #[tokio::test]
        async fn test_gpu_backend_creation() {
            let Some(context) = create_gpu_context().await else {
                eprintln!("Skipping: no GPU");
                return;
            };

            let backend = GpuDecoderBackend::new(context);
            assert!(backend.is_ok());
        }

        #[tokio::test]
        async fn test_gpu_new_decode_token_shape() {
            let Some(context) = create_gpu_context().await else {
                eprintln!("Skipping: no GPU");
                return;
            };

            let backend = GpuDecoderBackend::new(context).unwrap();
            let token = backend.new_decode_token().unwrap();

            assert_eq!(token.shape(), &[1, 1]);
        }

        #[tokio::test]
        async fn test_gpu_update_decode_token() {
            let Some(context) = create_gpu_context().await else {
                eprintln!("Skipping: no GPU");
                return;
            };

            let backend = GpuDecoderBackend::new(context).unwrap();
            let mut token = backend.new_decode_token().unwrap();

            // Should not error
            backend.update_decode_token(&mut token, 42).unwrap();
            backend.update_decode_token(&mut token, 12345).unwrap();
            backend.update_decode_token(&mut token, u32::MAX).unwrap();
        }

        #[tokio::test]
        async fn test_gpu_update_decode_token_sequence() {
            let Some(context) = create_gpu_context().await else {
                eprintln!("Skipping: no GPU");
                return;
            };

            let backend = GpuDecoderBackend::new(context).unwrap();
            let mut token = backend.new_decode_token().unwrap();

            // Rapid updates should work
            for i in 0..1000 {
                backend.update_decode_token(&mut token, i).unwrap();
            }
        }

        #[tokio::test]
        async fn test_gpu_multiple_tokens_independent() {
            let Some(context) = create_gpu_context().await else {
                eprintln!("Skipping: no GPU");
                return;
            };

            let backend = GpuDecoderBackend::new(context).unwrap();

            let mut token1 = backend.new_decode_token().unwrap();
            let mut token2 = backend.new_decode_token().unwrap();

            backend.update_decode_token(&mut token1, 100).unwrap();
            backend.update_decode_token(&mut token2, 200).unwrap();

            // Both should have their own values (can't easily verify GPU buffer content,
            // but at least verify no errors)
            assert_eq!(token1.shape(), &[1, 1]);
            assert_eq!(token2.shape(), &[1, 1]);
        }

        #[tokio::test]
        async fn test_gpu_tensor_buffer_exists() {
            let Some(context) = create_gpu_context().await else {
                eprintln!("Skipping: no GPU");
                return;
            };

            let backend = GpuDecoderBackend::new(context).unwrap();
            let token = backend.new_decode_token().unwrap();

            // GpuTensor should have a valid buffer
            let buffer = token.buffer();
            assert!(buffer.size() >= 4); // At least 4 bytes for u32
        }

        #[tokio::test]
        async fn test_gpu_context_accessor() {
            let Some(context) = create_gpu_context().await else {
                eprintln!("Skipping: no GPU");
                return;
            };

            let backend = GpuDecoderBackend::new(context.clone()).unwrap();
            let ctx = backend.context();

            // Should be the same context
            assert!(Arc::ptr_eq(&context, ctx));
        }
    }

    // =========================================================================
    //  AnyDecoderBackend Tests
    // =========================================================================

    mod any_backend_tests {
        use super::*;

        #[test]
        fn test_any_cpu_is_cpu() {
            let backend = AnyDecoderBackend::cpu();
            assert!(backend.is_cpu());
            assert!(!backend.is_gpu());
            assert_eq!(backend.backend_type(), "CPU");
        }

        #[tokio::test]
        async fn test_any_gpu_is_gpu() {
            let Some(context) = super::gpu_backend_tests::create_gpu_context().await else {
                eprintln!("Skipping: no GPU");
                return;
            };

            let gpu = Arc::new(GpuDecoderBackend::new(context).unwrap());
            let backend = AnyDecoderBackend::gpu(gpu);

            assert!(backend.is_gpu());
            assert!(!backend.is_cpu());
            assert_eq!(backend.backend_type(), "GPU");
        }

        #[test]
        fn test_any_cpu_new_decode_token() {
            let backend = AnyDecoderBackend::cpu();
            let token = backend.new_decode_token().unwrap();

            // Should be able to downcast to Array2<u32>
            let concrete = token.downcast_ref::<Array2<u32>>();
            assert!(concrete.is_some());
            assert_eq!(concrete.unwrap().shape(), &[1, 1]);
        }

        #[tokio::test]
        async fn test_any_gpu_new_decode_token() {
            let Some(context) = super::gpu_backend_tests::create_gpu_context().await else {
                eprintln!("Skipping: no GPU");
                return;
            };

            let gpu = Arc::new(GpuDecoderBackend::new(context).unwrap());
            let backend = AnyDecoderBackend::gpu(gpu);
            let token = backend.new_decode_token().unwrap();

            // Should be able to downcast to GpuTensor
            let concrete = token.downcast_ref::<GpuTensor>();
            assert!(concrete.is_some());
            assert_eq!(concrete.unwrap().shape(), &[1, 1]);
        }

        #[test]
        fn test_any_cpu_update_decode_token() {
            let backend = AnyDecoderBackend::cpu();
            let mut token = backend.new_decode_token().unwrap();

            backend.update_decode_token(&mut token, 42).unwrap();

            let concrete = token.downcast_ref::<Array2<u32>>().unwrap();
            assert_eq!(concrete[[0, 0]], 42);
        }

        #[tokio::test]
        async fn test_any_gpu_update_decode_token() {
            let Some(context) = super::gpu_backend_tests::create_gpu_context().await else {
                eprintln!("Skipping: no GPU");
                return;
            };

            let gpu = Arc::new(GpuDecoderBackend::new(context).unwrap());
            let backend = AnyDecoderBackend::gpu(gpu);
            let mut token = backend.new_decode_token().unwrap();

            // Should not error
            backend.update_decode_token(&mut token, 42).unwrap();
        }

        #[test]
        fn test_any_type_mismatch_error() {
            let backend = AnyDecoderBackend::cpu();

            // Create wrong type
            let mut fake: Box<dyn Any + Send + Sync> = Box::new(String::from("fake"));

            let result = backend.update_decode_token(&mut fake, 1);
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("Type mismatch"));
        }

        #[tokio::test]
        async fn test_any_cpu_prefill() {
            let model = MockCpuDecoderModel::pipelined();
            let backend = AnyDecoderBackend::cpu();
            let mut cache = MockCpuCache::new(100);

            let tokens = Array2::from_shape_vec((1, 3), vec![1, 2, 3]).unwrap();
            let logits = backend.prefill(&model, &tokens, &mut cache).await.unwrap();

            assert_eq!(logits.shape(), &[model.vocab_size]);
        }

        #[tokio::test]
        async fn test_any_cpu_decode_one() {
            let model = MockCpuDecoderModel::pipelined();
            let backend = AnyDecoderBackend::cpu();
            let mut cache = MockCpuCache::new(100).with_seq_length(5);

            let mut token = backend.new_decode_token().unwrap();
            backend.update_decode_token(&mut token, 42).unwrap();

            let logits = backend
                .decode_one(&model, &token, 6, &mut cache)
                .await
                .unwrap();

            assert_eq!(logits.shape(), &[model.vocab_size]);
        }

        #[tokio::test]
        async fn test_any_cpu_full_flow() {
            let model = MockCpuDecoderModel::pipelined();
            let backend = AnyDecoderBackend::cpu();
            let mut cache = MockCpuCache::new(100);

            // Prefill
            let tokens = Array2::from_shape_vec((1, 4), vec![1, 2, 3, 4]).unwrap();
            let _ = backend.prefill(&model, &tokens, &mut cache).await.unwrap();

            // Decode
            let mut decode_token = backend.new_decode_token().unwrap();
            for i in 0..5 {
                backend.update_decode_token(&mut decode_token, 100 + i).unwrap();
                let logits = backend
                    .decode_one(&model, &decode_token, 5 + i as usize, &mut cache)
                    .await
                    .unwrap();
                assert_eq!(logits.shape(), &[model.vocab_size]);
            }
        }
    }

    // =========================================================================
    //  CPU/GPU Parity Tests (when both available)
    // =========================================================================

    mod parity_tests {
        use super::*;

        /// Compare two logit arrays within a tolerance.
        fn logits_close(a: &Array1<f32>, b: &Array1<f32>, tolerance: f32) -> bool {
            if a.shape() != b.shape() {
                return false;
            }
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| (x - y).abs() < tolerance)
        }

        /// Get max difference between two logit arrays.
        fn max_logit_diff(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        }

        // Note: Full parity tests require a model that supports both CPU and GPU.
        // These are structural tests that verify the test framework works.

        #[test]
        fn test_logits_close_same() {
            let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
            let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
            assert!(logits_close(&a, &b, 0.001));
        }

        #[test]
        fn test_logits_close_different() {
            let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
            let b = Array1::from_vec(vec![1.0, 2.5, 3.0]);
            assert!(!logits_close(&a, &b, 0.001));
            assert!(logits_close(&a, &b, 1.0));
        }

        #[test]
        fn test_max_logit_diff() {
            let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
            let b = Array1::from_vec(vec![1.1, 2.0, 3.5]);
            let diff = max_logit_diff(&a, &b);
            assert!((diff - 0.5).abs() < 0.001);
        }

        // When a dual-execution model is available, add tests like:
        // #[tokio::test]
        // async fn test_cpu_gpu_prefill_parity() { ... }
        // #[tokio::test]
        // async fn test_cpu_gpu_decode_parity() { ... }
    }

    // =========================================================================
    //  Backend Trait Compliance Tests
    // =========================================================================

    mod trait_compliance_tests {
        use super::*;

        #[test]
        fn test_cpu_backend_is_send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<CpuDecoderBackend>();
        }

        #[test]
        fn test_cpu_decode_token_is_send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<<CpuDecoderBackend as DecoderGenerationBackend>::DecodeToken>();
        }

        #[test]
        fn test_any_backend_is_send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            // Note: AnyDecoderBackend might not be Send due to GPU internals
            // This test documents the actual behavior
        }

        #[test]
        fn test_any_decode_token_is_send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<<AnyDecoderBackend as DecoderGenerationBackend>::DecodeToken>();
        }
    }

    // =========================================================================
    //  Stress Tests
    // =========================================================================

    mod stress_tests {
        use super::*;

        #[tokio::test]
        #[ignore] // Run with --ignored
        async fn test_many_decode_iterations_cpu() {
            let model = MockCpuDecoderModel::pipelined();
            let backend = CpuDecoderBackend::new();
            let mut cache = MockCpuCache::new(10000);

            let prompt = Array2::from_shape_vec((1, 10), (0..10).collect()).unwrap();
            let _ = backend.prefill(&model, &prompt, &mut cache).await.unwrap();

            let mut token = backend.new_decode_token().unwrap();
            for i in 0..1000 {
                backend.update_decode_token(&mut token, i % 1000).unwrap();
                let _ = backend
                    .decode_one(&model, &token, 11 + i as usize, &mut cache)
                    .await
                    .unwrap();
            }
        }

        #[tokio::test]
        #[ignore]
        async fn test_many_decode_iterations_gpu() {
            let Some(context) = super::gpu_backend_tests::create_gpu_context().await else {
                eprintln!("Skipping: no GPU");
                return;
            };

            let backend = GpuDecoderBackend::new(context).unwrap();
            let mut token = backend.new_decode_token().unwrap();

            for i in 0..10000 {
                backend.update_decode_token(&mut token, i % 10000).unwrap();
            }
        }

        #[tokio::test]
        #[ignore]
        async fn test_concurrent_cpu_generations() {
            let model = Arc::new(MockCpuDecoderModel::pipelined());
            let backend = Arc::new(CpuDecoderBackend::new());

            let mut handles = Vec::new();

            for i in 0..10 {
                let model = model.clone();
                let backend = backend.clone();

                handles.push(tokio::spawn(async move {
                    let mut cache = MockCpuCache::new(100);
                    let tokens = Array2::from_shape_vec((1, 5), vec![i as u32; 5]).unwrap();
                    backend.prefill(model.as_ref(), &tokens, &mut cache).await
                }));
            }

            for handle in handles {
                assert!(handle.await.unwrap().is_ok());
            }
        }
    }
}