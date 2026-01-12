use crate::activations::Activation;
use crate::cpu::decoder::{CpuDecoderBackend, CpuRoPEDecoderLayer};
use crate::decoder::traits::{
    CpuDecoder, CpuDecoderOps, DecoderGenerationBackend, DecoderLanguageModel, GpuDecoderOps,
};
use crate::linear_layer::LinearLayer;
use crate::models::base::{AutoregressiveLoop, ModelInput};
use crate::normalization::RMSNorm;
use crate::rope::RoPE;
use crate::traits::InferenceModel;
use crate::{Cache, Device, LanguageModel, Normalization, WgpuContext};
use crate::{cpu::decoder::DecoderAttention, feedforward::SwiGluFeedForward};
use anyhow::Result;
use ndarray::{Array1, Array2, Array3};
use std::any::Any;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokenizers::Tokenizer;
// =========================================================================
//  Mocks
// =========================================================================

#[cfg(test)]
mod decoder_backend_test {
    use super::*;
    struct MockCache;
    impl Cache for MockCache {
        fn get_seq_length(&self) -> usize {
            0
        }
        fn set_seq_length(&mut self, _: usize) {}
        fn increment_len(&mut self, _: usize) {}
        fn clear(&mut self) {}
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn clone_box(&self) -> Box<dyn Cache> {
            Box::new(MockCache)
        }
    }

    struct MockDecoder;
    impl CpuDecoder for MockDecoder {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn num_layers(&self) -> usize {
            1
        }

        // We only care that forward is called. Return dummy data.
        fn forward(
            &self,
            input: ModelInput<'_>,
            _mask: &Array2<f32>,
            _pos: usize,
            _cache: Option<&mut dyn Cache>,
        ) -> Result<Array3<f32>> {
            let (b, s) = match input {
                ModelInput::TokensCpu(t) => (t.shape()[0], t.shape()[1]),
                _ => (1, 1),
            };
            Ok(Array3::zeros((b, s, 10))) // Hidden size 10
        }

        // Unused in backend tests
        fn forward_layers(
            &self,
            _: &Array3<f32>,
            _: &Array2<f32>,
            _: usize,
            _: Option<&mut dyn Cache>,
            _: usize,
            _: usize,
        ) -> Result<Array3<f32>> {
            unimplemented!()
        }

        fn embed(
            &self,
            input: ModelInput<'_>,
            position_offset: usize,
        ) -> anyhow::Result<Array3<f32>> {
            todo!()
        }

        fn embed_and_normalize(
            &self,
            input: ModelInput<'_>,
            position_offset: usize,
        ) -> anyhow::Result<Array3<f32>> {
            todo!()
        }
    }

    // The Ops Mock captures calls to verify logic
    struct MockOps {
        loop_type: AutoregressiveLoop,
        forward_calls: AtomicUsize,
        embed_calls: AtomicUsize,
    }

    impl MockOps {
        fn new(loop_type: AutoregressiveLoop) -> Self {
            Self {
                loop_type,
                forward_calls: AtomicUsize::new(0),
                embed_calls: AtomicUsize::new(0),
            }
        }
    }

    // We implement a dummy DecoderLanguageModel just to return our MockOps
    struct MockModel {
        ops: MockOps,
    }

    // 2. Implement InferenceModel
    impl InferenceModel for MockModel {
        fn device(&self) -> Device {
            crate::traits::Device::Cpu
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
            100
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
            100
        }
        fn tokenizer(&self) -> &Tokenizer {
            unimplemented!()
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
            Ok(Box::new(MockCache))
        }
    }
    impl DecoderLanguageModel for MockModel {
        fn autoregressive_loop(&self) -> AutoregressiveLoop {
            self.ops.loop_type
        }
        fn decoder_cpu_ops(&self) -> Option<&dyn CpuDecoderOps> {
            Some(&self.ops)
        }

        // Unused

        fn chat_template(&self) -> Option<&dyn crate::ChatTemplate> {
            None
        }
        fn is_instruct_model(&self) -> bool {
            false
        }

        #[doc = " Access GPU operations strategy. Returns `None` if model is CPU-only."]
        fn decoder_gpu_ops(&self) -> Option<&dyn GpuDecoderOps> {
            todo!()
        }
    }

    impl CpuDecoderOps for MockOps {
        fn decoder(&self) -> &dyn CpuDecoder {
            // Count forward calls implicitly via this accessor? No, we need to spy on forward.
            // But since CpuDecoderBackend calls ops.decoder().forward(), we can't easily spy unless
            // we wrap the decoder.
            // Actually, we can count inside project_to_logits since that happens after forward.
            &MockDecoder
        }

        fn embed(&self, _tokens: &[u32], _pos: usize) -> Result<Array3<f32>> {
            self.embed_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Array3::zeros((1, _tokens.len(), 10)))
        }

        fn project_to_logits(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
            self.forward_calls.fetch_add(1, Ordering::SeqCst);
            // Project hidden [1, seq, 10] -> logits [1, seq, vocab=5]
            let (b, s, _) = hidden.dim();
            Ok(Array3::zeros((b, s, 5)))
        }

        fn get_attention_mask(&self, seq_len: usize, _past: usize) -> Result<Array2<f32>> {
            Ok(Array2::zeros((1, seq_len)))
        }
    }

    // =========================================================================
    //  Tests
    // =========================================================================

    #[test]
    fn test_prime_and_update_tokens() {
        let backend = CpuDecoderBackend;

        // 1. Prime
        let tokens = vec![1, 2, 3];
        let tensor = backend.prime_tokens(&tokens).unwrap();
        assert_eq!(tensor.shape(), &[1, 3]);
        assert_eq!(tensor[[0, 0]], 1);
        assert_eq!(tensor[[0, 2]], 3);

        // 2. New Token Tensor
        let mut step_tensor = backend.new_token_tensor().unwrap();
        assert_eq!(step_tensor.shape(), &[1, 1]);
        assert_eq!(step_tensor[[0, 0]], 0); // Init zero

        // 3. Update
        backend.update_token_tensor(&mut step_tensor, 99).unwrap();
        assert_eq!(step_tensor[[0, 0]], 99);
    }

    #[tokio::test]
    async fn test_prefill_pipelined() -> Result<()> {
        let model = MockModel {
            ops: MockOps::new(AutoregressiveLoop::Pipelined),
        };
        let backend = CpuDecoderBackend;
        let mut cache = MockCache;
        let tokens = vec![1, 2, 3, 4];

        let logits = backend.prefill(&model, &tokens, &mut cache).await?;

        // Verify logic:
        // Pipelined should call forward (project_to_logits) EXACTLY ONCE
        assert_eq!(model.ops.forward_calls.load(Ordering::SeqCst), 1);
        // Should call embed ONCE (if using embed architecture, otherwise 0 if using Tokens)
        // Note: Your current implementation uses ModelInput::from_tokens, skipping embed() call on Ops.

        // Output shape should be [vocab_size=5]
        assert_eq!(logits.shape(), &[5]);

        Ok(())
    }

    #[tokio::test]
    async fn test_prefill_legacy() -> Result<()> {
        let model = MockModel {
            ops: MockOps::new(AutoregressiveLoop::Legacy),
        };
        let backend = CpuDecoderBackend;
        let mut cache = MockCache;
        let tokens = vec![1, 2, 3, 4];

        let logits = backend.prefill(&model, &tokens, &mut cache).await?;

        // Legacy should call forward TWICE:
        // 1. Cache fill (tokens 0..3) -> No projection (usually)
        // 2. Last token (token 3) -> Projection
        // Wait, your implementation calls project_to_logits only on the 2nd pass?
        // prefill_legacy:
        //   Phase 1: ops.decoder().forward(...) -> Result ignored? No project called.
        //   Phase 2: ops.decoder().forward(...) -> project called.

        // So project_to_logits count should be 1.
        assert_eq!(model.ops.forward_calls.load(Ordering::SeqCst), 1);

        // Verify output shape
        assert_eq!(logits.shape(), &[5]);

        Ok(())
    }

    #[tokio::test]
    async fn test_decode_one() -> Result<()> {
        let model = MockModel {
            ops: MockOps::new(AutoregressiveLoop::Pipelined),
        };
        let backend = CpuDecoderBackend;
        let mut cache = MockCache;

        // Token tensor [1, 1] with ID 50
        let token_tensor = Array2::from_elem((1, 1), 50u32);

        let logits = backend
            .decode_one(&model, &token_tensor, 10, &mut cache)
            .await?;

        // Should call forward once
        assert_eq!(model.ops.forward_calls.load(Ordering::SeqCst), 1);
        assert_eq!(logits.shape(), &[5]);

        Ok(())
    }

    #[tokio::test]
    async fn test_prefill_empty_error() {
        let model = MockModel {
            ops: MockOps::new(AutoregressiveLoop::Pipelined),
        };
        let backend = CpuDecoderBackend;
        let mut cache = MockCache;

        let err = backend.prefill(&model, &[], &mut cache).await;
        assert!(err.is_err());
        assert_eq!(
            err.unwrap_err().to_string(),
            "Cannot prefill with empty prompt"
        );
    }


    
}
