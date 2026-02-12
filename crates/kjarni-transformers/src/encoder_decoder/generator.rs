use std::any::Any;

use anyhow::{Result, anyhow};
use async_stream::try_stream;
use async_trait::async_trait;
use futures::{
    pin_mut,
    stream::{Stream, StreamExt},
};
use ndarray::Array3;

use crate::cache::Cache;
use crate::common::{DecodingStrategy, GenerationConfig, StreamedToken};
use crate::encoder_decoder::cpu_backend::{self, CpuBackend};
use crate::encoder_decoder::traits::{
    EncoderDecoderGenerationBackend, EncoderDecoderLanguageModel,
};
use crate::encoder_decoder::{run_beam_search, run_beam_search_stream};
use crate::gpu::{GpuEncoderDecoderBackend, GpuSeq2SeqState};
use crate::prelude::*;

#[derive(Debug)]
pub enum AnyEncoderDecoderBackend {
    Cpu(CpuBackend),
    Gpu(GpuEncoderDecoderBackend),
}

#[async_trait]
impl EncoderDecoderGenerationBackend for AnyEncoderDecoderBackend {
    type Tensor = Box<dyn Any + Send + Sync>;

    async fn encode(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        tokens: &[u32],
        num_beams: usize,
    ) -> Result<Self::Tensor> {
        match self {
            AnyEncoderDecoderBackend::Cpu(b) => {
                let tensor = b.encode(model, tokens, num_beams).await?;
                Ok(Box::new(tensor))
            }
            AnyEncoderDecoderBackend::Gpu(b) => {
                let tensor = b.encode(model, tokens, num_beams).await?;
                Ok(Box::new(tensor))
            }
        }
    }

    async fn decode_step(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        decoder_tokens: &Self::Tensor,
        encoder_state: &Self::Tensor,
        cache: &mut dyn Cache,
    ) -> Result<Array3<f32>> {
        match self {
            AnyEncoderDecoderBackend::Cpu(b) => {
                let tokens = decoder_tokens
                    .downcast_ref::<cpu_backend::CpuSeq2SeqState>()
                    .ok_or_else(|| anyhow!("mismatched tensor type for cpu backend"))?;
                let state = encoder_state
                    .downcast_ref::<cpu_backend::CpuSeq2SeqState>()
                    .ok_or_else(|| anyhow!("mismatched tensor type for cpu backend"))?;
                b.decode_step(model, tokens, state, cache).await
            }
            AnyEncoderDecoderBackend::Gpu(b) => {
                let tokens = decoder_tokens
                    .downcast_ref::<GpuSeq2SeqState>()
                    .ok_or_else(|| anyhow!("mismatched tensor type for gpu backend"))?;
                let state: &crate::gpu::GpuSeq2SeqState = encoder_state
                    .downcast_ref::<GpuSeq2SeqState>()
                    .ok_or_else(|| anyhow!("mismatched tensor type for gpu backend"))?;
                b.decode_step(model, tokens, state, cache).await
            }
        }
    }

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor> {
        match self {
            AnyEncoderDecoderBackend::Cpu(b) => {
                let tensor = b.create_token_tensor(tokens, num_beams)?;
                Ok(Box::new(tensor))
            }
            AnyEncoderDecoderBackend::Gpu(b) => {
                let tensor = b.create_token_tensor(tokens, num_beams)?;
                Ok(Box::new(tensor))
            }
        }
    }

    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()> {
        match self {
            AnyEncoderDecoderBackend::Cpu(b) => {
                let concrete_tensor = tensor
                    .downcast_mut::<cpu_backend::CpuSeq2SeqState>()
                    .ok_or_else(|| anyhow!("mismatched tensor type for cpu backend"))?;
                b.update_token_tensor(concrete_tensor, new_tokens)
            }
            AnyEncoderDecoderBackend::Gpu(b) => {
                let concrete_tensor = tensor
                    .downcast_mut::<GpuSeq2SeqState>()
                    .ok_or_else(|| anyhow!("mismatched tensor type for gpu backend"))?;
                b.update_token_tensor(concrete_tensor, new_tokens)
            }
        }
    }

    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()> {
        match self {
            AnyEncoderDecoderBackend::Cpu(b) => b.reorder_cache(cache, indices),
            AnyEncoderDecoderBackend::Gpu(b) => b.reorder_cache(cache, indices),
        }
    }
}

#[derive(Debug)]
pub struct EncoderDecoderGenerator {
    pub model: Box<dyn EncoderDecoderLanguageModel>,
    backend: AnyEncoderDecoderBackend,
}

impl std::fmt::Debug for Box<dyn EncoderDecoderLanguageModel> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EncoderDecoder")
    }
}

impl EncoderDecoderGenerator {
    pub fn new(model: Box<dyn EncoderDecoderLanguageModel>) -> Result<Self> {
        let backend = match model.device() {
            Device::Cpu => AnyEncoderDecoderBackend::Cpu(CpuBackend),
            Device::Wgpu => {
                let context = model
                    .context()
                    .ok_or_else(|| anyhow!("gpu model missing WgpuContext"))?;
                AnyEncoderDecoderBackend::Gpu(GpuEncoderDecoderBackend::new(context)?)
            }
        };
        Ok(Self { model, backend })
    }

    pub async fn generate(
        &self,
        input_text: &str,
        config: Option<&GenerationConfig>,
    ) -> Result<String> {
        let t_start = std::time::Instant::now();
        let generation_config = config
            .cloned()
            .unwrap_or_else(|| self.model.get_generation_config_for_input(input_text));

        let result = run_beam_search(
            self.model.as_ref(),
            &self.backend,
            input_text,
            &generation_config,
        )
        .await;

        let elapsed = t_start.elapsed();
        if let Ok(ref text) = result {
            let num_tokens = self
                .model
                .tokenizer()
                .encode(text.as_str(), false)
                .map_or(0, |e| e.len());
            if num_tokens > 0 && elapsed.as_secs_f32() > 0.0 {
                let tps = num_tokens as f32 / elapsed.as_secs_f32();
                log::info!(
                    "seq2seq generated {} tokens in {:?}, {:.2} t/s",
                    num_tokens,
                    elapsed,
                    tps
                );
            } else {
                log::info!("seq2seq generation time: {:?}", elapsed);
            }
        } else {
            log::info!("seq2seq generation failed in {:?}", elapsed);
        }

        result
    }

    pub fn generate_stream<'a>(
        &'a self,
        input_text: &'a str,
        config: Option<&GenerationConfig>,
    ) -> impl Stream<Item = Result<StreamedToken>> + 'a {
        let owned_config =
            config.map_or_else(|| self.model.get_default_generation_config(), |c| c.clone());

        try_stream! {
            if let DecodingStrategy::BeamSearch(params) = &owned_config.strategy {
                if params.num_beams > 1 {
                    log::warn!(
                        "streaming with beam search enabled, output may differ from non-streaming generate"
                    );
                }
            }

            let stream = run_beam_search_stream(
                self.model.as_ref(),
                &self.backend,
                input_text,
                &owned_config,
            );

            let t_start = std::time::Instant::now();
            let mut token_count = 0;

            pin_mut!(stream);
            while let Some(token) = StreamExt::next(&mut stream).await {
                token_count += 1;
                let elapsed = t_start.elapsed();
                if elapsed.as_secs_f32() > 0.0 {
                    let tps = token_count as f32 / elapsed.as_secs_f32();
                    log::debug!("stream token #{}, {:.2} t/s", token_count, tps);
                }
                yield token?;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::collections::HashSet;
    use std::sync::Arc;

    use async_trait::async_trait;
    use ndarray::Array4;
    use tokenizers::Tokenizer;

    use super::*;
    use crate::cache::Cache;
    use crate::cpu::encoder::traits::EncoderLanguageModel;
    use crate::cpu::encoder::{CpuEncoderOps, GpuEncoderOps};
    use crate::encoder_decoder::traits::{CpuEncoderDecoderOps, GpuEncoderDecoderOps};
    use crate::traits::{Device, InferenceModel};

    struct MockModel {
        device: Device,
    }

    impl InferenceModel for MockModel {
        fn device(&self) -> Device {
            self.device
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
            10
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
            HashSet::default()
        }
        fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Cache>> {
            unimplemented!()
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
            0
        }
        fn get_default_generation_config(&self) -> GenerationConfig {
            GenerationConfig::default()
        }
    }

    #[test]
    fn test_generator_new_cpu() {
        let model = MockModel {
            device: Device::Cpu,
        };
        let generator = EncoderDecoderGenerator::new(Box::new(model));
        assert!(generator.is_ok());
    }

    #[tokio::test]
    async fn test_generator_new_gpu_missing_context() {
        let model = MockModel {
            device: Device::Wgpu,
        };
        let generator = EncoderDecoderGenerator::new(Box::new(model));
        assert!(generator.is_err());
        assert!(
            generator
                .unwrap_err()
                .to_string()
                .contains("gpu model missing")
        );
    }

    #[test]
    fn test_backend_dispatch_cpu() {
        let backend =
            AnyEncoderDecoderBackend::Cpu(crate::encoder_decoder::cpu_backend::CpuBackend);

        let tokens = vec![1, 2, 3];
        let tensor = backend.create_token_tensor(&tokens, 1);
        assert!(tensor.is_ok());

        let boxed = tensor.unwrap();
        let concrete = boxed.downcast_ref::<crate::encoder_decoder::cpu_backend::CpuSeq2SeqState>();
        assert!(concrete.is_some());
    }

    #[test]
    fn test_backend_mismatch_error() {
        let backend =
            AnyEncoderDecoderBackend::Cpu(crate::encoder_decoder::cpu_backend::CpuBackend);

        let mut fake_tensor: Box<dyn Any + Send + Sync> = Box::new(String::from("fake"));

        let res = backend.update_token_tensor(&mut fake_tensor, &[1]);
        assert!(res.is_err());
        assert!(
            res.unwrap_err()
                .to_string()
                .contains("mismatched tensor type")
        );
    }

    #[test]
    fn test_any_backend_debug_cpu() {
        let backend = AnyEncoderDecoderBackend::Cpu(CpuBackend);
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("Cpu"));
    }

    #[test]
    fn test_any_backend_create_token_tensor_cpu() {
        let backend = AnyEncoderDecoderBackend::Cpu(CpuBackend);

        let tokens = vec![1u32, 2, 3, 4];
        let tensor = backend.create_token_tensor(&tokens, 2).unwrap();

        // Verify it's the right type
        let concrete = tensor.downcast_ref::<cpu_backend::CpuSeq2SeqState>();
        assert!(concrete.is_some());
    }

    #[test]
    fn test_any_backend_update_token_tensor_cpu() {
        let backend = AnyEncoderDecoderBackend::Cpu(CpuBackend);

        let tokens = vec![1u32, 2];
        let mut tensor = backend.create_token_tensor(&tokens, 1).unwrap();

        let new_tokens = vec![10u32];
        let result = backend.update_token_tensor(&mut tensor, &new_tokens);
        assert!(result.is_ok());
    }

    #[test]
    fn test_any_backend_update_token_tensor_wrong_type_cpu() {
        let backend = AnyEncoderDecoderBackend::Cpu(CpuBackend);

        // Create a tensor of wrong type
        let mut fake_tensor: Box<dyn Any + Send + Sync> = Box::new(42i32);

        let result = backend.update_token_tensor(&mut fake_tensor, &[1]);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("mismatched tensor type")
        );
    }

    #[test]
    fn test_any_backend_reorder_cache_cpu() {
        let backend = AnyEncoderDecoderBackend::Cpu(CpuBackend);

        let num_layers = 2;
        let num_heads = 4;
        let max_seq = 128;
        let head_dim = 64;
        let batch_size = 4;

        let mut cache = crate::cache::CpuBeamKVCache::new(num_layers, num_heads, max_seq, head_dim);
        let seq_len = 1;
        for layer in 0..num_layers {
            let k = Array3::<f32>::ones((batch_size, num_heads, head_dim));
            let v = Array3::<f32>::ones((batch_size, num_heads, head_dim));
            cache.update(layer, &k, &v);
        }
        cache.increment_len(seq_len);

        assert_eq!(
            cache.get_seq_length(),
            1,
            "Cache should have seq_length=1 after update"
        );

        let indices = vec![1, 0, 2, 3];
        let result = backend.reorder_cache(&mut cache, &indices);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generator_debug() {
        let model = MockModel {
            device: Device::Cpu,
        };
        let generator = EncoderDecoderGenerator::new(Box::new(model)).unwrap();

        let debug_str = format!("{:?}", generator);
        assert!(debug_str.contains("EncoderDecoderGenerator"));
    }

    #[test]
    fn test_boxed_model_debug() {
        let model: Box<dyn EncoderDecoderLanguageModel> = Box::new(MockModel {
            device: Device::Cpu,
        });
        let debug_str = format!("{:?}", model);
        assert_eq!(debug_str, "EncoderDecoder");
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
        fn increment_len(&mut self, n: usize) {
            self.len += n;
        }
        fn clone_box(&self) -> Box<dyn Cache> {
            Box::new(self.clone())
        }
    }

    #[tokio::test]
    async fn test_any_backend_decode_step_wrong_decoder_tokens_type() {
        let backend = AnyEncoderDecoderBackend::Cpu(CpuBackend);
        let model = MockModel {
            device: Device::Cpu,
        };

        let decoder_tokens: Box<dyn Any + Send + Sync> = Box::new("wrong type");
        let encoder_state: Box<dyn Any + Send + Sync> = Box::new(
            cpu_backend::CpuSeq2SeqState::U32(ndarray::Array2::zeros((1, 1))),
        );

        let mut cache = MockCache { len: 0 };

        let result = backend
            .decode_step(&model, &decoder_tokens, &encoder_state, &mut cache)
            .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("mismatched tensor type")
        );
    }

    #[tokio::test]
    async fn test_any_backend_decode_step_wrong_encoder_state_type() {
        let backend = AnyEncoderDecoderBackend::Cpu(CpuBackend);
        let model = MockModel {
            device: Device::Cpu,
        };

        let decoder_tokens: Box<dyn Any + Send + Sync> = Box::new(
            cpu_backend::CpuSeq2SeqState::U32(ndarray::Array2::zeros((1, 1))),
        );
        let encoder_state: Box<dyn Any + Send + Sync> = Box::new("wrong type");

        let mut cache = MockCache { len: 0 };

        let result = backend
            .decode_step(&model, &decoder_tokens, &encoder_state, &mut cache)
            .await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("mismatched tensor type")
        );
    }

    #[tokio::test]
    async fn test_any_backend_encode_cpu_returns_boxed() {
        let backend = AnyEncoderDecoderBackend::Cpu(CpuBackend);
        assert!(matches!(backend, AnyEncoderDecoderBackend::Cpu(_)));
    }

    struct MockModelWithContext {
        device: Device,
        context: Option<Arc<WgpuContext>>,
    }

    impl InferenceModel for MockModelWithContext {
        fn device(&self) -> Device {
            self.device
        }
        fn context(&self) -> Option<Arc<WgpuContext>> {
            self.context.clone()
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl LanguageModel for MockModelWithContext {
        fn vocab_size(&self) -> usize {
            100
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
            HashSet::default()
        }
        fn new_cache(&self, _: usize, _: usize, _: usize) -> Result<Box<dyn Cache>> {
            unimplemented!()
        }
    }

    #[async_trait]
    impl EncoderLanguageModel for MockModelWithContext {
        fn encoder_cpu_ops(&self) -> Option<&dyn CpuEncoderOps> {
            None
        }
        fn encoder_gpu_ops(&self) -> Option<&dyn GpuEncoderOps> {
            None
        }
    }

    #[async_trait]
    impl EncoderDecoderLanguageModel for MockModelWithContext {
        fn encoder_decoder_cpu_ops(&self) -> Option<&dyn CpuEncoderDecoderOps> {
            None
        }
        fn encoder_decoder_gpu_ops(&self) -> Option<&dyn GpuEncoderDecoderOps> {
            None
        }
        fn decoder_start_token_id(&self) -> u32 {
            0
        }
        fn get_default_generation_config(&self) -> GenerationConfig {
            GenerationConfig::default()
        }
    }

    #[tokio::test]
    async fn test_generator_new_gpu_with_context() {
        let context = WgpuContext::new().await.expect("Failed to create context");

        let model = MockModelWithContext {
            device: Device::Wgpu,
            context: Some(context),
        };

        let generator = EncoderDecoderGenerator::new(Box::new(model));
        assert!(generator.is_ok());
        match &generator.unwrap().backend {
            AnyEncoderDecoderBackend::Gpu(_) => {}
            _ => panic!("Expected GPU backend"),
        }
    }

    #[test]
    fn test_any_backend_create_tensor_multiple_beams() {
        let backend = AnyEncoderDecoderBackend::Cpu(CpuBackend);

        // 4 beams, 3 tokens each
        let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let tensor = backend.create_token_tensor(&tokens, 4).unwrap();

        let concrete = tensor
            .downcast_ref::<cpu_backend::CpuSeq2SeqState>()
            .unwrap();
        match concrete {
            cpu_backend::CpuSeq2SeqState::U32(arr) => {
                assert_eq!(arr.shape(), &[4, 3]);
            }
            _ => panic!("Expected U32 state"),
        }
    }
    struct WrongCacheType;

    impl Cache for WrongCacheType {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn get_seq_length(&self) -> usize {
            0
        }
        fn set_seq_length(&mut self, _: usize) {}
        fn clear(&mut self) {}
        fn increment_len(&mut self, _: usize) {}
        fn clone_box(&self) -> Box<dyn Cache> {
            Box::new(WrongCacheType)
        }
    }

    #[test]
    fn test_any_backend_reorder_cache_wrong_cache_type() {
        let backend = AnyEncoderDecoderBackend::Cpu(CpuBackend);

        let mut cache = WrongCacheType;
        let indices = vec![0, 1];

        let result = backend.reorder_cache(&mut cache, &indices);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("CpuBeamKVCache"));
    }

    #[test]
    fn test_any_backend_create_tensor_empty_tokens() {
        let backend = AnyEncoderDecoderBackend::Cpu(CpuBackend);

        let tokens: Vec<u32> = vec![];
        let tensor = backend.create_token_tensor(&tokens, 1);

        assert!(tensor.is_ok());
    }

    #[test]
    fn test_any_backend_update_empty_tokens() {
        let backend = AnyEncoderDecoderBackend::Cpu(CpuBackend);

        let tokens = vec![1u32, 2];
        let mut tensor = backend.create_token_tensor(&tokens, 1).unwrap();
        let result = backend.update_token_tensor(&mut tensor, &[]);
        let _ = result;
    }
}
