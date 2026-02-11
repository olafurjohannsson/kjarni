//! traits for Encoder-Decoder 
use crate::cache::Cache;
use crate::common::GenerationConfig;
use crate::cpu::encoder::prelude::EncoderLanguageModel;
use crate::cpu::encoder_decoder::CrossDecoderLayer;
use crate::gpu::{GpuFrameContext, GpuTensor, GpuTensorPool};
use crate::gpu_ops::blocks::layers::GpuCrossDecoderLayer;
use crate::models::base::ModelInput;
use crate::pipeline::EncoderDecoderPipeline;
use anyhow::Result;
use async_trait::async_trait;
use ndarray::{Array2, Array3, Array4};
use wgpu::CommandEncoder;

/// A container for the pre-computed cross-attention Key/Value cache on the CPU.
#[derive(Debug, Default)]
pub struct CpuCrossAttentionKVCache(pub Vec<(Array4<f32>, Array4<f32>)>);

/// A container for the pre-computed cross-attention Key/Value cache on the GPU.
#[derive(Debug, Default)]
pub struct GpuCrossAttentionKVCache(pub Vec<(GpuTensor, GpuTensor)>);

/// The output of a single step from a `CpuCrossDecoder`.
pub struct CpuCrossDecoderOutput {
    /// The final hidden states from the decoder stack. Shape: `[batch, seq, hidden]`.
    pub last_hidden_state: Array3<f32>,
    pub new_self_attn_kv: Vec<(Array3<f32>, Array3<f32>)>,
}

/// The output of a single step from a `GpuCrossDecoder`.
pub struct GpuCrossDecoderOutput {
    /// The final hidden states on the GPU. Shape: `[batch, seq, hidden]`.
    pub last_hidden_state: GpuTensor,
    pub new_self_attn_kv: Vec<(GpuTensor, GpuTensor)>,
}

#[async_trait]
pub trait CpuCrossDecoder: Send + Sync {
    #[deprecated]
    fn embed(&self, decoder_input_ids: &Array2<u32>, position_offset: usize) -> Array3<f32> {
        unimplemented!()
    }

    #[deprecated]
    fn embed_and_normalize(
        &self,
        input_ids: &Array2<u32>,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        unimplemented!()
    }
    #[deprecated]
    fn forward(
        &self,
        decoder_input_ids: &Array2<u32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_padding_mask: Option<&Array2<f32>>, 
        encoder_padding_mask: Option<&Array2<f32>>, 
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
    ) -> Result<CpuCrossDecoderOutput> {
        unimplemented!()
    }

    fn embed_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>>;
    fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>>;
    fn precompute_cross_attention_kv(
        &self,
        encoder_hidden_states: &Array3<f32>,
    ) -> Result<CpuCrossAttentionKVCache>;
    fn layers(&self) -> &Vec<CrossDecoderLayer>;
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_padding_mask: Option<&Array2<f32>>,
        encoder_padding_mask: Option<&Array2<f32>>,
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<CpuCrossDecoderOutput>;

    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
}

#[async_trait]
pub trait GpuCrossDecoder: Send + Sync {
    #[deprecated]
    fn embed(
        &self,
        _encoder: &mut CommandEncoder,
        _pool: &mut GpuTensorPool,
        _input: ModelInput<'_>,
        _position_offset: usize,
    ) -> Result<GpuTensor> {
        unimplemented!("Use GpuCrossDecoderOps::embed_tokens() instead")
    }

    #[deprecated]
    fn embed_and_normalize(
        &self,
        _encoder: &mut CommandEncoder,
        _pool: &mut GpuTensorPool,
        _input: ModelInput<'_>,
        _position_offset: usize,
    ) -> Result<GpuTensor> {
        unimplemented!("Use GpuCrossDecoderOps::embed_tokens() then decoder.embed_norm() instead")
    }

    #[deprecated]
    fn forward(
        &self,
        _encoder: &mut CommandEncoder,
        _pool: &mut GpuTensorPool,
        _decoder_input: ModelInput<'_>,
        _encoder_hidden_states: &GpuTensor,
        _decoder_attention_mask: &GpuTensor,
        _cache: Option<&mut dyn Cache>,
        _cross_kv_cache: Option<&GpuCrossAttentionKVCache>,
    ) -> Result<GpuCrossDecoderOutput> {
        unimplemented!("Use GpuCrossDecoderOps::forward_tokens() instead")
    }

    fn embed_norm(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor>;

    fn final_norm(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor>;

    fn precompute_cross_attention_kv(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        encoder_hidden_states: &GpuTensor,
    ) -> Result<GpuCrossAttentionKVCache>;

    fn layers(&self) -> &Vec<GpuCrossDecoderLayer>;

    fn forward_layers(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        encoder_hidden_states: &GpuTensor,
        decoder_attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&GpuCrossAttentionKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuCrossDecoderOutput>;

    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn as_any(&self) -> &dyn std::any::Any;
}

pub trait CpuEncoderDecoderOps: Send + Sync {
    fn decoder(&self) -> &dyn CpuCrossDecoder;
    fn embed_decoder_tokens(
        &self,
        input_ids: &Array2<u32>,
        position_offset: usize,
    ) -> Result<Array3<f32>>;
    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>>;
    fn broadcast_encoder_states(
        &self,
        encoder_hidden_states: &Array3<f32>,
        num_beams: usize,
    ) -> Result<Array3<f32>>;
    fn get_decoder_mask(&self, seq_len: usize, past_len: usize) -> Option<Array2<f32>>;
}

pub trait GpuEncoderDecoderOps: Send + Sync {
    fn decoder(&self) -> &dyn GpuCrossDecoder;
    fn embed_decoder_tokens(
        &self,
        encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor>;
    fn project_to_logits(
        &self,
        frame: &mut GpuFrameContext,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor>;
    fn broadcast_encoder_states(
        &self,
        frame: &mut GpuFrameContext,
        encoder_hidden_states: &GpuTensor,
        num_beams: usize,
    ) -> Result<GpuTensor>;
}

#[async_trait]
pub trait EncoderDecoderLanguageModel: EncoderLanguageModel {
    fn get_pipeline(&self) -> &EncoderDecoderPipeline {
        unimplemented!()
    }
    /// Provides access to the CPU-specific operations for the combined model.
    fn encoder_decoder_cpu_ops(&self) -> Option<&dyn CpuEncoderDecoderOps>;
    /// Provides access to the GPU-specific operations for the combined model.
    fn encoder_decoder_gpu_ops(&self) -> Option<&dyn GpuEncoderDecoderOps>;

    /// The token ID that should be used to start the decoding process.
    fn decoder_start_token_id(&self) -> u32;
    /// Returns the default generation configuration for this model.
    fn get_default_generation_config(&self) -> GenerationConfig;

    fn get_generation_config_for_input(&self, _input: &str) -> GenerationConfig {
        self.get_default_generation_config()
    }
}

#[async_trait]
pub trait EncoderDecoderGenerationBackend: Send + Sync {
    type Tensor: Send + Sync;

    async fn encode(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        tokens: &[u32],
        num_beams: usize,
    ) -> Result<Self::Tensor>;

    async fn decode_step(
        &self,
        model: &dyn EncoderDecoderLanguageModel,
        decoder_tokens: &Self::Tensor,
        encoder_state: &Self::Tensor,
        cache: &mut dyn Cache,
    ) -> Result<Array3<f32>>;

    fn create_token_tensor(&self, tokens: &[u32], num_beams: usize) -> Result<Self::Tensor>;
    fn update_token_tensor(&self, tensor: &mut Self::Tensor, new_tokens: &[u32]) -> Result<()>;
    fn reorder_cache(&self, cache: &mut dyn Cache, indices: &[usize]) -> Result<()>;
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3, Array4};

    #[test]
    fn test_cpu_cross_attention_kv_cache_default() {
        let cache = CpuCrossAttentionKVCache::default();
        assert!(cache.0.is_empty());
    }

    #[test]
    fn test_cpu_cross_attention_kv_cache_with_data() {
        let k = Array4::<f32>::zeros((1, 2, 3, 4));
        let v = Array4::<f32>::zeros((1, 2, 3, 4));
        let cache = CpuCrossAttentionKVCache(vec![(k.clone(), v.clone())]);
        assert_eq!(cache.0.len(), 1);
        assert_eq!(cache.0[0].0.shape(), &[1, 2, 3, 4]);
        assert_eq!(cache.0[0].1.shape(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_gpu_cross_attention_kv_cache_default() {
        let cache = GpuCrossAttentionKVCache::default();
        assert!(cache.0.is_empty());
    }
    #[test]
    fn test_cpu_cross_decoder_output_creation() {
        let hidden = Array3::<f32>::zeros((1, 5, 64));
        let k = Array3::<f32>::zeros((1, 5, 64));
        let v = Array3::<f32>::zeros((1, 5, 64));
        
        let output = CpuCrossDecoderOutput {
            last_hidden_state: hidden.clone(),
            new_self_attn_kv: vec![(k, v)],
        };
        
        assert_eq!(output.last_hidden_state.shape(), &[1, 5, 64]);
        assert_eq!(output.new_self_attn_kv.len(), 1);
    }

    #[test]
    fn test_cpu_cross_decoder_output_multiple_layers() {
        let hidden = Array3::<f32>::zeros((2, 10, 128));
        let kvs: Vec<_> = (0..6)
            .map(|_| {
                (
                    Array3::<f32>::zeros((2, 10, 128)),
                    Array3::<f32>::zeros((2, 10, 128)),
                )
            })
            .collect();
        
        let output = CpuCrossDecoderOutput {
            last_hidden_state: hidden,
            new_self_attn_kv: kvs,
        };
        
        assert_eq!(output.new_self_attn_kv.len(), 6);
    }
    struct MockCpuCrossDecoder {
        layers: Vec<CrossDecoderLayer>,
        hidden_size: usize,
        has_embed_norm: bool,
        has_final_norm: bool,
    }

    impl MockCpuCrossDecoder {
        fn new(num_layers: usize, hidden_size: usize) -> Self {
            Self {
                layers: Vec::new(),
                hidden_size,
                has_embed_norm: true,
                has_final_norm: false,
            }
        }

        fn with_norms(mut self, embed_norm: bool, final_norm: bool) -> Self {
            self.has_embed_norm = embed_norm;
            self.has_final_norm = final_norm;
            self
        }
    }

    #[async_trait]
    impl CpuCrossDecoder for MockCpuCrossDecoder {
        fn embed_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
            if self.has_embed_norm {
                // Simple mock: just return clone (real impl would apply LayerNorm)
                Ok(hidden_states.clone())
            } else {
                Ok(hidden_states.clone())
            }
        }

        fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
            if self.has_final_norm {
                Ok(hidden_states.clone())
            } else {
                Ok(hidden_states.clone())
            }
        }

        fn precompute_cross_attention_kv(
            &self,
            encoder_hidden_states: &Array3<f32>,
        ) -> Result<CpuCrossAttentionKVCache> {
            // Mock: create empty cache
            Ok(CpuCrossAttentionKVCache::default())
        }

        fn layers(&self) -> &Vec<CrossDecoderLayer> {
            &self.layers
        }

        fn forward_layers(
            &self,
            hidden_states: &Array3<f32>,
            _encoder_hidden_states: &Array3<f32>,
            _decoder_padding_mask: Option<&Array2<f32>>,
            _encoder_padding_mask: Option<&Array2<f32>>,
            _cache: Option<&mut dyn Cache>,
            _cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
            _start_layer: usize,
            _end_layer: usize,
        ) -> Result<CpuCrossDecoderOutput> {
            Ok(CpuCrossDecoderOutput {
                last_hidden_state: hidden_states.clone(),
                new_self_attn_kv: vec![],
            })
        }

        fn num_layers(&self) -> usize {
            self.layers.len()
        }

        fn hidden_size(&self) -> usize {
            self.hidden_size
        }
    }

    #[test]
    fn test_mock_cpu_decoder_metadata() {
        let decoder = MockCpuCrossDecoder::new(6, 768);
        assert_eq!(decoder.num_layers(), 0); // Empty layers vec
        assert_eq!(decoder.hidden_size(), 768);
    }

    #[test]
    fn test_mock_cpu_decoder_embed_norm() {
        let decoder = MockCpuCrossDecoder::new(6, 64).with_norms(true, false);
        let input = Array3::<f32>::ones((1, 3, 64));
        
        let output = decoder.embed_norm(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_mock_cpu_decoder_final_norm() {
        let decoder = MockCpuCrossDecoder::new(6, 64).with_norms(false, true);
        let input = Array3::<f32>::ones((1, 3, 64));
        
        let output = decoder.final_norm(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_mock_cpu_decoder_precompute_cross_kv() {
        let decoder = MockCpuCrossDecoder::new(6, 64);
        let encoder_hidden = Array3::<f32>::ones((1, 10, 64));
        
        let cache = decoder.precompute_cross_attention_kv(&encoder_hidden).unwrap();
        assert!(cache.0.is_empty()); // Mock returns empty
    }

    #[test]
    fn test_mock_cpu_decoder_forward_layers() {
        let decoder = MockCpuCrossDecoder::new(6, 64);
        let hidden = Array3::<f32>::ones((1, 5, 64));
        let encoder_hidden = Array3::<f32>::ones((1, 10, 64));
        
        let output = decoder
            .forward_layers(&hidden, &encoder_hidden, None, None, None, None, 0, 6)
            .unwrap();
        
        assert_eq!(output.last_hidden_state.shape(), &[1, 5, 64]);
        assert!(output.new_self_attn_kv.is_empty());
    }

    #[test]
    #[should_panic(expected = "not implemented")]
    fn test_deprecated_embed_panics() {
        let decoder = MockCpuCrossDecoder::new(6, 64);
        let input = Array2::<u32>::zeros((1, 5));
        #[allow(deprecated)]
        let _ = decoder.embed(&input, 0);
    }

    #[test]
    #[should_panic(expected = "not implemented")]
    fn test_deprecated_embed_and_normalize_panics() {
        let decoder = MockCpuCrossDecoder::new(6, 64);
        let input = Array2::<u32>::zeros((1, 5));
        #[allow(deprecated)]
        let _ = decoder.embed_and_normalize(&input, 0);
    }

    #[test]
    #[should_panic(expected = "not implemented")]
    fn test_deprecated_forward_panics() {
        let decoder = MockCpuCrossDecoder::new(6, 64);
        let input = Array2::<u32>::zeros((1, 5));
        let encoder_hidden = Array3::<f32>::zeros((1, 10, 64));
        #[allow(deprecated)]
        let _ = decoder.forward(&input, &encoder_hidden, None, None, None, None);
    }
    #[test]
    fn test_bart_style_decoder_config() {
        // BART: has embed_norm, no final_norm
        let decoder = MockCpuCrossDecoder::new(6, 768).with_norms(true, false);
        assert!(decoder.has_embed_norm);
        assert!(!decoder.has_final_norm);
    }

    #[test]
    fn test_t5_style_decoder_config() {
        // T5: no embed_norm, has final_norm
        let decoder = MockCpuCrossDecoder::new(6, 768).with_norms(false, true);
        assert!(!decoder.has_embed_norm);
        assert!(decoder.has_final_norm);
    }

    #[test]
    fn test_whisper_style_decoder_config() {
        let decoder = MockCpuCrossDecoder::new(6, 768).with_norms(true, true);
        assert!(decoder.has_embed_norm);
        assert!(decoder.has_final_norm);
    }
    #[test]
    fn test_cpu_decoder_full_flow() {
        let decoder = MockCpuCrossDecoder::new(6, 64).with_norms(true, false);
        
        let embedded = Array3::<f32>::ones((1, 5, 64)); // Would come from ops.embed_decoder_tokens()
        let encoder_hidden = Array3::<f32>::ones((1, 10, 64));
        
        let normed = decoder.embed_norm(&embedded).unwrap();
        assert_eq!(normed.shape(), &[1, 5, 64]);
        
        let cross_kv = decoder.precompute_cross_attention_kv(&encoder_hidden).unwrap();
        
        let output = decoder
            .forward_layers(&normed, &encoder_hidden, None, None, None, Some(&cross_kv), 0, 0)
            .unwrap();
        assert_eq!(output.last_hidden_state.shape(), &[1, 5, 64]);
        
        let final_output = decoder.final_norm(&output.last_hidden_state).unwrap();
        assert_eq!(final_output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_empty_sequence() {
        let decoder = MockCpuCrossDecoder::new(6, 64);
        let hidden = Array3::<f32>::zeros((1, 0, 64));
        let encoder_hidden = Array3::<f32>::zeros((1, 10, 64));
        
        let output = decoder
            .forward_layers(&hidden, &encoder_hidden, None, None, None, None, 0, 0)
            .unwrap();
        
        assert_eq!(output.last_hidden_state.shape(), &[1, 0, 64]);
    }

    #[test]
    fn test_batch_size_greater_than_one() {
        let decoder = MockCpuCrossDecoder::new(6, 64);
        let hidden = Array3::<f32>::ones((4, 5, 64)); // batch=4
        let encoder_hidden = Array3::<f32>::ones((4, 10, 64));
        
        let output = decoder
            .forward_layers(&hidden, &encoder_hidden, None, None, None, None, 0, 0)
            .unwrap();
        
        assert_eq!(output.last_hidden_state.shape(), &[4, 5, 64]);
    }

    #[test]
    fn test_layer_range_partial() {
        let decoder = MockCpuCrossDecoder::new(12, 64);
        let hidden = Array3::<f32>::ones((1, 5, 64));
        let encoder_hidden = Array3::<f32>::ones((1, 10, 64));
        
        // Only run layers 0-6 (first half)
        let output = decoder
            .forward_layers(&hidden, &encoder_hidden, None, None, None, None, 0, 6)
            .unwrap();
        
        assert_eq!(output.last_hidden_state.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_layer_range_second_half() {
        let decoder = MockCpuCrossDecoder::new(12, 64);
        let hidden = Array3::<f32>::ones((1, 5, 64));
        let encoder_hidden = Array3::<f32>::ones((1, 10, 64));
        
        // Only run layers 6-12 (second half)
        let output = decoder
            .forward_layers(&hidden, &encoder_hidden, None, None, None, None, 6, 12)
            .unwrap();
        
        assert_eq!(output.last_hidden_state.shape(), &[1, 5, 64]);
    }
}