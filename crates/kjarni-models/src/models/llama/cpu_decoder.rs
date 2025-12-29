// --- Standard Library ---
use std::sync::Arc;

// --- External Crates ---
use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, Axis, s};

use crate::models::llama::config::LlamaConfig;

use kjarni_transformers::{
    Normalization, WgpuContext,
    cache::CpuKVCache,
    decoder::prelude::*,
    embeddings::Embeddings,
    feedforward::SwiGluFeedForward,
    linear_layer::LinearLayer,
    models::base::ModelInput,
    normalization::RMSNorm,
    pipeline::CpuLayerFactory,
    rope::RoPE,
    tensor::DType,
    traits::{Cache, Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

pub struct LlamaCpuDecoder {
    pub embeddings: Embeddings,
    pub layers: Vec<CpuRoPEDecoderLayer>,
    pub final_norm: RMSNorm,
    pub metadata: ModelMetadata,
}


impl LlamaCpuDecoder {
    pub fn new(
        weights: &ModelWeights,
        metadata: ModelMetadata,
        layout: ModelLayout,
        rope: Arc<RoPE>,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("Llama layout must have a decoder section");

        let embeddings = Embeddings::from_weights(
            weights,
            &layout.token_embedding,
            decoder_layout.position_embedding.as_deref(), // Correctly access nested field
            decoder_layout.token_type_embedding.as_deref(),
        )?;

        let final_norm = RMSNorm::new(
            weights.get_array1(decoder_layout.final_norm_weight.as_ref().unwrap())?,
            metadata.norm_eps,
        );

        let mut layers = Vec::with_capacity(metadata.num_layers);
        for i in 0..metadata.num_layers {
            layers.push(Self::build_layer(
                weights,
                &metadata,
                &layout,
                i,
                rope.clone(),
                target_dtype,
            )?);
        }

        Ok(Self {
            embeddings,
            layers,
            final_norm,
            metadata,
        })
    }

    fn build_layer(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        i: usize,
        rope: Arc<RoPE>,
        target_dtype: Option<DType>,
    ) -> Result<CpuRoPEDecoderLayer> {
        // Get the specific nested layouts for the decoder.
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("Llama layout must have a decoder section");
        let layer_layout = &decoder_layout.layer;

        let attention: DecoderAttention = CpuLayerFactory::build_decoder_attention(
            weights,
            meta,
            &decoder_layout.layer.self_attn,
            i,
            target_dtype,
        )?;

        let feed_forward =
            CpuLayerFactory::build_swiglu_ffn(weights, &decoder_layout.layer.ffn, i, target_dtype)?;

        let attention_norm = CpuLayerFactory::build_norm(
            weights,
            &layer_layout.self_attn.norm_weight,
            &layer_layout.self_attn.norm_bias,
            meta.norm_eps,
            i,
        )?;

        let ffn_norm = CpuLayerFactory::build_norm(
            weights,
            &layer_layout.ffn.norm_weight,
            &layer_layout.ffn.norm_bias,
            meta.norm_eps,
            i,
        )?;

        Ok(CpuRoPEDecoderLayer {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            rope,
        })
    }
}

impl InferenceModel for LlamaCpuDecoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        None
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CpuDecoder for LlamaCpuDecoder {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn embed(&self, input: ModelInput<'_>, position_offset: usize) -> Result<Array3<f32>> {
        match input {
            ModelInput::TokensCpu(ids) => {
                let seq_len = ids.len();
                // let input_ids = Array2::from_shape_vec((1, seq_len), ids.to_vec())?;

                Ok(self
                    .embeddings
                    .forward(&ids.to_owned(), None, position_offset, false))
            }
            ModelInput::HiddenCpu(hidden) => Ok(hidden.to_owned()),
            _ => Err(anyhow!(
                "LlamaCpuDecoder received GPU input. Transfer to CPU first."
            )),
        }
    }

    fn embed_and_normalize(
        &self,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        // Llama is Pre-Norm (Norm is inside the layer).
        // No initial LayerNorm exists before the first block.
        self.embed(input, position_offset)
    }

    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        mut cache: Option<&mut dyn Cache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>> {
        let mut hidden = hidden_states.clone();
        let seq_len = hidden.shape()[1];

        // 1. Downcast to CpuKVCache to access specific get/update methods
        // We use a mutable option so we can borrow it mutably later for updates
        let mut cpu_cache_opt = cache.and_then(|c| c.as_any_mut().downcast_mut::<CpuKVCache>());

        // 2. Store new K/V pairs temporarily.
        // We cannot update the cache *inside* the loop because `cpu_cache_opt` is borrowed
        // to get `past_kv` (immutable borrow), preventing a mutable borrow for `update`.
        let mut new_key_values = Vec::with_capacity(end_layer - start_layer);

        for i in start_layer..end_layer {
            if i >= self.layers.len() {
                break;
            }
            let layer = &self.layers[i];

            // 3. Get Past KV View
            // We use .as_ref() to borrow the Option content without moving it
            let past_kv = cpu_cache_opt.as_ref().and_then(|c| c.get(i));

            // Map the Tuple(Array3, Array3) to Tuple(ArrayView3, ArrayView3)
            let past_kv_view = past_kv.as_ref().map(|(k, v)| (k.view(), v.view()));

            // 4. Layer Forward
            let (new_hidden, new_k, new_v) =
                layer.forward(&hidden, attention_mask, position_offset, past_kv_view)?;

            hidden = new_hidden;
            new_key_values.push((new_k, new_v));
        }

        // 5. Update Cache (Batch update after the loop)
        if let Some(cache) = cpu_cache_opt {
            for (local_idx, (k, v)) in new_key_values.into_iter().enumerate() {
                let layer_idx = start_layer + local_idx;
                cache.update(layer_idx, &k, &v)?;
            }
            // Important: Increment length so next step knows the offset
            cache.increment_len(seq_len);
        }

        Ok(hidden)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward(
        &self,
        input: ModelInput<'_>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        let hidden = self.embed_and_normalize(input, position_offset)?;

        // 2. Run Layers
        let mut output = self.forward_layers(
            &hidden,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )?;

        output = self.final_norm.forward_3d(&output);

        Ok(output)
    }
}

mod llama_test {

    use super::*;
    use crate::models::llama::config::LlamaConfig;
    const SAFETENSORS_PATH: &str = "/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B-Instruct";
    use kjarni_transformers::{
        kernels::{
            dequantize::{dequantize_q4_k_block, dequantize_q6_k_block},
            q_common::{BlockQ4_K, BlockQ6_K},
        },
        linear_layer::LinearData,
        tensor::CpuTensor,
        weights::{ModelWeights, cast_or_copy},
    };
    use std::path::Path;
    const GGUF_PATH: &str = "/home/olafurj/.cache/kjarni/llama-3.2-1b-instruct-q4_k_m/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
    #[test]
    fn test_check_all_matrix_sizes_interleaving() {
        let gguf_path = Path::new(GGUF_PATH);
        let st_path = Path::new(SAFETENSORS_PATH);

        let gguf_weights = ModelWeights::new(gguf_path).unwrap();
        let st_weights = ModelWeights::new(st_path).unwrap();

        let tensors_to_check = [
            ("model.layers.0.self_attn.q_proj.weight", [2048, 2048]), // Q
            ("model.layers.0.self_attn.k_proj.weight", [512, 2048]),  // K (smaller)
            ("model.layers.0.self_attn.v_proj.weight", [512, 2048]),  // V (smaller)
            ("model.layers.0.self_attn.o_proj.weight", [2048, 2048]), // O
            ("model.layers.0.mlp.gate_proj.weight", [8192, 2048]),    // Gate (larger)
            ("model.layers.0.mlp.up_proj.weight", [8192, 2048]),      // Up (larger)
            ("model.layers.0.mlp.down_proj.weight", [2048, 8192]),    // Down
        ];

        for (name, expected_shape) in tensors_to_check {
            println!(
                "\n=== {} [{}, {}] ===",
                name, expected_shape[0], expected_shape[1]
            );

            let raw = gguf_weights.get_raw(name).unwrap();
            println!("Actual shape: {:?}, dtype: {:?}", raw.shape, raw.dtype);

            // Get ST reference (if exists)
            let st_f32 = match st_weights.get_typed_tensor(name) {
                Ok(t) => t.to_array2_f32().ok(),
                Err(_) => None,
            };

            if st_f32.is_none() {
                println!("ST tensor not found, skipping");
                continue;
            }
            let st_f32 = st_f32.unwrap();

            // Get ORIGINAL GGUF blocks
            let blocks_per_row = raw.shape[1] / 256;

            // Check block group mapping for first few groups
            print!("Block mapping: ");

            match raw.dtype {
                DType::Q4_K => {
                    let blocks: Vec<BlockQ4_K> = cast_or_copy(&raw.bytes);
                    for block_group in [0, 1, 2, 3] {
                        let block_idx = block_group * blocks_per_row;
                        if block_idx >= blocks.len() {
                            continue;
                        }

                        let mut block_data = [0.0f32; 256];
                        dequantize_q4_k_block(&blocks[block_idx], &mut block_data);

                        // Find best ST row
                        let mut best_row = 0;
                        let mut best_diff = f32::MAX;
                        for st_row in 0..64.min(raw.shape[0]) {
                            let diff: f32 = block_data
                                .iter()
                                .zip(st_f32.row(st_row).iter().take(256))
                                .map(|(a, b)| (a - b).abs())
                                .sum();
                            if diff < best_diff {
                                best_diff = diff;
                                best_row = st_row;
                            }
                        }
                        print!("{}→{} ", block_group, best_row);
                    }
                }
                DType::Q6_K => {
                    let blocks: Vec<BlockQ6_K> = cast_or_copy(&raw.bytes);
                    for block_group in [0, 1, 2, 3] {
                        let block_idx = block_group * blocks_per_row;
                        if block_idx >= blocks.len() {
                            continue;
                        }

                        let mut block_data = [0.0f32; 256];
                        dequantize_q6_k_block(&blocks[block_idx], &mut block_data);

                        let mut best_row = 0;
                        let mut best_diff = f32::MAX;
                        for st_row in 0..64.min(raw.shape[0]) {
                            let diff: f32 = block_data
                                .iter()
                                .zip(st_f32.row(st_row).iter().take(256))
                                .map(|(a, b)| (a - b).abs())
                                .sum();
                            if diff < best_diff {
                                best_diff = diff;
                                best_row = st_row;
                            }
                        }
                        print!("{}→{} ", block_group, best_row);
                    }
                }
                _ => println!("Unsupported dtype"),
            }
            println!();
        }
    }
}
