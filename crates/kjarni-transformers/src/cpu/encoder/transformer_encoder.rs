use std::sync::Arc;

use anyhow::{Context, Result};
use ndarray::{Array2, Array3, s};

use crate::Normalization;
use crate::activations::Activation;
use crate::cpu::encoder::buffers::EncoderBuffers;
use crate::cpu::encoder::traits::CpuEncoderOutput;
use crate::cpu::encoder::{
    CpuEncoder, encoder_layer::EncoderLayer, encoder_self_attention::EncoderSelfAttention,
};
use crate::linear_layer::{F32MatmulStrategy, LinearLayer};
use crate::models::base::ModelLoadConfig;
use crate::normalization::RMSNorm;
use crate::rope::RoPE;
use crate::traits::{Device, InferenceModel, ModelLayout, ModelMetadata, NormalizationStrategy};
use crate::weights::ModelWeights;
use crate::{Embeddings, FeedForward, normalization::LayerNorm};

pub struct CpuTransformerEncoder {
    embeddings: Embeddings,
    embeddings_layer_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    pub metadata: ModelMetadata,
    pub rope: Option<Arc<RoPE>>,
}

impl CpuTransformerEncoder {
    pub fn new(
        weights: &ModelWeights,
        meta: ModelMetadata,
        layout: ModelLayout,
        load_cfg: ModelLoadConfig,
    ) -> Result<Self> {
        let dtype = load_cfg.target_dtype;

        // 1. Get Encoder Layout
        let encoder_layout = layout
            .encoder
            .as_ref()
            .context("ModelLayout is missing the required 'encoder' layout")?;

        // 2. Embeddings
        let embeddings = Embeddings::from_weights(
            weights,
            &layout.token_embedding,
            encoder_layout.position_embedding.as_deref(),
            encoder_layout.token_type_embedding.as_deref(),
            load_cfg.target_dtype,
        )?;

        // 3. Embedding LayerNorm
        let emb_norm_w = encoder_layout
            .embedding_norm_weight
            .as_ref()
            .context("Encoder layout requires embedding_norm_weight")?;
        let emb_norm_b = encoder_layout
            .embedding_norm_bias
            .as_ref()
            .context("Encoder layout requires embedding_norm_bias")?;

        let embeddings_layer_norm = LayerNorm::new(
            weights.get_array1(emb_norm_w)?,
            weights.get_array1(emb_norm_b)?,
            meta.norm_eps,
        );

        // 4. Build Layers
        let mut layers = Vec::with_capacity(meta.num_layers);

        for i in 0..meta.num_layers {
            let idx = i.to_string();
            let name = |template: &String| template.replace("{}", &idx);

            // Helper to resolve optional bias names safely
            let resolve_bias = |opt: &Option<String>| opt.as_ref().map(|s| name(s));

            let f32_strategy = F32MatmulStrategy::CustomSimd;

            let q_name = name(&encoder_layout.layer.self_attn.q_weight);
            let k_name = name(&encoder_layout.layer.self_attn.k_weight);
            let v_name = name(&encoder_layout.layer.self_attn.v_weight);

            let q_bias = resolve_bias(&encoder_layout.layer.self_attn.q_bias);
            let k_bias = resolve_bias(&encoder_layout.layer.self_attn.k_bias);
            let v_bias = resolve_bias(&encoder_layout.layer.self_attn.v_bias);

            // Check for Fused QKV (Nomic/Mosaic Style)
            // If Q, K, and V all point to the same tensor file, we slice it.
            let (q_proj, k_proj, v_proj) = if q_name == k_name && k_name == v_name {
                let hidden = meta.hidden_size;

                // 1. Load the raw fused weights [3 * Hidden, Hidden]
                let fused_w = weights.get_array2(&q_name)?;

                // 2. Slice weights
                let q_w = fused_w.slice(s![0..hidden, ..]).to_owned();
                let k_w = fused_w.slice(s![hidden..2 * hidden, ..]).to_owned();
                let v_w = fused_w.slice(s![2 * hidden..3 * hidden, ..]).to_owned();

                // 3. Handle Fused Bias (if present)
                let (q_b, k_b, v_b) = if let Some(b_name) = &q_bias {
                    let fused_b = weights.get_array1(b_name)?;
                    (
                        Some(fused_b.slice(s![0..hidden]).to_owned()),
                        Some(fused_b.slice(s![hidden..2 * hidden]).to_owned()),
                        Some(fused_b.slice(s![2 * hidden..3 * hidden]).to_owned()),
                    )
                } else {
                    (None, None, None)
                };

                (
                    LinearLayer::new_f32(q_w, q_b),
                    LinearLayer::new_f32(k_w, k_b),
                    LinearLayer::new_f32(v_w, v_b),
                )
            } else {
                // Standard Separate Loading (BERT/MiniLM)
                (
                    LinearLayer::builder(weights, &q_name)
                        .with_optional_bias(q_bias.as_deref())
                        .with_target_dtype(dtype)
                        .with_f32_strategy(Some(f32_strategy))
                        .build()?,
                    LinearLayer::builder(weights, &k_name)
                        .with_optional_bias(k_bias.as_deref())
                        .with_target_dtype(dtype)
                        .with_f32_strategy(Some(f32_strategy))
                        .build()?,
                    LinearLayer::builder(weights, &v_name)
                        .with_optional_bias(v_bias.as_deref())
                        .with_target_dtype(dtype)
                        .with_f32_strategy(Some(f32_strategy))
                        .build()?,
                )
            };

            let out_proj =
                LinearLayer::builder(weights, &name(&encoder_layout.layer.self_attn.o_weight))
                    .with_optional_bias(
                        resolve_bias(&encoder_layout.layer.self_attn.o_bias).as_deref(),
                    )
                    .with_target_dtype(dtype)
                    .with_f32_strategy(Some(f32_strategy))
                    .build()?;

            let self_attn = EncoderSelfAttention::new(
                meta.hidden_size,
                meta.num_attention_heads,
                q_proj.clone(),
                k_proj.clone(),
                v_proj.clone(),
                out_proj.clone(),
            );

            // --- FEED FORWARD LOADING ---

            let up_proj = LinearLayer::builder(weights, &name(&encoder_layout.layer.ffn.up_weight))
                .with_optional_bias(resolve_bias(&encoder_layout.layer.ffn.up_bias).as_deref())
                .with_target_dtype(dtype)
                .with_f32_strategy(Some(f32_strategy))
                .build()?;

            let down_proj =
                LinearLayer::builder(weights, &name(&encoder_layout.layer.ffn.down_weight))
                    .with_optional_bias(
                        resolve_bias(&encoder_layout.layer.ffn.down_bias).as_deref(),
                    )
                    .with_target_dtype(dtype)
                    .with_f32_strategy(Some(f32_strategy))
                    .build()?;

            // Handle Gate (Optional, for SwiGLU models like Nomic)
            let gate_proj = if let Some(gate_name) = &encoder_layout.layer.ffn.gate_weight {
                Some(
                    LinearLayer::builder(weights, &name(gate_name))
                        .with_target_dtype(dtype)
                        .with_f32_strategy(Some(f32_strategy))
                        .build()?,
                )
            } else {
                None
            };

            let feedforward = if let Some(gate) = gate_proj {
                // If we have a Gate layer, it's SwiGLU (Nomic/Llama)
                FeedForward::SwiGLU(crate::feedforward::SwiGluFeedForward::new(
                    gate,              // Gate
                    up_proj.clone(),   // Up
                    down_proj.clone(), // Down
                    Activation::SilU,
                ))
            } else {
                // If no Gate, it's Standard (BERT/MiniLM/DistilBERT)
                FeedForward::StandardNew(crate::feedforward::StdFeedForwardNew::new(
                    up_proj.clone(),   // FC1
                    down_proj.clone(), // FC2
                    meta.activation,   // Gelu/Relu
                ))
            };

            // --- NORMALIZATION ---

            let attn_norm_w = name(&encoder_layout.layer.self_attn.norm_weight);
            let attn_norm_b = resolve_bias(&encoder_layout.layer.self_attn.norm_bias);

            let self_attn_layer_norm =
                if meta.normalization_strategy == NormalizationStrategy::LayerNorm {
                    Normalization::LayerNorm(LayerNorm::new(
                        weights.get_array1(&attn_norm_w)?,
                        weights.get_array1(&attn_norm_b.unwrap())?, // LayerNorm usually requires bias
                        meta.norm_eps,
                    ))
                } else {
                    Normalization::RMSNorm(RMSNorm::new(
                        weights.get_array1(&attn_norm_w)?,
                        meta.norm_eps,
                    ))
                };

            let ffn_norm_w = name(&encoder_layout.layer.ffn.norm_weight);
            let ffn_norm_b = resolve_bias(&encoder_layout.layer.ffn.norm_bias);

            let ffn_layer_norm = if meta.normalization_strategy == NormalizationStrategy::LayerNorm
            {
                Normalization::LayerNorm(LayerNorm::new(
                    weights.get_array1(&ffn_norm_w)?,
                    weights.get_array1(&ffn_norm_b.unwrap())?,
                    meta.norm_eps,
                ))
            } else {
                Normalization::RMSNorm(RMSNorm::new(
                    weights.get_array1(&ffn_norm_w)?,
                    meta.norm_eps,
                ))
            };

            layers.push(EncoderLayer {
                self_attn,
                self_attn_layer_norm,
                feedforward,
                ffn_layer_norm,
            });
        }

        // 5. Initialize RoPE (If Metadata says so)
        let rope = if let Some(theta) = meta.rope_theta {
            Some(Arc::new(RoPE::new(meta.head_dim, meta.max_seq_len, theta)))
        } else {
            None
        };

        Ok(Self {
            embeddings,
            embeddings_layer_norm,
            layers,
            metadata: meta,
            rope,
        })
    }
}

impl CpuTransformerEncoder {
    /// Forward pass through layers with pre-allocated buffers (no allocation in hot path).
    ///
    /// # Note
    ///
    /// Still allocates: RoPE, 4D attention matmuls, reshape/permute, layer norm.
    pub fn forward_layers_noalloc(
        &self,
        hidden_states: &mut Array3<f32>,
        attention_mask: &Array2<f32>,
        start_layer: usize,
        end_layer: usize,
        buffers: &mut EncoderBuffers,
    ) -> Result<()> {
        let (batch, seq, _) = hidden_states.dim();

        #[cfg(debug_assertions)]
        buffers.ensure_capacity(batch, seq);

        for layer in &self.layers[start_layer..end_layer] {
            layer.forward_noalloc(
                hidden_states,
                attention_mask,
                None,
                self.metadata.is_prenorm,
                self.rope.as_deref(),
                buffers,
            )?;
        }

        Ok(())
    }

    /// Full forward pass with external buffer (caller owns buffers).
    ///
    /// This is the recommended API for hot paths where you reuse buffers.
    ///
    /// # Note
    ///
    /// Still allocates: embeddings, layer norm, RoPE, 4D attention matmuls.
    pub fn forward_noalloc(
        &self,
        input_ids: &Array2<u32>,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<u32>>,
        buffers: &mut EncoderBuffers,
    ) -> Result<CpuEncoderOutput> {
        // Embeddings still allocate (could optimize later)
        let mut hidden = self.embed_and_normalize(input_ids, token_type_ids);

        // Forward through layers (noalloc for layer computations)
        self.forward_layers_noalloc(&mut hidden, attention_mask, 0, self.num_layers(), buffers)?;

        Ok(CpuEncoderOutput {
            last_hidden_state: hidden,
        })
    }

    /// Creates appropriately sized buffers for this encoder.
    ///
    /// Call once and reuse across forward passes.
    pub fn create_buffers(&self, max_batch: usize, max_seq: usize) -> EncoderBuffers {
        EncoderBuffers::new_auto(
            max_batch,
            max_seq,
            self.metadata.hidden_size,
            self.metadata.num_attention_heads,
            self.metadata.intermediate_size,
        )
    }
}

impl InferenceModel for CpuTransformerEncoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CpuEncoder for CpuTransformerEncoder {
    fn create_buffers(&self, max_batch: usize, max_seq: usize) -> EncoderBuffers {
        EncoderBuffers::new_auto(
            max_batch,
            max_seq,
            self.metadata.hidden_size,
            self.metadata.num_attention_heads,
            self.metadata.intermediate_size,
        )
    }
    fn forward_with_buffers(
        &self,
        input_ids: &Array2<u32>,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<u32>>,
        buffers: &mut EncoderBuffers,
    ) -> Result<CpuEncoderOutput> {
        let mut hidden = self.embed_and_normalize(input_ids, token_type_ids);
        
        self.forward_layers_noalloc(
            &mut hidden,
            attention_mask,
            0,
            self.num_layers(),
            buffers,
        )?;
        
        Ok(CpuEncoderOutput {
            last_hidden_state: hidden,
        })
    }
    fn embed(&self, input_ids: &Array2<u32>, token_type_ids: Option<&Array2<u32>>) -> Array3<f32> {
        self.embeddings.forward(
            input_ids,
            token_type_ids,
            self.metadata.extra_pos_embeddings,
            self.metadata.scale_embeddings,
        )
    }

    fn embed_and_normalize(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Array3<f32> {
        let hidden = self.embed(input_ids, token_type_ids);
        self.embeddings_layer_norm.forward_3d(&hidden)
    }

    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>> {
        let mut hidden = hidden_states.clone();
        let is_prenorm = self.metadata.is_prenorm;
        for (i, layer) in self.layers[start_layer..end_layer].iter().enumerate() {
            hidden = layer.forward(
                hidden,
                attention_mask,
                None,
                is_prenorm,
                self.rope.as_deref(),
            )?;
        }
        Ok(hidden)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }
}
