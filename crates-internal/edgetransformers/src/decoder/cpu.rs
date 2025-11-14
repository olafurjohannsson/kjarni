use crate::cache::CpuKVCache;
use crate::decoder_layer::DecoderLayer;
use crate::feedforward::SwiGluFeedForward;
use crate::rope::RoPE;
use crate::traits::{Cache, Decoder, DecoderArchitecture, DecoderOutput, Device, TransformerModel};
use crate::weights::ModelWeights;
use crate::{
    Embeddings, FeedForward, MultiHeadAttention, feedforward::StdFeedForward,
    normalization::LayerNorm, normalization::Normalization, normalization::RMSNorm,
};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use log::{debug, info};
use ndarray::{Array1, Array2, Array3, Axis, s};
use std::sync::Arc;

/// The CPU backend implementation for the generic `TransformerDecoder`.
pub struct CpuTransformerDecoder {
    embeddings: Embeddings,
    final_layer_norm: Normalization,
    layers: Vec<DecoderLayer>,
    config: Arc<dyn DecoderArchitecture + Send + Sync>,
    rope: Option<Arc<RoPE>>,
}

impl CpuTransformerDecoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn DecoderArchitecture + Send + Sync>,
        rope: Option<Arc<RoPE>>,
    ) -> Result<Self> {
        debug!("Building CpuTransformerDecoder...");
        debug!("[CPU Decoder] Layers: {}", config.num_hidden_layers());
        debug!("[CPU Decoder] Hidden size: {}", config.hidden_size());
        debug!("[CPU Decoder] Attention heads: {}", config.num_attention_heads());
        debug!("[CPU Decoder] Pre-norm: {}", config.is_prenorm());
        debug!("[CPU Decoder] RoPE: {}", rope.is_some());

        // Load embedding weights with optional position embeddings
        let (word_w, pos_w, _) = config.get_embedding_weight_names();

        debug!("[CPU Decoder] Loading embeddings...");
        debug!("    Word embeddings: {}", word_w);

        let word_embeddings = weights.get_array2(word_w)?;
        let position_embeddings = if !pos_w.is_empty() {
            debug!("[CPU Decoder] Position embeddings: {}", pos_w);
            Some(weights.get_array2(pos_w)?)
        } else {
            debug!("[CPU Decoder] Position embeddings: None (using RoPE)");
            None
        };

        let embeddings = if let Some(pos_emb) = position_embeddings {
            Embeddings::new(word_embeddings, pos_emb, None)
        } else {
            // For models without position embeddings (LLaMA with RoPE)
            Embeddings::new_without_position(word_embeddings, None)
        };

        // Get final layer norm
        let (norm_w, norm_b) = config.get_final_layer_norm_names();
        debug!("  Loading final layer norm...");

        let final_layer_norm =
            Self::load_normalization(weights, &(norm_w, norm_b), config.layer_norm_eps())?
                .ok_or_else(|| anyhow!("Final layer normalization is required"))?;

        // Build decoder layers
        debug!(
            "  Building {} decoder layers...",
            config.num_hidden_layers()
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers());

        for i in 0..config.num_hidden_layers() {
            let layer = Self::build_layer(weights, config.as_ref(), i, rope.clone())?;
            layers.push(layer);
        }

        debug!("✓ CpuTransformerDecoder built successfully");

        Ok(Self {
            embeddings,
            final_layer_norm,
            layers,
            config: config as Arc<dyn DecoderArchitecture + Send + Sync>,
            rope,
        })
    }

    /// Build a single decoder layer (works for both GPT-2 and LLaMA)
    fn build_layer(
        weights: &ModelWeights,
        config: &dyn DecoderArchitecture,
        layer_idx: usize,
        rope: Option<Arc<RoPE>>,
    ) -> Result<DecoderLayer> {
        let attn_names = config.get_attention_names(layer_idx);

        let ffn_names = config.get_feed_forward_names(layer_idx);
        let hidden_size = config.hidden_size();
        let kv_dim = config.kv_dim();

        // Load attention weights (handle both combined QKV and separate Q/K/V)
        let (q_weight, k_weight, v_weight, o_weight, q_bias, k_bias, v_bias, o_bias) =
            if !attn_names.qkv_weight.is_empty() {
                // GPT-2 style: Combined QKV
                let qkv_weight = weights.get_array2(&attn_names.qkv_weight)?;
                let qkv_bias = weights.get_array1(&attn_names.qkv_bias)?;

                let q_weight = qkv_weight.slice(s![.., 0..hidden_size]).to_owned();
                let k_weight = qkv_weight
                    .slice(s![.., hidden_size..2 * hidden_size])
                    .to_owned();
                let v_weight = qkv_weight
                    .slice(s![.., 2 * hidden_size..3 * hidden_size])
                    .to_owned();
                let o_weight = weights.get_array2(&attn_names.output_weight)?;

                let q_bias = qkv_bias.slice(s![0..hidden_size]).to_owned();
                let k_bias = qkv_bias.slice(s![hidden_size..2 * hidden_size]).to_owned();
                let v_bias = qkv_bias
                    .slice(s![2 * hidden_size..3 * hidden_size])
                    .to_owned();
                let o_bias = weights.get_array1(&attn_names.output_bias)?;

                (
                    q_weight, k_weight, v_weight, o_weight, q_bias, k_bias, v_bias, o_bias,
                )
            } else {
                let layer_attn_names = config.get_layer_attention_names(layer_idx);
                // LLaMA style: Separate Q, K, V
                let q_weight = weights.get_linear_weight(&layer_attn_names.q_weight)?;
                let k_weight = weights.get_linear_weight(&layer_attn_names.k_weight)?;
                let v_weight = weights.get_linear_weight(&layer_attn_names.v_weight)?;
                let o_weight = weights.get_linear_weight(&layer_attn_names.output_weight)?;

                let q_bias =
                    Self::load_optional_bias(weights, &layer_attn_names.q_bias, hidden_size)?;
                let k_bias =
                    Self::load_optional_bias(weights, &layer_attn_names.k_bias, kv_dim)?;
                let v_bias =
                    Self::load_optional_bias(weights, &layer_attn_names.v_bias, kv_dim)?;
                let o_bias =
                    Self::load_optional_bias(weights, &layer_attn_names.output_bias, hidden_size)?;

                (
                    q_weight, k_weight, v_weight, o_weight, q_bias, k_bias, v_bias, o_bias,
                )
            };
        
        let attention = MultiHeadAttention::new(
            hidden_size,
            config.num_attention_heads(),
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            o_weight,
            o_bias,
            Some(config.num_key_value_heads()),
        );
        
        // Load FFN (standard or SwiGLU)
        let feed_forward = if let Some(gate_weight_name) = &ffn_names.gate_weight {
            // SwiGLU (LLaMA)
            let gate_weight = weights.get_linear_weight(gate_weight_name)?;
            let up_weight = weights.get_linear_weight(&ffn_names.intermediate_weight)?;
            let down_weight = weights.get_linear_weight(&ffn_names.output_weight)?;

            FeedForward::SwiGLU(SwiGluFeedForward::new(gate_weight, up_weight, down_weight))
        } else {
            // Standard FFN (GPT-2)
            let intermediate_weight = if config.transpose_ffn_weights() {
                weights.get_linear_weight(&ffn_names.intermediate_weight)?
            } else {
                weights.get_array2(&ffn_names.intermediate_weight)?
            };

            let output_weight = if config.transpose_ffn_weights() {
                weights.get_linear_weight(&ffn_names.output_weight)?
            } else {
                weights.get_array2(&ffn_names.output_weight)?
            };

            let intermediate_bias = Self::load_optional_bias(
                weights,
                &ffn_names.intermediate_bias,
                config.intermediate_size(),
            )?;

            let output_bias =
                Self::load_optional_bias(weights, &ffn_names.output_bias, hidden_size)?;

            FeedForward::Standard(StdFeedForward::new(
                intermediate_weight,
                intermediate_bias,
                output_weight,
                output_bias,
                config.activation_function(),
            ))
        };
        // Load normalization layers
        let self_attn_layer_norm = if !attn_names.qkv_weight.is_empty() {
            Self::load_normalization(
                weights,
                &(
                    attn_names.norm_weight.as_str(),
                    attn_names.norm_bias.as_str(),
                ),
                config.layer_norm_eps(),
            )?
            .ok_or_else(|| anyhow!("Attention normalization required for layer {}", layer_idx))?
        } else {
            let layer_attn_names = config.get_layer_attention_names(layer_idx);
            Self::load_normalization(
                weights,
                &(
                    layer_attn_names.norm_weight.as_str(),
                    layer_attn_names.norm_bias.as_str(),
                ),
                config.layer_norm_eps(),
            )?
            .ok_or_else(|| anyhow!("Attention normalization required for layer {}", layer_idx))?
        };

        let ffn_layer_norm = Self::load_normalization(
            weights,
            &(ffn_names.norm_weight.as_str(), ffn_names.norm_bias.as_str()),
            config.layer_norm_eps(),
        )?
        .ok_or_else(|| anyhow!("FFN normalization required for layer {}", layer_idx))?;

        Ok(DecoderLayer {
            self_attn: attention,
            self_attn_layer_norm,
            feedforward: feed_forward,
            ffn_layer_norm,
            is_prenorm: config.is_prenorm(),
            rope: rope,
        })
    }

    /// Load optional bias (returns zeros if name is empty)
    fn load_optional_bias(
        weights: &ModelWeights,
        name: &str,
        expected_size: usize,
    ) -> Result<Array1<f32>> {
        if name.is_empty() {
            // LLaMA has no biases - return zeros
            Ok(Array1::zeros(expected_size))
        } else {
            weights.get_array1(name)
        }
    }

    /// Load normalization layer (LayerNorm or RMSNorm)
    fn load_normalization(
        weights: &ModelWeights,
        names: &(&str, &str),
        eps: f32,
    ) -> Result<Option<Normalization>> {
        let (weight_name, bias_name) = names;

        if weight_name.is_empty() {
            return Ok(None);
        }

        let weight = weights.get_array1(weight_name)?;

        if bias_name.is_empty() {
            // RMSNorm (no bias) - LLaMA
            Ok(Some(Normalization::RMSNorm(RMSNorm::new(weight, eps))))
        } else {
            // LayerNorm (with bias) - GPT-2
            let bias = weights.get_array1(bias_name)?;
            Ok(Some(Normalization::LayerNorm(LayerNorm::new(
                weight, bias, eps,
            ))))
        }
    }
}

impl TransformerModel for CpuTransformerDecoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
}

#[async_trait]
impl Decoder for CpuTransformerDecoder {
    type Input = Array2<u32>;
    type Output = DecoderOutput;

    async fn forward(
        &self,
        input_ids: &Self::Input,
        attention_mask: &Array2<f32>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());
        let seq_len = input_ids.shape()[1];
        debug!(
            "[CPU Decoder] Forward pass started. position_offset: {}",
            position_offset
        );

        let mut hidden_states = self.embed_with_offset(input_ids, position_offset);
        debug!(
            "[CPU Decoder] Initial hidden_states shape: {:?}",
            hidden_states.shape()
        );
        debug!(
            "[CPU Decoder] Attention mask shape: {:?}",
            attention_mask.shape()
        );

        let mut cpu_cache_opt = cache.and_then(|c| c.as_any_mut().downcast_mut::<CpuKVCache>());
        let mut new_key_values = Vec::with_capacity(self.layers.len());

        info!(
            "[CPU Decoder] Executing {} decoder layers...",
            self.layers.len()
        );
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            debug!("[CPU Decoder] --- Layer {} ---", layer_idx);

            // Retrieve the past K/V state for THIS specific layer from the cache.
            let past_kv_owned = cpu_cache_opt
                .as_ref()
                .and_then(|cache| cache.get(layer_idx));

            let past_kv_views = past_kv_owned.as_ref().map(|(k, v)| (k.view(), v.view()));

            // --- STRATEGIC LOGGING ---
            debug!(
                "[CPU Decoder] Layer {} input hidden_states shape: {:?}",
                layer_idx,
                hidden_states.shape()
            );
            if let Some((k, v)) = past_kv_views {
                debug!(
                    "[CPU Decoder] Layer {} past_k shape for forward: {:?}",
                    layer_idx,
                    k.shape()
                );
                debug!(
                    "[CPU Decoder] Layer {} past_v shape for forward: {:?}",
                    layer_idx,
                    v.shape()
                );
            } else {
                debug!(
                    "[CPU Decoder] Layer {} past_kv is None (priming pass).",
                    layer_idx
                );
            }

            let (new_hidden_states, (new_k, new_v)) = layer.forward(
                &hidden_states,
                attention_mask,
                position_offset,
                past_kv_views,
            )?;

            debug!(
                "[CPU Decoder] Layer {} output hidden_states shape: {:?}",
                layer_idx,
                new_hidden_states.shape()
            );
            debug!(
                "[CPU Decoder] Layer {} new_k shape: {:?}",
                layer_idx,
                new_k.shape()
            );
            debug!(
                "[CPU Decoder] Layer {} new_v shape: {:?}",
                layer_idx,
                new_v.shape()
            );

            hidden_states = new_hidden_states;
            new_key_values.push((new_k, new_v));
        }

        // --- 4. Final Layer Normalization ---
        info!("[CPU Decoder] Applying final layer normalization...");
        hidden_states = self.final_layer_norm.forward(&hidden_states);
        debug!(
            "[CPU Decoder] Final hidden_states shape after LN: {:?}",
            hidden_states.shape()
        );

        // --- 5. Cache Update Step ---
        if let Some(cache) = cpu_cache_opt {
            for (layer_idx, (k, v)) in new_key_values.into_iter().enumerate() {
                cache.update(layer_idx, &k, &v)?;
            }
            cache.increment_len(seq_len);
            debug!(
                "[CPU Decoder] Incremented cache length by {}. New length: {}",
                seq_len,
                cache.get_seq_length()
            );
        }

        Ok(DecoderOutput {
            last_hidden_state: hidden_states,
            past_key_values: None,
        })
    }
    async fn get_hidden_states(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
    ) -> Result<Array3<f32>> {
        let output = self.forward(input, attention_mask, None).await?;
        Ok(output.last_hidden_state)
    }
}

impl CpuTransformerDecoder {
    /// Embed tokens with optional position offset for autoregressive generation
    ///
    /// For models with learned position embeddings (GPT-2):
    ///   - Adds position embeddings using absolute position (position_offset + j)
    ///
    /// For models with RoPE (LLaMA):
    ///   - Only returns word embeddings (RoPE is applied in attention layer)
    ///   - position_offset is still tracked for RoPE in attention
    fn embed_with_offset(&self, input_ids: &Array2<u32>, position_offset: usize) -> Array3<f32> {
        let (batch_size, seq_len) = input_ids.dim();
        let hidden_size = self.config.hidden_size();

        // Convert input_ids to a flattened slice of usize for indexing
        // The .as_slice() method is only available on contiguous arrays, which is the default.
        let indices_slice = input_ids.as_slice().unwrap().iter().map(|&x| x as usize).collect::<Vec<_>>();

        // Chain the operations to correctly infer the final 3D type.
        let mut hidden_states = self
            .embeddings
            .word_embeddings
            .select(Axis(0), &indices_slice) // select returns a 2D array
            .into_shape_with_order((batch_size, seq_len, hidden_size)) // reshape into 3D
            .unwrap()
            .to_owned(); // Convert the view from into_shape into an owned array


        // Position embeddings with offset (only if present - GPT-2 has them, LLaMA doesn't)
        if let Some(ref pos_embeddings) = self.embeddings.position_embeddings {
            for j in 0..seq_len {
                let pos_idx = position_offset + j;
                let pos_emb = pos_embeddings.row(pos_idx);
                for i in 0..batch_size {
                    hidden_states
                        .slice_mut(s![i, j, ..])
                        .scaled_add(1.0, &pos_emb);
                }
            }
        }
        // For RoPE models (LLaMA), position is handled in attention layer

        hidden_states
    }
}
