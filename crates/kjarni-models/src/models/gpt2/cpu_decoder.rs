
use std::sync::Arc;
use anyhow::{Result, anyhow};
use log::debug;
use ndarray::{Array1, Array2, Array3, s};

use kjarni_transformers::{
    Embeddings, FeedForward, MultiHeadAttention, Normalization, cache::{Cache, CpuKVCache}, decoder::prelude::*, feedforward::{LegacyFeedForward, StdFeedForward}, linear_layer::LinearLayer, models::base::ModelInput, normalization::LayerNorm, tensor::DType, traits::{InferenceModel, ModelConfig, ModelLayout, ModelMetadata}, weights::ModelWeights
};

// --- Crate-Specific ---
use crate::models::gpt2::config::Gpt2Config;

/// The CPU-native implementation of the GPT-2 decoder architecture.
pub struct Gpt2CpuDecoder {
    pub embeddings: Embeddings,
    pub layers: Vec<GptPreNormDecoderLayer>,
    pub final_layer_norm: Normalization,
    pub config: Arc<Gpt2Config>,
    pub meta: ModelMetadata,
    pub layout: ModelLayout,
}

impl Gpt2CpuDecoder {
    pub fn new(weights: &ModelWeights, config: Arc<Gpt2Config>) -> Result<Self> {
        log::info!("Building GPT-2 CPU decoder...");

        let meta = config.metadata();
        let layout = config.layout();
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("GPT-2 layout must have a decoder section");

        // 1. Embeddings
        let word_embeddings = weights.get_array2(&layout.token_embedding)?;
        let position_embeddings = if let Some(pos_name) = &decoder_layout.position_embedding {
            debug!("[CPU Decoder] Position embeddings: {}", pos_name);
            Some(weights.get_array2(pos_name)?)
        } else {
            debug!("[CPU Decoder] Position embeddings: None");
            None
        };

        let embeddings = Embeddings::new(
            kjarni_transformers::EmbeddingData::F32(Arc::new(word_embeddings)),
            position_embeddings,
            None,
        );

        // 2. Final Layer Norm
        debug!("  Loading final layer norm...");
        let final_norm_weight = decoder_layout.final_norm_weight.as_ref().unwrap();
        let final_norm_bias = decoder_layout.final_norm_bias.as_deref().unwrap_or("");
        let final_layer_norm = Self::load_normalization(
            weights,
            &(final_norm_weight, final_norm_bias),
            meta.norm_eps,
        )?
        .ok_or_else(|| anyhow!("Final layer normalization is required"))?;

        // 3. Build decoder layers
        debug!("  Building {} decoder layers...", meta.num_layers);
        let mut layers = Vec::with_capacity(meta.num_layers);

        for i in 0..meta.num_layers {
            let layer = Self::build_layer(weights, &meta, &layout, i)?;
            layers.push(layer);
        }

        debug!("✓ Gpt2CpuDecoder built successfully");

        Ok(Self {
            embeddings,
            final_layer_norm,
            layers,
            config,
            meta,
            layout,
        })
    }

    fn build_layer(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        layer_idx: usize,
    ) -> Result<GptPreNormDecoderLayer> {
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("GPT-2 layout must have a decoder section");
        let layer_layout = &decoder_layout.layer;
        let self_attn_layout = &layer_layout.self_attn;
        let ffn_layout = &layer_layout.ffn;

        let idx = layer_idx.to_string();
        let name = |t: &String| t.replace("{}", &idx);
        let opt_name = |t: &Option<String>| {
            t.as_ref()
                .map(|s| s.replace("{}", &idx))
                .unwrap_or_default()
        };

        let hidden_size = meta.hidden_size;
        let head_dim = meta.head_dim;
        let kv_dim = meta.num_kv_heads * head_dim;

        // --- 1. Load Attention Weights ---
        // GPT-2 uses fused QKV (c_attn).
        let qkv_weight_name = name(&self_attn_layout.q_weight);
        let qkv_bias_name = opt_name(&self_attn_layout.q_bias);

        // The logic for splitting the fused tensor is preserved.
        let (q_weight, k_weight, v_weight, o_weight, q_bias, k_bias, v_bias, o_bias) =
            if !qkv_weight_name.is_empty() {
                // GPT-2 style: Combined QKV
                let qkv_weight = weights.get_array2(&qkv_weight_name)?;
                let qkv_bias = weights.get_array1(&qkv_bias_name)?;

                let q_weight = qkv_weight.slice(s![.., 0..hidden_size]).to_owned();
                let k_weight = qkv_weight
                    .slice(s![.., hidden_size..2 * hidden_size])
                    .to_owned();
                let v_weight = qkv_weight
                    .slice(s![.., 2 * hidden_size..3 * hidden_size])
                    .to_owned();

                let o_weight = weights.get_array2(&name(&self_attn_layout.o_weight))?;

                let q_bias = qkv_bias.slice(s![0..hidden_size]).to_owned();
                let k_bias = qkv_bias.slice(s![hidden_size..2 * hidden_size]).to_owned();
                let v_bias = qkv_bias
                    .slice(s![2 * hidden_size..3 * hidden_size])
                    .to_owned();
                let o_bias = weights.get_array1(&opt_name(&self_attn_layout.o_bias))?;

                (
                    q_weight, k_weight, v_weight, o_weight, q_bias, k_bias, v_bias, o_bias,
                )
            } else {
                // Fallback for separate weights (not used by GPT-2 but preserved for robustness)
                let q_weight = weights.get_array2(&name(&self_attn_layout.q_weight))?;
                let k_weight = weights.get_array2(&name(&self_attn_layout.k_weight))?;
                let v_weight = weights.get_array2(&name(&self_attn_layout.v_weight))?;
                let o_weight = weights.get_array2(&name(&self_attn_layout.o_weight))?;

                let q_bias = Self::load_optional_bias(
                    weights,
                    &opt_name(&self_attn_layout.q_bias),
                    hidden_size,
                )?;
                let k_bias =
                    Self::load_optional_bias(weights, &opt_name(&self_attn_layout.k_bias), kv_dim)?;
                let v_bias =
                    Self::load_optional_bias(weights, &opt_name(&self_attn_layout.v_bias), kv_dim)?;
                let o_bias = Self::load_optional_bias(
                    weights,
                    &opt_name(&self_attn_layout.o_bias),
                    hidden_size,
                )?;

                (
                    q_weight, k_weight, v_weight, o_weight, q_bias, k_bias, v_bias, o_bias,
                )
            };

        let attention = MultiHeadAttention::new(
            hidden_size,
            meta.num_attention_heads,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            o_weight,
            o_bias,
            Some(meta.num_kv_heads),
        );

        // --- 2. Load FFN ---
        let feed_forward = {
            let intermediate_weight = if meta.transpose_ffn_weights {
                weights
                    .get_array2(&name(&ffn_layout.up_weight))?
                    .t()
                    .as_standard_layout()
                    .to_owned()
            } else {
                weights.get_array2(&name(&ffn_layout.up_weight))?
            };

            let output_weight = if meta.transpose_ffn_weights {
                weights
                    .get_array2(&name(&ffn_layout.down_weight))?
                    .t()
                    .as_standard_layout()
                    .to_owned()
            } else {
                weights.get_array2(&name(&ffn_layout.down_weight))?
            };
            let intermediate_size = meta.hidden_size * 4;
            let intermediate_bias = Self::load_optional_bias(
                weights,
                &opt_name(&ffn_layout.up_bias),
                intermediate_size,
            )?;
            let output_bias =
                Self::load_optional_bias(weights, &opt_name(&ffn_layout.down_bias), hidden_size)?;

            // Logic to choose between Legacy and Standard FFN path is preserved
            FeedForward::Legacy(LegacyFeedForward::new(
                intermediate_weight,
                intermediate_bias,
                output_weight,
                output_bias,
                meta.activation,
            ))
        };

        // --- 3. Load Normalizations ---
        let attn_norm_name = name(&self_attn_layout.norm_weight);
        let attn_norm_bias = opt_name(&self_attn_layout.norm_bias);

        let self_attn_layer_norm =
            Self::load_normalization(weights, &(&attn_norm_name, &attn_norm_bias), meta.norm_eps)?
                .ok_or_else(|| anyhow!("Attn Norm required"))?;

        let ffn_norm_name = name(&ffn_layout.norm_weight);
        let ffn_norm_bias = opt_name(&ffn_layout.norm_bias);

        let ffn_layer_norm =
            Self::load_normalization(weights, &(&ffn_norm_name, &ffn_norm_bias), meta.norm_eps)?
                .ok_or_else(|| anyhow!("FFN Norm required"))?;

        Ok(GptPreNormDecoderLayer {
            self_attn: attention,
            self_attn_layer_norm,
            feedforward: feed_forward,
            ffn_layer_norm,
        })
    }
    fn load_optional_bias(weights: &ModelWeights, name: &str, size: usize) -> Result<Array1<f32>> {
        if name.is_empty() {
            Ok(Array1::zeros(size))
        } else {
            weights.get_array1(name)
        }
    }

    fn load_normalization(
        weights: &ModelWeights,
        names: &(&str, &str),
        eps: f32,
    ) -> Result<Option<Normalization>> {
        let (w_name, b_name) = names;
        if w_name.is_empty() {
            return Ok(None);
        }

        let weight = weights.get_array1(w_name)?;
        if b_name.is_empty() {
            Ok(Some(Normalization::RMSNorm(
                kjarni_transformers::normalization::RMSNorm::new(weight, eps),
            )))
        } else {
            let bias = weights.get_array1(b_name)?;
            Ok(Some(Normalization::LayerNorm(LayerNorm::new(
                weight, bias, eps,
            ))))
        }
    }
}

// --- Trait Implementation ---

impl CpuDecoder for Gpt2CpuDecoder {
    // fn embed(&self, input: ModelInput<'_>, position_offset: usize) -> Result<Array3<f32>> {
    //     match input {
    //         ModelInput::TokensCpu(ids) => {
    //             let seq_len = ids.len();
    //             // let input_ids = Array2::from_shape_vec((1, seq_len), ids.to_vec())?;

    //             // GPT-2 uses absolute position embeddings, so position_offset matters
    //             Ok(self.embeddings.forward(
    //                 &ids.to_owned(),
    //                 None,
    //                 position_offset,
    //                 self.meta.scale_embeddings,
    //             ))
    //         }
    //         ModelInput::HiddenCpu(hidden) => Ok(hidden.to_owned()),
    //         _ => Err(anyhow!(
    //             "Gpt2CpuDecoder received GPU input. Transfer to CPU first."
    //         )),
    //     }
    // }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        Ok(self.final_layer_norm.forward(&hidden_states))
    }
    fn head_dim(&self) -> usize {
        0
    }
    fn hidden_size(&self) -> usize {
        0
    }
    fn num_attention_heads(&self) -> usize {
        0
    }
    fn num_kv_heads(&self) -> usize {
        0
    }
    // fn embed_and_normalize(
    //     &self,
    //     input: ModelInput<'_>,
    //     position_offset: usize,
    // ) -> Result<Array3<f32>> {
    //     // GPT-2 is Pre-Norm (norms inside layers). No initial norm.
    //     self.embed(input, position_offset)
    // }

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

        // 1. Downcast Cache
        let mut cpu_cache_opt = cache.and_then(|c| c.as_any_mut().downcast_mut::<CpuKVCache>());

        // 2. Temp storage for new KV
        let mut new_key_values = Vec::with_capacity(end_layer - start_layer);

        for i in start_layer..end_layer {
            if i >= self.layers.len() {
                break;
            }
            let layer = &self.layers[i];

            // 3. Get View
            let past_kv_owned = cpu_cache_opt.as_ref().and_then(|c| c.get(i));
            let past_kv_views = past_kv_owned.as_ref().map(|(k, v)| (k.view(), v.view()));

            // 4. Forward
            let (new_hidden, (new_k, new_v)) =
                layer.forward(&hidden, attention_mask, past_kv_views)?;

            hidden = new_hidden;
            new_key_values.push((new_k, new_v));
        }

        // 5. Update Cache
        if let Some(cache) = cpu_cache_opt {
            for (local_idx, (k, v)) in new_key_values.into_iter().enumerate() {
                cache.update(start_layer + local_idx, &k, &v)?;
            }
            cache.increment_len(seq_len);
        }

        Ok(hidden)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward(
        &self,
        hidden: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        // 1. Embed
        // let hidden = self.embed_and_normalize(input, position_offset)?;

        // 2. Layers
        let mut output = self.forward_layers(
            &hidden,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )?;

        // 3. Final Norm (GPT-2 has a final LayerNorm)
        output = self.final_layer_norm.forward(&output);

        Ok(output)
    }
}

// GPT 2 is pre norm and load In, Out

pub struct GptPreNormDecoderLayer {
    pub self_attn: MultiHeadAttention,
    pub self_attn_layer_norm: Normalization,
    pub feedforward: FeedForward,
    pub ffn_layer_norm: Normalization,
}

impl GptPreNormDecoderLayer {
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, (Array3<f32>, Array3<f32>))> {
        let residual = hidden_states.clone();
        let ln1_out = self.self_attn_layer_norm.forward(hidden_states);
        let (attn_out, new_k, new_v) = self.self_attn.forward_with_cache(
            &ln1_out,
            None,
            Some(attention_mask),
            true,
            past_kv,
            None,
        )?;
        let attn_block_output = residual + attn_out;
        let residual = attn_block_output.clone();
        let ln2_out = self.ffn_layer_norm.forward(&attn_block_output);
        let ffn_out = self.feedforward.forward(&ln2_out)?;
        let final_output = residual + ffn_out;
        Ok((final_output, (new_k, new_v)))
    }
}
