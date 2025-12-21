// --- Standard Library ---
use std::sync::Arc;

// --- External Crates ---
use anyhow::{anyhow, Result};
use log::debug;
use ndarray::{s, Array1, Array2, Array3};

// --- Workspace Crates ---
use kjarni_transformers::{
    cache::{Cache, CpuKVCache},
    decoder::prelude::*,
    feedforward::{LegacyFeedForward, StdFeedForward},
    linear_layer::LinearLayer,
    normalization::LayerNorm,
    traits::{DecoderArchitecture, LanguageModelConfig},
    weights::{ModelWeights},
    tensor::DType,
    Embeddings, FeedForward, MultiHeadAttention, Normalization, TransformerConfig,
};

// --- Crate-Specific ---
use crate::models::gpt2::config::Gpt2Config;

/// The CPU-native implementation of the GPT-2 decoder architecture.
pub struct Gpt2CpuDecoder {
    pub embeddings: Embeddings,
    pub layers: Vec<GptPreNormDecoderLayer>,
    pub final_layer_norm: Normalization, // Use generic Normalization enum
    pub config: Arc<Gpt2Config>,
}

impl Gpt2CpuDecoder {
    pub fn new(weights: &ModelWeights, config: Arc<Gpt2Config>) -> Result<Self> {
        log::info!("Building GPT-2 CPU decoder...");

        // 1. Embeddings
        let (word_w, pos_w, _) = config.get_embedding_weight_names();
        let word_embeddings = weights.get_array2(word_w)?;

        let position_embeddings = if !pos_w.is_empty() {
            debug!("[CPU Decoder] Position embeddings: {}", pos_w);
            Some(weights.get_array2(pos_w)?)
        } else {
            debug!("[CPU Decoder] Position embeddings: None");
            None
        };

        let embeddings = Embeddings::new(word_embeddings, position_embeddings, None);

        // 2. Final Layer Norm
        let (norm_w, norm_b) = config.get_final_layer_norm_names();
        debug!("  Loading final layer norm...");

        let final_layer_norm =
            Self::load_normalization(weights, &(norm_w, norm_b), config.layer_norm_eps())?
                .ok_or_else(|| anyhow!("Final layer normalization is required"))?;

        // 3. Build decoder layers
        debug!(
            "  Building {} decoder layers...",
            config.num_hidden_layers()
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers());

        // Cast config to trait object for generic layer building
        let dyn_config = config.clone() as Arc<dyn DecoderArchitecture + Send + Sync>;

        for i in 0..config.num_hidden_layers() {
            let layer = Self::build_layer(weights, dyn_config.as_ref(), i)?;
            layers.push(layer);
        }

        debug!("✓ Gpt2CpuDecoder built successfully");

        Ok(Self {
            embeddings,
            final_layer_norm,
            layers,
            config,
        })
    }

    fn build_layer(
        weights: &ModelWeights,
        config: &dyn DecoderArchitecture,
        layer_idx: usize,
    ) -> Result<GptPreNormDecoderLayer> {
        let attn_names = config.get_attention_names(layer_idx);
        let ffn_names = config.get_feed_forward_names(layer_idx);
        let hidden_size = config.hidden_size();
        let kv_dim = config.kv_dim();

        // Load attention weights (handle combined QKV for GPT-2)
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
                // Fallback for separate weights (unlikely for standard GPT-2)
                let layer_attn_names = config.get_layer_attention_names(layer_idx);
                let q_weight = weights.get_array2(&layer_attn_names.q_weight)?;
                let k_weight = weights.get_array2(&layer_attn_names.k_weight)?;
                let v_weight = weights.get_array2(&layer_attn_names.v_weight)?;
                let o_weight = weights.get_array2(&layer_attn_names.output_weight)?;

                let q_bias =
                    Self::load_optional_bias(weights, &layer_attn_names.q_bias, hidden_size)?;
                let k_bias = Self::load_optional_bias(weights, &layer_attn_names.k_bias, kv_dim)?;
                let v_bias = Self::load_optional_bias(weights, &layer_attn_names.v_bias, kv_dim)?;
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

        // Load FFN
        let feed_forward = {
            // GPT-2 Standard FFN
            let intermediate_weight = if config.transpose_ffn_weights() {
                let arr = weights.get_array2(&ffn_names.intermediate_weight)?;
                arr.t().as_standard_layout().to_owned()
            } else {
                weights.get_array2(&ffn_names.intermediate_weight)?
            };

            let output_weight = if config.transpose_ffn_weights() {
                let arr = weights.get_array2(&ffn_names.output_weight)?;
                arr.t().as_standard_layout().to_owned()
            } else {
                weights.get_array2(&ffn_names.output_weight)?
            };

            // Layout assertions (Legacy/Standard handling)
            if config.legacy_ffn_weights() {
                let expected_inter_shape = [config.hidden_size(), config.intermediate_size()];
                let expected_out_shape = [config.intermediate_size(), config.hidden_size()];
                assert_eq!(
                    intermediate_weight.shape(),
                    expected_inter_shape,
                    "FFN Inter shape mismatch"
                );
                assert_eq!(
                    output_weight.shape(),
                    expected_out_shape,
                    "FFN Out shape mismatch"
                );
            } else {
                let expected_inter_shape = [config.intermediate_size(), config.hidden_size()];
                let expected_out_shape = [config.hidden_size(), config.intermediate_size()];
                assert_eq!(
                    intermediate_weight.shape(),
                    expected_inter_shape,
                    "FFN Inter shape mismatch"
                );
                assert_eq!(
                    output_weight.shape(),
                    expected_out_shape,
                    "FFN Out shape mismatch"
                );
            }

            let intermediate_bias = Self::load_optional_bias(
                weights,
                &ffn_names.intermediate_bias,
                config.intermediate_size(),
            )?;
            let output_bias =
                Self::load_optional_bias(weights, &ffn_names.output_bias, hidden_size)?;

            if config.legacy_ffn_weights() {
                FeedForward::Legacy(LegacyFeedForward::new(
                    intermediate_weight,
                    intermediate_bias,
                    output_weight,
                    output_bias,
                    config.activation_function(),
                ))
            } else {
                FeedForward::Standard(StdFeedForward::new(
                    intermediate_weight,
                    intermediate_bias,
                    output_weight,
                    output_bias,
                    config.activation_function(),
                ))
            }
        };

        // Load Normalization
        let (attn_norm_name, attn_norm_bias) = if !attn_names.qkv_weight.is_empty() {
            (
                attn_names.norm_weight.as_str(),
                attn_names.norm_bias.as_str(),
            )
        } else {
            unimplemented!()
        };

        let self_attn_layer_norm = Self::load_normalization(
            weights,
            &(attn_norm_name, attn_norm_bias),
            config.layer_norm_eps(),
        )?
        .ok_or_else(|| anyhow!("Attn Norm required"))?;

        let ffn_layer_norm = Self::load_normalization(
            weights,
            &(ffn_names.norm_weight.as_str(), ffn_names.norm_bias.as_str()),
            config.layer_norm_eps(),
        )?
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
    fn embed(&self, input: DecoderInput<'_>, position_offset: usize) -> Result<Array3<f32>> {
        match input {
            DecoderInput::TokensCpu(ids) => {
                let seq_len = ids.len();
                let input_ids = Array2::from_shape_vec((1, seq_len), ids.to_vec())?;

                // GPT-2 uses absolute position embeddings, so position_offset matters
                Ok(self.embeddings.forward(
                    &input_ids,
                    None,
                    position_offset,
                    self.config.scale_embeddings(),
                ))
            }
            DecoderInput::HiddenCpu(hidden) => Ok(hidden.clone()),
            _ => Err(anyhow!(
                "Gpt2CpuDecoder received GPU input. Transfer to CPU first."
            )),
        }
    }

    fn embed_and_normalize(
        &self,
        input: DecoderInput<'_>,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        // GPT-2 is Pre-Norm (norms inside layers). No initial norm.
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
        input: DecoderInput<'_>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        // 1. Embed
        let hidden = self.embed_and_normalize(input, position_offset)?;

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

