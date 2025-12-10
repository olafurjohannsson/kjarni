
use crate::models::bart::config::BartConfig;
use anyhow::Result;
use async_trait::async_trait;
use edgetransformers::cache::CpuBeamKVCache;
use edgetransformers::activations::Activation;
use edgetransformers::encoder_decoder::{DecoderCrossAttention, DecoderSelfAttention};
use edgetransformers::decoder_cross_attn_layer::DecoderCrossAttentionLayer;
use edgetransformers::embeddings::Embeddings;
use edgetransformers::encoder::TransformerEncoder;
use edgetransformers::feedforward::{FeedForward, LegacyFeedForward, StdFeedForward};
use edgetransformers::linear_layer::LinearLayer;
use edgetransformers::normalization::LayerNorm;
use edgetransformers::prelude::*;
use edgetransformers::traits::{DecoderOutput};
use edgetransformers::encoder_decoder::traits::CrossAttentionDecoder;
use edgetransformers::weights::ModelWeights;
use ndarray::{Array2, Array3};
use std::sync::Arc;
use std::time::Instant;

pub struct BartCpuDecoder {
    embeddings: Embeddings,
    layers: Vec<DecoderCrossAttentionLayer>,
    // BART puts layer norm *after* embedding
    embed_layer_norm: LayerNorm, 
    config: Arc<BartConfig>,
}

impl BartCpuDecoder {
    pub fn new(weights: &ModelWeights, config: Arc<BartConfig>) -> Result<Self> {
        // 1. Embeddings
        // "model.shared.weight" is the word embedding
        // "model.decoder.embed_positions.weight" is the learned positional embedding
        let embeddings = Embeddings::new(
            weights.get_array2("model.shared.weight")?,
            Some(weights.get_array2("model.decoder.embed_positions.weight")?),
            None, // No token_type_embeddings in BART
        );

        // 2. Layernorm
        let embed_layer_norm = LayerNorm::new(
            weights.get_array1("model.decoder.layernorm_embedding.weight")?,
            weights.get_array1("model.decoder.layernorm_embedding.bias")?,
            config.layer_norm_eps,
        );

        // 3. Layers
        let mut layers = Vec::with_capacity(config.decoder_layers);
        for i in 0..config.decoder_layers {
            layers.push(Self::load_layer(weights, &config, i)?);
        }

        Ok(Self {
            embeddings,
            layers,
            embed_layer_norm,
            config,
        })
    }

    fn load_layer(weights: &ModelWeights, config: &BartConfig, i: usize) -> Result<DecoderCrossAttentionLayer> {
        let prefix = format!("model.decoder.layers.{}", i);
        let dtype = None; // Add BF16 support here later if needed

        // A. Self Attention
        let self_attn = DecoderSelfAttention::new(
            config.d_model,
            config.decoder_attention_heads,
            LinearLayer::from_weights(weights, &format!("{}.self_attn.q_proj.weight", prefix), dtype)?,
            LinearLayer::from_weights(weights, &format!("{}.self_attn.k_proj.weight", prefix), dtype)?,
            LinearLayer::from_weights(weights, &format!("{}.self_attn.v_proj.weight", prefix), dtype)?,
            LinearLayer::from_weights(weights, &format!("{}.self_attn.out_proj.weight", prefix), dtype)?,
        );
        let self_attn_norm = LayerNorm::new(
            weights.get_array1(&format!("{}.self_attn_layer_norm.weight", prefix))?,
            weights.get_array1(&format!("{}.self_attn_layer_norm.bias", prefix))?,
            config.layer_norm_eps,
        );

        // B. Cross Attention
        let cross_attn = DecoderCrossAttention::new(
            config.d_model,
            config.decoder_attention_heads,
            LinearLayer::from_weights(weights, &format!("{}.encoder_attn.q_proj.weight", prefix), dtype)?,
            LinearLayer::from_weights(weights, &format!("{}.encoder_attn.k_proj.weight", prefix), dtype)?,
            LinearLayer::from_weights(weights, &format!("{}.encoder_attn.v_proj.weight", prefix), dtype)?,
            LinearLayer::from_weights(weights, &format!("{}.encoder_attn.out_proj.weight", prefix), dtype)?,
        );
        let cross_attn_norm = LayerNorm::new(
            weights.get_array1(&format!("{}.encoder_attn_layer_norm.weight", prefix))?,
            weights.get_array1(&format!("{}.encoder_attn_layer_norm.bias", prefix))?,
            config.layer_norm_eps,
        );

        // C. Feed Forward (FC1 -> FC2)
        // Note: Using raw arrays for StdFeedForward until it supports LinearLayer
        let fc1 = weights.get_array2(&format!("{}.fc1.weight", prefix))?;
        let fc2 = weights.get_array2(&format!("{}.fc2.weight", prefix))?;
        
        let ffn = FeedForward::Legacy(LegacyFeedForward::new(
            fc1.t().as_standard_layout().to_owned(), // Transpose
            weights.get_array1(&format!("{}.fc1.bias", prefix))?,
            fc2.t().as_standard_layout().to_owned(), // Transpose
            weights.get_array1(&format!("{}.fc2.bias", prefix))?,
            Activation::Gelu, // Hardcoded for BART
        ));
        let ffn_norm = LayerNorm::new(
            weights.get_array1(&format!("{}.final_layer_norm.weight", prefix))?,
            weights.get_array1(&format!("{}.final_layer_norm.bias", prefix))?,
            config.layer_norm_eps,
        );

        Ok(DecoderCrossAttentionLayer {
            self_attn,
            self_attn_layer_norm: self_attn_norm,
            cross_attn,
            cross_attn_layer_norm: cross_attn_norm,
            feedforward: ffn,
            ffn_layer_norm: ffn_norm,
        })
    }
}


impl TransformerModel for BartCpuDecoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait(?Send)]
impl CrossAttentionDecoder for BartCpuDecoder {
    type TokenInput = Array2<u32>;
    type EncoderStateInput = Array3<f32>;
    type MaskInput = Array2<f32>;
    type Output = DecoderOutput;

    fn precompute_cross_attention_kv(
        &self,
        encoder_state: &Self::EncoderStateInput,
    ) -> Result<Vec<(ndarray::Array4<f32>, ndarray::Array4<f32>)>> {
        let mut cache = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            // Call the method on the generic DecoderCrossAttention struct
            let (k, v) = layer.cross_attn.precompute_encoder_kv(encoder_state)?;
            cache.push((k, v));
        }
        Ok(cache)
    }

    async fn forward<'a>(
        &self,
        decoder_input_ids: &Self::TokenInput,
        encoder_hidden_states: &'a Self::EncoderStateInput,
        encoder_attention_mask: Option<&'a Self::MaskInput>,
        decoder_attention_mask: Option<&'a Self::MaskInput>,
        cache: Option<&mut dyn Cache>,
        cross_kv_caches: Option<&Vec<(ndarray::Array4<f32>, ndarray::Array4<f32>)>>, 
    ) -> Result<Self::Output> {
        let t_total = Instant::now();
        let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());
        
        // --- 1. Embeddings ---
        let t_embed = Instant::now();
        let mut hidden_states = self.embeddings.forward(
            decoder_input_ids,
            None,
            position_offset + 2,
            false,
        );
        log::info!("[CpuDecoder] Embeddings took: {:?}", t_embed.elapsed());

        // --- 2. Layer Norm ---
        let t_ln_embed = Instant::now();
        hidden_states = self.embed_layer_norm.forward_3d(&hidden_states);
        log::info!("[CpuDecoder] Embed LayerNorm took: {:?}", t_ln_embed.elapsed());

        let mut cpu_cache_opt = cache.and_then(|c| c.as_any_mut().downcast_mut::<CpuBeamKVCache>());
        let mut new_kv = Vec::with_capacity(self.layers.len());

        // --- 3. Layer Loop ---
        for (i, layer) in self.layers.iter().enumerate() {
            let t_layer = Instant::now();
            let past_kv_owned = cpu_cache_opt.as_ref().and_then(|c| c.get(i));
            let past_kv_view = past_kv_owned.as_ref().map(|(k, v)| (k.view(), v.view()));
            let layer_cross = cross_kv_caches.map(|v| &v[i]);

            let (out, (nk, nv)) = layer.forward(
                &hidden_states,
                encoder_hidden_states,
                decoder_attention_mask,
                encoder_attention_mask,
                past_kv_view,
                layer_cross,
            )?;
            
            hidden_states = out;
            new_kv.push((nk, nv));
            log::info!("[CpuDecoder] Layer {} took: {:?}", i, t_layer.elapsed());
        }

        // --- 4. Update Cache ---
        if let Some(cache) = cpu_cache_opt {
            for (i, (k, v)) in new_kv.into_iter().enumerate() {
                cache.update(i, &k, &v)?;
            }
        }
        
        log::info!("[CpuDecoder] Total forward pass took: {:?}", t_total.elapsed());
        Ok(DecoderOutput { last_hidden_state: hidden_states, past_key_values: None })
    }
}