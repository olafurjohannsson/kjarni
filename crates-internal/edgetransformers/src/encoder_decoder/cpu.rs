use crate::activations::Activation;
use crate::encoder::TransformerEncoder;
use crate::traits::{
    CrossAttentionDecoder, DecoderOutput, Device, EncoderDecoderArchitecture, TransformerModel,
};
use crate::cache::CpuBeamKVCache;
use crate::traits::{
    CrossAttentionDecoderArchitecture, DecoderArchitecture, EncoderArchitecture, LanguageModelConfig,
    LayerAttentionNames, LayerDecoderAttentionNames, LayerFeedForwardNames, TransformerConfig,
};
use crate::weights::ModelWeights;
use crate::{
    decoder_cross_attn_layer::DecoderCrossAttentionLayer, feedforward::StdFeedForward, normalization::LayerNorm, Cache, CpuKVCache,
    Embeddings, FeedForward,
    MultiHeadAttention,
};
use anyhow::Result;
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use std::any::Any;
use std::sync::Arc;

/// The CPU backend implementation for a generic `TransformerEncoderDecoder`.
pub struct CpuTransformerEncoderDecoder {
    pub encoder: TransformerEncoder,
    pub decoder_layers: Vec<DecoderCrossAttentionLayer>,
    pub decoder_embeddings: Embeddings,
    pub decoder_embed_layer_norm: LayerNorm,
    config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
}

impl CpuTransformerEncoderDecoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
    ) -> Result<Self> {
        let encoder_config_adapter = Arc::new(EncoderConfigAdapter(config.clone()));
        let decoder_config_adapter = Arc::new(DecoderConfigAdapter(config.clone()));

        let encoder = TransformerEncoder::new(weights, encoder_config_adapter, Device::Cpu, None)?;

        let (word_w, pos_w, _) = config.get_decoder_embedding_names();
        let decoder_embeddings = Embeddings::new(
            weights.get_array2(word_w)?,
            Some(weights.get_array2(pos_w)?),
            None,
        );

        let (embed_norm_w, embed_norm_b) = config.get_decoder_embedding_ln_names();
        let decoder_embed_layer_norm = LayerNorm::new(
            weights.get_array1(embed_norm_w)?,
            weights.get_array1(embed_norm_b)?,
            decoder_config_adapter.layer_norm_eps(),
        );

        let mut decoder_layers = Vec::with_capacity(config.num_decoder_layers());
        for i in 0..config.num_decoder_layers() {
            let self_attn_names = config.get_decoder_self_attention_names(i);
            let cross_attn_names = config.get_decoder_cross_attention_names(i);
            let ffn_names = config.get_decoder_feed_forward_names(i);

            let self_attn = MultiHeadAttention::new(
                config.hidden_size(),
                config.num_attention_heads(),
                weights.get_linear_weight(&self_attn_names.q_weight)?,
                weights.get_array1(&self_attn_names.q_bias)?,
                weights.get_linear_weight(&self_attn_names.k_weight)?,
                weights.get_array1(&self_attn_names.k_bias)?,
                weights.get_linear_weight(&self_attn_names.v_weight)?,
                weights.get_array1(&self_attn_names.v_bias)?,
                weights.get_linear_weight(&self_attn_names.output_weight)?,
                weights.get_array1(&self_attn_names.output_bias)?,
                None,
            );
            let self_attn_layer_norm = LayerNorm::new(
                weights.get_array1(&self_attn_names.norm_weight)?,
                weights.get_array1(&self_attn_names.norm_bias)?,
                config.layer_norm_eps(),
            );

            let cross_attn = MultiHeadAttention::new(
                config.hidden_size(),
                config.num_attention_heads(),
                weights.get_linear_weight(&cross_attn_names.q_weight)?,
                weights.get_array1(&cross_attn_names.q_bias)?,
                weights.get_linear_weight(&cross_attn_names.k_weight)?,
                weights.get_array1(&cross_attn_names.k_bias)?,
                weights.get_linear_weight(&cross_attn_names.v_weight)?,
                weights.get_array1(&cross_attn_names.v_bias)?,
                weights.get_linear_weight(&cross_attn_names.output_weight)?,
                weights.get_array1(&cross_attn_names.output_bias)?,
                None,
            );
            let cross_attn_layer_norm = LayerNorm::new(
                weights.get_array1(&cross_attn_names.norm_weight)?,
                weights.get_array1(&cross_attn_names.norm_bias)?,
                config.layer_norm_eps(),
            );

            let raw_intermediate_w = weights.get_array2(&ffn_names.intermediate_weight)?;
            let raw_output_w = weights.get_array2(&ffn_names.output_weight)?;

            // The loader is now responsible for handling the transpose convention.
            // It prepares the weights into the [in, out] format that CpuFeedForward expects.
            let fc1_weight_for_constructor = if config.transpose_ffn_weights() {
                // The raw weight is [out, in]. Transpose it to [in, out].
                raw_intermediate_w.t().as_standard_layout().to_owned()
            } else {
                // The raw weight is already [in, out]. Use it as is.
                raw_intermediate_w
            };

            let fc2_weight_for_constructor = if config.transpose_ffn_weights() {
                raw_output_w.t().as_standard_layout().to_owned()
            } else {
                raw_output_w
            };

            // Now, call the simple "dumb" constructor with the correctly prepared weights.

            let feedforward = FeedForward::Standard(StdFeedForward::new(
                fc1_weight_for_constructor,
                weights.get_array1(&ffn_names.intermediate_bias)?,
                fc2_weight_for_constructor,
                weights.get_array1(&ffn_names.output_bias)?,
                config.activation_function(),
            ));

            let ffn_layer_norm = LayerNorm::new(
                weights.get_array1(&ffn_names.norm_weight)?,
                weights.get_array1(&ffn_names.norm_bias)?,
                config.layer_norm_eps(),
            );

            decoder_layers.push(DecoderCrossAttentionLayer {
                self_attn,
                self_attn_layer_norm,
                cross_attn: cross_attn,
                cross_attn_layer_norm: cross_attn_layer_norm,
                feedforward,
                ffn_layer_norm,
            });
        }

        Ok(Self {
            encoder,
            decoder_layers,
            decoder_embeddings,
            decoder_embed_layer_norm,
            config,
        })
    }

    pub fn encoder(&self) -> &TransformerEncoder {
        &self.encoder
    }
}

#[async_trait(?Send)]
impl CrossAttentionDecoder for CpuTransformerEncoderDecoder {
    type TokenInput = Array2<u32>;
    type EncoderStateInput = Array3<f32>;
    type MaskInput = Array2<f32>;
    type Output = DecoderOutput;

    async fn forward<'a>(
        &self,
        decoder_input_ids: &Self::TokenInput,
        encoder_hidden_states: &'a Self::EncoderStateInput,
        encoder_attention_mask: Option<&'a Self::MaskInput>,
        decoder_attention_mask: Option<&'a Self::MaskInput>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        // The generator now provides the position offset.
        let position_offset =
            cache.as_ref().map_or(0, |c| c.get_seq_length());
        let seq_len = decoder_input_ids.shape()[1];
        let total_len = position_offset + seq_len;

        // self.decoder_embeddings.forward(decoder_input_ids
        let mut hidden_states = self.decoder_embeddings.forward(
            decoder_input_ids,
            None,
            position_offset + self.config.extra_pos_embeddings(), // some models offset their embeddings like BART
            self.config.scale_embeddings(),
        );
        // 1. Embed the decoder input tokens and apply layer norm.
        hidden_states = self.decoder_embed_layer_norm.forward_3d(&hidden_states);

        let mut cpu_cache_opt = cache.and_then(|c| c.as_any_mut().downcast_mut::<CpuBeamKVCache>());
        let mut new_key_values = Vec::with_capacity(self.decoder_layers.len());

        // 2. Iterate through the decoder layers.
        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            let past_kv_owned = cpu_cache_opt
                .as_ref()
                .and_then(|cache| cache.get(layer_idx));
            let past_kv_views = past_kv_owned.as_ref().map(|(k, v)| (k.view(), v.view()));

            // Cross attention decoding
            let (new_hidden, (new_k, new_v)) = layer.forward(
                &hidden_states,
                encoder_hidden_states,
                decoder_attention_mask,
                encoder_attention_mask,
                past_kv_views,
            )?;

            hidden_states = new_hidden;
            new_key_values.push((new_k, new_v));
        }

        // 3. Update the cache *after* all layers have been processed.
        if let Some(cache) = cpu_cache_opt {
            for (layer_idx, (k, v)) in new_key_values.into_iter().enumerate() {
                cache.update(layer_idx, &k, &v)?;
            }
            // cache.increment_len(seq_len);
            // cache.set_seq_length(total_len);
        }

        Ok(DecoderOutput {
            last_hidden_state: hidden_states,
            past_key_values: None, // We manage the cache externally now
        })
        // --- FIX END ---
    }
}

impl TransformerModel for CpuTransformerEncoderDecoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub(super) struct EncoderConfigAdapter(
    pub(super) Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
);
pub(super) struct DecoderConfigAdapter(
    pub(super) Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
);

// --- Encoder Adapter Implementation ---
impl TransformerConfig for EncoderConfigAdapter {
    fn hidden_size(&self) -> usize {
        self.0.hidden_size()
    }
    fn num_attention_heads(&self) -> usize {
        self.0.num_attention_heads()
    }
    fn num_hidden_layers(&self) -> usize {
        self.0.num_encoder_layers()
    }
    fn layer_norm_eps(&self) -> f32 {
        self.0.layer_norm_eps()
    }
    fn is_causal(&self) -> bool {
        false
    }
    fn is_prenorm(&self) -> bool {
        self.0.is_prenorm()
    }
    fn extra_pos_embeddings(&self) -> usize {
        self.0.extra_pos_embeddings()
    }
}
impl LanguageModelConfig for EncoderConfigAdapter {
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn decoder_start_token_id(&self) -> u32 {
        self.0.decoder_start_token_id()
    }
    fn max_position_embeddings(&self) -> usize {
        self.0.max_position_embeddings()
    }
    fn intermediate_size(&self) -> usize {
        self.0.intermediate_size()
    }
    fn transpose_ffn_weights(&self) -> bool {
        self.0.transpose_ffn_weights()
    }
    fn transpose_attention_weights(&self) -> bool {
        // self.0.transpose_attention_weights()
        false // TODO: this flag is  inverted, we actually WANT to transpose in seq2seq
        // TODO: also consider moving this value to BartConfig
    }
    fn as_any(&self) -> &dyn Any {
        self // Simply return a reference to self as a `&dyn Any`
    }
    fn activation_function(&self) -> Activation {
        self.0.activation_function()
    }
    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        self.0.get_encoder_embedding_names()
    }
}
impl EncoderArchitecture for EncoderConfigAdapter {
    // fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
    //     self.0.get_encoder_embedding_names()
    // }
    fn get_embedding_layer_norm_names(&self) -> (&str, &str) {
        self.0.get_encoder_embedding_ln_names()
    }
    fn get_attention_names(&self, layer: usize) -> LayerAttentionNames {
        self.0.get_encoder_attention_names(layer)
    }
    fn get_feed_forward_names(&self, layer: usize) -> LayerFeedForwardNames {
        self.0.get_encoder_feed_forward_names(layer)
    }
}

impl TransformerConfig for DecoderConfigAdapter {
    fn hidden_size(&self) -> usize {
        self.0.hidden_size()
    }
    fn num_attention_heads(&self) -> usize {
        self.0.num_attention_heads()
    }
    fn num_hidden_layers(&self) -> usize {
        self.0.num_hidden_layers()
    }
    fn layer_norm_eps(&self) -> f32 {
        self.0.layer_norm_eps()
    }
    fn is_causal(&self) -> bool {
        true
    }
    fn is_prenorm(&self) -> bool {
        self.0.is_prenorm()
    }
}
impl LanguageModelConfig for DecoderConfigAdapter {
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn decoder_start_token_id(&self) -> u32 {
        self.0.decoder_start_token_id()
    }
    fn max_position_embeddings(&self) -> usize {
        self.0.max_position_embeddings()
    }
    fn intermediate_size(&self) -> usize {
        self.0.intermediate_size()
    }
    fn transpose_ffn_weights(&self) -> bool {
        true
    }
    fn transpose_attention_weights(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn Any {
        self // Simply return a reference to self as a `&dyn Any`
    }
    fn activation_function(&self) -> Activation {
        self.0.activation_function()
    }
    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        self.0.get_decoder_embedding_names()
    }
}
impl CrossAttentionDecoderArchitecture for DecoderConfigAdapter {
    fn num_decoder_layers(&self) -> usize {
        self.0.num_decoder_layers()
    }

    fn get_decoder_embedding_ln_names(&self) -> (&str, &str) {
        self.0.get_decoder_embedding_ln_names()
    }

    fn get_decoder_self_attention_names(&self, layer_index: usize) -> LayerAttentionNames {
        self.0.get_decoder_self_attention_names(layer_index)
    }

    fn get_decoder_cross_attention_names(&self, layer_index: usize) -> LayerAttentionNames {
        self.0.get_decoder_cross_attention_names(layer_index)
    }

    fn get_decoder_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames {
        self.0.get_decoder_feed_forward_names(layer_index)
    }
}
impl DecoderArchitecture for DecoderConfigAdapter {
    // fn get_embedding_weight_names(&self) -> (&str, &str) {
    //     self.0.get_decoder_embedding_names()
    // }

    // fn as_any(&self) -> &dyn Any {
    //     self // Simply return a reference to self as a `&dyn Any`
    // }
    fn get_final_layer_norm_names(&self) -> (&str, &str) {
        unimplemented!()
    }
    fn get_lm_head_name(&self) -> &str {
        "model.shared.weight"
    }
    // Note: The generic decoder doesn't have a concept of separate self/cross attention names.
    // This highlights why we build the cross-attention decoder manually.
    fn get_attention_names(&self, _layer: usize) -> LayerDecoderAttentionNames {
        unimplemented!()
    }
    fn get_feed_forward_names(&self, layer: usize) -> LayerFeedForwardNames {
        self.0.get_decoder_feed_forward_names(layer)
    }
    fn get_layer_attention_names(&self, layer_index: usize) -> LayerAttentionNames {
        unimplemented!()
    }
}
