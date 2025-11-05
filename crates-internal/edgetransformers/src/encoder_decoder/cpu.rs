use crate::encoder::TransformerEncoder;
use crate::traits::{
    CrossAttentionDecoder, DecoderOutput, Device, Encoder, EncoderDecoderArchitecture,
    EncoderOutput as EncoderOutputTrait, TransformerModel,
};
use crate::traits::{
    DecoderArchitecture, EncoderArchitecture, LanguageModelConfig, LayerAttentionNames,
    LayerDecoderAttentionNames, LayerFeedForwardNames, TransformerConfig,
};
use crate::weights::ModelWeights;
use anyhow::Result;
use async_trait::async_trait;
use ndarray::{Array2, Array3, s};
use std::sync::Arc;

use crate::{
    Cache, CpuKVCache, Embeddings, FeedForward, LayerNorm, MultiHeadAttention, TransformerLayer,
};

/// The CPU backend implementation for a generic `TransformerEncoderDecoder`.
pub struct CpuTransformerEncoderDecoder {
    encoder: TransformerEncoder,
    decoder_layers: Vec<TransformerLayer>,
    decoder_embeddings: Embeddings,
    decoder_embed_layer_norm: LayerNorm,
    config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
}

impl CpuTransformerEncoderDecoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
    ) -> Result<Self> {
        let encoder_config_adapter = Arc::new(EncoderConfigAdapter(config.clone()));
        let encoder = TransformerEncoder::new(weights, encoder_config_adapter, Device::Cpu, None)?;

        let (word_w, pos_w) = config.get_decoder_embedding_names();
        let decoder_embeddings = Embeddings::new(
            weights.get_array2(word_w)?,
            weights.get_array2(pos_w)?,
            None,
        );

        let (embed_norm_w, embed_norm_b) = config.get_decoder_embedding_ln_names();
        let decoder_embed_layer_norm = LayerNorm::new(weights.get_array1(embed_norm_w)?, weights.get_array1(embed_norm_b)?, config.layer_norm_eps());

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
            let feedforward = FeedForward::new(
                fc1_weight_for_constructor,
                weights.get_array1(&ffn_names.intermediate_bias)?,
                fc2_weight_for_constructor,
                weights.get_array1(&ffn_names.output_bias)?,
                crate::activations::Activation::Gelu,
            );

            let ffn_layer_norm = LayerNorm::new(
                weights.get_array1(&ffn_names.norm_weight)?,
                weights.get_array1(&ffn_names.norm_bias)?,
                config.layer_norm_eps(),
            );

            decoder_layers.push(TransformerLayer {
                self_attn,
                self_attn_layer_norm,
                cross_attn: Some(cross_attn),
                cross_attn_layer_norm: Some(cross_attn_layer_norm),
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

    fn embed_decoder_with_offset(
        &self,
        input_ids: &Array2<f32>,
        position_offset: usize,
    ) -> Array3<f32> {
        let (_batch_size, seq_len) = input_ids.dim();
        let mut hidden_states = self.decoder_embeddings.forward_word_only(input_ids);
        
        // TODO: BART specific stuff
        let start_idx = position_offset + 2;
        let end_idx = position_offset + seq_len + 2;

        let pos_embeddings_to_add = self
            .decoder_embeddings
            .position_embeddings
            .slice(s![start_idx..end_idx, ..]);
        hidden_states += &pos_embeddings_to_add;
        hidden_states
    }
}

#[async_trait]
impl<'a> CrossAttentionDecoder<'a> for CpuTransformerEncoderDecoder {
    type Input = Array2<f32>;
    type Output = DecoderOutput;

    async fn forward(
        &self,
        encoder_input_ids: &Self::Input,
        decoder_input_ids: &Self::Input,
        encoder_attention_mask: &Array2<f32>,
        decoder_attention_mask_from_generator: &Array2<f32>, // Renamed for clarity
        cache: Option<&mut dyn Cache>,
        encoder_output_opt: Option<&'a EncoderOutputTrait>,
        
    ) -> Result<Self::Output> {


        let encoder_output_ref: &EncoderOutputTrait;
        let encoder_output_owned: EncoderOutputTrait; // Holder for the owned value if we must create it

        let encoder_output = match encoder_output_opt {
            Some(output_ref) => {
                encoder_output_ref = output_ref; // Borrow the existing output
                encoder_output_ref
            },
            None => {
                encoder_output_owned = self.encoder.forward(encoder_input_ids, encoder_attention_mask, None).await?; // TODO Token_type_ids ???
                &encoder_output_owned // Borrow the newly created output
            },
        };

        let position_offset = cache.as_ref().map(|c| c.get_seq_length()).unwrap_or(0);
        let seq_len = decoder_input_ids.shape()[1];
        let total_len = position_offset + seq_len;

        let mut hidden_states = self.embed_decoder_with_offset(decoder_input_ids, position_offset);
        hidden_states = self.decoder_embed_layer_norm.forward_3d(&hidden_states);

        let mut cpu_cache_opt = cache.and_then(|c| c.as_any_mut().downcast_mut::<CpuKVCache>());
        
        // FIX #1: Use the correctly-sized mask passed in from the generator,
        // rather than ignoring it with `_` and creating a new, potentially incorrect one.
        // This ensures the mask length always matches the total key length.
        let decoder_self_attention_mask = decoder_attention_mask_from_generator;

        for (layer_idx, layer) in self.decoder_layers.iter().enumerate() {
            hidden_states = layer.forward_cross_attention(
                hidden_states,
                &encoder_output.last_hidden_state,
                decoder_self_attention_mask, // Use the correct mask
                encoder_attention_mask,
                self.config.as_ref(),
                layer_idx,
                cpu_cache_opt.as_deref_mut(),
            )?;
        }

        // FIX #2: Atomically update the cache's sequence length AFTER all layers
        // have been processed. This is the exact same fix we applied to the other decoder.
        if let Some(cache) = cpu_cache_opt {
            cache.set_seq_length(total_len);
        }

        Ok(DecoderOutput {
            last_hidden_state: hidden_states,
            past_key_values: None,
        })
    }
}

impl TransformerModel for CpuTransformerEncoderDecoder {
    fn device(&self) -> Device {
        Device::Cpu
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
}
impl LanguageModelConfig for EncoderConfigAdapter {
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
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
        self.0.transpose_attention_weights()
    }
}
impl EncoderArchitecture for EncoderConfigAdapter {
    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        self.0.get_encoder_embedding_names()
    }
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
        self.0.transpose_attention_weights()
    }
}
impl DecoderArchitecture for DecoderConfigAdapter {
    fn get_embedding_weight_names(&self) -> (&str, &str) {
        self.0.get_decoder_embedding_names()
    }
    fn get_final_layer_norm_names(&self) -> (&str, &str) {
        unimplemented!()
    }
    fn get_lm_head_name(&self) -> &str {
        self.0.get_lm_head_name()
    }
    // Note: The generic decoder doesn't have a concept of separate self/cross attention names.
    // This highlights why we build the cross-attention decoder manually.
    fn get_attention_names(&self, _layer: usize) -> LayerDecoderAttentionNames {
        unimplemented!()
    }
    fn get_feed_forward_names(&self, layer: usize) -> LayerFeedForwardNames {
        self.0.get_decoder_feed_forward_names(layer)
    }
}
