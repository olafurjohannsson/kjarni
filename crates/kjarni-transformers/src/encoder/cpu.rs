use anyhow::Result;
use ndarray::{Array2, Array3};
use std::sync::Arc;

use crate::encoder::CpuEncoder;
use crate::encoder::encoder_self_attention::EncoderSelfAttention;
use crate::encoder::traits::EncoderArchitecture;
use crate::feedforward::LegacyFeedForward;
use crate::linear_layer_old::LinearLayer;
use crate::traits::{Device, TransformerModel};
use crate::weights_old::ModelWeights;
use crate::{
    Embeddings, FeedForward, encoder::encoder_layer::EncoderLayer, normalization::LayerNorm,
};

pub struct CpuTransformerEncoder {
    embeddings: Embeddings,
    embeddings_layer_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    config: Arc<dyn EncoderArchitecture + Send + Sync>,
}

impl CpuTransformerEncoder {
    
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn EncoderArchitecture + Send + Sync>,
    ) -> Result<Self> {
        let dtype = None;
        // Load embedding weights using the names provided by the config.
        let (word_w, pos_w, type_w) = config.get_embedding_weight_names();
        let token_type_embeddings = match type_w {
            Some(name) => Some(weights.get_array2(name)?), // Load if present
            None => None,
        };
        let position_embeddings = if pos_w.is_empty() {
            None
        } else {
            Some(weights.get_array2(pos_w)?)
        };
        let embeddings = Embeddings::new(
            weights.get_array2(word_w)?,
            position_embeddings,
            token_type_embeddings,
        );

        let (norm_w, norm_b) = config.get_embedding_layer_norm_names();
        let embeddings_layer_norm = LayerNorm::new(
            weights.get_array1(norm_w)?,
            weights.get_array1(norm_b)?,
            config.layer_norm_eps(),
        );

        // LayerNorm::from_weights(weights, norm_w, norm_b, config.layer_norm_eps())?;

        // Build each transformer layer.
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let attn_names = config.get_attention_names(i);
            let ffn_names = config.get_feed_forward_names(i);

            let self_attn = EncoderSelfAttention::new(
                config.hidden_size(),
                config.num_attention_heads(),
                LinearLayer::from_weight_and_bias(
                    weights,
                    &attn_names.q_weight,
                    Some(&attn_names.q_bias),
                    false,
                    dtype,
                )?,
                LinearLayer::from_weight_and_bias(
                    weights,
                    &attn_names.k_weight,
                    Some(&attn_names.k_bias),
                    false,
                    dtype,
                )?,
                LinearLayer::from_weight_and_bias(
                    weights,
                    &attn_names.v_weight,
                    Some(&attn_names.v_bias),
                    false,
                    dtype,
                )?,
                LinearLayer::from_weight_and_bias(
                    weights,
                    &attn_names.output_weight,
                    Some(&attn_names.output_bias),
                    false,
                    dtype,
                )?,
            );

            let raw_intermediate_w = weights.get_array2(&ffn_names.intermediate_weight)?;
            let raw_output_w = weights.get_array2(&ffn_names.output_weight)?;

            let fc1_weight_for_constructor = if config.transpose_ffn_weights() {
                raw_intermediate_w.t().as_standard_layout().to_owned()
            } else {
                raw_intermediate_w
            };

            let fc2_weight_for_constructor = if config.transpose_ffn_weights() {
                raw_output_w.t().as_standard_layout().to_owned()
            } else {
                raw_output_w
            };

            let feed_forward = FeedForward::Legacy(LegacyFeedForward::new(
                fc1_weight_for_constructor,
                weights.get_array1(&ffn_names.intermediate_bias)?,
                fc2_weight_for_constructor,
                weights.get_array1(&ffn_names.output_bias)?,
                config.activation_function(),
            ));

            let self_attn_layer_norm = LayerNorm::new(
                weights.get_array1(&attn_names.norm_weight)?,
                weights.get_array1(&attn_names.norm_bias)?,
                config.layer_norm_eps(),
            );

            let ffn_layer_norm = LayerNorm::new(
                weights.get_array1(&ffn_names.norm_weight)?,
                weights.get_array1(&ffn_names.norm_bias)?,
                config.layer_norm_eps(),
            );

            layers.push(EncoderLayer {
                self_attn: self_attn,
                self_attn_layer_norm,
                feedforward: feed_forward,
                ffn_layer_norm,
            });
        }

        Ok(Self {
            embeddings,
            embeddings_layer_norm,
            layers,
            config: config as Arc<dyn EncoderArchitecture + Send + Sync>,
        })
    }
    pub fn config(&self) -> &Arc<dyn EncoderArchitecture + Send + Sync> {
        &self.config
    }
}

impl TransformerModel for CpuTransformerEncoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CpuEncoder for CpuTransformerEncoder {
    fn embed(&self, input_ids: &Array2<u32>, token_type_ids: Option<&Array2<u32>>) -> Array3<f32> {
        self.embeddings.forward(
            input_ids,
            token_type_ids,
            self.config.extra_pos_embeddings(), // Assumes config has this
            self.config.scale_embeddings(),     // Assumes config has this
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
        let is_prenorm = self.config.is_prenorm();
        for layer in &self.layers[start_layer..end_layer] {
            // Note: Your EncoderLayer::forward might need to be non-async
            hidden = layer.forward(hidden, attention_mask, None, is_prenorm)?;
        }
        Ok(hidden)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size()
    }

    // The full `forward` method is provided by the trait's default implementation,
    // so you don't need to write it here unless you want to override it.
}

// #[async_trait]
// impl Encoder for CpuTransformerEncoder {
//     type Input = Array2<u32>; // The direct input is token IDs
//     type Output = EncoderOutput;

//     /// Executes the forward pass for the CPU encoder.
//     /// It's marked `async` to match the trait, but all operations are synchronous.
//     async fn forward(
//         &self,
//         input_ids: &Self::Input,
//         attention_mask: &Array2<f32>,
//         token_type_ids: Option<&Array2<u32>>,
//     ) -> Result<Self::Output> {
//         let mut hidden_states = self.embeddings.forward(
//             input_ids,
//             token_type_ids,
//             self.config.extra_pos_embeddings(),
//             self.config.scale_embeddings(),
//         );
//         hidden_states = self.embeddings_layer_norm.forward_3d(&hidden_states);
//         let is_prenorm = self.config().is_prenorm();
//         for layer in &self.layers {
//             hidden_states = layer.forward(hidden_states, attention_mask, None, is_prenorm)?;
//         }

//         Ok(EncoderOutput {
//             last_hidden_state: hidden_states,
//         })
//     }
//     async fn get_hidden_states(
//         &self,
//         input: &Self::Input,
//         attention_mask: &Array2<f32>,
//         token_type_ids: Option<&Array2<u32>>,
//     ) -> Result<Array3<f32>> {
//         let output = self.forward(input, attention_mask, token_type_ids).await?;
//         Ok(output.last_hidden_state)
//     }
// }
