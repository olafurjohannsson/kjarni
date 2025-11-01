use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use std::sync::Arc;

use crate::traits::{
    Device, Encoder, EncoderArchitecture, EncoderOutput, ModelConfig, TransformerConfig,
    TransformerModel,
};
use crate::weights::ModelWeights;
use crate::{Embeddings, FeedForward, LayerNorm, MultiHeadAttention, TransformerLayer};

/// The CPU backend implementation for the generic `TransformerEncoder`.
///
/// This struct holds the model's components (`ndarray`-based) in memory and
/// executes the forward pass using standard CPU computations.
pub struct CpuTransformerEncoder {
    embeddings: Embeddings,
    embeddings_layer_norm: LayerNorm,
    layers: Vec<TransformerLayer>,
    config: Arc<dyn EncoderArchitecture + Send + Sync>,
}

impl CpuTransformerEncoder {
    /// Constructs a new `CpuTransformerEncoder`.
    ///
    /// This function uses the `EncoderArchitecture` trait to dynamically look up
    /// the names of all required weight tensors and constructs the full model stack.
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn EncoderArchitecture + Send + Sync>,
    ) -> Result<Self> {
        // Load embedding weights using the names provided by the config.
        let (word_w, pos_w, type_w) = config.get_embedding_weight_names();
        let token_type_embeddings = match type_w {
            Some(name) => Some(weights.get_array2(name)?), // Load if present
            None => None,
        };
        let embeddings = Embeddings::new(
            weights.get_array2(word_w)?,
            weights.get_array2(pos_w)?,
            token_type_embeddings,
        );

        let (norm_w, norm_b) = config.get_embedding_layer_norm_names();
        let embeddings_layer_norm = LayerNorm::new(
            weights.get_array1(norm_w)?,
            weights.get_array1(norm_b)?,
            config.layer_norm_eps(),
        );

        // Build each transformer layer.
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let attn_names = config.get_attention_names(i);
            let ffn_names = config.get_feed_forward_names(i);

            let attention = MultiHeadAttention::new(
                config.hidden_size(),
                config.num_attention_heads(),
                weights.get_array2(&attn_names.q_weight)?.t().to_owned(),
                weights.get_array1(&attn_names.q_bias)?,
                weights.get_array2(&attn_names.k_weight)?.t().to_owned(),
                weights.get_array1(&attn_names.k_bias)?,
                weights.get_array2(&attn_names.v_weight)?.t().to_owned(),
                weights.get_array1(&attn_names.v_bias)?,
                weights
                    .get_array2(&attn_names.output_weight)?
                    .t()
                    .to_owned(),
                weights.get_array1(&attn_names.output_bias)?,
            );

            let feed_forward = FeedForward::new(
                weights
                    .get_array2(&ffn_names.intermediate_weight)?
                    .t()
                    .to_owned(),
                weights.get_array1(&ffn_names.intermediate_bias)?,
                weights.get_array2(&ffn_names.output_weight)?.t().to_owned(),
                weights.get_array1(&ffn_names.output_bias)?,
            );

            // The original BERT has two layer norms per block.
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

            layers.push(TransformerLayer {
                self_attn: attention,
                self_attn_layer_norm,
                cross_attn: None, // An encoder layer has no cross-attention
                cross_attn_layer_norm: None,
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
}

#[async_trait]
impl Encoder for CpuTransformerEncoder {
    type Input = Array2<f32>; // The direct input is token IDs
    type Output = EncoderOutput;

    /// Executes the forward pass for the CPU encoder.
    /// It's marked `async` to match the trait, but all operations are synchronous.
    async fn forward(
        &self,
        input_ids: &Self::Input,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<f32>>,
    ) -> Result<Self::Output> {
        // Embed inputs
        let mut hidden_states = self.embeddings.forward(input_ids, token_type_ids);

        // Apply embeddings layer norm
        hidden_states = self.embeddings_layer_norm.forward_3d(&hidden_states);

        // Transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(hidden_states, attention_mask, self.config.as_ref())?;
        }

        Ok(EncoderOutput {
            last_hidden_state: hidden_states,
        })
    }
    async fn get_hidden_states(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<f32>>,
    ) -> Result<Array3<f32>> {
        let output = self.forward(input, attention_mask, token_type_ids).await?;
        Ok(output.last_hidden_state)
    }
}
