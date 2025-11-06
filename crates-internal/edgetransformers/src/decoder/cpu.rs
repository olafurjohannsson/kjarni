use crate::cache::CpuKVCache;
use crate::decoder_layer::PreNormDecoderLayer;
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3, Array4, s};
use rand::seq;
use std::sync::Arc;

use crate::traits::{
    Cache, CrossAttentionDecoder, Decoder, DecoderArchitecture, DecoderOutput, Device,
    EncoderOutput as EncoderOutputTrait, TransformerConfig, TransformerModel,
};
use crate::utils::{create_causal_mask, create_padding_mask_from_tokens};
use crate::weights::ModelWeights;
use crate::{
    Embeddings, FeedForward, LayerNorm, MultiHeadAttention, TransformerLayer,
    decoder_layer::DecoderLayer,
};

/// The CPU backend implementation for the generic `TransformerDecoder`.
pub struct CpuTransformerDecoder {
    embeddings: Embeddings,
    final_layer_norm: LayerNorm,
    layers: Vec<DecoderLayer>,
    config: Arc<dyn DecoderArchitecture + Send + Sync>,
}

impl CpuTransformerDecoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn DecoderArchitecture + Send + Sync>,
    ) -> Result<Self> {
        // Load embedding weights (no token_type for decoder)
        let (word_w, pos_w) = config.get_embedding_weight_names();
        let embeddings = Embeddings::new(
            weights.get_array2(word_w)?,
            weights.get_array2(pos_w)?,
            None, // No token_type_embeddings for decoder
        );

        // Get final layer norm (not embedding layer norm!)
        let (norm_w, norm_b) = config.get_final_layer_norm_names();
        let final_layer_norm = LayerNorm::new(
            weights.get_array1(norm_w)?,
            weights.get_array1(norm_b)?,
            config.layer_norm_eps(),
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let attn_names = config.get_attention_names(i);
            let ffn_names = config.get_feed_forward_names(i);

            // For GPT-2: QKV is combined, stored as [hidden, 3*hidden] - NO transpose needed!
            let qkv_weight = weights.get_array2(&attn_names.qkv_weight)?; // Remove .t()
            let qkv_bias = weights.get_array1(&attn_names.qkv_bias)?;

            let hidden_size = config.hidden_size();

            // Split along axis 1 (columns) into Q, K, V
            let q_weight = qkv_weight.slice(s![.., 0..hidden_size]).to_owned();
            let k_weight = qkv_weight
                .slice(s![.., hidden_size..2 * hidden_size])
                .to_owned();
            let v_weight = qkv_weight
                .slice(s![.., 2 * hidden_size..3 * hidden_size])
                .to_owned();

            let q_bias = qkv_bias.slice(s![0..hidden_size]).to_owned();
            let k_bias = qkv_bias.slice(s![hidden_size..2 * hidden_size]).to_owned();
            let v_bias = qkv_bias
                .slice(s![2 * hidden_size..3 * hidden_size])
                .to_owned();

            let attention = MultiHeadAttention::new(
                config.hidden_size(),
                config.num_attention_heads(),
                q_weight, // No .t() - GPT-2 weights are already [in, out]
                q_bias,
                k_weight,
                k_bias,
                v_weight,
                v_bias,
                weights.get_array2(&attn_names.output_weight)?, // No .t()
                weights.get_array1(&attn_names.output_bias)?,
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
            let feed_forward = FeedForward::new(
                fc1_weight_for_constructor,
                weights.get_array1(&ffn_names.intermediate_bias)?,
                fc2_weight_for_constructor,
                weights.get_array1(&ffn_names.output_bias)?,
                crate::activations::Activation::Gelu,
            );

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
            layers.push(DecoderLayer::PreNorm(PreNormDecoderLayer {
                self_attn: attention,
                self_attn_layer_norm,
                feedforward: feed_forward,
                ffn_layer_norm,
            }));
            // layers.push(TransformerLayer {
            //     self_attn: attention,
            //     self_attn_layer_norm,
            //     cross_attn: None, // A decoder-only model has no cross-attention
            //     cross_attn_layer_norm: None,
            //     feedforward: feed_forward,
            //     ffn_layer_norm,
            // });
        }

        Ok(Self {
            embeddings,
            final_layer_norm,
            layers,
            config: config as Arc<dyn DecoderArchitecture + Send + Sync>,
        })
    }
}

impl TransformerModel for CpuTransformerDecoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
}

#[async_trait]
impl Decoder for CpuTransformerDecoder {
    type Input = Array2<f32>;
    type Output = DecoderOutput;

    // async fn forward(
    //     &self,
    //     input_ids: &Self::Input,
    //     attention_mask: &Array2<f32>, // This is the top-level mask from the generator
    //     cache: Option<&mut dyn Cache>,
    // ) -> Result<Self::Output> {
    //     let position_offset = cache.as_ref().map(|c| c.get_seq_length()).unwrap_or(0);

    //     let mut hidden_states = self.embed_with_offset(input_ids, position_offset);

    //     let mut cpu_cache_opt = cache.and_then(|c| c.as_any_mut().downcast_mut::<CpuKVCache>());

    //     // let (batch_size, seq_len) = input_ids.dim();
    //     let seq_len = input_ids.shape()[1];
    //     let total_len = position_offset + seq_len;

    //     // let internal_mask = Array2::ones((batch_size, total_len));

    //     for (layer_idx, layer) in self.layers.iter().enumerate() {
    //         hidden_states = layer.forward_with_cache(
    //             hidden_states,
    //             // &internal_mask,
    //             &attention_mask,
    //             self.config.as_ref(),
    //             layer_idx,
    //             cpu_cache_opt.as_deref_mut(),
    //         )?;
    //     }

    //     // Apply final layer norm
    //     hidden_states = self.final_layer_norm.forward_3d(&hidden_states);

    //     // FIX: Update the cache's sequence length here, AFTER all layers are processed.
    //     if let Some(cache) = cpu_cache_opt {
    //         // cache.set_seq_length(total_len);
    //         cache.increment_len(seq_len);
    //     }

    //     Ok(DecoderOutput {
    //         last_hidden_state: hidden_states,
    //         past_key_values: None,
    //     })
    // }
    async fn forward(
        &self,
        input_ids: &Self::Input,
        attention_mask: &Array2<f32>, // The full attention mask for this step
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        // Determine the starting position for this forward pass.
        let position_offset = cache.as_ref().map(|c| c.get_seq_length()).unwrap_or(0);
        let seq_len = input_ids.shape()[1];

        // Perform the initial embedding lookup.
        let mut hidden_states = self.embed_with_offset(input_ids, position_offset);

        // Downcast the generic `Cache` trait to our specific `CpuKVCache`.
        let mut cpu_cache_opt = cache.and_then(|c| c.as_any_mut().downcast_mut::<CpuKVCache>());

        // A temporary vector to hold the new (K, V) pairs generated by each layer.
        // We collect them here and update the cache in a separate loop for clarity.
        let mut new_key_values = Vec::with_capacity(self.layers.len());

        // --- The New Orchestration Loop ---
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Step 1: Retrieve the past K/V state for THIS specific layer from the cache.
            let past_kv_owned = cpu_cache_opt
                .as_ref()
                .and_then(|cache| cache.get(layer_idx));

            // Step 2: Create references to the owned data *if it exists*.
            // `as_ref()` converts `&Option<T>` to `Option<&T>`.
            // We then map over the inner `&(k, v)` to get `(&k, &v)`.
            let past_kv_refs = past_kv_owned.as_ref().map(|(k, v)| (k.view(), v.view()));

            // Step 3: Call the layer's new, stateless forward method with the correct types.
            let (new_hidden_states, (new_k, new_v)) = layer.forward(
                &hidden_states,
                attention_mask,
                position_offset,
                past_kv_refs, // Now this correctly passes an Option of views
            )?;

            // The output of this layer becomes the input for the next.
            hidden_states = new_hidden_states;
            // Store the newly generated K/V state.
            new_key_values.push((new_k, new_v));
        }

        // Apply the final layer normalization after all layers have run.
        hidden_states = self.final_layer_norm.forward_3d(&hidden_states);

        // --- Cache Update Step ---
        // AFTER all layers are processed, commit the new K/V states to the cache.
        if let Some(cache) = cpu_cache_opt {
            for (layer_idx, (k, v)) in new_key_values.into_iter().enumerate() {
                // The cache's update method handles appending the new data.
                cache.update(layer_idx, &k, &v)?;
            }
            // Finally, increment the cache's internal length counter.
            cache.increment_len(seq_len);
        }

        Ok(DecoderOutput {
            last_hidden_state: hidden_states,
            past_key_values: None, // The `Cache` trait now manages this state externally.
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
    fn embed_with_offset(&self, input_ids: &Array2<f32>, position_offset: usize) -> Array3<f32> {
        let (batch_size, seq_len) = input_ids.dim();
        let hidden_size = self.config.hidden_size();

        let mut hidden_states = Array3::<f32>::zeros((batch_size, seq_len, hidden_size));

        // Word embeddings
        for i in 0..batch_size {
            for j in 0..seq_len {
                let token_id = input_ids[[i, j]] as usize;
                let word_emb = self.embeddings.word_embeddings.row(token_id);
                hidden_states.slice_mut(s![i, j, ..]).assign(&word_emb);
            }
        }

        // Position embeddings with offset
        for j in 0..seq_len {
            let pos_idx = position_offset + j;
            let pos_emb = self.embeddings.position_embeddings.row(pos_idx);
            for i in 0..batch_size {
                hidden_states
                    .slice_mut(s![i, j, ..])
                    .scaled_add(1.0, &pos_emb);
            }
        }

        hidden_states
    }
}
