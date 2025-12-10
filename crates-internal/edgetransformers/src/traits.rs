//! Core model traits and data structures for transformer architectures.
//!

use crate::activations::Activation;
pub use crate::cache::Cache;
use crate::WgpuContext;
use anyhow::Result;
use async_trait::async_trait;
use ndarray::{Array2, Array3, Array4};
use std::any::Any;
use std::sync::Arc;

/// Supported computation backends.
///
/// A model is typically initialized for a specific device and will use that
/// device for all its computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// Let Kjarni decide on which to use
    // Auto,
    /// Execute computations on the CPU.
    Cpu,
    /// Execute computations on the GPU via WGPU.
    Wgpu,
}


/// A base marker trait for all models in the library.
///
/// Provides a common interface for identifying the model's computation device.
/// It requires `Send + Sync` to ensure models can be safely used across threads.
pub trait TransformerModel: Send + Sync {
    /// Returns the computation device this model instance is configured to use.
    fn device(&self) -> Device;
    /// Returns the GPU context if this model is running on GPU, None for CPU.
    fn context(&self) -> Option<Arc<WgpuContext>> {
        None // Default implementation for CPU models
    }
    fn as_any(&self) -> &dyn Any;
}

/// A marker trait for model configuration structs (e.g., `BertConfig`, `GptConfig`).
///
/// This allows for generic model loading and initialization from configuration data.
pub trait ModelConfig: Send + Sync + Any {}

/// The standard output from an encoder model.
#[derive(Clone)]
pub struct EncoderOutput<T = f32> {
    /// The final hidden states of the encoder.
    /// Shape: `(batch_size, sequence_length, hidden_size)`.
    pub last_hidden_state: Array3<T>,
}

/// The standard output from a decoder model.
pub struct DecoderOutput<T = f32> {
    /// The final hidden states of the decoder.
    /// Shape: `(batch_size, sequence_length, hidden_size)`.
    pub last_hidden_state: Array3<T>,
    /// The updated Key-Value cache after this forward pass.
    /// This can be fed back into the next generation step.
    pub past_key_values: Option<Vec<(Array4<T>, Array4<T>)>>,
}

pub trait CpuClassificationHead {
    fn project(&self, hidden_states: &Array3<f32>) -> Array2<f32>;  // [batch, num_classes]
}

/// Defines the asynchronous interface for an encoder model (e.g., BERT).
///
/// An encoder processes an entire input sequence at once, creating a
/// rich, contextualized representation.
#[async_trait]
pub trait Encoder: TransformerModel {
    type Input;
    type Output;

    /// Asynchronously performs a forward pass through the encoder.
    ///
    /// # Arguments
    /// * `input` - The input tensor (e.g., token embeddings).
    /// * `attention_mask` - A mask to prevent attention to padding tokens.
    ///
    /// # Returns
    /// An `EncoderOutput` containing the final hidden states.
    async fn forward(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Result<Self::Output>;

    /// Get raw hidden states
    ///
    /// Default implementation calls forward and extracts last_hidden_state.
    async fn get_hidden_states(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Result<Array3<f32>>;
}


// ============================================================================
// CPU ENCODER
// ============================================================================

/// Output from a CPU encoder.
#[derive(Clone, Debug)]
pub struct CpuEncoderOutput {
    /// Final hidden states: `[batch_size, sequence_length, hidden_size]`
    pub last_hidden_state: Array3<f32>,
}

/// CPU-based transformer encoder trait.
///
/// Provides methods for embedding lookup, normalization, and layer execution
/// with support for partial layer execution (for hybrid CPU/GPU workflows).
///
/// # Example
/// ```rust
/// // Full forward pass
/// let output = encoder.forward(&input_ids, &attention_mask, None)?;
///
/// // Partial execution for hybrid workflow
/// let hidden = encoder.embed_and_normalize(&input_ids, None);
/// let partial = encoder.forward_layers(&hidden, &mask, 0, 6)?;  // First 6 layers
/// // ... transfer to GPU and continue ...
/// ```
pub trait CpuEncoder: Send + Sync {
    /// Compute embeddings only (word + position + token_type).
    ///
    /// Does NOT apply the initial layer normalization.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs `[batch_size, sequence_length]`
    /// * `token_type_ids` - Optional token type IDs for models like BERT
    ///
    /// # Returns
    /// Hidden states `[batch_size, sequence_length, hidden_size]`
    fn embed(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Array3<f32>;

    /// Compute embeddings + initial normalization.
    ///
    /// This produces hidden states ready to be processed by encoder layers.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs `[batch_size, sequence_length]`
    /// * `token_type_ids` - Optional token type IDs
    ///
    /// # Returns
    /// Normalized hidden states `[batch_size, sequence_length, hidden_size]`
    fn embed_and_normalize(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Array3<f32>;

    /// Run layers `[start_layer, end_layer)` on pre-computed hidden states.
    ///
    /// Useful for:
    /// - Hybrid CPU/GPU execution
    /// - Layer-by-layer debugging
    /// - Partial model execution
    ///
    /// # Arguments
    /// * `hidden_states` - Input hidden states `[batch_size, seq_len, hidden_size]`
    /// * `attention_mask` - Attention mask `[batch_size, seq_len]`
    /// * `start_layer` - First layer to execute (inclusive)
    /// * `end_layer` - Last layer to execute (exclusive)
    ///
    /// # Returns
    /// Hidden states after processing through the specified layers
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>>;

    /// Number of encoder layers in this model.
    fn num_layers(&self) -> usize;

    /// Hidden dimension of the model.
    fn hidden_size(&self) -> usize;

    /// Full forward pass through the encoder.
    ///
    /// Default implementation calls embed_and_normalize + forward_layers(0, num_layers).
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs `[batch_size, sequence_length]`
    /// * `attention_mask` - Attention mask `[batch_size, sequence_length]`
    /// * `token_type_ids` - Optional token type IDs
    ///
    /// # Returns
    /// Encoder output containing the final hidden states
    fn forward(
        &self,
        input_ids: &Array2<u32>,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Result<CpuEncoderOutput> {
        let hidden = self.embed_and_normalize(input_ids, token_type_ids);
        let output = self.forward_layers(&hidden, attention_mask, 0, self.num_layers())?;
        Ok(CpuEncoderOutput {
            last_hidden_state: output,
        })
    }
}


/// Defines the asynchronous interface for a standalone decoder model (e.g., GPT-2).
///
/// A decoder is typically used for autoregressive generation, where it predicts
/// one token at a time. It uses a causal attention mask to ensure that a given
/// position can only attend to previous positions.
#[async_trait(?Send)]
pub trait Decoder: TransformerModel {
    type Input;
    type Output;

    /// Asynchronously performs a forward pass through the decoder.
    ///
    /// # Arguments
    /// * `input` - The input tensor for the current step(s).
    /// * `attention_mask` - The causal attention mask.
    /// * `cache` - An optional mutable reference to a `Cache` object to enable
    ///   efficient, incremental decoding by reusing past Key-Value states.
    ///
    /// # Returns
    /// A `DecoderOutput` containing the new hidden states and the updated cache.
    async fn forward(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output>;

    /// Get raw hidden states
    ///
    /// Default implementation calls forward with no cache and extracts last_hidden_state.
    async fn get_hidden_states(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
    ) -> Result<Array3<f32>>;
}


/// A trait providing high-level configuration shared by all transformer models.
///
/// This provides the essential hyperparameters needed to construct the layers
/// of a transformer model, such as the hidden dimensions and the number of
/// layers to build.
pub trait TransformerConfig: Send + Sync {
    /// The size of the hidden states and embedding dimensions.
    fn hidden_size(&self) -> usize;
    /// The number of attention heads in each multi-head attention layer.
    fn num_attention_heads(&self) -> usize;
    /// The total number of transformer layers (or blocks) in the model stack.
    fn num_hidden_layers(&self) -> usize;
    /// The epsilon value to use in LayerNorm layers for numerical stability.
    fn layer_norm_eps(&self) -> f32;
    ///
    fn is_causal(&self) -> bool;
    ///
    fn is_prenorm(&self) -> bool;
    fn extra_pos_embeddings(&self) -> usize {
        0
    }
}

/// Language model configuration (extends TransformerConfig)
///
/// Adds language-model-specific properties like vocabulary size,
/// sequence length, and intermediate dimensions.
pub trait LanguageModelConfig: TransformerConfig {
    /// Size of the vocabulary
    fn vocab_size(&self) -> usize;

    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>);

    /// Maximum sequence length (position embeddings)
    fn max_position_embeddings(&self) -> usize;

    /// Size of the intermediate (feedforward) layer
    fn intermediate_size(&self) -> usize;

    /// The number of key/value heads (for Grouped-Query Attention).
    /// Defaults to the number of attention heads for standard multi-head attention.
    fn num_key_value_heads(&self) -> usize {
        self.num_attention_heads()
    }

    /// The total dimensionality of the key/value projections.
    fn kv_dim(&self) -> usize {
        let head_dim = self.hidden_size() / self.num_attention_heads();
        self.num_key_value_heads() * head_dim
    }

    fn is_encoder_decoder(&self) -> Option<bool> {
        None
    }
    fn model_type(&self) -> Option<String> {
        None
    }

    /// The Beginning-Of-Sequence token ID, if specified by the model config.
    fn bos_token_id(&self) -> Option<u32> {
        None
    }

    /// The End-Of-Sequence token ID, if specified by the model config.
    fn eos_token_id(&self) -> Option<u32> {
        None
    }

    /// The Padding token ID, if specified by the model config.
    fn pad_token_id(&self) -> Option<u32> {
        None
    }
    fn sliding_window_size(&self) -> Option<usize> {
        None // Default to no sliding window
    }
    /// If we should transpose the feedforward weights.
    fn transpose_ffn_weights(&self) -> bool {
        false
    }

    // In, Out layout for Legacy FFN weights
    fn legacy_ffn_weights(&self) -> bool {
        false
    }

    /// If we should transpose the attention weights.
    fn transpose_attention_weights(&self) -> bool {
        false
    }
    fn as_any(&self) -> &dyn Any;

    fn activation_function(&self) -> Activation;
    fn decoder_start_token_id(&self) -> u32;
    /// Whether to scale the word + position embeddings by sqrt(hidden_size).
    ///
    /// This is a specific quirk used by some models, like the original facebook/bart-large.
    fn scale_embeddings(&self) -> bool {
        false // Default to false
    }
}

/// Describes the architectural specifics of a Decoder that uses cross-attention (e.g., BART, T5).
///
/// This trait provides the blueprint for building the decoder part of an encoder-decoder model,
/// specifying the tensor names for self-attention, cross-attention, and feed-forward blocks.
pub trait CrossAttentionDecoderArchitecture: LanguageModelConfig {
    /// The number of layers in the decoder stack.
    fn num_decoder_layers(&self) -> usize;

    /// The tensor names for the LayerNorm applied after the decoder's embedding layer.
    fn get_decoder_embedding_ln_names(&self) -> (&str, &str);

    /// The names for the self-attention component of a specific decoder layer.
    fn get_decoder_self_attention_names(&self, layer_index: usize) -> LayerAttentionNames;

    /// The names for the cross-attention component of a specific decoder layer.
    fn get_decoder_cross_attention_names(&self, layer_index: usize) -> LayerAttentionNames;

    /// The names for the feed-forward component of a specific decoder layer.
    fn get_decoder_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;
}


/// Describes the architectural specifics of a Decoder-only model (e.g., GPT-2, Llama).
///
/// This trait will enable the creation of a generic `TransformerDecoder` for
/// autoregressive language models by providing the necessary weight tensor names.
pub trait DecoderArchitecture: LanguageModelConfig {
    /// Returns the tensor names for the word and position embeddings.
    //fn get_embedding_weight_names(&self) -> (&str, &str);
    /// Returns the tensor names for the final LayerNorm before the LM head.
    fn get_final_layer_norm_names(&self) -> (&str, &str);
    /// Returns the name of the language modeling head weight tensor, which projects to the vocabulary.
    fn get_lm_head_name(&self) -> &str;
    /// Returns the names for the single, combined QKV projection in a decoder layer's attention block.
    fn get_attention_names(&self, layer_index: usize) -> LayerDecoderAttentionNames;
    /// Returns the names for the feed-forward block in a decoder layer.
    fn get_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;

    fn get_layer_attention_names(&self, layer_index: usize) -> LayerAttentionNames;

    // fn as_any(&self) -> &dyn Any;
    // KV projection dimension
    // fn kv_dim(&self) -> usize {
    //     let head_dim = self.hidden_size() / self.num_attention_heads();
    //     self.num_key_value_heads() * head_dim
    // }
}

/// A container for the concrete tensor names of an attention block in a transformer layer.
///
/// An instance of this struct is returned by `EncoderArchitecture::get_attention_names`,
/// providing the generic `TransformerEncoder` with all the keys it needs to load the
/// weights for a single attention component from a `ModelWeights` object.
#[derive(Debug)]
pub struct LayerAttentionNames {
    /// Weight tensor for the Query projection.
    pub q_weight: String,
    /// Bias tensor for the Query projection.
    pub q_bias: String,
    /// Weight tensor for the Key projection.
    pub k_weight: String,
    /// Bias tensor for the Key projection.
    pub k_bias: String,
    /// Weight tensor for the Value projection.
    pub v_weight: String,
    /// Bias tensor for the Value projection.
    pub v_bias: String,
    /// Weight tensor for the output projection.
    pub output_weight: String,
    /// Bias tensor for the output projection.
    pub output_bias: String,
    /// Weight tensor for the LayerNorm following the attention block.
    pub norm_weight: String,
    /// Bias tensor for the LayerNorm following the attention block.
    pub norm_bias: String,
}

/// A container for the concrete tensor names of a feed-forward block in a transformer layer.
///
/// An instance of this struct is returned by `EncoderArchitecture::get_feed_forward_names`,
/// providing the generic `TransformerEncoder` with all the keys it needs to load the
/// weights for a single feed-forward component from a `ModelWeights` object.
#[derive(Debug)]
pub struct LayerFeedForwardNames {
    /// Weight tensor for the intermediate (first) dense layer.
    pub intermediate_weight: String,
    /// Bias tensor for the intermediate (first) dense layer.
    pub intermediate_bias: String,
    /// Weight tensor for the output (second) dense layer.
    pub output_weight: String,
    /// Bias tensor for the output (second) dense layer.
    pub output_bias: String,
    /// Weight tensor for the LayerNorm following the feed-forward block.
    pub norm_weight: String,
    /// Bias tensor for the LayerNorm following the feed-forward block.
    pub norm_bias: String,
    /// Gate projection weight (only for SwiGLU/LLaMA)
    /// If None, uses standard FFN. If Some, uses SwiGLU.
    pub gate_weight: Option<String>,
}

/// A container for the concrete tensor names of a decoder's causal self-attention block.
///
/// This is often different from an encoder's attention, sometimes using a single
/// combined projection matrix for Q, K, and V.
pub struct LayerDecoderAttentionNames {
    /// Weight for the combined Query, Key, and Value projection.
    pub qkv_weight: String,
    /// Bias for the combined Query, Key, and Value projection.
    pub qkv_bias: String,
    /// Weight for the output projection.
    pub output_weight: String,
    /// Bias for the output projection.
    pub output_bias: String,
    /// Weight for the LayerNorm preceding the attention block.
    pub norm_weight: String,
    /// Bias for the LayerNorm preceding the attention block.
    pub norm_bias: String,
}

/// Describes the architectural specifics of an Encoder-Decoder model (e.g., BART, T5).
///
/// This trait will enable the creation of a generic `TransformerEncoderDecoder` for
/// sequence-to-sequence tasks. It provides methods to get tensor names for all
/// components: the shared embeddings, the encoder stack, and the decoder stack
/// (including its self-attention and cross-attention blocks).
pub trait EncoderDecoderArchitecture: LanguageModelConfig + Any {
    // <-- Inherit from LanguageModelConfig
    // --- Shared ---
    fn get_shared_embedding_weight_name(&self) -> &str;
    fn get_lm_head_name(&self) -> &str;
    fn get_final_logits_bias_name(&self) -> Option<&str>;

    fn num_encoder_layers(&self) -> usize;
    fn num_decoder_layers(&self) -> usize;
    // fn as_any(&self) -> &dyn Any;
    // --- Encoder Methods ---
    fn get_encoder_embedding_names(&self) -> (&str, &str, Option<&str>);
    fn get_encoder_embedding_ln_names(&self) -> (&str, &str);
    fn get_encoder_attention_names(&self, layer_index: usize) -> LayerAttentionNames;
    fn get_encoder_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;

    // --- Decoder Methods ---
    fn get_decoder_embedding_names(&self) -> (&str, &str, Option<&str>);
    fn get_decoder_embedding_ln_names(&self) -> (&str, &str);
    fn get_decoder_self_attention_names(&self, layer_index: usize) -> LayerAttentionNames;
    fn get_decoder_cross_attention_names(&self, layer_index: usize) -> LayerAttentionNames;
    fn get_decoder_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;

    // fn eos_token_id(&self) -> u32;

    // fn decoder_start_token_id(&self) -> u32;
}
