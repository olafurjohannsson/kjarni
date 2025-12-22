//! Core model traits and data structures for the Kjarni Inference Engine.
//! This is the unified interface for Encoders, Decoders, and Seq2Seq models.

use crate::activations::Activation;
pub use crate::cache::Cache;
use crate::models::base::RopeScalingConfig;
use crate::WgpuContext;
use std::any::Any;
use std::sync::Arc;

// ============================================================================
//  1. Backend & Runtime
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Wgpu,
}

impl Device {
    pub fn is_cpu(&self) -> bool { matches!(self, Device::Cpu) }
    pub fn is_gpu(&self) -> bool { matches!(self, Device::Wgpu) }
}

/// A handle to a loaded model instance.
/// Renamed from TransformerModel to better reflect its role as a runtime handle.
pub trait InferenceModel: Send + Sync {
    /// The device this model instance is running on.
    fn device(&self) -> Device;

    /// The GPU context (if applicable).
    fn context(&self) -> Option<Arc<WgpuContext>> { None }

    /// Downcasting support for custom model logic.
    fn as_any(&self) -> &dyn Any;
}

// ============================================================================
//  2. The Unified Blueprint
// ============================================================================

/// The mathematical "Ground Truth" of a model.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub norm_eps: f32,
    pub activation: Activation,
    pub rope_theta: Option<f32>,
    pub rope_scaling: Option<RopeScalingConfig>,
    pub scale_embeddings: bool,
    pub extra_pos_embeddings: usize,
    pub transpose_ffn_weights: bool,
    pub is_prenorm: bool,
}

/// The naming templates for weights inside the file (Safetensors or GGUF).
#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub token_embedding: String,
    pub position_embedding: Option<String>,
    pub token_type_embedding: Option<String>,
    pub embedding_norm: Option<String>,
    pub embedding_norm_bias: Option<String>,
    pub final_norm: String,
    pub lm_head: String,

    // Layer templates (use "{}" for index)
    pub attn_q: String,
    pub attn_q_bias: Option<String>, // Added for Encoders
    pub attn_k: String,
    pub attn_k_bias: Option<String>,
    pub attn_v: String,
    pub attn_v_bias: Option<String>,
    pub attn_o: String,
    pub attn_o_bias: Option<String>,
    pub attn_norm: String,
    pub attn_norm_bias: Option<String>,

    pub ffn_up: String,
    pub ffn_up_bias: Option<String>,
    pub ffn_down: String,
    pub ffn_down_bias: Option<String>,
    pub ffn_norm: String,
    pub ffn_norm_bias: Option<String>,
    pub ffn_gate: Option<String>,

    pub cross_attn_q: Option<String>,
    pub cross_attn_k: Option<String>,
    pub cross_attn_v: Option<String>,
    pub cross_attn_o: Option<String>,
    pub cross_attn_norm: Option<String>,
}

/// The single source of truth for model configuration.
pub trait ModelConfig: Send + Sync {
    fn metadata(&self) -> ModelMetadata;
    fn layout(&self) -> ModelLayout;
    fn model_type(&self) -> &str;
}

// //! Core model traits and data structures for transformer architectures.
// //!
//
// use crate::activations::Activation;
// pub use crate::cache::Cache;
// use crate::models::base::RopeScalingConfig;
// use crate::WgpuContext;
// use std::any::Any;
// use std::sync::Arc;
//
// /// Supported computation backends.
// ///
// /// A model is typically initialized for a specific device and will use that
// /// device for all its computations.
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum Device {
//     /// Let Kjarni decide on which to use
//     // Auto,
//     /// Execute computations on the CPU.
//     Cpu,
//     /// Execute computations on the GPU via WGPU.
//     Wgpu,
// }
//
// impl Device {
//     /// Check if the device is CPU
//     pub fn is_cpu(&self) -> bool {
//         matches!(self, Device::Cpu)
//     }
//
//     /// Check if the device is WGPU (GPU)
//     pub fn is_gpu(&self) -> bool {
//         matches!(self, Device::Wgpu)
//     }
// }
//
// /// The mathematical ground truth of a model.
// /// Used by kernels to allocate buffers and perform SIMD operations.
// #[derive(Debug, Clone)]
// pub struct ModelMetadata {
//     pub hidden_size: usize,
//     pub num_layers: usize,
//     pub num_attention_heads: usize,
//     pub num_kv_heads: usize,
//     pub head_dim: usize,
//     pub vocab_size: usize,
//     pub max_seq_len: usize,
//     pub norm_eps: f32,
//     pub activation: Activation,
//     pub rope_theta: Option<f32>,
//     pub scale_embeddings: bool,
//     pub extra_pos_embeddings: usize,
//     pub rope_scaling: Option<RopeScalingConfig>,
// }
//
// /// The naming templates for tensors.
// /// Uses "{}" as a placeholder for the layer index.
// #[derive(Debug, Clone)]
// pub struct ModelLayout {
//     pub token_embedding: String,
//     pub position_embedding: Option<String>, // None for Llama/RoPE
//     pub final_norm: String,
//     pub lm_head: String,
//
//     // Layer templates
//     pub attn_q: String,
//     pub attn_k: String,
//     pub attn_v: String,
//     pub attn_o: String,
//     pub attn_norm: String,
//
//     pub ffn_gate: Option<String>, // Some for Llama (SwiGLU), None for BERT
//     pub ffn_up: String,
//     pub ffn_down: String,
//     pub ffn_norm: String,
//
//     pub cross_attn_q: Option<String>,
//     pub cross_attn_k: Option<String>,
//     pub cross_attn_v: Option<String>,
//     pub cross_attn_o: Option<String>,
//     pub cross_attn_norm: Option<String>,
// }
//
// /// The ONLY trait a model configuration needs to implement.
// pub trait ModelConfig: Send + Sync {
//     fn metadata(&self) -> ModelMetadata;
//     fn layout(&self) -> ModelLayout;
//     fn model_type(&self) -> &str;
// }
//
// /// A base marker trait for all models in the library.
// ///
// /// Provides a common interface for identifying the model's computation device.
// /// It requires `Send + Sync` to ensure models can be safely used across threads.
// pub trait TransformerModel: Send + Sync {
//     /// Returns the computation device this model instance is configured to use.
//     fn device(&self) -> Device;
//     /// Returns the GPU context if this model is running on GPU, None for CPU.
//     fn context(&self) -> Option<Arc<WgpuContext>> {
//         None // Default implementation for CPU models
//     }
//     fn as_any(&self) -> &dyn Any;
// }
//
// /// A trait providing high-level configuration shared by all transformer models.
// ///
// /// This provides the essential hyperparameters needed to construct the layers
// /// of a transformer model, such as the hidden dimensions and the number of
// /// layers to build.
// pub trait TransformerConfig: Send + Sync {
//     /// The size of the hidden states and embedding dimensions.
//     fn hidden_size(&self) -> usize;
//     /// The number of attention heads in each multi-head attention layer.
//     fn num_attention_heads(&self) -> usize;
//     /// The total number of transformer layers (or blocks) in the model stack.
//     fn num_hidden_layers(&self) -> usize;
//     /// The epsilon value to use in LayerNorm layers for numerical stability.
//     fn layer_norm_eps(&self) -> f32;
//     ///
//     fn is_causal(&self) -> bool;
//     ///
//     fn is_prenorm(&self) -> bool;
//     fn extra_pos_embeddings(&self) -> usize {
//         0
//     }
// }
//
// /// Language model configuration (extends TransformerConfig)
// ///
// /// Adds language-model-specific properties like vocabulary size,
// /// sequence length, and intermediate dimensions.
// pub trait LanguageModelConfig: TransformerConfig {
//     /// Size of the vocabulary
//     fn vocab_size(&self) -> usize;
//
//     fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>);
//
//     /// Maximum sequence length (position embeddings)
//     fn max_position_embeddings(&self) -> usize;
//
//     /// Size of the intermediate (feedforward) layer
//     fn intermediate_size(&self) -> usize;
//
//     /// The number of key/value heads (for Grouped-Query Attention).
//     /// Defaults to the number of attention heads for standard multi-head attention.
//     fn num_key_value_heads(&self) -> usize {
//         self.num_attention_heads()
//     }
//
//     /// The total dimensionality of the key/value projections.
//     fn kv_dim(&self) -> usize {
//         let head_dim = self.hidden_size() / self.num_attention_heads();
//         self.num_key_value_heads() * head_dim
//     }
//
//     fn is_encoder_decoder(&self) -> Option<bool> {
//         None
//     }
//     fn model_type(&self) -> Option<String> {
//         None
//     }
//
//     /// The Beginning-Of-Sequence token ID, if specified by the model config.
//     fn bos_token_id(&self) -> Option<u32> {
//         None
//     }
//
//     /// The End-Of-Sequence token ID, if specified by the model config.
//     fn eos_token_id(&self) -> Option<u32> {
//         None
//     }
//
//     /// The Padding token ID, if specified by the model config.
//     fn pad_token_id(&self) -> Option<u32> {
//         None
//     }
//     fn sliding_window_size(&self) -> Option<usize> {
//         None // Default to no sliding window
//     }
//     /// If we should transpose the feedforward weights.
//     fn transpose_ffn_weights(&self) -> bool {
//         false
//     }
//
//     // In, Out layout for Legacy FFN weights
//     fn legacy_ffn_weights(&self) -> bool {
//         false
//     }
//
//     /// If we should transpose the attention weights.
//     fn transpose_attention_weights(&self) -> bool {
//         false
//     }
//     fn as_any(&self) -> &dyn Any;
//
//     fn activation_function(&self) -> Activation;
//     fn decoder_start_token_id(&self) -> u32;
//     /// Whether to scale the word + position embeddings by sqrt(hidden_size).
//     ///
//     /// This is a specific quirk used by some models, like the original facebook/bart-large.
//     fn scale_embeddings(&self) -> bool {
//         false // Default to false
//     }
//     fn forced_bos_token_id(&self) -> Option<u32> {
//         None // Default
//     }
// }
//
// /// Describes the architectural specifics of a Decoder that uses cross-attention (e.g., BART, T5).
// ///
// /// This trait provides the blueprint for building the decoder part of an encoder-decoder model,
// /// specifying the tensor names for self-attention, cross-attention, and feed-forward blocks.
// pub trait CrossAttentionDecoderArchitecture: LanguageModelConfig {
//     /// The number of layers in the decoder stack.
//     fn num_decoder_layers(&self) -> usize;
//
//     /// The tensor names for the LayerNorm applied after the decoder's embedding layer.
//     fn get_decoder_embedding_ln_names(&self) -> (&str, &str);
//
//     /// The names for the self-attention component of a specific decoder layer.
//     fn get_decoder_self_attention_names(&self, layer_index: usize) -> LayerAttentionNames;
//
//     /// The names for the cross-attention component of a specific decoder layer.
//     fn get_decoder_cross_attention_names(&self, layer_index: usize) -> LayerAttentionNames;
//
//     /// The names for the feed-forward component of a specific decoder layer.
//     fn get_decoder_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;
// }
//
// /// Describes the architectural specifics of a Decoder-only model (e.g., GPT-2, Llama).
// ///
// /// This trait will enable the creation of a generic `TransformerDecoder` for
// /// autoregressive language models by providing the necessary weight tensor names.
// pub trait DecoderArchitecture: LanguageModelConfig {
//     /// Returns the tensor names for the word and position embeddings.
//     //fn get_embedding_weight_names(&self) -> (&str, &str);
//     /// Returns the tensor names for the final LayerNorm before the LM head.
//     fn get_final_layer_norm_names(&self) -> (&str, &str);
//     /// Returns the name of the language modeling head weight tensor, which projects to the vocabulary.
//     fn get_lm_head_name(&self) -> &str;
//     /// Returns the names for the single, combined QKV projection in a decoder layer's attention block.
//     fn get_attention_names(&self, layer_index: usize) -> LayerDecoderAttentionNames;
//     /// Returns the names for the feed-forward block in a decoder layer.
//     fn get_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;
//
//     fn get_layer_attention_names(&self, layer_index: usize) -> LayerAttentionNames;
//
//     // fn as_any(&self) -> &dyn Any;
//     // KV projection dimension
//     // fn kv_dim(&self) -> usize {
//     //     let head_dim = self.hidden_size() / self.num_attention_heads();
//     //     self.num_key_value_heads() * head_dim
//     // }
// }
//
// /// A container for the concrete tensor names of an attention block in a transformer layer.
// ///
// /// An instance of this struct is returned by `EncoderArchitecture::get_attention_names`,
// /// providing the generic `TransformerEncoder` with all the keys it needs to load the
// /// weights for a single attention component from a `ModelWeights` object.
// #[derive(Debug)]
// pub struct LayerAttentionNames {
//     /// Weight tensor for the Query projection.
//     pub q_weight: String,
//     /// Bias tensor for the Query projection.
//     pub q_bias: String,
//     /// Weight tensor for the Key projection.
//     pub k_weight: String,
//     /// Bias tensor for the Key projection.
//     pub k_bias: String,
//     /// Weight tensor for the Value projection.
//     pub v_weight: String,
//     /// Bias tensor for the Value projection.
//     pub v_bias: String,
//     /// Weight tensor for the output projection.
//     pub output_weight: String,
//     /// Bias tensor for the output projection.
//     pub output_bias: String,
//     /// Weight tensor for the LayerNorm following the attention block.
//     pub norm_weight: String,
//     /// Bias tensor for the LayerNorm following the attention block.
//     pub norm_bias: String,
// }
//
// /// A container for the concrete tensor names of a feed-forward block in a transformer layer.
// ///
// /// An instance of this struct is returned by `EncoderArchitecture::get_feed_forward_names`,
// /// providing the generic `TransformerEncoder` with all the keys it needs to load the
// /// weights for a single feed-forward component from a `ModelWeights` object.
// #[derive(Debug)]
// pub struct LayerFeedForwardNames {
//     /// Weight tensor for the intermediate (first) dense layer.
//     pub intermediate_weight: String,
//     /// Bias tensor for the intermediate (first) dense layer.
//     pub intermediate_bias: String,
//     /// Weight tensor for the output (second) dense layer.
//     pub output_weight: String,
//     /// Bias tensor for the output (second) dense layer.
//     pub output_bias: String,
//     /// Weight tensor for the LayerNorm following the feed-forward block.
//     pub norm_weight: String,
//     /// Bias tensor for the LayerNorm following the feed-forward block.
//     pub norm_bias: String,
//     /// Gate projection weight (only for SwiGLU/LLaMA)
//     /// If None, uses standard FFN. If Some, uses SwiGLU.
//     pub gate_weight: Option<String>,
// }
//
// /// A container for the concrete tensor names of a decoder's causal self-attention block.
// ///
// /// This is often different from an encoder's attention, sometimes using a single
// /// combined projection matrix for Q, K, and V.
// pub struct LayerDecoderAttentionNames {
//     /// Weight for the combined Query, Key, and Value projection.
//     pub qkv_weight: String,
//     /// Bias for the combined Query, Key, and Value projection.
//     pub qkv_bias: String,
//     /// Weight for the output projection.
//     pub output_weight: String,
//     /// Bias for the output projection.
//     pub output_bias: String,
//     /// Weight for the LayerNorm preceding the attention block.
//     pub norm_weight: String,
//     /// Bias for the LayerNorm preceding the attention block.
//     pub norm_bias: String,
// }
//
// /// Describes the architectural specifics of an Encoder-Decoder model (e.g., BART, T5).
// ///
// /// This trait will enable the creation of a generic `TransformerEncoderDecoder` for
// /// sequence-to-sequence tasks. It provides methods to get tensor names for all
// /// components: the shared embeddings, the encoder stack, and the decoder stack
// /// (including its self-attention and cross-attention blocks).
// pub trait EncoderDecoderArchitecture: LanguageModelConfig + Any {
//     // <-- Inherit from LanguageModelConfig
//     // --- Shared ---
//     fn get_shared_embedding_weight_name(&self) -> &str;
//     fn get_lm_head_name(&self) -> &str;
//     fn get_final_logits_bias_name(&self) -> Option<&str>;
//
//     fn num_encoder_layers(&self) -> usize;
//     fn num_decoder_layers(&self) -> usize;
//     // fn as_any(&self) -> &dyn Any;
//     // --- Encoder Methods ---
//     fn get_encoder_embedding_names(&self) -> (&str, &str, Option<&str>);
//     fn get_encoder_embedding_ln_names(&self) -> (&str, &str);
//     fn get_encoder_attention_names(&self, layer_index: usize) -> LayerAttentionNames;
//     fn get_encoder_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;
//
//     // --- Decoder Methods ---
//     fn get_decoder_embedding_names(&self) -> (&str, &str, Option<&str>);
//     fn get_decoder_embedding_ln_names(&self) -> (&str, &str);
//     fn get_decoder_self_attention_names(&self, layer_index: usize) -> LayerAttentionNames;
//     fn get_decoder_cross_attention_names(&self, layer_index: usize) -> LayerAttentionNames;
//     fn get_decoder_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames;
//
//     // fn eos_token_id(&self) -> u32;
//
//     // fn decoder_start_token_id(&self) -> u32;
// }
