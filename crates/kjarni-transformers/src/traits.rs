//! Core model traits and data structures for the Kjarni Inference Engine.
//! This is the unified interface for Encoders, Decoders, and Seq2Seq models.

use crate::WgpuContext;
use crate::activations::Activation;
pub use crate::cache::Cache;
use crate::models::base::RopeScalingConfig;
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
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }
    pub fn is_gpu(&self) -> bool {
        matches!(self, Device::Wgpu)
    }
}

/// A handle to a loaded model instance.
/// Renamed from TransformerModel to better reflect its role as a runtime handle.
pub trait InferenceModel: Send + Sync {
    /// The device this model instance is running on.
    fn device(&self) -> Device;

    /// The GPU context (if applicable).
    fn context(&self) -> Option<Arc<WgpuContext>> {
        None
    }

    /// Downcasting support for custom model logic.
    fn as_any(&self) -> &dyn Any;
}

// ============================================================================
//  2. The Unified Blueprint
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum NormalizationStrategy {
    LayerNorm,
    RMSNorm,
}

#[derive(Debug, Clone, PartialEq)]
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
    pub transpose_attention_weights: bool,
    pub is_prenorm: bool,
    /// Whether to scale embeddings by sqrt(d_model)
    /// Note: This is DIFFERENT from scale_embeddings!
    /// - scale_embeddings: embeddings *= sqrt(d_model) (scaling up)
    /// - normalize_embedding: layer norm after embedding lookup
    pub normalize_embedding: bool,
    pub normalization_strategy: NormalizationStrategy,
}

fn default_normalization_strategy() -> NormalizationStrategy {
    NormalizationStrategy::LayerNorm
}

/// Naming templates for a standard attention block (self- or cross-attention).
#[derive(Debug, Clone)]
pub struct AttentionLayout {
    pub q_weight: String,
    pub q_bias: Option<String>,
    pub k_weight: String,
    pub k_bias: Option<String>,
    pub v_weight: String,
    pub v_bias: Option<String>,
    pub o_weight: String,
    pub o_bias: Option<String>,
    pub norm_weight: String,
    pub norm_bias: Option<String>,
}

/// Naming templates for a standard feed-forward network block.
#[derive(Debug, Clone)]
pub struct FeedForwardLayout {
    pub up_weight: String,
    pub up_bias: Option<String>,
    pub down_weight: String,
    pub down_bias: Option<String>,
    pub gate_weight: Option<String>, // For SwiGLU/GEGLU variants
    pub norm_weight: String,
    pub norm_bias: Option<String>,
}

/// Naming templates for a complete encoder transformer layer.
#[derive(Debug, Clone)]
pub struct EncoderLayerLayout {
    pub self_attn: AttentionLayout,
    pub ffn: FeedForwardLayout,
}

/// Naming templates for a complete decoder transformer layer.
#[derive(Debug, Clone)]
pub struct DecoderLayerLayout {
    pub self_attn: AttentionLayout,
    pub cross_attn: Option<AttentionLayout>, // Only present in encoder-decoder models
    pub ffn: FeedForwardLayout,
}

/// Naming templates for all components of an encoder block.
#[derive(Debug, Clone)]
pub struct EncoderLayout {
    pub position_embedding: Option<String>,
    pub token_type_embedding: Option<String>,
    pub embedding_norm_weight: Option<String>,
    pub embedding_norm_bias: Option<String>,
    pub final_norm_weight: Option<String>,
    pub final_norm_bias: Option<String>,
    pub layer: EncoderLayerLayout,
}

/// Naming templates for all components of a decoder block.
#[derive(Debug, Clone)]
pub struct DecoderLayout {
    pub position_embedding: Option<String>,
    pub token_type_embedding: Option<String>,
    pub embedding_norm_weight: Option<String>,
    pub embedding_norm_bias: Option<String>,
    pub final_norm_weight: Option<String>,
    pub final_norm_bias: Option<String>,
    pub layer: DecoderLayerLayout,
}

/// The top-level, comprehensive layout for any transformer model.
///
/// This struct acts as the single source of truth for all tensor names.
/// - For a **decoder-only** model (Llama), `encoder` will be `None`.
/// - For an **encoder-only** model (BERT), `decoder` and `lm_head` will be `None`.
/// - For an **encoder-decoder** model (BART), both `encoder` and `decoder` will be `Some`.
#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub token_embedding: String,
    pub lm_head: String,
    pub encoder: Option<EncoderLayout>,
    pub decoder: Option<DecoderLayout>,
}
pub trait ModelConfig: Send + Sync {
    fn metadata(&self) -> ModelMetadata;
    fn layout(&self) -> ModelLayout;
    fn model_type(&self) -> &str;
    fn vocab_size(&self) -> usize {
        self.metadata().vocab_size
    }
    fn num_heads(&self) -> usize {
        self.metadata().num_attention_heads
    }
    fn context_size(&self) -> usize {
        self.metadata().max_seq_len
    }
    fn hidden_size(&self) -> usize {
        self.metadata().hidden_size
    }
    fn num_attention_heads(&self) -> usize {
        self.metadata().num_attention_heads
    }
    fn num_layers(&self) -> usize {
        self.metadata().num_layers
    }
    fn layer_norm_eps(&self) -> f32 {
        self.metadata().norm_eps
    }
    fn activation(&self) -> Activation {
        self.metadata().activation
    }
}
