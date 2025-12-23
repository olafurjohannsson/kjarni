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
    pub transpose_attention_weights: bool,
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
    pub final_norm_bias: Option<String>,
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
    pub cross_attn_q_bias: Option<String>,
    pub cross_attn_k: Option<String>,
    pub cross_attn_k_bias: Option<String>,
    pub cross_attn_v: Option<String>,
    pub cross_attn_v_bias: Option<String>,
    pub cross_attn_o: Option<String>,
    pub cross_attn_o_bias: Option<String>,
    pub cross_attn_norm: Option<String>,
    pub cross_attn_norm_bias: Option<String>,
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
