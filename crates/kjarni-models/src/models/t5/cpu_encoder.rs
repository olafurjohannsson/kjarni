//! T5 CPU Encoder implementation
//!
//! Key differences from BART:
//! - Pre-norm (layer norm before attention/ff)
//! - No absolute position embeddings (uses relative position bias)
//! - RMSNorm instead of LayerNorm (no bias)
//! - Gated FFN in FLAN-T5

use crate::models::t5::config::T5Config;
use anyhow::Result;
use async_trait::async_trait;
use kjarni_transformers::{
    WgpuContext,
    activations::Activation,
    encoder::prelude::*,
    linear_layer::LinearLayer,
    normalization::RMSNorm,
    weights::ModelWeights,
    traits::{Device, TransformerModel, TransformerConfig},
};
use ndarray::{Array2, Array3, Array4, Axis, s};
use std::sync::Arc;

/// T5 Encoder Layer
pub struct T5EncoderLayer {
    /// Self attention
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,
    pub self_attn_norm: RMSNorm,
    
    /// Feed forward
    pub wi: LinearLayer,      // or wi_0 for gated
    pub wi_gate: Option<LinearLayer>, // wi_1 for gated activation
    pub wo: LinearLayer,
    pub ffn_norm: RMSNorm,
    
    /// Config
    pub num_heads: usize,
    pub head_dim: usize,
    pub activation: Activation,
}

impl T5EncoderLayer {
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
        position_bias: Option<&Array4<f32>>,
    ) -> Result<Array3<f32>> {
        let (batch, seq_len, hidden_size) = hidden_states.dim();
        
        // 1. Pre-norm for self attention
        let normed = self.self_attn_norm.forward_3d(hidden_states);
        
        // 2. Self attention
        let q = self.q_proj.forward_3d(&normed);
        let k = self.k_proj.forward_3d(&normed);
        let v = self.v_proj.forward_3d(&normed);
        
        // Reshape for multi-head attention: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q = q.into_shape((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);
        let k = k.into_shape((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);
        let v = v.into_shape((batch, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);
        
        // Attention scores: Q @ K^T / sqrt(d_k)
        // T5 doesn't scale by sqrt(d_k) - it's baked into the weights
        let mut scores = matmul_qk(&q, &k);
        
        // Add position bias if provided
        if let Some(bias) = position_bias {
            scores = scores + bias;
        }
        
        // Apply attention mask
        if let Some(mask) = attention_mask {
            let mask_4d = mask.clone().insert_axis(Axis(1)).insert_axis(Axis(2));
            // T5 uses additive mask: -inf for masked positions
            let mask_value = -1e9_f32;
            scores = scores.mapv(|s| s) + mask_4d.mapv(|m| if m == 0.0 { mask_value } else { 0.0 });
        }
        
        // Softmax
        let attn_weights = softmax_4d(&scores);
        
        // Attention output: weights @ V
        let attn_output = matmul_qv(&attn_weights, &v);
        
        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output = attn_output.permuted_axes([0, 2, 1, 3])
            .into_shape((batch, seq_len, hidden_size))?
            .as_standard_layout()
            .to_owned();
        
        // Output projection + residual
        let attn_output = self.o_proj.forward_3d(&attn_output);
        let hidden_states = hidden_states + &attn_output;
        
        // 3. Pre-norm for FFN
        let normed = self.ffn_norm.forward_3d(&hidden_states);
        
        // 4. Feed forward
        let ffn_output = if let Some(ref gate) = self.wi_gate {
            // Gated FFN: gelu(wi_0(x)) * wi_1(x)
            let gate_output = self.wi.forward_3d(&normed).mapv(|x| gelu(x));
            let hidden = gate.forward_3d(&normed);
            gate_output * hidden
        } else {
            // Standard FFN: relu(wi(x))
            self.wi.forward_3d(&normed).mapv(|x| x.max(0.0))
        };
        
        let ffn_output = self.wo.forward_3d(&ffn_output);
        
        // Residual
        Ok(&hidden_states + &ffn_output)
    }
}

pub struct T5CpuEncoder {
    /// Shared embeddings (no position embeddings in T5)
    pub embed_tokens: Array2<f32>,
    
    /// Encoder layers
    pub layers: Vec<T5EncoderLayer>,
    
    /// Final layer norm
    pub final_layer_norm: RMSNorm,
    
    /// Relative position bias (only in first layer, shared across heads)
    pub relative_attention_bias: Option<Array2<f32>>,
    
    /// Config
    pub config: Arc<T5Config>,
}

impl T5CpuEncoder {
    pub fn new(weights: &ModelWeights, config: Arc<T5Config>) -> Result<Self> {
        // 1. Embeddings
        let embed_tokens = weights.get_array2("shared.weight")?;
        
        // 2. Relative position bias (only first layer has it)
        let relative_attention_bias = weights
            .get_array2("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
            .ok();
        
        // 3. Layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(Self::load_layer(weights, &config, i)?);
        }
        
        // 4. Final layer norm
        let final_layer_norm = RMSNorm::new(
            weights.get_array1("encoder.final_layer_norm.weight")?,
            config.layer_norm_epsilon,
        );
        
        Ok(Self {
            embed_tokens,
            layers,
            final_layer_norm,
            relative_attention_bias,
            config,
        })
    }
    
    fn load_layer(weights: &ModelWeights, config: &T5Config, i: usize) -> Result<T5EncoderLayer> {
        let prefix = format!("encoder.block.{}", i);
        
        // Self attention projections (no bias in T5)
        let q_proj = LinearLayer::from_weights(
            weights,
            &format!("{}.layer.0.SelfAttention.q.weight", prefix),
            None,
        )?;
        let k_proj = LinearLayer::from_weights(
            weights,
            &format!("{}.layer.0.SelfAttention.k.weight", prefix),
            None,
        )?;
        let v_proj = LinearLayer::from_weights(
            weights,
            &format!("{}.layer.0.SelfAttention.v.weight", prefix),
            None,
        )?;
        let o_proj = LinearLayer::from_weights(
            weights,
            &format!("{}.layer.0.SelfAttention.o.weight", prefix),
            None,
        )?;
        
        let self_attn_norm = RMSNorm::new(
            weights.get_array1(&format!("{}.layer.0.layer_norm.weight", prefix))?,
            config.layer_norm_epsilon,
        );
        
        // FFN - check if gated
        let (wi, wi_gate) = if config.is_gated_activation() {
            let wi = LinearLayer::from_weights(
                weights,
                &format!("{}.layer.1.DenseReluDense.wi_0.weight", prefix),
                None,
            )?;
            let gate = LinearLayer::from_weights(
                weights,
                &format!("{}.layer.1.DenseReluDense.wi_1.weight", prefix),
                None,
            )?;
            (wi, Some(gate))
        } else {
            let wi = LinearLayer::from_weights(
                weights,
                &format!("{}.layer.1.DenseReluDense.wi.weight", prefix),
                None,
            )?;
            (wi, None)
        };
        
        let wo = LinearLayer::from_weights(
            weights,
            &format!("{}.layer.1.DenseReluDense.wo.weight", prefix),
            None,
        )?;
        
        let ffn_norm = RMSNorm::new(
            weights.get_array1(&format!("{}.layer.1.layer_norm.weight", prefix))?,
            config.layer_norm_epsilon,
        );
        
        Ok(T5EncoderLayer {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            self_attn_norm,
            wi,
            wi_gate,
            wo,
            ffn_norm,
            num_heads: config.num_heads,
            head_dim: config.d_kv,
            activation: config.activation_function(),
        })
    }
    
    /// Compute relative position bias for attention
    fn compute_position_bias(&self, query_length: usize, key_length: usize) -> Option<Array4<f32>> {
        let bias_weights = self.relative_attention_bias.as_ref()?;
        
        // T5 uses bucketed relative positions
        let num_buckets = self.config.relative_attention_num_buckets;
        let max_distance = self.config.relative_attention_max_distance;
        
        let mut bias = Array4::<f32>::zeros((1, bias_weights.shape()[1], query_length, key_length));
        
        for q_pos in 0..query_length {
            for k_pos in 0..key_length {
                let relative_position = k_pos as i32 - q_pos as i32;
                let bucket = Self::relative_position_bucket(
                    relative_position,
                    false, // encoder is bidirectional
                    num_buckets,
                    max_distance,
                );
                
                // bias_weights: [num_buckets, num_heads]
                for head in 0..bias_weights.shape()[1] {
                    bias[[0, head, q_pos, k_pos]] = bias_weights[[bucket, head]];
                }
            }
        }
        
        Some(bias)
    }
    
    /// Compute bucket for relative position (T5 algorithm)
    fn relative_position_bucket(
        relative_position: i32,
        bidirectional: bool,
        num_buckets: usize,
        max_distance: usize,
    ) -> usize {
        let mut bucket = 0i32;
        let mut rel_pos = -relative_position;
        
        let num_buckets = num_buckets as i32;
        
        if bidirectional {
            let half = num_buckets / 2;
            bucket += if rel_pos > 0 { half } else { 0 };
            rel_pos = rel_pos.abs();
        } else {
            rel_pos = rel_pos.max(0);
        }
        
        let max_exact = num_buckets / 2;
        let is_small = rel_pos < max_exact;
        
        let rel_pos_if_large = (max_exact as f32
            + (rel_pos as f32 / max_exact as f32).ln() / (max_distance as f32 / max_exact as f32).ln()
                * (num_buckets - max_exact) as f32)
            .min((num_buckets - 1) as f32) as i32;
        
        bucket += if is_small { rel_pos } else { rel_pos_if_large };
        
        bucket as usize
    }
}

impl TransformerModel for T5CpuEncoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        None
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl CpuEncoder for T5CpuEncoder {
    fn embed(&self, input_ids: &Array2<u32>, _token_type_ids: Option<&Array2<u32>>) -> Array3<f32> {
        let (batch, seq_len) = input_ids.dim();
        let hidden_size = self.embed_tokens.shape()[1];
        
        let mut embeddings = Array3::<f32>::zeros((batch, seq_len, hidden_size));
        
        for b in 0..batch {
            for s in 0..seq_len {
                let token_id = input_ids[[b, s]] as usize;
                for h in 0..hidden_size {
                    embeddings[[b, s, h]] = self.embed_tokens[[token_id, h]];
                }
            }
        }
        
        embeddings
    }
    
    fn embed_and_normalize(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Array3<f32> {
        // T5 doesn't normalize embeddings at the start
        self.embed(input_ids, token_type_ids)
    }
    
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>> {
        let seq_len = hidden_states.shape()[1];
        
        // Compute position bias once (only needed for first layer, then cached)
        let position_bias = if start_layer == 0 {
            self.compute_position_bias(seq_len, seq_len)
        } else {
            None
        };
        
        let mut hidden = hidden_states.clone();
        
        for (i, layer) in self.layers.iter().enumerate().skip(start_layer).take(end_layer - start_layer) {
            let bias = if i == 0 { position_bias.as_ref() } else { None };
            hidden = layer.forward(&hidden, Some(attention_mask), bias)?;
        }
        
        // Apply final layer norm
        if end_layer == self.layers.len() {
            hidden = self.final_layer_norm.forward_3d(&hidden);
        }
        
        Ok(hidden)
    }
    
    fn num_layers(&self) -> usize {
        self.layers.len()
    }
    
    fn hidden_size(&self) -> usize {
        self.config.d_model
    }
}

// Helper functions
fn matmul_qk(q: &Array4<f32>, k: &Array4<f32>) -> Array4<f32> {
    // [batch, heads, seq_q, head_dim] @ [batch, heads, head_dim, seq_k]
    let (batch, heads, seq_q, head_dim) = q.dim();
    let seq_k = k.shape()[2];
    
    let mut output = Array4::<f32>::zeros((batch, heads, seq_q, seq_k));
    
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq_q {
                for j in 0..seq_k {
                    let mut sum = 0.0f32;
                    for d in 0..head_dim {
                        sum += q[[b, h, i, d]] * k[[b, h, j, d]];
                    }
                    output[[b, h, i, j]] = sum;
                }
            }
        }
    }
    
    output
}

fn matmul_qv(attn: &Array4<f32>, v: &Array4<f32>) -> Array4<f32> {
    let (batch, heads, seq_q, seq_k) = attn.dim();
    let head_dim = v.shape()[3];
    
    let mut output = Array4::<f32>::zeros((batch, heads, seq_q, head_dim));
    
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq_q {
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..seq_k {
                        sum += attn[[b, h, i, j]] * v[[b, h, j, d]];
                    }
                    output[[b, h, i, d]] = sum;
                }
            }
        }
    }
    
    output
}

fn softmax_4d(x: &Array4<f32>) -> Array4<f32> {
    let mut output = x.clone();
    let (batch, heads, seq_q, seq_k) = x.dim();
    
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..seq_q {
                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..seq_k {
                    max_val = max_val.max(x[[b, h, i, j]]);
                }
                
                // Compute exp and sum
                let mut sum = 0.0f32;
                for j in 0..seq_k {
                    let exp_val = (x[[b, h, i, j]] - max_val).exp();
                    output[[b, h, i, j]] = exp_val;
                    sum += exp_val;
                }
                
                // Normalize
                for j in 0..seq_k {
                    output[[b, h, i, j]] /= sum;
                }
            }
        }
    }
    
    output
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}