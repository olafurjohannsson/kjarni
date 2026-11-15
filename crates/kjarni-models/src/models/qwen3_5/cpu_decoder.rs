//! The Qwen 3.5 decoder on the CPU.

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::WgpuContext;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use kjarni_transformers::{
    activations::Activation,
    decoder::prelude::*,
    feedforward::SwiGluFeedForward,
    linear_layer::LinearLayer,
    normalization::RMSNorm,
    rope::RoPE,
    tensor::DType,
    traits::{Cache, Device, InferenceModel},
    weights::ModelWeights,
};
use ndarray::{Array2, Array3};

use super::cache::{HybridCache, HybridDims, LinearState, SNAPSHOT_TAIL};
use super::config::{LayerKind, Qwen3_5Config};
use super::layers::{GatedAttention, GatedDeltaNet, offset_norm};

pub enum Mixer {
    /// Uses the KV cache slot given.
    Attention(GatedAttention, usize),
    /// Uses the linear state slot given.
    DeltaNet(GatedDeltaNet, usize),
}

pub struct Qwen3_5Layer {
    pub input_norm: RMSNorm,
    pub mixer: Mixer,
    pub post_norm: RMSNorm,
    pub mlp: SwiGluFeedForward,
}

pub struct Qwen3_5CpuDecoder {
    pub layers: Vec<Qwen3_5Layer>,
    pub final_norm: RMSNorm,
    pub rope: Arc<RoPE>,
    pub config: Arc<Qwen3_5Config>,
}

impl Qwen3_5CpuDecoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<Qwen3_5Config>,
        rope: Arc<RoPE>,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        anyhow::ensure!(
            rope.head_dim == config.rotary_dim,
            "the rotary table is {} wide; Qwen 3.5 rotates {}",
            rope.head_dim,
            config.rotary_dim
        );
        let c = &config;
        let eps = c.rms_norm_eps;
        let linear = |name: String| -> Result<LinearLayer> {
            LinearLayer::builder(weights, &name)
                .with_target_dtype(target_dtype)
                .build()
                .with_context(|| format!("loading {name}"))
        };
        let vector = |name: String| {
            weights
                .get_array1(&name)
                .with_context(|| format!("loading {name}"))
        };

        let (mut attention_slot, mut linear_slot) = (0, 0);
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for (i, kind) in c.layer_kinds.iter().enumerate() {
            let t = |name: &str| c.layer_tensor(i, name);
            let mixer = match kind {
                LayerKind::FullAttention => {
                    let attention = GatedAttention {
                        q_proj: linear(t("self_attn.q_proj.weight"))?,
                        k_proj: linear(t("self_attn.k_proj.weight"))?,
                        v_proj: linear(t("self_attn.v_proj.weight"))?,
                        o_proj: linear(t("self_attn.o_proj.weight"))?,
                        q_norm: offset_norm(vector(t("self_attn.q_norm.weight"))?, eps),
                        k_norm: offset_norm(vector(t("self_attn.k_norm.weight"))?, eps),
                        num_heads: c.num_attention_heads,
                        num_kv_heads: c.num_key_value_heads,
                        head_dim: c.head_dim,
                        rotary_dim: c.rotary_dim,
                    };
                    attention_slot += 1;
                    Mixer::Attention(attention, attention_slot - 1)
                }
                LayerKind::LinearAttention => {
                    let name = t("linear_attn.conv1d.weight");
                    let conv = weights
                        .get_array3(&name)
                        .with_context(|| format!("loading {name}"))?;
                    let (channels, _, kernel) = conv.dim();
                    anyhow::ensure!(
                        channels == c.conv_dim() && kernel == c.conv_kernel,
                        "{name} is {:?}; expected [{}, 1, {}]",
                        conv.dim(),
                        c.conv_dim(),
                        c.conv_kernel
                    );
                    let conv_weight = conv.into_shape_with_order((channels, kernel))?;
                    let net = GatedDeltaNet {
                        in_proj_qkv: linear(t("linear_attn.in_proj_qkv.weight"))?,
                        in_proj_z: linear(t("linear_attn.in_proj_z.weight"))?,
                        in_proj_b: linear(t("linear_attn.in_proj_b.weight"))?,
                        in_proj_a: linear(t("linear_attn.in_proj_a.weight"))?,
                        out_proj: linear(t("linear_attn.out_proj.weight"))?,
                        conv_weight,
                        neg_a: vector(t("linear_attn.A_log"))?.mapv(|v| -v.exp()),
                        dt_bias: vector(t("linear_attn.dt_bias"))?,
                        norm_weight: vector(t("linear_attn.norm.weight"))?,
                        eps,
                        num_k_heads: c.linear_num_key_heads,
                        num_v_heads: c.linear_num_value_heads,
                        key_dim: c.linear_key_head_dim,
                        value_dim: c.linear_value_head_dim,
                    };
                    linear_slot += 1;
                    Mixer::DeltaNet(net, linear_slot - 1)
                }
            };
            layers.push(Qwen3_5Layer {
                input_norm: offset_norm(vector(t("input_layernorm.weight"))?, eps),
                mixer,
                post_norm: offset_norm(vector(t("post_attention_layernorm.weight"))?, eps),
                mlp: SwiGluFeedForward::new(
                    linear(t("mlp.gate_proj.weight"))?,
                    linear(t("mlp.up_proj.weight"))?,
                    linear(t("mlp.down_proj.weight"))?,
                    Activation::SilU,
                ),
            });
        }

        Ok(Self {
            layers,
            final_norm: offset_norm(vector(c.tensor("norm.weight"))?, eps),
            rope,
            config,
        })
    }

    pub fn cache_dims(config: &Qwen3_5Config) -> HybridDims {
        HybridDims {
            attention_layers: config.num_full_attention_layers(),
            kv_dim: config.num_key_value_heads * config.head_dim,
            linear_layers: config.num_hidden_layers - config.num_full_attention_layers(),
            conv_channels: config.conv_dim(),
            conv_window: config.conv_kernel - 1,
            heads: config.linear_num_value_heads,
            key_dim: config.linear_key_head_dim,
            value_dim: config.linear_value_head_dim,
        }
    }

    fn run(
        &self,
        hidden_states: &Array3<f32>,
        position_offset: usize,
        cache: &mut HybridCache,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>> {
        cache.check_usable()?;
        let seq = hidden_states.shape()[1];
        let whole = start_layer == 0 && end_layer == self.layers.len();
        // A prefill also keeps the state from just before its end; see the cache.
        let capture = (whole && seq > SNAPSHOT_TAIL).then(|| seq - SNAPSHOT_TAIL);
        let mut captured: Vec<LinearState> = Vec::new();
        let mut hidden = hidden_states.clone();
        let (kv, states) = cache.parts_mut();
        for layer in &self.layers[start_layer..end_layer] {
            let normed = layer.input_norm.forward_3d(&hidden);
            let mixed = match &layer.mixer {
                Mixer::Attention(attention, slot) => {
                    let (k, v) = kv.get_context_view_mut(*slot, seq)?;
                    attention.forward(&normed, k, v, position_offset, &self.rope)?
                }
                Mixer::DeltaNet(net, slot) => {
                    let (out, kept) = net.forward(&normed, &mut states[*slot], capture)?;
                    captured.extend(kept);
                    out
                }
            };
            hidden += &mixed;
            let normed = layer.post_norm.forward_3d(&hidden);
            hidden += &layer.mlp.forward(&normed)?;
        }
        if end_layer == self.layers.len() {
            if let Some(c) = capture {
                let at = cache.get_seq_length() + c;
                cache.keep_snapshot(at, captured);
            }
            cache.increment_len(seq);
            if seq > 1 {
                cache.snapshot();
            }
        }
        Ok(hidden)
    }
}

impl InferenceModel for Qwen3_5CpuDecoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    fn context(&self) -> Option<Arc<WgpuContext>> {
        None
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CpuDecoder for Qwen3_5CpuDecoder {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn head_dim(&self) -> usize {
        self.config.head_dim
    }
    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
    fn num_attention_heads(&self) -> usize {
        self.config.num_attention_heads
    }
    fn num_kv_heads(&self) -> usize {
        self.config.num_key_value_heads
    }
    fn num_layers(&self) -> usize {
        self.layers.len()
    }
    fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        Ok(self.final_norm.forward_3d(hidden_states))
    }

    /// Positions are causal by construction, in the attention layers and in the
    /// recurrence alike, so the mask is not read.
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        _attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>> {
        match cache {
            Some(cache) => {
                let cache = cache
                    .as_any_mut()
                    .downcast_mut::<HybridCache>()
                    .ok_or_else(|| anyhow!("Qwen 3.5 needs its own hybrid cache"))?;
                self.run(
                    hidden_states,
                    position_offset,
                    cache,
                    start_layer,
                    end_layer,
                )
            }
            None => {
                anyhow::ensure!(
                    position_offset == 0,
                    "without a cache there is no earlier state to continue from"
                );
                let (batch, seq, _) = hidden_states.dim();
                let mut cache = HybridCache::new(&Self::cache_dims(&self.config), batch, seq);
                self.run(hidden_states, 0, &mut cache, start_layer, end_layer)
            }
        }
    }
}
