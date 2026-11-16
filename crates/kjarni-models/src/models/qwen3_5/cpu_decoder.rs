//! The Qwen 3.5 decoder on the CPU.

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::WgpuContext;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use kjarni_transformers::{
    activations::Activation,
    decoder::prelude::*,
    feedforward::{MoeFeedForward, SharedExpert, SwiGluFeedForward},
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

pub enum FeedForward {
    Dense(SwiGluFeedForward),
    Experts(MoeFeedForward),
}

impl FeedForward {
    fn forward(&self, x: &Array3<f32>) -> Result<Array3<f32>> {
        match self {
            FeedForward::Dense(mlp) => mlp.forward(x),
            FeedForward::Experts(moe) => moe.forward(x),
        }
    }
}

pub struct Qwen3_5Layer {
    pub input_norm: RMSNorm,
    pub mixer: Mixer,
    pub post_norm: RMSNorm,
    pub mlp: FeedForward,
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
        // A llama.cpp GGUF stores these norms with their +1 already added.
        let norm = |weight: ndarray::Array1<f32>| {
            if c.llama_cpp_layout {
                RMSNorm::new(weight, eps)
            } else {
                offset_norm(weight, eps)
            }
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
                        q_norm: norm(vector(t("self_attn.q_norm.weight"))?),
                        k_norm: norm(vector(t("self_attn.k_norm.weight"))?),
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
                    // `[channels, 1, kernel]`, or `[channels, kernel]` from llama.cpp.
                    let conv_weight = if c.llama_cpp_layout {
                        weights.get_array2(&name)
                    } else {
                        weights.get_array3(&name).and_then(|w| {
                            let (channels, _, kernel) = w.dim();
                            Ok(w.into_shape_with_order((channels, kernel))?)
                        })
                    }
                    .with_context(|| format!("loading {name}"))?;
                    anyhow::ensure!(
                        conv_weight.dim() == (c.conv_dim(), c.conv_kernel),
                        "{name} is {:?}; expected {} channels of {}",
                        conv_weight.dim(),
                        c.conv_dim(),
                        c.conv_kernel
                    );
                    let net = GatedDeltaNet {
                        in_proj_qkv: linear(t("linear_attn.in_proj_qkv.weight"))?,
                        in_proj_z: linear(t("linear_attn.in_proj_z.weight"))?,
                        in_proj_b: linear(t("linear_attn.in_proj_b.weight"))?,
                        in_proj_a: linear(t("linear_attn.in_proj_a.weight"))?,
                        out_proj: linear(t("linear_attn.out_proj.weight"))?,
                        conv_weight,
                        neg_a: if c.llama_cpp_layout {
                            vector(t("linear_attn.A_log"))?
                        } else {
                            vector(t("linear_attn.A_log"))?.mapv(|v| -v.exp())
                        },
                        dt_bias: vector(t("linear_attn.dt_bias"))?,
                        norm_weight: vector(t("linear_attn.norm.weight"))?,
                        eps,
                        num_k_heads: c.linear_num_key_heads,
                        num_v_heads: c.linear_num_value_heads,
                        key_dim: c.linear_key_head_dim,
                        value_dim: c.linear_value_head_dim,
                        tiled_value_heads: c.llama_cpp_layout,
                    };
                    linear_slot += 1;
                    Mixer::DeltaNet(net, linear_slot - 1)
                }
            };
            let swiglu = |gate: String, up: String, down: String| -> Result<SwiGluFeedForward> {
                Ok(SwiGluFeedForward::new(
                    linear(gate)?,
                    linear(up)?,
                    linear(down)?,
                    Activation::SilU,
                ))
            };
            let mlp = match c.moe {
                None => FeedForward::Dense(swiglu(
                    t("mlp.gate_proj.weight"),
                    t("mlp.up_proj.weight"),
                    t("mlp.down_proj.weight"),
                )?),
                Some(moe) => {
                    let experts = (0..moe.num_experts)
                        .map(|e| {
                            let [gate, up, down] = expert_names(c, i, e, moe.expert_size, weights);
                            swiglu(gate, up, down)
                                .with_context(|| format!("expert {e} of layer {i}"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let shared = if moe.shared_size > 0 {
                        let name = t("mlp.shared_expert_gate.weight");
                        // `[1, hidden]`, or `[hidden]` from llama.cpp.
                        let gate = if weights.tensor_shape(&name)?.len() == 2 {
                            weights.get_array2(&name)?.row(0).to_owned()
                        } else {
                            vector(name)?
                        };
                        Some(SharedExpert {
                            mlp: swiglu(
                                t("mlp.shared_expert.gate_proj.weight"),
                                t("mlp.shared_expert.up_proj.weight"),
                                t("mlp.shared_expert.down_proj.weight"),
                            )?,
                            gate: Some(gate),
                        })
                    } else {
                        None
                    };
                    FeedForward::Experts(MoeFeedForward {
                        router: linear(t("mlp.gate.weight"))?,
                        experts,
                        top_k: moe.top_k,
                        normalize_top_k: true,
                        shared,
                    })
                }
            };
            layers.push(Qwen3_5Layer {
                input_norm: norm(vector(t("input_layernorm.weight"))?),
                mixer,
                post_norm: norm(vector(t("post_attention_layernorm.weight"))?),
                mlp,
            });
        }

        Ok(Self {
            layers,
            final_norm: norm(vector(c.tensor("norm.weight"))?),
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

/// The gate, up and down matrices of expert `e` in layer `i`, as names the
/// weights resolve. A checkpoint may hold one tensor per expert, or all of a
/// layer's experts stacked in one, whose rows for expert `e` are named with a
/// `[start:end]` range: transformers stacks gate over up in `gate_up_proj`, and
/// llama.cpp keeps `gate`, `up` and `down` apart.
fn expert_names(
    c: &Qwen3_5Config,
    i: usize,
    e: usize,
    size: usize,
    weights: &ModelWeights,
) -> [String; 3] {
    let t = |name: &str| c.layer_tensor(i, name);
    let one = |part: &str| t(&format!("mlp.experts.{e}.{part}.weight"));
    if weights.contains(&one("gate_proj")) {
        return [one("gate_proj"), one("up_proj"), one("down_proj")];
    }
    let hidden = c.hidden_size;
    let down = format!(
        "{}[{}:{}]",
        t("mlp.experts.down_proj"),
        e * hidden,
        (e + 1) * hidden
    );
    let rows = |name: &str, from: usize| format!("{}[{from}:{}]", t(name), from + size);
    if c.llama_cpp_layout {
        [
            rows("mlp.experts.gate_proj", e * size),
            rows("mlp.experts.up_proj", e * size),
            down,
        ]
    } else {
        [
            rows("mlp.experts.gate_up_proj", 2 * e * size),
            rows("mlp.experts.gate_up_proj", 2 * e * size + size),
            down,
        ]
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
