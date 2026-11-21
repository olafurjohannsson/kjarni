//! The Qwen 3.5 decoder on the GPU, layer for layer what the CPU decoder
//! computes: the projections, norms and feed-forward use the shared GPU
//! blocks, and the parts only Qwen 3.5 has run in the kernels beside this file.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use kjarni_transformers::{
    EmbeddingConfig, LoadedEmbeddings, WgpuContext,
    decoder::prelude::*,
    gpu::{
        ComputeKernel, GpuRoPEAttention, GpuTensor, GpuTensorPool, Kernel,
        cache::{GpuKVCache, GpuRecurrentState},
        normalization::{GpuRMSNorm, GpuRMSNormWeights},
    },
    gpu_ops::{
        blocks::{GpuSwiGLUFFN, GpuSwiGLUFFNWeights, rope::GpuRoPE},
        primitives::{add::GpuAdd, linear::GpuLinearLayer},
    },
    models::base::ModelInput,
    tensor::DType,
    weights::ModelWeights,
};
use ndarray::{Array1, Array2, Axis, concatenate};

use super::cache::SNAPSHOT_TAIL;
use super::config::{LayerKind, Qwen3_5Config};

/// Snapshots of the recurrent state kept on the GPU, as on the CPU.
const SNAPSHOTS: usize = 8;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PrepareQInfo {
    rows: u32,
    seq: u32,
    heads: u32,
    head_dim: u32,
    rotary_half: u32,
    position_offset: u32,
    rope_stride: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PrepareKvInfo {
    rows: u32,
    seq: u32,
    kv_heads: u32,
    head_dim: u32,
    rotary_half: u32,
    position_offset: u32,
    rope_stride: u32,
    eps: f32,
    write_offset: u32,
    key_stride: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GateInfo {
    rows: u32,
    heads: u32,
    head_dim: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvInfo {
    batch: u32,
    seq: u32,
    channels: u32,
    kernel: u32,
    capture: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DeltaInfo {
    batch: u32,
    seq: u32,
    k_heads: u32,
    v_heads: u32,
    key_dim: u32,
    value_dim: u32,
    tiled: u32,
    capture: u32,
    channels: u32,
    scale: f32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GatedNormInfo {
    rows: u32,
    heads: u32,
    dim: u32,
    eps: f32,
}

/// The largest key head and convolution the kernels hold in private memory.
const MAX_KEY_DIM: usize = 128;
const MAX_CONV_KERNEL: usize = 8;
/// Threads sharing one head of the gated norm.
const GATED_NORM_THREADS: usize = 64;
/// Threads per (row, head) in prepare_q.wgsl and prepare_kv.wgsl.
const PREPARE_THREADS: usize = 64;
/// Value columns per workgroup of the delta rule.
const DELTA_COLS: usize = 32;

struct Kernels {
    prepare_q: ComputeKernel,
    prepare_kv: ComputeKernel,
    gate: ComputeKernel,
    conv: ComputeKernel,
    delta: ComputeKernel,
    /// The delta rule for one token per sequence, as in decoding.
    delta_step: ComputeKernel,
    gated_norm: ComputeKernel,
}

impl Kernels {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            prepare_q: ComputeKernel::new(
                device,
                "Qwen 3.5 prepare q",
                wgpu::include_wgsl!("gpu/prepare_q.wgsl"),
                &[true, true, true, true, false],
            ),
            prepare_kv: ComputeKernel::new(
                device,
                "Qwen 3.5 prepare kv",
                wgpu::include_wgsl!("gpu/prepare_kv.wgsl"),
                &[true, true, true, true, true, false, false],
            ),
            gate: ComputeKernel::new(
                device,
                "Qwen 3.5 output gate",
                wgpu::include_wgsl!("gpu/gate.wgsl"),
                &[true, false],
            ),
            conv: ComputeKernel::new(
                device,
                "Qwen 3.5 conv",
                wgpu::include_wgsl!("gpu/conv.wgsl"),
                &[true, true, false, false, false],
            ),
            delta: ComputeKernel::new(
                device,
                "Qwen 3.5 delta rule",
                wgpu::include_wgsl!("gpu/delta.wgsl"),
                &[true, true, true, true, false, false, false],
            ),
            delta_step: ComputeKernel::new(
                device,
                "Qwen 3.5 delta step",
                wgpu::include_wgsl!("gpu/delta_step.wgsl"),
                &[true, true, true, true, false, false, false],
            ),
            gated_norm: ComputeKernel::new(
                device,
                "Qwen 3.5 gated norm",
                wgpu::include_wgsl!("gpu/gated_norm.wgsl"),
                &[true, true, false],
            ),
        }
    }
}

struct Attention {
    /// Query and gate per head, `[H * 2D, hidden]`.
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    o: GpuTensor,
    q_norm: GpuTensor,
    k_norm: GpuTensor,
    /// The KV cache layer it uses.
    slot: usize,
}

struct DeltaNet {
    qkv: GpuTensor,
    z: GpuTensor,
    b: GpuTensor,
    a: GpuTensor,
    out: GpuTensor,
    /// `[channels, kernel]`.
    conv: GpuTensor,
    /// `-exp(A_log)` then `dt_bias`, one of each per value head.
    params: GpuTensor,
    norm: GpuTensor,
    /// The first of its two recurrent state buffers, convolution then delta.
    slot: usize,
}

enum Mixer {
    Attention(Attention),
    DeltaNet(DeltaNet),
}

struct Layer {
    input_norm: GpuRMSNormWeights,
    mixer: Mixer,
    post_norm: GpuRMSNormWeights,
    mlp: GpuSwiGLUFFNWeights,
}

pub struct Qwen3_5GpuDecoder {
    context: Arc<WgpuContext>,
    config: Arc<Qwen3_5Config>,
    embeddings: LoadedEmbeddings,
    layers: Vec<Layer>,
    final_norm: GpuRMSNormWeights,
    norm: GpuRMSNorm,
    linear: GpuLinearLayer,
    add: GpuAdd,
    ffn: GpuSwiGLUFFN,
    attention: GpuRoPEAttention,
    rope: Arc<GpuRoPE>,
    kernels: Kernels,
}

impl Qwen3_5GpuDecoder {
    pub fn new(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<Qwen3_5Config>,
        rope: Arc<GpuRoPE>,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        let c = &config;
        ensure!(
            c.moe.is_none(),
            "Qwen 3.5's mixture-of-experts models run on the CPU for now"
        );
        ensure!(
            c.linear_key_head_dim <= MAX_KEY_DIM && c.conv_kernel <= MAX_CONV_KERNEL,
            "the GPU kernels take key heads of up to {MAX_KEY_DIM} and convolutions of up to {MAX_CONV_KERNEL}"
        );
        let eps = c.rms_norm_eps;
        let matrix = |name: String| -> Result<GpuTensor> {
            GpuTensor::from_model_weights(context, weights, &name, target_dtype, &name)
                .with_context(|| format!("loading {name}"))
        };
        let vector = |name: String| -> Result<Array1<f32>> {
            weights
                .get_array1(&name)
                .with_context(|| format!("loading {name}"))
        };
        let upload = |v: &Array1<f32>| GpuTensor::from_ndarray(context, v);
        // A llama.cpp GGUF stores these norms with their +1 already added.
        let norm = |name: String| -> Result<GpuRMSNormWeights> {
            let w = vector(name)?;
            let w = if c.llama_cpp_layout { w } else { w + 1.0 };
            GpuRMSNormWeights::new(upload(&w)?)
        };

        let (mut attention_slot, mut state_slot) = (0, 0);
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for (i, kind) in c.layer_kinds.iter().enumerate() {
            let t = |name: &str| c.layer_tensor(i, name);
            let mixer = match kind {
                LayerKind::FullAttention => {
                    let head_norm = |name: &str| -> Result<GpuTensor> {
                        let w = vector(t(name))?;
                        upload(&if c.llama_cpp_layout { w } else { w + 1.0 })
                    };
                    attention_slot += 1;
                    Mixer::Attention(Attention {
                        q: matrix(t("self_attn.q_proj.weight"))?,
                        k: matrix(t("self_attn.k_proj.weight"))?,
                        v: matrix(t("self_attn.v_proj.weight"))?,
                        o: matrix(t("self_attn.o_proj.weight"))?,
                        q_norm: head_norm("self_attn.q_norm.weight")?,
                        k_norm: head_norm("self_attn.k_norm.weight")?,
                        slot: attention_slot - 1,
                    })
                }
                LayerKind::LinearAttention => {
                    let name = t("linear_attn.conv1d.weight");
                    let conv: Array2<f32> = if c.llama_cpp_layout {
                        weights.get_array2(&name)
                    } else {
                        weights.get_array3(&name).and_then(|w| {
                            let (channels, _, kernel) = w.dim();
                            Ok(w.into_shape_with_order((channels, kernel))?)
                        })
                    }
                    .with_context(|| format!("loading {name}"))?;
                    ensure!(
                        conv.dim() == (c.conv_dim(), c.conv_kernel),
                        "{name} is {:?}; expected {} channels of {}",
                        conv.dim(),
                        c.conv_dim(),
                        c.conv_kernel
                    );
                    let a_log = vector(t("linear_attn.A_log"))?;
                    let neg_a = if c.llama_cpp_layout {
                        a_log
                    } else {
                        a_log.mapv(|v| -v.exp())
                    };
                    let params = concatenate![Axis(0), neg_a, vector(t("linear_attn.dt_bias"))?];
                    state_slot += 2;
                    Mixer::DeltaNet(DeltaNet {
                        qkv: matrix(t("linear_attn.in_proj_qkv.weight"))?,
                        z: matrix(t("linear_attn.in_proj_z.weight"))?,
                        b: matrix(t("linear_attn.in_proj_b.weight"))?,
                        a: matrix(t("linear_attn.in_proj_a.weight"))?,
                        out: matrix(t("linear_attn.out_proj.weight"))?,
                        conv: GpuTensor::from_ndarray(
                            context,
                            &conv.as_standard_layout().to_owned(),
                        )?,
                        params: upload(&params)?,
                        norm: upload(&vector(t("linear_attn.norm.weight"))?)?,
                        slot: state_slot - 2,
                    })
                }
            };
            layers.push(Layer {
                input_norm: norm(t("input_layernorm.weight"))?,
                mixer,
                post_norm: norm(t("post_attention_layernorm.weight"))?,
                mlp: GpuSwiGLUFFNWeights::new(
                    matrix(t("mlp.gate_proj.weight"))?,
                    matrix(t("mlp.up_proj.weight"))?,
                    matrix(t("mlp.down_proj.weight"))?,
                )?,
            });
        }

        let embeddings = LoadedEmbeddings::new(
            Some(context),
            weights,
            EmbeddingConfig::new(c.tensor("embed_tokens.weight"), c.hidden_size),
            false,
            true,
            target_dtype,
        )?;
        let heads = c.num_attention_heads;
        Ok(Self {
            context: context.clone(),
            embeddings,
            layers,
            final_norm: norm(c.tensor("norm.weight"))?,
            norm: GpuRMSNorm::new(context, eps),
            linear: GpuLinearLayer::new(context),
            add: GpuAdd::new(context),
            ffn: GpuSwiGLUFFN::new(context)?,
            // Its scores, softmax and output read `head_dim` from the width.
            attention: GpuRoPEAttention::new(
                context,
                (heads * c.head_dim) as u32,
                heads as u32,
                c.num_key_value_heads as u32,
            ),
            rope,
            kernels: Kernels::new(&context.device),
            config,
        })
    }

    /// A cache for this decoder: K and V for its attention layers, and the
    /// convolution window and delta-rule state of each Gated DeltaNet layer.
    pub fn new_cache(
        context: &Arc<WgpuContext>,
        config: &Qwen3_5Config,
        batch: usize,
        max_len: usize,
    ) -> Result<GpuKVCache> {
        let linear_layers = config.num_hidden_layers - config.num_full_attention_layers();
        let mut shapes = Vec::with_capacity(2 * linear_layers);
        for _ in 0..linear_layers {
            shapes.push(vec![batch, config.conv_dim(), config.conv_kernel - 1]);
            shapes.push(vec![
                batch,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            ]);
        }
        let recurrent = GpuRecurrentState::new(context, shapes, SNAPSHOTS)?;
        Ok(GpuKVCache::new(
            context,
            config.num_full_attention_layers(),
            batch,
            config.num_key_value_heads,
            config.head_dim,
            max_len,
        )?
        .with_recurrent(recurrent))
    }

    fn linear(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: &GpuTensor,
        weight: &GpuTensor,
    ) -> GpuTensor {
        let rows = input.num_elements() / input.shape().last().unwrap();
        let out = pool.get(vec![rows, weight.shape()[0]]);
        self.linear.encode(encoder, input, weight, &out);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        layer: &Attention,
        x: &GpuTensor,
        mask: &GpuTensor,
        batch: usize,
        seq: usize,
        position_offset: usize,
        cache: &GpuKVCache,
    ) -> Result<GpuTensor> {
        let c = &self.config;
        let (heads, kv_heads, d) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let rows = batch * seq;
        let qg = self.linear(encoder, pool, x, &layer.q);
        let k = self.linear(encoder, pool, x, &layer.k);
        let v = self.linear(encoder, pool, x, &layer.v);

        let (k_store, v_store) = cache
            .get(layer.slot)
            .ok_or_else(|| anyhow!("the cache has no attention layer {}", layer.slot))?;
        let capacity = k_store.shape()[2];
        ensure!(
            position_offset + seq <= capacity,
            "KV cache holds {capacity} tokens, {position_offset} + {seq} do not fit"
        );
        let (cos, sin) = self.rope.tables();
        let rope_stride = cos.shape()[1] as u32;

        let q = pool.get(vec![batch, heads, seq, d]);
        let info = PrepareQInfo {
            rows: rows as u32,
            seq: seq as u32,
            heads: heads as u32,
            head_dim: d as u32,
            rotary_half: (c.rotary_dim / 2) as u32,
            position_offset: position_offset as u32,
            rope_stride,
            eps: c.rms_norm_eps,
        };
        self.kernels.prepare_q.encode(
            &self.context,
            encoder,
            &self.context.uniform_buffer(&info),
            &[&qg, &layer.q_norm, cos, sin, &q],
            rows * heads * PREPARE_THREADS,
            PREPARE_THREADS,
        );
        let info = PrepareKvInfo {
            rows: rows as u32,
            seq: seq as u32,
            kv_heads: kv_heads as u32,
            head_dim: d as u32,
            rotary_half: (c.rotary_dim / 2) as u32,
            position_offset: position_offset as u32,
            rope_stride,
            eps: c.rms_norm_eps,
            write_offset: position_offset as u32,
            key_stride: capacity as u32,
            _pad: [0; 2],
        };
        self.kernels.prepare_kv.encode(
            &self.context,
            encoder,
            &self.context.uniform_buffer(&info),
            &[&k, &v, &layer.k_norm, cos, sin, &k_store, &v_store],
            rows * kv_heads * PREPARE_THREADS,
            PREPARE_THREADS,
        );

        let attended = self.attention.attend(
            encoder,
            &q,
            &k_store,
            &v_store,
            mask,
            position_offset,
            position_offset + seq,
            pool,
        );
        let info = GateInfo {
            rows: rows as u32,
            heads: heads as u32,
            head_dim: d as u32,
            _pad: 0,
        };
        self.kernels.gate.encode(
            &self.context,
            encoder,
            &self.context.uniform_buffer(&info),
            &[&qg, &attended],
            rows * heads * d,
            64,
        );
        Ok(self.linear(encoder, pool, &attended, &layer.o))
    }

    /// Runs `x` on from the layer's states in `states` (its window, then its
    /// delta-rule state), writing into `captured` as well the states after
    /// `capture` tokens when that is set.
    #[allow(clippy::too_many_arguments)]
    fn delta_net(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        layer: &DeltaNet,
        x: &GpuTensor,
        batch: usize,
        seq: usize,
        states: &[GpuTensor],
        captured: Option<&[GpuTensor]>,
        capture: usize,
    ) -> GpuTensor {
        let c = &self.config;
        let rows = batch * seq;
        let channels = c.conv_dim();
        let (nk, nv, dk, dv) = (
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        );
        let (window, recurrent) = (&states[layer.slot], &states[layer.slot + 1]);
        // Without a capture the kernels still bind a buffer there, and never write it.
        let (window_out, recurrent_out) = match captured {
            Some(slots) => (&slots[layer.slot], &slots[layer.slot + 1]),
            None => (window, recurrent),
        };

        let qkv = self.linear(encoder, pool, x, &layer.qkv);
        let z = self.linear(encoder, pool, x, &layer.z);
        let b = self.linear(encoder, pool, x, &layer.b);
        let a = self.linear(encoder, pool, x, &layer.a);

        let convolved = pool.get(vec![rows, channels]);
        let info = ConvInfo {
            batch: batch as u32,
            seq: seq as u32,
            channels: channels as u32,
            kernel: c.conv_kernel as u32,
            capture: capture as u32,
            _pad: [0; 3],
        };
        self.kernels.conv.encode(
            &self.context,
            encoder,
            &self.context.uniform_buffer(&info),
            &[&qkv, &layer.conv, window, window_out, &convolved],
            batch * channels,
            64,
        );

        let core = pool.get(vec![rows, nv * dv]);
        let info = DeltaInfo {
            batch: batch as u32,
            seq: seq as u32,
            k_heads: nk as u32,
            v_heads: nv as u32,
            key_dim: dk as u32,
            value_dim: dv as u32,
            tiled: c.llama_cpp_layout as u32,
            capture: capture as u32,
            channels: channels as u32,
            scale: (dk as f32).powf(-0.5),
            _pad: [0; 2],
        };
        let delta = if seq == 1 {
            &self.kernels.delta_step
        } else {
            &self.kernels.delta
        };
        delta.encode_workgroups(
            &self.context,
            encoder,
            &self.context.uniform_buffer(&info),
            &[
                &convolved,
                &a,
                &b,
                &layer.params,
                recurrent,
                recurrent_out,
                &core,
            ],
            [dv.div_ceil(DELTA_COLS) as u32, nv as u32, batch as u32],
        );

        let info = GatedNormInfo {
            rows: rows as u32,
            heads: nv as u32,
            dim: dv as u32,
            eps: c.rms_norm_eps,
        };
        self.kernels.gated_norm.encode(
            &self.context,
            encoder,
            &self.context.uniform_buffer(&info),
            &[&z, &layer.norm, &core],
            rows * nv * GATED_NORM_THREADS,
            GATED_NORM_THREADS,
        );
        self.linear(encoder, pool, &core, &layer.out)
    }
}

impl GpuDecoder for Qwen3_5GpuDecoder {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn embed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        self.embeddings
            .embed(encoder, pool, input, None, position_offset)
    }

    fn embed_and_normalize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        self.embed(encoder, pool, input, position_offset)
    }

    fn final_norm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        let out = pool.get(hidden_states.shape().to_vec());
        self.norm
            .encode(encoder, &self.final_norm, hidden_states, &out);
        Ok(out)
    }

    /// Positions are causal by construction in the recurrence, and the
    /// attention layers read the mask as every other GPU decoder does. A pass
    /// over all layers that prefills more than one token leaves snapshots of the
    /// recurrent state at its end and just before, as the CPU decoder does.
    fn forward_layers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuTensor> {
        let cache = cache.ok_or_else(|| {
            anyhow!("Qwen 3.5 on the GPU needs its cache; make one with the model's new_cache")
        })?;
        let (batch, seq, hidden) = hidden_states.dims3();
        let whole = start_layer == 0 && end_layer == self.layers.len();
        let capture = if whole && seq > SNAPSHOT_TAIL {
            seq - SNAPSHOT_TAIL
        } else {
            0
        };

        let recurrent = cache
            .recurrent_mut()
            .ok_or_else(|| anyhow!("Qwen 3.5 needs a cache with recurrent states"))?;
        recurrent.check_usable()?;
        let states = recurrent.states().to_vec();
        let captured = if capture > 0 {
            Some(recurrent.take_slot()?)
        } else {
            None
        };

        let rows = batch * seq;
        let mut current = hidden_states.view(vec![rows, hidden]);
        let checkpoint = pool.checkpoint();
        for (i, layer) in self
            .layers
            .iter()
            .enumerate()
            .take(end_layer)
            .skip(start_layer)
        {
            let normed = pool.get(vec![rows, hidden]);
            self.norm
                .encode(encoder, &layer.input_norm, &current, &normed);
            let mixed = match &layer.mixer {
                Mixer::Attention(attention) => self.attention(
                    encoder,
                    pool,
                    attention,
                    &normed.view(vec![batch, seq, hidden]),
                    attention_mask,
                    batch,
                    seq,
                    position_offset,
                    cache,
                )?,
                Mixer::DeltaNet(net) => self.delta_net(
                    encoder,
                    pool,
                    net,
                    &normed,
                    batch,
                    seq,
                    &states,
                    captured.as_deref(),
                    capture,
                ),
            };
            let mixed_in = pool.get(vec![rows, hidden]);
            self.add.encode(
                encoder,
                &[&current, &mixed.view(vec![rows, hidden])],
                &mixed_in,
            );
            let normed = pool.get(vec![rows, hidden]);
            self.norm
                .encode(encoder, &layer.post_norm, &mixed_in, &normed);
            let mlp = pool.get(vec![rows, hidden]);
            self.ffn.encode(encoder, &layer.mlp, &normed, &mlp, pool);
            let next = pool.get(vec![rows, hidden]);
            self.add.encode(encoder, &[&mixed_in, &mlp], &next);
            current = next;
            pool.release_since(&checkpoint, &[&current]);
            kjarni_transformers::gpu::submit_between_layers(&self.context, encoder, i, end_layer);
        }

        if end_layer == self.layers.len() {
            let start = cache.get_seq_length();
            let recurrent = cache
                .recurrent_mut()
                .expect("checked above that the cache has recurrent states");
            if let Some(slot) = captured {
                recurrent.keep(start + capture, slot);
            }
            if seq > 1 {
                recurrent.snapshot(encoder, start + seq)?;
            }
        }
        Ok(current.view(vec![batch, seq, hidden]))
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: ModelInput<'_>,
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuKVCache>,
        _encoder_hidden_states: Option<&GpuTensor>,
    ) -> Result<GpuTensor> {
        let hidden = self.embed(encoder, pool, input, position_offset)?;
        let hidden = self.forward_layers(
            encoder,
            pool,
            &hidden,
            attention_mask,
            position_offset,
            cache,
            0,
            self.layers.len(),
        )?;
        GpuDecoder::final_norm(self, encoder, pool, &hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn random(n: usize, seed: u64) -> Array1<f32> {
        let mut state = seed | 1;
        Array1::from_shape_fn(n, |_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 8_388_608.0 - 1.0
        })
    }

    /// The one-token delta rule gives what the general one gives for a sequence
    /// of one: output, state and the captured state, in both head layouts, with
    /// value columns that do not fill the last workgroup.
    #[tokio::test]
    #[ignore = "GPU required"]
    async fn the_delta_step_matches_the_delta_rule_for_one_token() {
        let ctx = WgpuContext::new().await.unwrap();
        let kernels = Kernels::new(&ctx.device);
        let (batch, nk, nv, dk, dv) = (2usize, 2usize, 4usize, 128usize, 48usize);
        let channels = 2 * nk * dk + nv * dv;
        let tensor = |a: Array1<f32>| GpuTensor::from_ndarray(&ctx, &a).unwrap();
        let qkv = random(batch * channels, 1);
        let a = random(batch * nv, 2);
        let b = random(batch * nv, 3);
        let mut params = random(2 * nv, 4);
        params
            .slice_mut(ndarray::s![..nv])
            .mapv_inplace(|x| -x.abs() - 0.1);
        let state = random(batch * nv * dk * dv, 5);

        for tiled in [0u32, 1] {
            let run = |kernel: &ComputeKernel| {
                let (qkv, a, b, params) = (
                    tensor(qkv.clone()),
                    tensor(a.clone()),
                    tensor(b.clone()),
                    tensor(params.clone()),
                );
                let state = tensor(state.clone());
                let captured = tensor(Array1::zeros(batch * nv * dk * dv));
                let out = tensor(Array1::zeros(batch * nv * dv));
                let info = DeltaInfo {
                    batch: batch as u32,
                    seq: 1,
                    k_heads: nk as u32,
                    v_heads: nv as u32,
                    key_dim: dk as u32,
                    value_dim: dv as u32,
                    tiled,
                    capture: 1,
                    channels: channels as u32,
                    scale: (dk as f32).powf(-0.5),
                    _pad: [0; 2],
                };
                let mut encoder = ctx.device.create_command_encoder(&Default::default());
                kernel.encode_workgroups(
                    &ctx,
                    &mut encoder,
                    &ctx.uniform_buffer(&info),
                    &[&qkv, &a, &b, &params, &state, &captured, &out],
                    [dv.div_ceil(DELTA_COLS) as u32, nv as u32, batch as u32],
                );
                ctx.queue.submit(Some(encoder.finish()));
                async move {
                    (
                        out.to_ndarray_1d::<f32>().await.unwrap(),
                        state.to_ndarray_1d::<f32>().await.unwrap(),
                        captured.to_ndarray_1d::<f32>().await.unwrap(),
                    )
                }
            };
            let want = run(&kernels.delta).await;
            let got = run(&kernels.delta_step).await;
            for (what, g, w) in [
                ("output", &got.0, &want.0),
                ("state", &got.1, &want.1),
                ("captured", &got.2, &want.2),
            ] {
                let worst = g
                    .iter()
                    .zip(w.iter())
                    .map(|(g, w)| (g - w).abs())
                    .fold(0.0f32, f32::max);
                assert!(worst < 1e-5, "{what}, tiled {tiled}: differs by {worst:e}");
            }
            assert!(want.0.iter().any(|x| x.abs() > 1e-3));
        }
    }
}
