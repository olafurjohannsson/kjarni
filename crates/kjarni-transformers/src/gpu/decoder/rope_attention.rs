//! RoPE-based causal attention for modern LLMs.

use std::sync::Arc;

use anyhow::{Result, ensure};

use crate::WgpuContext;
use crate::gpu::{GpuTensor, GpuTensorPool};
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::attention::ops::AttentionOps;
use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::primitives::softmax::GpuSoftmax;
use crate::gpu_ops::utils::dispatch_1d;

/// Threads in a workgroup of all three kernels.
const WORKGROUP_THREADS: usize = 32;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RoPEWriteKvInfo {
    batch: u32,
    seq: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    position_offset: u32,
    write_offset: u32,
    key_stride: u32,
    rope_stride: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RoPEAttendInfo {
    batch: u32,
    seq: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    position_offset: u32,
    key_stride: u32,
    keys: u32,
    mask_stride: u32,
    mask_rows: u32,
    _pad: [u32; 2],
}

/// A compute pipeline and its bind group layout: a uniform at binding 0, then
/// one storage buffer per entry of `read_only`.
struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl Kernel {
    fn new(
        device: &wgpu::Device,
        label: &str,
        shader: wgpu::ShaderModuleDescriptor<'_>,
        read_only: &[bool],
    ) -> Self {
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for (i, read_only) in read_only.iter().enumerate() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32 + 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: *read_only,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(shader);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// One dispatch of `threads` threads over `tensors`, at bindings 1 and up.
    fn encode(
        &self,
        context: &WgpuContext,
        encoder: &mut wgpu::CommandEncoder,
        uniforms: &wgpu::Buffer,
        tensors: &[&GpuTensor],
        threads: usize,
    ) {
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniforms.as_entire_binding(),
        }];
        for (i, tensor) in tensors.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: tensor.buffer().as_entire_binding(),
            });
        }
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &entries,
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        dispatch_1d(&mut pass, threads.div_ceil(WORKGROUP_THREADS) as u32);
    }
}

/// Causal attention with rotary positions.
///
/// One kernel rotates the step's queries and keys and writes keys and values
/// straight into the KV store. The scores and the output then read the store in
/// place. A dispatch costs about 15 microseconds on an RTX A2000 however little
/// data it touches, so this does in eight what splitting heads, rotating, copying
/// the cache, expanding the KV heads, permuting, masking, merging heads and writing
/// the cache back as kernels of their own did in over twenty per layer per token.
///
/// The two matmuls take one thread per output instead of a tiled GEMM: at decode
/// sizes they are short rows over few heads, where a tile's barriers cost more than
/// its reuse saves. Prefill skips the keys after each query, which is half the work.
pub struct GpuRoPEAttention {
    context: Arc<WgpuContext>,
    ops: AttentionOps,
    softmax: GpuSoftmax,
    write_kv: Kernel,
    scores: Kernel,
    output: Kernel,
}

impl GpuRoPEAttention {
    pub fn new(
        context: &Arc<WgpuContext>,
        hidden_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
    ) -> Self {
        let device = &context.device;
        Self {
            context: context.clone(),
            ops: AttentionOps::new(context, hidden_size, num_heads, num_kv_heads),
            softmax: GpuSoftmax::new(context),
            write_kv: Kernel::new(
                device,
                "RoPE write KV",
                wgpu::include_wgsl!("./rope_write_kv.wgsl"),
                &[true, true, true, true, true, false, false, false],
            ),
            scores: Kernel::new(
                device,
                "RoPE scores",
                wgpu::include_wgsl!("./rope_scores.wgsl"),
                &[true, true, true, false],
            ),
            output: Kernel::new(
                device,
                "RoPE context",
                wgpu::include_wgsl!("./rope_context.wgsl"),
                &[true, true, false],
            ),
        }
    }

    pub fn ops(&self) -> &AttentionOps {
        &self.ops
    }

    pub fn num_heads(&self) -> u32 {
        self.ops.num_heads()
    }

    pub fn num_kv_heads(&self) -> u32 {
        self.ops.num_kv_heads()
    }

    pub fn gqa_ratio(&self) -> u32 {
        self.ops.num_heads() / self.ops.num_kv_heads()
    }

    /// Runs attention over `hidden_states`, `[B, S, H]`, and returns `[B, S, H]`.
    ///
    /// With `cached_kv`, the new keys and values are written into it at
    /// `position_offset`, so the caller's KV cache is up to date afterwards. It
    /// must have room for them.
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        weights: &GpuAttentionWeights,
        rope: &GpuRoPE,
        attention_mask: &GpuTensor,
        cached_kv: Option<(&GpuTensor, &GpuTensor)>,
        position_offset: usize,
        pool: &mut GpuTensorPool,
    ) -> Result<GpuTensor> {
        let (batch, seq, hidden) = hidden_states.dims3();
        let heads = self.ops.num_heads() as usize;
        let kv_heads = self.ops.num_kv_heads() as usize;
        let head_dim = self.ops.head_dim() as usize;

        let q_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.q_weight,
            weights.q_bias.as_ref(),
            pool,
        );
        let k_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.k_weight,
            weights.k_bias.as_ref(),
            pool,
        );
        let v_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.v_weight,
            weights.v_bias.as_ref(),
            pool,
        );

        let (k_store, v_store, write_offset) = match cached_kv {
            Some((k, v)) => {
                let capacity = k.shape()[2];
                ensure!(
                    position_offset + seq <= capacity,
                    "KV cache holds {capacity} tokens, {position_offset} + {seq} do not fit"
                );
                (k.clone(), v.clone(), position_offset)
            }
            None => {
                let shape = vec![batch, kv_heads, seq, head_dim];
                (pool.get(shape.clone()), pool.get(shape), 0)
            }
        };
        let keys = write_offset + seq;

        let q_rotated = pool.get(vec![batch, heads, seq, head_dim]);
        self.encode_write_kv(
            encoder,
            &q_proj,
            &k_proj,
            &v_proj,
            rope,
            &q_rotated,
            &k_store,
            &v_store,
            position_offset,
            write_offset,
        );

        let mask_shape = attention_mask.shape();
        let info = RoPEAttendInfo {
            batch: batch as u32,
            seq: seq as u32,
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            head_dim: head_dim as u32,
            position_offset: position_offset as u32,
            key_stride: k_store.shape()[2] as u32,
            keys: keys as u32,
            mask_stride: mask_shape[mask_shape.len() - 1] as u32,
            mask_rows: mask_shape[0] as u32,
            _pad: [0; 2],
        };
        let uniforms = self.context.uniform_buffer(&info);

        let rows = batch * heads * seq;
        let scores = pool.get(vec![rows, keys]);
        self.scores.encode(
            &self.context,
            encoder,
            &uniforms,
            &[&q_rotated, &k_store, attention_mask, &scores],
            rows * keys,
        );
        self.softmax
            .encode(encoder, &scores, self.ops.scale_factor());

        let attended = pool.get(vec![batch, seq, hidden]);
        self.output.encode(
            &self.context,
            encoder,
            &uniforms,
            &[&scores, &v_store, &attended],
            rows * head_dim,
        );

        Ok(self.ops.project(
            encoder,
            &attended,
            &weights.output_weight,
            weights.output_bias.as_ref(),
            pool,
        ))
    }

    fn encode_write_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_proj: &GpuTensor,
        k_proj: &GpuTensor,
        v_proj: &GpuTensor,
        rope: &GpuRoPE,
        q_rotated: &GpuTensor,
        k_store: &GpuTensor,
        v_store: &GpuTensor,
        position_offset: usize,
        write_offset: usize,
    ) {
        let (batch, seq, _) = q_proj.dims3();
        let heads = self.ops.num_heads();
        let (cos_table, sin_table) = rope.tables();
        let info = RoPEWriteKvInfo {
            batch: batch as u32,
            seq: seq as u32,
            heads,
            kv_heads: self.ops.num_kv_heads(),
            head_dim: self.ops.head_dim(),
            position_offset: position_offset as u32,
            write_offset: write_offset as u32,
            key_stride: k_store.shape()[2] as u32,
            rope_stride: cos_table.shape()[1] as u32,
            _pad: [0; 3],
        };
        let uniforms = self.context.uniform_buffer(&info);
        self.write_kv.encode(
            &self.context,
            encoder,
            &uniforms,
            &[
                q_proj, k_proj, v_proj, cos_table, sin_table, q_rotated, k_store, v_store,
            ],
            batch * seq * heads as usize,
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::{Array1, Array2, Array3, Array4, s};

    use super::*;
    use crate::WgpuContext;
    use crate::gpu::GpuTensorPool;
    use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
    use crate::gpu_ops::blocks::rope::GpuRoPE;
    use crate::rope::RoPE;

    async fn setup() -> Arc<WgpuContext> {
        WgpuContext::new().await.unwrap()
    }

    fn create_dummy_weights(
        ctx: &Arc<WgpuContext>,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
    ) -> GpuAttentionWeights {
        let head_dim = hidden_size / num_heads;
        let q_dim = hidden_size;
        let kv_dim = num_kv_heads * head_dim;

        let q_weight =
            GpuTensor::from_ndarray(ctx, &Array2::<f32>::zeros((q_dim, hidden_size))).unwrap();
        let q_bias = GpuTensor::from_ndarray(ctx, &Array1::<f32>::zeros(q_dim)).unwrap();

        let k_weight =
            GpuTensor::from_ndarray(ctx, &Array2::<f32>::zeros((kv_dim, hidden_size))).unwrap();
        let k_bias = GpuTensor::from_ndarray(ctx, &Array1::<f32>::zeros(kv_dim)).unwrap();
        let v_weight =
            GpuTensor::from_ndarray(ctx, &Array2::<f32>::zeros((kv_dim, hidden_size))).unwrap();
        let v_bias = GpuTensor::from_ndarray(ctx, &Array1::<f32>::zeros(kv_dim)).unwrap();

        let o_weight =
            GpuTensor::from_ndarray(ctx, &Array2::<f32>::zeros((hidden_size, hidden_size)))
                .unwrap();
        let o_bias = GpuTensor::from_ndarray(ctx, &Array1::<f32>::zeros(hidden_size)).unwrap();

        GpuAttentionWeights::new(
            q_weight,
            Some(q_bias),
            k_weight,
            Some(k_bias),
            v_weight,
            Some(v_bias),
            o_weight,
            Some(o_bias),
        )
        .unwrap()
    }

    fn create_rope(ctx: &Arc<WgpuContext>, head_dim: usize, max_seq_len: usize) -> GpuRoPE {
        let cpu_rope = RoPE::new(head_dim, max_seq_len, 10000.0);
        GpuRoPE::new(ctx, &cpu_rope.cos_cache, &cpu_rope.sin_cache).unwrap()
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_rope_attention_prefill_no_cache() {
        let ctx = setup().await;

        let hidden_size = 256;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = hidden_size / num_heads;
        let batch_size = 1;
        let seq_len = 10;

        let attn = GpuRoPEAttention::new(
            &ctx,
            hidden_size as u32,
            num_heads as u32,
            num_kv_heads as u32,
        );
        let weights = create_dummy_weights(&ctx, hidden_size, num_heads, num_kv_heads);
        let rope = create_rope(&ctx, head_dim, 128);

        let hidden = Array3::<f32>::ones((batch_size, seq_len, hidden_size));
        let hidden_gpu = GpuTensor::from_ndarray(&ctx, &hidden).unwrap();

        let mask = Array2::<f32>::ones((1, seq_len));
        let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask).unwrap();

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(ctx.clone());

        let result = attn.forward(
            &mut encoder,
            &hidden_gpu,
            &weights,
            &rope,
            &mask_gpu,
            None,
            0,
            &mut pool,
        );

        assert!(result.is_ok(), "prefill should succeed: {:?}", result.err());

        let output = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn test_rope_attention_decode_with_cache() {
        let ctx = setup().await;

        let hidden_size = 256;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = hidden_size / num_heads;
        let batch_size = 1;
        let cache_len = 10;
        let query_len = 1;
        let total_len = cache_len + query_len;

        let attn = GpuRoPEAttention::new(
            &ctx,
            hidden_size as u32,
            num_heads as u32,
            num_kv_heads as u32,
        );
        let weights = create_dummy_weights(&ctx, hidden_size, num_heads, num_kv_heads);
        let rope = create_rope(&ctx, head_dim, 128);

        let hidden = Array3::<f32>::ones((batch_size, query_len, hidden_size));
        let hidden_gpu = GpuTensor::from_ndarray(&ctx, &hidden).unwrap();

        let mask = Array2::<f32>::ones((1, total_len));
        let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask).unwrap();

        let cached_k = Array4::<f32>::zeros((batch_size, num_kv_heads, total_len, head_dim));
        let cached_v = Array4::<f32>::zeros((batch_size, num_kv_heads, total_len, head_dim));
        let cached_k_gpu = GpuTensor::from_ndarray(&ctx, &cached_k).unwrap();
        let cached_v_gpu = GpuTensor::from_ndarray(&ctx, &cached_v).unwrap();

        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(ctx.clone());

        let result = attn.forward(
            &mut encoder,
            &hidden_gpu,
            &weights,
            &rope,
            &mask_gpu,
            Some((&cached_k_gpu, &cached_v_gpu)),
            cache_len,
            &mut pool,
        );

        assert!(
            result.is_ok(),
            "decode with cache should succeed: {:?}",
            result.err()
        );

        let output = result.unwrap();
        assert_eq!(output.shape(), &[batch_size, query_len, hidden_size]);
    }

    /// A pseudo-random `[rows, cols]` matrix with entries in [-1, 1).
    fn random(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut state = seed | 1;
        Array2::from_shape_fn((rows, cols), |_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 8_388_608.0 - 1.0
        })
    }

    /// Attention spelled out on the CPU: project, rotate, softmax over the keys a
    /// query may see, project. `mask` is one row per batch row.
    fn reference(
        hidden: &Array3<f32>,
        mask: &Array2<f32>,
        [q_w, k_w, v_w, o_w]: [&Array2<f32>; 4],
        rope: &RoPE,
        heads: usize,
        kv_heads: usize,
    ) -> Array3<f32> {
        let (batch, seq, size) = hidden.dim();
        let d = size / heads;
        let rotate = |row: &[f32], pos: usize| -> Vec<f32> {
            let mut out = row.to_vec();
            for i in 0..d / 2 {
                let (c, s) = (rope.cos_cache[[pos, i]], rope.sin_cache[[pos, i]]);
                out[i] = row[i] * c - row[i + d / 2] * s;
                out[i + d / 2] = row[i + d / 2] * c + row[i] * s;
            }
            out
        };
        let mut output = Array3::<f32>::zeros((batch, seq, size));
        for b in 0..batch {
            let h = hidden.index_axis(ndarray::Axis(0), b);
            let (q, k, v) = (h.dot(&q_w.t()), h.dot(&k_w.t()), h.dot(&v_w.t()));
            let mut attended = Array2::<f32>::zeros((seq, size));
            for t in 0..seq {
                for head in 0..heads {
                    let qh = rotate(&q.row(t).as_slice().unwrap()[head * d..(head + 1) * d], t);
                    let kv = head / (heads / kv_heads);
                    let mut scores = vec![];
                    for key in 0..=t {
                        if mask[[b, key]] == 0.0 {
                            continue;
                        }
                        let kr = rotate(&k.row(key).as_slice().unwrap()[kv * d..(kv + 1) * d], key);
                        let dot: f32 = qh.iter().zip(&kr).map(|(a, b)| a * b).sum();
                        scores.push((key, dot / (d as f32).sqrt()));
                    }
                    let largest = scores.iter().map(|s| s.1).fold(f32::MIN, f32::max);
                    let total: f32 = scores.iter().map(|s| (s.1 - largest).exp()).sum();
                    for (key, score) in scores {
                        let w = (score - largest).exp() / total;
                        for i in 0..d {
                            attended[[t, head * d + i]] += w * v[[key, kv * d + i]];
                        }
                    }
                }
            }
            output
                .index_axis_mut(ndarray::Axis(0), b)
                .assign(&attended.dot(&o_w.t()));
        }
        output
    }

    /// Writes into a cache, grouped-query heads and a padding mask, against the
    /// spelled-out computation: prefill all at once, then the same tokens split
    /// into a prefill and single-token steps.
    #[tokio::test]
    #[ignore = "GPU required"]
    async fn attention_matches_reference_in_one_step_and_token_by_token() {
        let ctx = setup().await;
        let (hidden_size, heads, kv_heads) = (64usize, 8usize, 2usize);
        let d = hidden_size / heads;
        let kv_dim = kv_heads * d;
        let (batch, seq) = (2usize, 7usize);

        let q_w = random(hidden_size, hidden_size, 11);
        let k_w = random(kv_dim, hidden_size, 22);
        let v_w = random(kv_dim, hidden_size, 33);
        let o_w = random(hidden_size, hidden_size, 44);
        let bias = |n: usize| GpuTensor::from_ndarray(&ctx, &Array1::<f32>::zeros(n)).unwrap();
        let tensor = |m: &Array2<f32>| GpuTensor::from_ndarray(&ctx, m).unwrap();
        let weights = GpuAttentionWeights::new(
            tensor(&q_w),
            Some(bias(hidden_size)),
            tensor(&k_w),
            Some(bias(kv_dim)),
            tensor(&v_w),
            Some(bias(kv_dim)),
            tensor(&o_w),
            Some(bias(hidden_size)),
        )
        .unwrap();

        let hidden = Array3::from_shape_vec(
            (batch, seq, hidden_size),
            random(batch * seq, hidden_size, 55)
                .into_raw_vec_and_offset()
                .0,
        )
        .unwrap();
        // The second batch row is padded on the left: its first two keys are masked.
        let mut mask = Array2::<f32>::ones((batch, seq));
        mask[[1, 0]] = 0.0;
        mask[[1, 1]] = 0.0;

        let cpu_rope = RoPE::new(d, 128, 10000.0);
        let rope = GpuRoPE::from_cpu_rope(&ctx, &cpu_rope).unwrap();
        let attn = GpuRoPEAttention::new(&ctx, hidden_size as u32, heads as u32, kv_heads as u32);
        let want = reference(
            &hidden,
            &mask,
            [&q_w, &k_w, &v_w, &o_w],
            &cpu_rope,
            heads,
            kv_heads,
        );

        let k_cache = Array4::<f32>::zeros((batch, kv_heads, 16, d));
        let k_cache = GpuTensor::from_ndarray(&ctx, &k_cache).unwrap();
        let v_cache = Array4::<f32>::zeros((batch, kv_heads, 16, d));
        let v_cache = GpuTensor::from_ndarray(&ctx, &v_cache).unwrap();

        let run = |input: &Array3<f32>, offset: usize, cached: bool| {
            let attn = &attn;
            let (weights, rope, k_cache, v_cache) = (&weights, &rope, &k_cache, &v_cache);
            let ctx = ctx.clone();
            // As wide as the keys seen so far, like the mask a decoder builds.
            let keys = offset + input.dim().1;
            let mask_gpu =
                GpuTensor::from_ndarray(&ctx, &mask.slice(s![.., ..keys]).to_owned()).unwrap();
            let input = GpuTensor::from_ndarray(&ctx, input).unwrap();
            async move {
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                let mut pool = GpuTensorPool::new(ctx.clone());
                let out = attn
                    .forward(
                        &mut enc,
                        &input,
                        weights,
                        rope,
                        &mask_gpu,
                        cached.then_some((k_cache, v_cache)),
                        offset,
                        &mut pool,
                    )
                    .unwrap();
                ctx.queue.submit(Some(enc.finish()));
                out.to_ndarray_3d::<f32>().await.unwrap()
            }
        };

        // A query on a padded position sees no key at all; what it holds is never used.
        let close =
            |got: &Array3<f32>, want: ndarray::ArrayView3<f32>, offset: usize, what: &str| {
                let worst = got
                    .indexed_iter()
                    .filter(|((b, t, _), _)| !(*b == 1 && offset + t < 2))
                    .map(|((b, t, i), g)| (g - want[[b, t, i]]).abs())
                    .fold(0.0f32, f32::max);
                println!("{what}: max abs diff {worst:.3e}");
                assert!(worst < 2e-4, "{what} differs by {worst:.3e}");
            };

        close(
            &run(&hidden, 0, false).await,
            want.view(),
            0,
            "without a cache",
        );
        close(
            &run(&hidden, 0, true).await,
            want.view(),
            0,
            "prefill into a cache",
        );

        let split = 4;
        let first = hidden.slice(s![.., ..split, ..]).to_owned();
        close(
            &run(&first, 0, true).await,
            want.slice(s![.., ..split, ..]),
            0,
            "shorter prefill",
        );
        for t in split..seq {
            let step = hidden.slice(s![.., t..t + 1, ..]).to_owned();
            close(
                &run(&step, t, true).await,
                want.slice(s![.., t..t + 1, ..]),
                t,
                &format!("token {t}"),
            );
        }
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn a_step_that_does_not_fit_in_the_cache_is_an_error() {
        let ctx = setup().await;
        let attn = GpuRoPEAttention::new(&ctx, 64, 8, 8);
        let weights = create_dummy_weights(&ctx, 64, 8, 8);
        let rope = create_rope(&ctx, 8, 32);
        let hidden = GpuTensor::from_ndarray(&ctx, &Array3::<f32>::ones((1, 2, 64))).unwrap();
        let mask = GpuTensor::from_ndarray(&ctx, &Array2::<f32>::ones((1, 5))).unwrap();
        let cache = Array4::<f32>::zeros((1, 8, 4, 8));
        let k = GpuTensor::from_ndarray(&ctx, &cache).unwrap();
        let v = GpuTensor::from_ndarray(&ctx, &cache).unwrap();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(ctx.clone());
        let result = attn.forward(
            &mut enc,
            &hidden,
            &weights,
            &rope,
            &mask,
            Some((&k, &v)),
            3,
            &mut pool,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_rope_attention_dimensions() {
        let hidden_size = 2048u32;
        let num_heads = 32u32;
        let num_kv_heads = 8u32;
        let head_dim = hidden_size / num_heads;
        let gqa_ratio = num_heads / num_kv_heads;

        assert_eq!(head_dim, 64);
        assert_eq!(gqa_ratio, 4);
    }

    #[test]
    fn test_no_gqa() {
        let num_heads = 32u32;
        let num_kv_heads = 32u32;
        let gqa_ratio = num_heads / num_kv_heads;

        assert_eq!(gqa_ratio, 1);
    }
}
