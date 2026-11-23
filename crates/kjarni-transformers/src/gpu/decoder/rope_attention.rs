//! RoPE-based causal attention for modern LLMs.

use std::sync::Arc;

use anyhow::{Result, ensure};

use crate::WgpuContext;
use crate::gpu::{ComputeKernel, DType, GpuTensor, GpuTensorPool, kv_shader};
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::attention::ops::AttentionOps;
use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::primitives::softmax::GpuSoftmax;

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
    query_start: u32,
    queries: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RoPEDecodeInfo {
    batch: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    position_offset: u32,
    key_stride: u32,
    keys: u32,
    mask_stride: u32,
    mask_rows: u32,
    chunks: u32,
    scale: f32,
    _pad: u32,
}

/// Keys per workgroup of the one-query attention, as in rope_decode_chunks.wgsl.
const DECODE_CHUNK: usize = 32;
/// The largest head and query-head group the one-query attention takes.
const DECODE_MAX_HEAD_DIM: usize = 256;
const DECODE_MAX_GROUP: usize = 8;

/// The most bytes of attention scores one dispatch writes. A prompt whose scores
/// would be larger is attended a slice of queries at a time.
const MAX_SCORE_BYTES: u64 = 512 << 20;

/// Queries per slice for scores `rows_per_query * keys` wide per query, within
/// `budget` bytes: at least one.
fn queries_per_slice(rows_per_query: usize, keys: usize, budget: u64) -> usize {
    let per_query = (rows_per_query * keys * std::mem::size_of::<f32>()) as u64;
    ((budget / per_query.max(1)) as usize).max(1)
}

/// The shared helper, with this module's workgroup size.
struct Kernel(ComputeKernel);

impl Kernel {
    fn new(
        device: &wgpu::Device,
        label: &str,
        shader: wgpu::ShaderModuleDescriptor<'_>,
        read_only: &[bool],
    ) -> Self {
        Self(ComputeKernel::new(device, label, shader, read_only))
    }

    /// One dispatch of `threads` threads over `tensors`, at bindings 1 and up.
    fn encode(
        &self,
        context: &WgpuContext,
        encoder: &mut crate::gpu::GpuEncoder,
        uniforms: &wgpu::Buffer,
        tensors: &[&GpuTensor],
        threads: usize,
    ) {
        self.0.encode(
            context,
            encoder,
            uniforms,
            tensors,
            threads,
            WORKGROUP_THREADS,
        );
    }
}

/// The kernels that write or read the KV store, built for one store dtype.
struct StoreKernels {
    write_kv: Kernel,
    scores: Kernel,
    output: Kernel,
    decode_chunks: Kernel,
}

impl StoreKernels {
    fn new(device: &wgpu::Device, dtype: DType) -> Self {
        let make = |label: &str, source: &str, read_only: &[bool]| {
            Kernel::new(device, label, kv_shader(label, source, dtype), read_only)
        };
        Self {
            write_kv: make(
                "RoPE write KV",
                include_str!("./rope_write_kv.wgsl"),
                &[true, true, true, true, true, false, false, false],
            ),
            scores: make(
                "RoPE scores",
                include_str!("./rope_scores.wgsl"),
                &[true, true, true, false],
            ),
            output: make(
                "RoPE context",
                include_str!("./rope_context.wgsl"),
                &[true, true, false],
            ),
            decode_chunks: make(
                "RoPE decode chunks",
                include_str!("./rope_decode_chunks.wgsl"),
                &[true, true, true, true, false, false],
            ),
        }
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
    /// The kernels that touch the KV store, for an F32 store.
    f32_store: StoreKernels,
    /// The same for an F16 store, when the context has f16 KV caches.
    f16_store: Option<StoreKernels>,
    decode_combine: Kernel,
    /// Bytes of scores per dispatch: the device's binding limit or
    /// `MAX_SCORE_BYTES`, whichever is smaller.
    score_budget: u64,
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
            f32_store: StoreKernels::new(device, DType::F32),
            f16_store: context
                .f16_kv_cache
                .then(|| StoreKernels::new(device, DType::F16)),
            decode_combine: Kernel::new(
                device,
                "RoPE decode combine",
                wgpu::include_wgsl!("./rope_decode_combine.wgsl"),
                &[true, true, false],
            ),
            score_budget: (context.device.limits().max_storage_buffer_binding_size as u64)
                .min(MAX_SCORE_BYTES),
        }
    }

    /// The kernels for a KV store of `store`'s dtype.
    fn store_kernels(&self, store: &GpuTensor) -> &StoreKernels {
        match store.dtype() {
            DType::F32 => &self.f32_store,
            DType::F16 => self
                .f16_store
                .as_ref()
                .expect("an F16 KV cache on a context without f16 KV caches"),
            other => panic!("a KV store holds F32 or F16, not {other:?}"),
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
        encoder: &mut crate::gpu::GpuEncoder,
        hidden_states: &GpuTensor,
        weights: &GpuAttentionWeights,
        rope: &GpuRoPE,
        attention_mask: &GpuTensor,
        cached_kv: Option<(&GpuTensor, &GpuTensor)>,
        position_offset: usize,
        pool: &mut GpuTensorPool,
    ) -> Result<GpuTensor> {
        let (batch, seq, _) = hidden_states.dims3();
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

        let attended = self.attend(
            encoder,
            &q_rotated,
            &k_store,
            &v_store,
            attention_mask,
            position_offset,
            keys,
            pool,
        );

        Ok(self.ops.project(
            encoder,
            &attended,
            &weights.output_weight,
            weights.output_bias.as_ref(),
            pool,
        ))
    }

    /// Attends with queries already rotated, `[B, H, S, D]`, over the first
    /// `keys` keys and values of the stores, `[B, KV, capacity, D]`, and
    /// returns the heads merged, `[B, S, H * D]`. A query at absolute position
    /// `position_offset + s` sees the keys up to it that the mask allows.
    pub fn attend(
        &self,
        encoder: &mut crate::gpu::GpuEncoder,
        q_rotated: &GpuTensor,
        k_store: &GpuTensor,
        v_store: &GpuTensor,
        attention_mask: &GpuTensor,
        position_offset: usize,
        keys: usize,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (batch, heads, seq) = (
            q_rotated.shape()[0],
            q_rotated.shape()[1],
            q_rotated.shape()[2],
        );
        let kv_heads = self.ops.num_kv_heads() as usize;
        let head_dim = self.ops.head_dim() as usize;
        let mask_shape = attention_mask.shape();
        if seq == 1
            && head_dim.is_multiple_of(4)
            && head_dim <= DECODE_MAX_HEAD_DIM
            && heads / kv_heads <= DECODE_MAX_GROUP
        {
            return self.attend_one(
                encoder,
                q_rotated,
                k_store,
                v_store,
                attention_mask,
                position_offset,
                keys,
                pool,
            );
        }
        let mut info = RoPEAttendInfo {
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
            query_start: 0,
            queries: 0,
        };

        // One scores buffer, reused by each slice of queries in turn.
        let slice = queries_per_slice(batch * heads, keys, self.score_budget).min(seq);
        let scores = pool.get(vec![batch * heads * slice, keys]);
        let attended = pool.get(vec![batch, seq, heads * head_dim]);
        for start in (0..seq).step_by(slice) {
            let queries = slice.min(seq - start);
            info.query_start = start as u32;
            info.queries = queries as u32;
            let uniforms = self.context.uniform_buffer(&info);
            let rows = batch * heads * queries;
            self.store_kernels(k_store).scores.encode(
                &self.context,
                encoder,
                &uniforms,
                &[q_rotated, k_store, attention_mask, &scores],
                rows * keys,
            );
            self.softmax
                .encode(encoder, &scores, self.ops.scale_factor());
            self.store_kernels(v_store).output.encode(
                &self.context,
                encoder,
                &uniforms,
                &[&scores, v_store, &attended],
                rows * head_dim,
            );
        }
        attended
    }

    /// `attend` for one query per sequence: the keys in chunks across
    /// workgroups, then the chunks merged.
    #[allow(clippy::too_many_arguments)]
    fn attend_one(
        &self,
        encoder: &mut crate::gpu::GpuEncoder,
        q_rotated: &GpuTensor,
        k_store: &GpuTensor,
        v_store: &GpuTensor,
        attention_mask: &GpuTensor,
        position_offset: usize,
        keys: usize,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (batch, heads) = (q_rotated.shape()[0], q_rotated.shape()[1]);
        let kv_heads = self.ops.num_kv_heads() as usize;
        let head_dim = self.ops.head_dim() as usize;
        let mask_shape = attention_mask.shape();
        let chunks = keys.div_ceil(DECODE_CHUNK);
        let info = RoPEDecodeInfo {
            batch: batch as u32,
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            head_dim: head_dim as u32,
            position_offset: position_offset as u32,
            key_stride: k_store.shape()[2] as u32,
            keys: keys as u32,
            mask_stride: mask_shape[mask_shape.len() - 1] as u32,
            mask_rows: mask_shape[0] as u32,
            chunks: chunks as u32,
            scale: self.ops.scale_factor(),
            _pad: 0,
        };
        let uniforms = self.context.uniform_buffer(&info);
        let stats = pool.get(vec![batch, heads, chunks, 2]);
        let partial = pool.get(vec![batch, heads, chunks, head_dim]);
        self.store_kernels(k_store)
            .decode_chunks
            .0
            .encode_workgroups(
                &self.context,
                encoder,
                &uniforms,
                &[
                    q_rotated,
                    k_store,
                    v_store,
                    attention_mask,
                    &stats,
                    &partial,
                ],
                [chunks as u32, kv_heads as u32, batch as u32],
            );
        let attended = pool.get(vec![batch, 1, heads * head_dim]);
        self.decode_combine.0.encode_workgroups(
            &self.context,
            encoder,
            &uniforms,
            &[&stats, &partial, &attended],
            [heads as u32, batch as u32, 1],
        );
        attended
    }

    fn encode_write_kv(
        &self,
        encoder: &mut crate::gpu::GpuEncoder,
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
        self.store_kernels(k_store).write_kv.encode(
            &self.context,
            encoder,
            &uniforms,
            &[
                q_proj, k_proj, v_proj, cos_table, sin_table, q_rotated, k_store, v_store,
            ],
            batch * seq * heads as usize * (self.ops.head_dim() as usize / 2),
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

        let mut encoder = crate::gpu::GpuEncoder::from_encoder(
            ctx.device.create_command_encoder(&Default::default()),
        );
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

        let mut encoder = crate::gpu::GpuEncoder::from_encoder(
            ctx.device.create_command_encoder(&Default::default()),
        );
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
    /// query may see, project. `mask` is one row per batch row. `f16_kv`: the
    /// rotated keys and the values rounded to f16, as an F16 KV store holds them.
    fn reference(
        hidden: &Array3<f32>,
        mask: &Array2<f32>,
        [q_w, k_w, v_w, o_w]: [&Array2<f32>; 4],
        rope: &RoPE,
        heads: usize,
        kv_heads: usize,
        f16_kv: bool,
    ) -> Array3<f32> {
        let store = |x: f32| {
            if f16_kv {
                half::f16::from_f32(x).to_f32()
            } else {
                x
            }
        };
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
                        let kr: Vec<f32> =
                            rotate(&k.row(key).as_slice().unwrap()[kv * d..(kv + 1) * d], key)
                                .into_iter()
                                .map(store)
                                .collect();
                        let dot: f32 = qh.iter().zip(&kr).map(|(a, b)| a * b).sum();
                        scores.push((key, dot / (d as f32).sqrt()));
                    }
                    let largest = scores.iter().map(|s| s.1).fold(f32::MIN, f32::max);
                    let total: f32 = scores.iter().map(|s| (s.1 - largest).exp()).sum();
                    for (key, score) in scores {
                        let w = (score - largest).exp() / total;
                        for i in 0..d {
                            attended[[t, head * d + i]] += w * store(v[[key, kv * d + i]]);
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
        attention_case(DType::F32).await;
    }

    /// The same over an F16 KV cache, against the reference with keys and values
    /// rounded to f16 where the cache holds them.
    #[tokio::test]
    #[ignore = "GPU required"]
    async fn attention_over_an_f16_cache_matches_reference() {
        attention_case(DType::F16).await;
    }

    async fn attention_case(store: DType) {
        let ctx = setup().await;
        if store == DType::F16 && !ctx.f16_kv_cache {
            println!("Skipping: no f16 KV caches on this device");
            return;
        }
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
        let want_with = |f16_kv| {
            reference(
                &hidden,
                &mask,
                [&q_w, &k_w, &v_w, &o_w],
                &cpu_rope,
                heads,
                kv_heads,
                f16_kv,
            )
        };
        // Without a cache the store is the pool's F32; with one, `store`.
        let uncached = want_with(false);
        let want = want_with(store == DType::F16);
        // Rounding to f16 moves the output by more than the tolerance below, so
        // an F16 cache that held f32, or an f32 one that rounded, fails.
        let rounding = (&want - &uncached)
            .iter()
            .fold(0.0f32, |m, x| m.max(x.abs()));
        if store == DType::F16 {
            assert!(
                rounding > 2e-4,
                "f16 rounding moves the output by {rounding:.1e}"
            );
        }

        let cache_shape = vec![batch, kv_heads, 16, d];
        let k_cache = GpuTensor::uninitialized(&ctx, cache_shape.clone(), store, "k cache");
        let v_cache = GpuTensor::uninitialized(&ctx, cache_shape, store, "v cache");

        // Scores for two queries at a time: a prompt is attended in slices, the last
        // one short, as a prompt too long for one buffer binding is.
        let mut sliced =
            GpuRoPEAttention::new(&ctx, hidden_size as u32, heads as u32, kv_heads as u32);
        sliced.score_budget = (2 * batch * heads * seq * 4) as u64;

        let run = |in_slices: bool, input: &Array3<f32>, offset: usize, cached: bool| {
            let attn = if in_slices { &sliced } else { &attn };
            let (weights, rope, k_cache, v_cache) = (&weights, &rope, &k_cache, &v_cache);
            let ctx = ctx.clone();
            // As wide as the keys seen so far, like the mask a decoder builds.
            let keys = offset + input.dim().1;
            let mask_gpu =
                GpuTensor::from_ndarray(&ctx, &mask.slice(s![.., ..keys]).to_owned()).unwrap();
            let input = GpuTensor::from_ndarray(&ctx, input).unwrap();
            async move {
                let mut enc = crate::gpu::GpuEncoder::from_encoder(
                    ctx.device.create_command_encoder(&Default::default()),
                );
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
            &run(false, &hidden, 0, false).await,
            uncached.view(),
            0,
            "without a cache",
        );
        close(
            &run(false, &hidden, 0, true).await,
            want.view(),
            0,
            "prefill into a cache",
        );

        let split = 4;
        let first = hidden.slice(s![.., ..split, ..]).to_owned();
        close(
            &run(false, &first, 0, true).await,
            want.slice(s![.., ..split, ..]),
            0,
            "shorter prefill",
        );
        for t in split..seq {
            let step = hidden.slice(s![.., t..t + 1, ..]).to_owned();
            close(
                &run(false, &step, t, true).await,
                want.slice(s![.., t..t + 1, ..]),
                t,
                &format!("token {t}"),
            );
        }

        close(
            &run(true, &hidden, 0, false).await,
            uncached.view(),
            0,
            "in slices without a cache",
        );
        close(
            &run(true, &first, 0, true).await,
            want.slice(s![.., ..split, ..]),
            0,
            "in slices into a cache",
        );
        let rest = hidden.slice(s![.., split.., ..]).to_owned();
        close(
            &run(true, &rest, split, true).await,
            want.slice(s![.., split.., ..]),
            split,
            "in slices after a cached prefix",
        );
    }

    /// One query over keys in several chunks: the chunks merged give the
    /// softmax over every key it may see, with a padded first chunk in one
    /// batch row, keys past the query's position and grouped KV heads.
    #[tokio::test]
    #[ignore = "GPU required"]
    async fn one_query_attends_across_chunks_like_the_reference() {
        let ctx = setup().await;
        let (batch, heads, kv_heads, d) = (2usize, 8usize, 2usize, 64usize);
        let (capacity, keys, position) = (320usize, 300usize, 290usize);
        let attn = GpuRoPEAttention::new(&ctx, (heads * d) as u32, heads as u32, kv_heads as u32);
        let q = random(batch * heads, d, 1);
        let k = random(batch * kv_heads * capacity, d, 2);
        let v = random(batch * kv_heads * capacity, d, 3);
        let mut mask = Array2::<f32>::ones((batch, keys));
        mask.slice_mut(s![1, ..130]).fill(0.0);

        let gpu = |a: &Array2<f32>, shape: [usize; 4]| {
            GpuTensor::from_ndarray(&ctx, &a.clone().into_shape_with_order(shape).unwrap()).unwrap()
        };
        let q_gpu = gpu(&q, [batch, heads, 1, d]);
        let k_gpu = gpu(&k, [batch, kv_heads, capacity, d]);
        let v_gpu = gpu(&v, [batch, kv_heads, capacity, d]);
        let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask).unwrap();
        let mut enc = crate::gpu::GpuEncoder::from_encoder(
            ctx.device.create_command_encoder(&Default::default()),
        );
        let mut pool = GpuTensorPool::new(ctx.clone());
        let out = attn.attend(
            &mut enc, &q_gpu, &k_gpu, &v_gpu, &mask_gpu, position, keys, &mut pool,
        );
        ctx.queue.submit(Some(enc.finish()));
        let got = out.to_ndarray_3d::<f32>().await.unwrap();

        let mut worst = 0.0f32;
        for b in 0..batch {
            for h in 0..heads {
                let kv = h / (heads / kv_heads);
                let qr = q.row(b * heads + h);
                let row = |key: usize| (b * kv_heads + kv) * capacity + key;
                let seen: Vec<usize> = (0..=position)
                    .filter(|&key| mask[[b, key]] != 0.0)
                    .collect();
                let scores: Vec<f32> = seen
                    .iter()
                    .map(|&key| qr.dot(&k.row(row(key))) / (d as f32).sqrt())
                    .collect();
                let largest = scores.iter().cloned().fold(f32::MIN, f32::max);
                let total: f32 = scores.iter().map(|s| (s - largest).exp()).sum();
                for i in 0..d {
                    let want: f32 = seen
                        .iter()
                        .zip(&scores)
                        .map(|(&key, s)| (s - largest).exp() / total * v[[row(key), i]])
                        .sum();
                    worst = worst.max((got[[b, 0, h * d + i]] - want).abs());
                }
            }
        }
        assert!(worst < 1e-5, "differs by {worst:e}");
    }

    #[test]
    fn slices_keep_the_scores_within_the_budget() {
        // The prompt that failed: 16 heads, 7,093 new tokens over 9,060 keys, whose
        // scores in one piece are 4.1 GB against a 2 GB binding limit.
        let slice = queries_per_slice(16, 9060, 2_147_483_647);
        assert_eq!(slice, 3703);
        assert!((16 * slice * 9060 * 4) as u64 <= 2_147_483_647);
        assert_eq!(queries_per_slice(16, 9060, MAX_SCORE_BYTES), 925);
        // A query too wide for the budget still gets a slice of one.
        assert_eq!(queries_per_slice(16, 9060, 1024), 1);
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
        let mut enc = crate::gpu::GpuEncoder::from_encoder(
            ctx.device.create_command_encoder(&Default::default()),
        );
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
