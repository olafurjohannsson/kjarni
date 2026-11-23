use crate::WgpuContext;
use crate::cache::Cache;
use crate::gpu::cache::GpuRecurrentState;
use crate::gpu::{DType, GpuTensor};
use crate::gpu_ops::blocks::cache::GpuUpdateCache;
use anyhow::{Result, anyhow, ensure};
use std::any::Any;
use std::sync::Arc;

pub struct GpuKVCache {
    // A vector of K-cache tensors, one for each decoder layer
    k_tensors: Vec<GpuTensor>,
    // A vector of V-cache tensors, one for each decoder layer
    v_tensors: Vec<GpuTensor>,
    // The current number of tokens stored in the cache.
    seq_length: usize,
    // The specialized kernel for performing in-place updates
    update_kernel: GpuUpdateCache,
    /// The recurrent states of a hybrid decoder's other layers, which move
    /// with the KV cache.
    recurrent: Option<GpuRecurrentState>,
}

impl GpuKVCache {
    /// Creates a new, pre-allocated GPU KV cache of f32.
    pub fn new(
        context: &Arc<WgpuContext>,
        num_layers: usize,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Result<Self> {
        Self::with_dtype(
            context,
            num_layers,
            batch_size,
            num_heads,
            head_dim,
            capacity,
            DType::F32,
        )
    }

    /// A cache of `dtype`, F32 or F16. An F16 cache is written and read only by
    /// the kernels built with [`kv_shader`]: those of `GpuRoPEAttention` and the
    /// decoders that share its attention.
    pub fn with_dtype(
        context: &Arc<WgpuContext>,
        num_layers: usize,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        capacity: usize,
        dtype: DType,
    ) -> Result<Self> {
        ensure!(
            matches!(dtype, DType::F32 | DType::F16),
            "a KV cache holds F32 or F16, not {dtype:?}"
        );
        ensure!(
            dtype != DType::F16 || head_dim.is_multiple_of(2),
            "an F16 KV cache needs an even head size, not {head_dim}"
        );
        if capacity == 0 {
            return Err(anyhow!("Cache capacity cannot be zero."));
        }
        log::info!(
            "[GpuKVCache] Starting new()... batch={}, heads={}, capacity={}",
            batch_size,
            num_heads,
            capacity
        );
        let cache_shape = vec![batch_size, num_heads, capacity, head_dim];
        let mut k_tensors = Vec::with_capacity(num_layers);
        let mut v_tensors = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            k_tensors.push(GpuTensor::uninitialized(
                context,
                cache_shape.clone(),
                dtype,
                &format!("Layer {} K-Cache", i),
            ));
            v_tensors.push(GpuTensor::uninitialized(
                context,
                cache_shape.clone(),
                dtype,
                &format!("Layer {} V-Cache", i),
            ));
        }

        Ok(Self {
            k_tensors,
            v_tensors,
            seq_length: 0,
            update_kernel: GpuUpdateCache::new(context),
            recurrent: None,
        })
    }

    /// This cache with `recurrent` states alongside, for a hybrid decoder whose
    /// attention layers use the KV half.
    pub fn with_recurrent(mut self, recurrent: GpuRecurrentState) -> Self {
        self.recurrent = Some(recurrent);
        self
    }

    pub fn recurrent(&self) -> Option<&GpuRecurrentState> {
        self.recurrent.as_ref()
    }

    pub fn recurrent_mut(&mut self) -> Option<&mut GpuRecurrentState> {
        self.recurrent.as_mut()
    }

    /// Encodes a GPU command to append new key and value tensors to the cache
    pub fn update(
        &self,
        encoder: &mut crate::gpu::GpuEncoder,
        layer_idx: usize,
        new_k: &GpuTensor,
        new_v: &GpuTensor,
        position_offset: usize,
    ) -> Result<()> {
        if layer_idx >= self.k_tensors.len() {
            anyhow::bail!("Layer index {} out of bounds", layer_idx);
        }
        if new_k.rank() != 3 || new_v.rank() != 3 {
            return Err(anyhow::anyhow!(
                "Input tensors for cache update must be rank 3"
            ));
        }
        ensure!(
            self.dtype() == DType::F32,
            "this update kernel writes F32 caches only, not {:?}",
            self.dtype()
        );
        let cache_k = &self.k_tensors[layer_idx];
        let cache_v = &self.v_tensors[layer_idx];
        // let position_offset = self.seq_length;
        self.update_kernel
            .encode(encoder, new_k, new_v, cache_k, cache_v, position_offset); //self.seq_length);

        Ok(())
    }

    /// Retrieves a view of the cached keys and values for a specific layer.
    pub fn get(&self, layer_idx: usize) -> Option<(GpuTensor, GpuTensor)> {
        if layer_idx >= self.k_tensors.len() {
            return None;
        }
        let cache_k = &self.k_tensors[layer_idx];
        let cache_v = &self.v_tensors[layer_idx];
        Some((cache_k.clone(), cache_v.clone()))
    }

    pub fn get_seq_length(&self) -> usize {
        self.seq_length
    }

    /// The keys' and values' element type, F32 or F16.
    pub fn dtype(&self) -> DType {
        self.k_tensors.first().map_or(DType::F32, |t| t.dtype())
    }

    pub fn max_seq_len(&self) -> usize {
        if self.k_tensors.is_empty() {
            0
        } else {
            self.k_tensors[0].shape()[2]
        }
    }

    pub fn increment_len(&mut self, new_tokens_len: usize) {
        let new_len = self.seq_length + new_tokens_len;
        let max = self.max_seq_len();

        if new_len > max {
            panic!(
                "GpuKVCache overflow: tried to set seq_length to {} but capacity is {}. \
             Current: {}, adding: {}. This indicates a bug in the generation loop.",
                new_len, max, self.seq_length, new_tokens_len
            );
        }

        self.seq_length = new_len;
    }
}

impl Cache for GpuKVCache {
    fn get_seq_length(&self) -> usize {
        self.seq_length
    }
    fn clone_box(&self) -> Box<dyn Cache> {
        let new_k_tensors = self
            .k_tensors
            .iter()
            .enumerate()
            .map(|(i, t)| t.deep_clone(&format!("Cloned Layer {} K-Cache", i)))
            .collect();

        let new_v_tensors = self
            .v_tensors
            .iter()
            .enumerate()
            .map(|(i, t)| t.deep_clone(&format!("Cloned Layer {} V-Cache", i)))
            .collect();

        let new_cache = GpuKVCache {
            k_tensors: new_k_tensors,
            v_tensors: new_v_tensors,
            seq_length: self.seq_length,
            update_kernel: self.update_kernel.clone(),
            recurrent: self.recurrent.as_ref().map(|r| r.deep_clone()),
        };
        Box::new(new_cache)
    }
    fn clear(&mut self) {
        self.seq_length = 0;
        if let Some(recurrent) = &mut self.recurrent {
            recurrent.clear();
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn set_seq_length(&mut self, len: usize) {
        if let Some(recurrent) = &mut self.recurrent {
            recurrent.set_len(self.seq_length, len);
        }
        self.seq_length = len;
    }

    fn reusable_prefix(&self, len: usize) -> usize {
        match &self.recurrent {
            Some(recurrent) => recurrent.reusable_prefix(self.seq_length, len),
            None => len,
        }
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn increment_len(&mut self, new_tokens_len: usize) {
        self.increment_len(new_tokens_len);
    }
}

/// A shader that reads or writes a KV store, for a store of `dtype`. The
/// source declares `alias KV = f32;` and converts at every access; for F16 that
/// becomes `enable f16; alias KV = f16;`.
pub fn kv_shader<'a>(
    label: &'a str,
    source: &str,
    dtype: DType,
) -> wgpu::ShaderModuleDescriptor<'a> {
    const ALIAS: &str = "alias KV = f32;";
    assert_eq!(source.matches(ALIAS).count(), 1, "{label}: no `{ALIAS}`");
    let source = match dtype {
        DType::F32 => source.to_string(),
        DType::F16 => source.replacen(ALIAS, "enable f16;\nalias KV = f16;", 1),
        other => panic!("{label}: a KV store holds F32 or F16, not {other:?}"),
    };
    wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    }
}

/// The KV cache dtype for a decoder whose attention reads F32 or F16 stores:
/// F16 over quantised weights when the context has f16 KV caches, as it does
/// unless exact mode is on. Float weights keep an F32 cache, so their GPU output
/// stays in step with the CPU's.
pub fn kv_cache_dtype(context: &WgpuContext, quantized_weights: bool) -> DType {
    if quantized_weights && context.f16_kv_cache {
        DType::F16
    } else {
        DType::F32
    }
}
