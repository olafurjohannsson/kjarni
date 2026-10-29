//! GPU tensor memory pool for temporary buffer reuse.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::WgpuContext;
use crate::gpu::{DType, GpuTensor};

#[cfg(debug_assertions)]
#[derive(Default, Debug)]
struct AllocationStats {
    gets: usize,
    reuses: usize,
    allocations: usize,
}

/// A frame-based memory pool for temporary GPU tensors
pub struct GpuTensorPool {
    context: Arc<WgpuContext>,
    current_frame_ids: HashSet<u64>,
    buffer_pool: HashMap<usize, Vec<GpuTensor>>,
    #[cfg(debug_assertions)]
    stats: AllocationStats,
}

impl GpuTensorPool {
    /// Creates a new empty tensor pool.
    pub fn new(context: Arc<WgpuContext>) -> Self {
        Self {
            context,
            current_frame_ids: HashSet::new(),
            buffer_pool: HashMap::new(),
            #[cfg(debug_assertions)]
            stats: AllocationStats::default(),
        }
    }

    /// Returns a temporary buffer guaranteed not to alias with other buffers from this frame.
    pub fn get(&mut self, shape: Vec<usize>) -> GpuTensor {
        #[cfg(debug_assertions)]
        {
            self.stats.gets += 1;
        }

        let needed_size = shape.iter().product::<usize>();

        let tensor = self
            .try_reuse_from_pool(needed_size, shape.clone())
            .unwrap_or_else(|| {
                #[cfg(debug_assertions)]
                {
                    self.stats.allocations += 1;
                }
                self.create_and_store_tensor(shape)
            });

        #[cfg_attr(not(debug_assertions), allow(unused_variables))]
        let was_newly_inserted = self.current_frame_ids.insert(tensor.buffer_id());

        #[cfg(debug_assertions)]
        {
            assert!(
                was_newly_inserted,
                "safety violation: buffer id {} was already active this frame",
                tensor.buffer_id()
            );
        }

        tensor
    }

    fn try_reuse_from_pool(&mut self, needed_size: usize, shape: Vec<usize>) -> Option<GpuTensor> {
        if let Some(buffers) = self.buffer_pool.get(&needed_size)
            && let Some(tensor_to_reuse) = buffers
                .iter()
                .find(|t| !self.current_frame_ids.contains(&t.buffer_id()))
        {
            #[cfg(debug_assertions)]
            {
                self.stats.reuses += 1;
            }

            return Some(if tensor_to_reuse.shape() == shape.as_slice() {
                tensor_to_reuse.clone()
            } else {
                tensor_to_reuse.view(shape)
            });
        }

        None
    }

    fn create_and_store_tensor(&mut self, shape: Vec<usize>) -> GpuTensor {
        let tensor = {
            #[cfg(test)]
            {
                GpuTensor::zeros(&self.context, shape.clone(), DType::F32, "GpuTensorPool")
                    .expect("failed to allocate temp tensor")
            }
            #[cfg(not(test))]
            {
                GpuTensor::uninitialized(&self.context, shape, DType::F32, "GpuTensorPool")
            }
        };

        let size = tensor.num_elements();
        self.buffer_pool
            .entry(size)
            .or_default()
            .push(tensor.clone());

        tensor
    }

    /// Advances to the next frame, making all buffers available for reuse.
    ///
    /// Buffers the frame just ended did not use are released. Decode grows the key
    /// length by one every step, so attention temporaries arrive in a new size each
    /// token; kept, a 3B model accumulates about 70MB of them per token until the
    /// card is full and the driver starts paging, at which point decode runs several
    /// times slower. A frame that repeats the previous frame's shapes reuses all of
    /// its buffers, so steady-state decode allocates only what changed size.
    pub fn next_frame(&mut self) {
        let used = &self.current_frame_ids;
        for buffers in self.buffer_pool.values_mut() {
            buffers.retain(|t| used.contains(&t.buffer_id()));
        }
        self.buffer_pool.retain(|_, buffers| !buffers.is_empty());
        self.current_frame_ids.clear();

        #[cfg(debug_assertions)]
        {
            if self.stats.gets > 0 {
                let reuse_rate = (self.stats.reuses as f32 / self.stats.gets as f32) * 100.0;
                log::info!(
                    "[GpuTensorPool] frame stats - gets: {}, reuses: {} ({:.1}%), allocations: {}. pool total: {}",
                    self.stats.gets,
                    self.stats.reuses,
                    reuse_rate,
                    self.stats.allocations,
                    self.buffer_pool.values().map(Vec::len).sum::<usize>()
                );
            }
            self.stats = AllocationStats::default();
        }
    }
}

/// The buffers in use at some point in a frame. See [`GpuTensorPool::release_since`].
pub struct PoolCheckpoint(HashSet<u64>);

impl GpuTensorPool {
    /// Records the buffers in use now, before a loop that will call
    /// [`Self::release_since`].
    pub fn checkpoint(&self) -> PoolCheckpoint {
        PoolCheckpoint(self.current_frame_ids.clone())
    }

    /// Makes the buffers handed out since `checkpoint` reusable, except those
    /// behind `live`, without ending the frame. Buffers in use at the
    /// checkpoint stay in use: the caller may still read them, as a test that
    /// runs the layers twice on the same input does.
    ///
    /// A deep forward pass is one frame, and within a frame no buffer is handed
    /// out twice, so every layer's intermediates stay allocated until the frame
    /// ends: about 200MB a layer for bge-m3 at 3000 tokens, 4.8GB over its 24
    /// layers, enough to push the weights out of VRAM. Calling this between
    /// layers with the hidden state that carries over keeps one layer's worth.
    ///
    /// Reuse inside one command encoder is safe: the commands that used a buffer
    /// earlier are already recorded, and wgpu orders the passes that touch it.
    pub fn release_since(&mut self, checkpoint: &PoolCheckpoint, live: &[&GpuTensor]) {
        self.next_frame();
        self.current_frame_ids.extend(checkpoint.0.iter().copied());
        for t in live {
            self.current_frame_ids.insert(t.buffer_id());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pooled(pool: &GpuTensorPool) -> usize {
        pool.buffer_pool.values().map(Vec::len).sum()
    }

    /// Decode asks for one more key position every step. If every size the pool has
    /// handed out stayed allocated, VRAM would grow with each token.
    #[tokio::test]
    #[ignore = "GPU required"]
    async fn growing_shapes_do_not_accumulate_across_frames() -> anyhow::Result<()> {
        let context = WgpuContext::new().await?;
        let mut pool = GpuTensorPool::new(context);

        for key_len in 1..=50 {
            let _fixed = pool.get(vec![1, 3072]);
            let _growing = pool.get(vec![24, key_len, 128]);
            pool.next_frame();
        }

        // Only the last frame's two buffers are kept, not one per key length.
        assert_eq!(pooled(&pool), 2);
        Ok(())
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn a_repeated_frame_reuses_its_buffers() -> anyhow::Result<()> {
        let context = WgpuContext::new().await?;
        let mut pool = GpuTensorPool::new(context);

        let first = pool.get(vec![1, 3072]).buffer_id();
        pool.next_frame();
        let second = pool.get(vec![1, 3072]).buffer_id();
        pool.next_frame();

        assert_eq!(first, second);
        assert_eq!(pooled(&pool), 1);
        Ok(())
    }

    /// Two live requests of the same size in one frame must never alias, and both
    /// survive into the next frame.
    #[tokio::test]
    #[ignore = "GPU required"]
    async fn same_size_requests_in_one_frame_get_distinct_buffers() -> anyhow::Result<()> {
        let context = WgpuContext::new().await?;
        let mut pool = GpuTensorPool::new(context);

        let a = pool.get(vec![4, 4]).buffer_id();
        let b = pool.get(vec![16]).buffer_id();
        assert_ne!(a, b);
        pool.next_frame();
        assert_eq!(pooled(&pool), 2);
        Ok(())
    }

    /// After release_since, buffers from the finished scope are reused, but
    /// never the live ones or those in use at the checkpoint: handing out a
    /// tensor someone is about to read as scratch space would corrupt it.
    #[tokio::test]
    #[ignore = "GPU required"]
    async fn release_since_reuses_only_the_finished_scope() -> anyhow::Result<()> {
        let context = WgpuContext::new().await?;
        let mut pool = GpuTensorPool::new(context);

        let input = pool.get(vec![8, 64]);
        let checkpoint = pool.checkpoint();
        let hidden = pool.get(vec![8, 64]);
        let scratch = pool.get(vec![8, 64]);
        pool.release_since(&checkpoint, &[&hidden]);

        let next = pool.get(vec![8, 64]);
        assert_eq!(
            next.buffer_id(),
            scratch.buffer_id(),
            "scratch was not reused"
        );
        let another = pool.get(vec![8, 64]);
        for kept in [&input, &hidden, &next] {
            assert_ne!(
                another.buffer_id(),
                kept.buffer_id(),
                "a buffer in use was handed out"
            );
        }
        Ok(())
    }
}
