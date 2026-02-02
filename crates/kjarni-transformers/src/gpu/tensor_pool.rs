//! GPU tensor memory pool for temporary buffer reuse.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::gpu::{DType, GpuTensor};
use crate::WgpuContext;

#[cfg(debug_assertions)]
#[derive(Default, Debug)]
struct AllocationStats {
    gets: usize,
    reuses: usize,
    allocations: usize,
}

/// A frame-based memory pool for temporary GPU tensors.
///
/// Maximizes performance by reusing buffers across forward passes while
/// preventing aliasing within a single frame.
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
        if let Some(buffers) = self.buffer_pool.get(&needed_size) {
            if let Some(tensor_to_reuse) = buffers
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
    /// Must be called after the associated `CommandEncoder` work is submitted.
    pub fn next_frame(&mut self) {
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