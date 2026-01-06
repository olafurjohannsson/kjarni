use crate::gpu_ops::{DType, GpuTensor};
use crate::WgpuContext;
use log::info;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[cfg(debug_assertions)]
#[derive(Default, Debug)]
struct AllocationStats {
    gets: usize,
    reuses: usize,
    allocations: usize,
}

/// A persistent, frame-based memory pool for temporary `GpuTensor`s.
///
/// This allocator is designed to be a long-lived member of a model struct (e.g., `GpuCrossAttentionDecoder`).
/// It maximizes performance by reusing buffers across multiple `forward` calls, while ensuring safety
/// by preventing buffer aliasing within a single frame of work.
pub struct GpuTensorPool {
    context: Arc<WgpuContext>,

    // Tracks IDs of buffers handed out THIS frame to prevent aliasing.
    current_frame_ids: HashSet<u64>,

    // The master list of all allocated buffers, ready for reuse.
    buffer_pool: HashMap<usize, Vec<GpuTensor>>,

    #[cfg(debug_assertions)]
    stats: AllocationStats,
}

impl GpuTensorPool {
    /// Creates a new, empty tensor pool.
    pub fn new(context: Arc<WgpuContext>) -> Self {
        Self {
            context,
            current_frame_ids: HashSet::new(),
            buffer_pool: HashMap::new(),
            #[cfg(debug_assertions)]
            stats: AllocationStats::default(),
        }
    }

    /// Gets a temporary buffer. Guaranteed not to alias with other buffers from this pool in the same frame.
    pub fn get(&mut self, shape: Vec<usize>) -> GpuTensor {
        #[cfg(debug_assertions)]
        {
            self.stats.gets += 1;
        }

        let needed_size = shape.iter().product::<usize>();

        // Try to find an available buffer in the pool.
        let tensor = self
            .try_reuse_from_pool(needed_size, shape.clone())
            .unwrap_or_else(|| {
                #[cfg(debug_assertions)]
                {
                    self.stats.allocations += 1;
                }
                self.create_and_store_tensor(shape)
            });

        // Mark this buffer's ID as "in-flight" for the current frame.
        let was_newly_inserted = self.current_frame_ids.insert(tensor.buffer_id());

        #[cfg(debug_assertions)]
        {
            assert!(
                was_newly_inserted,
                "Safety violation: Buffer ID {} was already active this frame!",
                tensor.buffer_id()
            );
        }

        tensor
    }

    /// Searches the pool for an idle buffer that can be reused.
    fn try_reuse_from_pool(&mut self, needed_size: usize, shape: Vec<usize>) -> Option<GpuTensor> {
        // --- START FIX ---
        // Only try for an exact size match.
        if let Some(buffers) = self.buffer_pool.get(&needed_size) {
            // Find the first buffer in this size bucket that is not currently in use.
            if let Some(tensor_to_reuse) = buffers
                .iter()
                .find(|t| !self.current_frame_ids.contains(&t.buffer_id()))
            {
                #[cfg(debug_assertions)]
                {
                    self.stats.reuses += 1;
                }

                // Since num_elements is guaranteed to be the same, we can safely view or clone.
                // Return a clone or a view of the tensor. The pool retains ownership of the original.
                return Some(if tensor_to_reuse.shape() == shape.as_slice() {
                    tensor_to_reuse.clone()
                } else {
                    tensor_to_reuse.view(shape)
                });
            }
        }

        // If no exact size match was found and available, return None to force a new allocation.
        None
        // --- END FIX ---
    }

    /// Creates a new tensor, stores it in the pool, and returns it.
    fn create_and_store_tensor(&mut self, shape: Vec<usize>) -> GpuTensor {
        let tensor = {
            #[cfg(test)]
            {
                GpuTensor::zeros(&self.context, shape.clone(), DType::F32, "GpuTensorPool")
                    .expect("Failed to allocate temp tensor")
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

    /// Advances the frame, making all buffers used in the last frame available for reuse.
    /// This MUST be called after the associated `CommandEncoder`'s work is submitted.
    pub fn next_frame(&mut self) {
        self.current_frame_ids.clear();

        #[cfg(debug_assertions)]
        {
            if self.stats.gets > 0 {
                let reuse_rate = if self.stats.gets > 0 {
                    (self.stats.reuses as f32 / self.stats.gets as f32) * 100.0
                } else {
                    0.0
                };
                info!(
                    "[GpuTensorPool] Frame stats - Gets: {}, Reuses: {} ({:.1}%), Allocations: {}. Pool total: {}",
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
