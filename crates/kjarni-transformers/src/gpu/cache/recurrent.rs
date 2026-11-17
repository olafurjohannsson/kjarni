//! The recurrent half of a hybrid decoder's cache on the GPU: state buffers
//! that every token is folded into, as Gated DeltaNet's convolution window and
//! delta-rule state are.
//!
//! A KV cache can be cut back to any length. A recurrent state cannot, so this
//! keeps copies of it at chosen lengths (snapshots), and a cut back is exact
//! only to one of those or to the start. A forward pass decides where they are
//! taken: at the end of a prefill, and just before it, as the CPU's hybrid
//! cache does.

use std::sync::Arc;

use anyhow::Result;

use crate::WgpuContext;
use crate::gpu::{DType, GpuTensor};

struct Snapshot {
    len: usize,
    states: Vec<GpuTensor>,
}

pub struct GpuRecurrentState {
    context: Arc<WgpuContext>,
    shapes: Vec<Vec<usize>>,
    states: Vec<GpuTensor>,
    snapshots: Vec<Snapshot>,
    /// Buffers of dropped snapshots, reused for the next ones.
    spare: Vec<Vec<GpuTensor>>,
    max_snapshots: usize,
    /// Set when cut back to a length no snapshot holds; a forward pass must
    /// refuse to run from a state at the wrong position.
    broken: Option<usize>,
}

impl GpuRecurrentState {
    /// Zeroed f32 states of `shapes`, keeping at most `max_snapshots` copies.
    pub fn new(
        context: &Arc<WgpuContext>,
        shapes: Vec<Vec<usize>>,
        max_snapshots: usize,
    ) -> Result<Self> {
        let states = Self::allocate(context, &shapes, "recurrent state")?;
        Ok(Self {
            context: context.clone(),
            shapes,
            states,
            snapshots: Vec::new(),
            spare: Vec::new(),
            max_snapshots,
            broken: None,
        })
    }

    fn allocate(
        context: &Arc<WgpuContext>,
        shapes: &[Vec<usize>],
        label: &str,
    ) -> Result<Vec<GpuTensor>> {
        shapes
            .iter()
            .map(|shape| GpuTensor::zeros(context, shape.clone(), DType::F32, label))
            .collect()
    }

    /// The live states, in the order of their shapes.
    pub fn states(&self) -> &[GpuTensor] {
        &self.states
    }

    /// Fails if a cut back left the states at the wrong position.
    pub fn check_usable(&self) -> Result<()> {
        match self.broken {
            Some(len) => anyhow::bail!(
                "the recurrent state cannot be cut back to {len} tokens; start from an empty cache"
            ),
            None => Ok(()),
        }
    }

    /// Buffers shaped like the states, for a pass to write a snapshot into;
    /// [`keep`](Self::keep) them once the pass is encoded. Their contents are
    /// whatever an earlier snapshot left.
    pub fn take_slot(&mut self) -> Result<Vec<GpuTensor>> {
        match self.spare.pop() {
            Some(slot) => Ok(slot),
            None => Self::allocate(&self.context, &self.shapes, "recurrent snapshot"),
        }
    }

    /// Records `states` as the state after `len` tokens, dropping the oldest
    /// snapshot when there are too many.
    pub fn keep(&mut self, len: usize, states: Vec<GpuTensor>) {
        if let Some(i) = self.snapshots.iter().position(|s| s.len == len) {
            let old = self.snapshots.remove(i);
            self.spare.push(old.states);
        }
        if self.snapshots.len() == self.max_snapshots {
            let old = self.snapshots.remove(0);
            self.spare.push(old.states);
        }
        self.snapshots.push(Snapshot { len, states });
    }

    /// Encodes a copy of the live states as the snapshot after `len` tokens.
    pub fn snapshot(&mut self, encoder: &mut wgpu::CommandEncoder, len: usize) -> Result<()> {
        let slot = self.take_slot()?;
        for (from, to) in self.states.iter().zip(&slot) {
            encoder.copy_buffer_to_buffer(from.buffer(), 0, to.buffer(), 0, from.buffer().size());
        }
        self.keep(len, slot);
        Ok(())
    }

    /// The longest length of at most `len` the states can be cut back to,
    /// with `current` tokens in them now.
    pub fn reusable_prefix(&self, current: usize, len: usize) -> usize {
        if len >= current && self.broken.is_none() {
            return current.min(len);
        }
        self.snapshots
            .iter()
            .map(|s| s.len)
            .filter(|&l| l <= len)
            .max()
            .unwrap_or(0)
    }

    /// Moves the states from `current` tokens to `len`: to the start, or back
    /// to a snapshot. Anywhere else marks them unusable.
    pub fn set_len(&mut self, current: usize, len: usize) {
        if len == current {
            return;
        }
        if len == 0 {
            self.clear();
            return;
        }
        if len < current
            && let Some(i) = self.snapshots.iter().position(|s| s.len == len)
        {
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("restore recurrent snapshot"),
                    });
            for (from, to) in self.snapshots[i].states.iter().zip(&self.states) {
                encoder.copy_buffer_to_buffer(
                    from.buffer(),
                    0,
                    to.buffer(),
                    0,
                    from.buffer().size(),
                );
            }
            self.context.queue.submit(Some(encoder.finish()));
            // Snapshots past `len` belong to a continuation that is gone.
            let (kept, dropped): (Vec<_>, Vec<_>) =
                self.snapshots.drain(..).partition(|s| s.len <= len);
            self.snapshots = kept;
            self.spare.extend(dropped.into_iter().map(|s| s.states));
            self.broken = None;
            return;
        }
        self.broken = Some(len);
    }

    /// Zeroes the states and forgets every snapshot.
    pub fn clear(&mut self) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("clear recurrent state"),
                });
        for state in &self.states {
            encoder.clear_buffer(state.buffer(), 0, None);
        }
        self.context.queue.submit(Some(encoder.finish()));
        self.spare
            .extend(self.snapshots.drain(..).map(|s| s.states));
        self.broken = None;
    }

    /// A copy with buffers of its own.
    pub fn deep_clone(&self) -> Self {
        let copy = |states: &[GpuTensor]| -> Vec<GpuTensor> {
            states
                .iter()
                .map(|t| t.deep_clone("recurrent state copy"))
                .collect()
        };
        Self {
            context: self.context.clone(),
            shapes: self.shapes.clone(),
            states: copy(&self.states),
            snapshots: self
                .snapshots
                .iter()
                .map(|s| Snapshot {
                    len: s.len,
                    states: copy(&s.states),
                })
                .collect(),
            spare: Vec::new(),
            max_snapshots: self.max_snapshots,
            broken: self.broken,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::Cache;
    use crate::gpu::GpuKVCache;

    async fn cache(max_snapshots: usize) -> GpuKVCache {
        let ctx = WgpuContext::new().await.unwrap();
        let recurrent =
            GpuRecurrentState::new(&ctx, vec![vec![2], vec![1, 3]], max_snapshots).unwrap();
        GpuKVCache::new(&ctx, 1, 1, 1, 4, 64)
            .unwrap()
            .with_recurrent(recurrent)
    }

    /// Stands in for a forward pass of `n` tokens that leaves every state
    /// value at `value`.
    fn advance(c: &mut GpuKVCache, n: usize, value: f32) {
        let r = c.recurrent().unwrap();
        for state in r.states() {
            let values = vec![value; state.num_elements()];
            r.context
                .queue
                .write_buffer(state.buffer(), 0, bytemuck::cast_slice(&values));
        }
        c.increment_len(n);
    }

    async fn values(c: &GpuKVCache) -> (Vec<f32>, Vec<f32>) {
        let states = c.recurrent().unwrap().states();
        let a = states[0].to_ndarray_1d::<f32>().await.unwrap().to_vec();
        let b = states[1].to_ndarray_2d::<f32>().await.unwrap();
        (a, b.iter().copied().collect())
    }

    fn snapshot(c: &mut GpuKVCache) {
        let len = c.get_seq_length();
        let r = c.recurrent_mut().unwrap();
        let mut encoder = r.context.device.create_command_encoder(&Default::default());
        r.snapshot(&mut encoder, len).unwrap();
        r.context.queue.submit(Some(encoder.finish()));
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn cutting_back_to_a_snapshot_restores_the_states_taken_there() {
        let mut c = cache(8).await;
        advance(&mut c, 10, 1.5);
        snapshot(&mut c);
        advance(&mut c, 5, 7.0);
        assert_eq!(c.reusable_prefix(12), 10);
        assert_eq!(c.reusable_prefix(15), 15);

        c.set_seq_length(10);
        assert_eq!(c.get_seq_length(), 10);
        assert!(c.recurrent().unwrap().check_usable().is_ok());
        assert_eq!(values(&c).await, (vec![1.5; 2], vec![1.5; 3]));
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn cutting_back_where_no_snapshot_is_refuses_until_cleared() {
        let mut c = cache(8).await;
        advance(&mut c, 10, 2.0);
        assert_eq!(c.reusable_prefix(9), 0);
        c.set_seq_length(4);
        assert!(c.recurrent().unwrap().check_usable().is_err());

        c.set_seq_length(0);
        assert!(c.recurrent().unwrap().check_usable().is_ok());
        assert_eq!(values(&c).await, (vec![0.0; 2], vec![0.0; 3]));
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn the_oldest_snapshot_goes_first_and_later_ones_go_with_a_cut() {
        let mut c = cache(2).await;
        for step in 1..=3 {
            advance(&mut c, 1, step as f32);
            snapshot(&mut c);
        }
        assert_eq!(c.reusable_prefix(1), 0, "the first of three was dropped");
        assert_eq!(c.reusable_prefix(2), 2);

        c.set_seq_length(2);
        assert_eq!(values(&c).await, (vec![2.0; 2], vec![2.0; 3]));
        // The state at 3 belonged to the continuation that was cut.
        advance(&mut c, 2, 9.0);
        assert_eq!(c.reusable_prefix(3), 2);
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn a_clone_has_states_of_its_own() {
        let mut c = cache(8).await;
        advance(&mut c, 3, 4.0);
        let copy = c.clone_box();
        advance(&mut c, 1, 5.0);
        let copy = copy.as_any().downcast_ref::<GpuKVCache>().unwrap();
        assert_eq!(values(copy).await, (vec![4.0; 2], vec![4.0; 3]));
    }
}
