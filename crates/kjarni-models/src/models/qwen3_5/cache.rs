//! The cache of a hybrid decoder: K and V for its attention layers, and a
//! convolution window and a recurrent state for its Gated DeltaNet layers.
//!
//! A KV cache can be cut back to any length by moving its end, because each
//! position keeps its own K and V. A recurrent state cannot: it is one matrix per
//! head that every token has already been folded into. So this cache keeps
//! snapshots of the recurrent half at the end of each prefill, and can be cut back
//! exactly to those points and nowhere else.
//!
//! Multi-turn chat is what this serves: the next turn's prompt starts with the
//! previous one. Not all of it, though. Chat templates often end a prompt with
//! a generation header that the next turn renders differently, as Qwen 3.5
//! renders an empty `<think></think>` block there and drops it from the history.
//! So a prefill also keeps the state from a few tokens before its end.

use std::any::Any;

use kjarni_transformers::cache::{Cache, CpuKVCache};
use ndarray::{Array3, Array4};

/// How many snapshots are kept. Each costs one copy of the recurrent state,
/// about 20 MB for the 0.8B model.
const SNAPSHOTS: usize = 8;

/// How far before the end of a prefill the second snapshot is taken: more than
/// a chat template's generation header, which is under a dozen tokens in the
/// templates Kjarni ships.
pub const SNAPSHOT_TAIL: usize = 16;

/// The state of one Gated DeltaNet layer.
#[derive(Clone)]
pub struct LinearState {
    /// The last `kernel - 1` inputs to the convolution, `[batch, channels, kernel - 1]`.
    pub conv: Array3<f32>,
    /// One `[key_dim, value_dim]` matrix per head, `[batch, heads, key_dim, value_dim]`.
    pub recurrent: Array4<f32>,
}

#[derive(Clone)]
struct Snapshot {
    len: usize,
    states: Vec<LinearState>,
}

#[derive(Clone)]
pub struct HybridCache {
    kv: CpuKVCache,
    linear: Vec<LinearState>,
    snapshots: Vec<Snapshot>,
    /// Set when asked to rewind to a point no snapshot holds. The next forward
    /// pass refuses to run rather than read a state from the wrong position.
    broken: Option<usize>,
}

pub struct HybridDims {
    pub attention_layers: usize,
    pub kv_dim: usize,
    pub linear_layers: usize,
    pub conv_channels: usize,
    pub conv_window: usize,
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl HybridCache {
    pub fn new(dims: &HybridDims, batch: usize, max_len: usize) -> Self {
        let state = LinearState {
            conv: Array3::zeros((batch, dims.conv_channels, dims.conv_window)),
            recurrent: Array4::zeros((batch, dims.heads, dims.key_dim, dims.value_dim)),
        };
        Self {
            kv: CpuKVCache::new(dims.attention_layers, batch, max_len, dims.kv_dim),
            linear: vec![state; dims.linear_layers],
            snapshots: Vec::new(),
            broken: None,
        }
    }

    pub fn kv_mut(&mut self) -> &mut CpuKVCache {
        &mut self.kv
    }

    /// The KV cache and the linear states at once, for a pass that writes both.
    pub fn parts_mut(&mut self) -> (&mut CpuKVCache, &mut [LinearState]) {
        (&mut self.kv, &mut self.linear)
    }

    /// Fails if a rewind left the recurrent state at the wrong position.
    pub fn check_usable(&self) -> anyhow::Result<()> {
        match self.broken {
            Some(len) => anyhow::bail!(
                "the recurrent state cannot be cut back to {len} tokens; start from an empty cache"
            ),
            None => Ok(()),
        }
    }

    /// Remembers the current state so a later prompt sharing everything up to
    /// here can start from it.
    pub fn snapshot(&mut self) {
        let len = self.kv.get_seq_length();
        self.keep_snapshot(len, self.linear.clone());
    }

    /// Remembers `states`, the linear states as they stood after `len` tokens.
    pub fn keep_snapshot(&mut self, len: usize, states: Vec<LinearState>) {
        self.snapshots.retain(|s| s.len != len);
        if self.snapshots.len() == SNAPSHOTS {
            self.snapshots.remove(0);
        }
        self.snapshots.push(Snapshot { len, states });
    }

    fn zero_states(&mut self) {
        for state in &mut self.linear {
            state.conv.fill(0.0);
            state.recurrent.fill(0.0);
        }
    }
}

impl Cache for HybridCache {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn get_seq_length(&self) -> usize {
        self.kv.get_seq_length()
    }

    /// Moves the end to `len`. Moving it back is exact only to the start or to
    /// a snapshot; [`Cache::reusable_prefix`] says which lengths those are.
    fn set_seq_length(&mut self, len: usize) {
        let current = self.kv.get_seq_length();
        if len == current {
            return;
        }
        if len == 0 {
            self.clear();
            return;
        }
        if len < current
            && let Some(snapshot) = self.snapshots.iter().find(|s| s.len == len)
        {
            self.linear = snapshot.states.clone();
            self.snapshots.retain(|s| s.len <= len);
            self.kv.set_seq_length(len);
            self.broken = None;
            return;
        }
        self.kv.set_seq_length(len);
        self.broken = Some(len);
    }

    fn reusable_prefix(&self, len: usize) -> usize {
        let current = self.kv.get_seq_length();
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

    fn clear(&mut self) {
        self.kv.clear();
        self.zero_states();
        self.snapshots.clear();
        self.broken = None;
    }
    fn increment_len(&mut self, new_tokens_len: usize) {
        self.kv.increment_len(new_tokens_len);
    }
    fn clone_box(&self) -> Box<dyn Cache> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> HybridCache {
        HybridCache::new(
            &HybridDims {
                attention_layers: 1,
                kv_dim: 4,
                linear_layers: 2,
                conv_channels: 3,
                conv_window: 3,
                heads: 2,
                key_dim: 2,
                value_dim: 2,
            },
            1,
            64,
        )
    }

    /// Stands in for a forward pass of `n` tokens: every state moves.
    fn advance(c: &mut HybridCache, n: usize) {
        for s in &mut c.linear {
            s.recurrent += n as f32;
            s.conv += n as f32;
        }
        c.increment_len(n);
    }

    #[test]
    fn rewinding_to_a_snapshot_restores_the_state_taken_there() {
        let mut c = cache();
        advance(&mut c, 10);
        c.snapshot();
        advance(&mut c, 5);
        assert_eq!(c.reusable_prefix(12), 10);
        c.set_seq_length(10);
        assert_eq!(c.get_seq_length(), 10);
        assert!(c.check_usable().is_ok());
        assert_eq!(c.linear[0].recurrent[[0, 0, 0, 0]], 10.0);
        assert_eq!(c.linear[1].conv[[0, 0, 0]], 10.0);
    }

    #[test]
    fn without_a_snapshot_only_the_start_is_reusable() {
        let mut c = cache();
        advance(&mut c, 10);
        assert_eq!(c.reusable_prefix(9), 0);
        // Asking for the current length or more costs nothing.
        assert_eq!(c.reusable_prefix(10), 10);
        assert_eq!(c.reusable_prefix(20), 10);
    }

    #[test]
    fn rewinding_where_no_snapshot_is_refuses_the_next_pass() {
        let mut c = cache();
        advance(&mut c, 10);
        c.set_seq_length(4);
        assert!(c.check_usable().is_err());
        c.set_seq_length(0);
        assert!(c.check_usable().is_ok());
        assert_eq!(c.linear[0].recurrent[[0, 0, 0, 0]], 0.0);
    }

    #[test]
    fn the_oldest_snapshot_goes_first() {
        let mut c = cache();
        for _ in 0..SNAPSHOTS + 1 {
            advance(&mut c, 1);
            c.snapshot();
        }
        assert_eq!(c.reusable_prefix(1), 0, "the first snapshot was dropped");
        assert_eq!(c.reusable_prefix(2), 2);
    }

    #[test]
    fn snapshots_past_a_rewind_are_forgotten() {
        let mut c = cache();
        advance(&mut c, 4);
        c.snapshot();
        advance(&mut c, 4);
        c.snapshot();
        c.set_seq_length(4);
        // A new continuation from 4 makes the old state at 8 meaningless.
        advance(&mut c, 4);
        assert_eq!(c.reusable_prefix(7), 4);
        c.set_seq_length(4);
        assert!(c.check_usable().is_ok());
    }
}
