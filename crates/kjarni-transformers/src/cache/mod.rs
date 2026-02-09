//! Cache implementations for transformer models

mod cpu;
mod cpu_beam;
pub use cpu_beam::CpuBeamKVCache;
pub use cpu::CpuKVCache;

use std::any::Any;

/// A type-erased, thread-safe container for mutable inference state.
pub trait Cache: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Get the current sequence length (number of cached tokens)
    fn get_seq_length(&self) -> usize;
    fn set_seq_length(&mut self, len: usize);
    /// Clear the cache
    fn clear(&mut self);
    fn increment_len(&mut self, new_tokens_len: usize);
    fn clone_box(&self) -> Box<dyn Cache>;
}

#[cfg(test)]
mod tests;