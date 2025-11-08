//! Cache implementations for transformer models

mod cpu;
mod gpu;

mod tests;

pub use cpu::CpuKVCache;
pub use gpu::GpuKVCache;

use std::any::Any;

/// A type-erased, thread-safe container for mutable inference state.
///
/// This is essential for efficient autoregressive generation. Concrete implementations
/// might store attention Key-Value tensors, beam search hypotheses, or other
/// intermediate state that needs to be preserved across generation steps.
///
/// The `AsAny` and `AsAnyMut` methods are crucial for downcasting a `&mut dyn Cache`
/// back to its concrete type within a model's `forward` implementation.
pub trait Cache: Send + Sync {
    /// Returns a reference to the underlying cache as a type-erased `Any` object.
    fn as_any(&self) -> &dyn Any;
    /// Returns a mutable reference to the underlying cache as a type-erased `Any` object.
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Get the current sequence length (number of cached tokens)
    fn get_seq_length(&self) -> usize;
    /// Clear the cache
    fn clear(&mut self);

    fn increment_len(&mut self, new_tokens_len: usize);

}


