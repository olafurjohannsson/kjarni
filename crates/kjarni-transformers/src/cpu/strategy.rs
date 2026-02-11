
// heuristics
pub mod thresholds {
    /// Below this token count, fused QKV always wins
    pub const FUSED_ALWAYS_WINS_TOKENS: usize = 64;
    
    /// Above this hidden size with large batches, separate wins
    pub const LARGE_HIDDEN_THRESHOLD: usize = 768;
    
    /// Above this token count, no-alloc with buffer reuse wins
    pub const NOALLOC_WINS_TOKENS: usize = 1000;
    
    /// Below this token count (decode)
    pub const DECODE_THRESHOLD: usize = 1;
    
    /// Threshold for switching from vec kernel to batched 4x3 kernel
    pub const BATCH_KERNEL_THRESHOLD: usize = 1000;
}

/// Compute strategy decisions for attention and matmul operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeStrategy {
    pub use_fused_qkv: bool,
    pub use_scratch_buffers: bool,
}

impl ComputeStrategy {
    /// Select optimal strategy based on workload characteristics
    pub fn select(tokens: usize, hidden: usize) -> Self {
        use thresholds::*;
        
        // FUSED DECISION
        let use_fused_qkv = if tokens <= FUSED_ALWAYS_WINS_TOKENS {
            true
        } else if hidden >= LARGE_HIDDEN_THRESHOLD && tokens >= 512 {
            false
        } else if hidden <= 384 {
            true
        } else {
            tokens < 256
        };

        // NO-ALLOC DECISION
        let use_scratch_buffers = tokens <= DECODE_THRESHOLD || tokens >= NOALLOC_WINS_TOKENS;

        Self { use_fused_qkv, use_scratch_buffers }
    }
    
    /// Strategy optimized for decode (autoregressive generation)
    pub fn decode() -> Self {
        Self { use_fused_qkv: true, use_scratch_buffers: true }
    }
    
    /// Strategy optimized for encoding large batches
    pub fn encode_batch(hidden: usize) -> Self {
        Self { 
            use_fused_qkv: hidden <= 512,
            use_scratch_buffers: true,
        }
    }
}