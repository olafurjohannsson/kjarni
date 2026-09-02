// heuristics
pub mod thresholds {
    /// Below this token count, fused QKV always wins
    pub const FUSED_ALWAYS_WINS_TOKENS: usize = 64;

    /// Above this hidden size with large batches, separate wins
    pub const LARGE_HIDDEN_THRESHOLD: usize = 768;

    /// Threshold for switching from vec kernel to batched 4x3 kernel
    pub const BATCH_KERNEL_THRESHOLD: usize = 1000;
}

/// Compute strategy decisions for attention and matmul operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeStrategy {
    pub use_fused_qkv: bool,
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

        // There is no no-alloc decision any more. The buffered path was measured
        // against the allocating one on five encoders, both entry points, batches
        // of 1 to 64 and two document lengths: faster in 45 of 46 cases, from
        // 1.09x to 4.31x, and never slower. The remaining case was within noise.
        Self { use_fused_qkv }
    }

    /// Strategy optimized for decode (autoregressive generation)
    pub fn decode() -> Self {
        Self {
            use_fused_qkv: true,
        }
    }

    /// Strategy optimized for encoding large batches
    pub fn encode_batch(hidden: usize) -> Self {
        Self {
            use_fused_qkv: hidden <= 512,
        }
    }
}
