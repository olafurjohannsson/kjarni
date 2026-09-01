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
        // The old rule reused the scratch buffers at a single token and above
        // NOALLOC_WINS_TOKENS, and took the allocating path in between, which is
        // where most requests land. The buffered path also builds its buffers per
        // call, so the in-between case was worth measuring rather than assuming.
        //
        // MiniLM, mean of five runs, repeated. The rightmost column is how many
        // allocations each rule made, which is what proves the two took different
        // paths at all:
        //
        //     docs x tok    total    always   old rule   allocations
        //        1 x  18       18     12 ms      24 ms   475 / 1190
        //        4 x  18       72     25         57      1745 / 3685
        //       16 x  18      288     41         85      8519 / 13649
        //        1 x 216      216     70         93      3471 / 4187
        //        4 x 216      864    156        233      15434 / 15669
        //       64 x  18     1152    159        159      28779 / 28778
        //       16 x 216     3456    575        550      56455 / 56454
        //
        // Below the old threshold, reusing the buffers is 1.3x to 2.2x faster, for
        // paragraphs as well as single sentences. At and above it both rules
        // already agreed: the allocation counts are identical, the code is the
        // same, and the remaining spread is measurement noise, which reaches 12%
        // on the largest batches on this machine.
        static AUTO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let auto = *AUTO.get_or_init(|| std::env::var("KJARNI_SCRATCH").as_deref() == Ok("auto"));
        let use_scratch_buffers = if auto {
            tokens <= DECODE_THRESHOLD || tokens >= NOALLOC_WINS_TOKENS
        } else {
            true
        };

        Self {
            use_fused_qkv,
            use_scratch_buffers,
        }
    }

    /// Strategy optimized for decode (autoregressive generation)
    pub fn decode() -> Self {
        Self {
            use_fused_qkv: true,
            use_scratch_buffers: true,
        }
    }

    /// Strategy optimized for encoding large batches
    pub fn encode_batch(hidden: usize) -> Self {
        Self {
            use_fused_qkv: hidden <= 512,
            use_scratch_buffers: true,
        }
    }
}
