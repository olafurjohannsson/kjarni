// kjarni-transformers/src/stats.rs

//! Generation statistics tracking for performance monitoring.
//!
//! Provides low-overhead metrics collection for text generation, including:
//! - Prefill throughput (tokens/second)
//! - Decode throughput (tokens/second)
//! - Per-phase timing breakdowns
//!
//! # Usage
//!
//! Statistics collection is disabled by default. Enable it globally:
//!
//! ```ignore
//! GenerationStats::enable();
//! ```
//!
//! Or via CLI flag:
//!
//! ```bash
//! kjarni chat --stats "Hello, world"
//! ```
//!
//! # Output Example
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │       Generation Statistics         │
//! ├─────────────────────────────────────┤
//! │ Prefill:     42 tokens @   850.3 t/s│
//! │ Decode:      58 tokens @    23.4 t/s│
//! │ Total:      100 tokens              │
//! └─────────────────────────────────────┘
//! ```

use log::info;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Global flag to enable/disable statistics collection.
///
/// When disabled (default), all stats methods are no-ops with minimal overhead.
static STATS_ENABLED: AtomicBool = AtomicBool::new(false);

/// Tracks generation performance metrics.
///
/// This struct is designed for low overhead in the hot path:
/// - No allocations after construction
/// - Atomic operations only for the global enable flag
/// - All timing uses `Instant` (monotonic, fast)
///
/// # Example
///
/// ```ignore
/// let mut stats = GenerationStats::new();
///
/// stats.start_prefill(prompt_tokens.len());
/// // ... run prefill ...
/// stats.end_prefill();
///
/// for token in generated_tokens {
///     // ... generate token ...
///     stats.record_token();
/// }
///
/// stats.print_summary();
/// ```
#[derive(Debug)]
pub struct GenerationStats {
    /// When prefill started
    prefill_start: Option<Instant>,
    /// Number of tokens in the prompt
    prefill_tokens: usize,
    /// Total prefill duration
    prefill_duration: Duration,

    /// When decode phase started (first token)
    decode_start: Option<Instant>,
    /// Number of tokens generated
    decode_tokens: usize,
    /// Time of the most recent token
    last_token_time: Option<Instant>,
}

impl GenerationStats {
    /// Creates a new stats tracker.
    ///
    /// Collection is only active if `GenerationStats::is_enabled()` returns true.
    pub fn new() -> Self {
        Self {
            prefill_start: None,
            prefill_tokens: 0,
            prefill_duration: Duration::ZERO,
            decode_start: None,
            decode_tokens: 0,
            last_token_time: None,
        }
    }

    // =========================================================================
    // Global Enable/Disable
    // =========================================================================

    /// Enables statistics collection globally.
    ///
    /// Call this once at startup (e.g., based on CLI flags).
    pub fn enable() {
        STATS_ENABLED.store(true, Ordering::Relaxed);
    }

    /// Disables statistics collection globally.
    pub fn disable() {
        STATS_ENABLED.store(false, Ordering::Relaxed);
    }

    /// Returns whether statistics collection is enabled.
    #[inline]
    pub fn is_enabled() -> bool {
        STATS_ENABLED.load(Ordering::Relaxed)
    }

    // =========================================================================
    // Prefill Phase
    // =========================================================================

    /// Marks the start of the prefill phase.
    ///
    /// # Arguments
    ///
    /// * `num_tokens` - Number of prompt tokens being processed
    #[inline]
    pub fn start_prefill(&mut self, num_tokens: usize) {
        if !Self::is_enabled() {
            return;
        }
        self.prefill_start = Some(Instant::now());
        self.prefill_tokens = num_tokens;
    }

    /// Marks the end of the prefill phase.
    #[inline]
    pub fn end_prefill(&mut self) {
        if let Some(start) = self.prefill_start.take() {
            self.prefill_duration = start.elapsed();
        }
    }

    /// Returns prefill throughput in tokens per second.
    pub fn prefill_tps(&self) -> f64 {
        let secs = self.prefill_duration.as_secs_f64();
        if secs > 0.0 {
            self.prefill_tokens as f64 / secs
        } else {
            0.0
        }
    }

    // =========================================================================
    // Decode Phase
    // =========================================================================

    /// Records a generated token.
    ///
    /// Call this after each token is sampled and yielded.
    #[inline]
    pub fn record_token(&mut self) {
        if !Self::is_enabled() {
            return;
        }

        let now = Instant::now();
        if self.decode_start.is_none() {
            self.decode_start = Some(now);
        }
        self.decode_tokens += 1;
        self.last_token_time = Some(now);
    }

    /// Returns decode throughput in tokens per second.
    ///
    /// This is the "generation speed" users typically care about.
    pub fn decode_tps(&self) -> f64 {
        if let (Some(start), Some(last)) = (self.decode_start, self.last_token_time) {
            let secs = last.duration_since(start).as_secs_f64();
            if secs > 0.0 && self.decode_tokens > 1 {
                // Exclude first token (it's part of prefill latency)
                return (self.decode_tokens - 1) as f64 / secs;
            }
        }
        0.0
    }

    /// Returns the total number of tokens generated.
    pub fn tokens_generated(&self) -> usize {
        self.decode_tokens
    }

    // =========================================================================
    // Reporting
    // =========================================================================

    /// Prints a summary of generation statistics.
    ///
    /// Only prints if statistics collection is enabled and tokens were generated.
    pub fn print_summary(&self) {
        if !Self::is_enabled() {
            return;
        }

        if self.prefill_tokens == 0 && self.decode_tokens == 0 {
            return;
        }

        let total_tokens = self.prefill_tokens + self.decode_tokens;

        info!("┌─────────────────────────────────────┐");
        info!("│       Generation Statistics         │");
        info!("├─────────────────────────────────────┤");
        info!(
            "│ Prefill: {:>5} tokens @ {:>7.1} t/s │",
            self.prefill_tokens,
            self.prefill_tps()
        );
        info!(
            "│ Decode:  {:>5} tokens @ {:>7.1} t/s │",
            self.decode_tokens,
            self.decode_tps()
        );
        info!("│ Total:   {:>5} tokens               │", total_tokens);
        info!("└─────────────────────────────────────┘");
    }

    /// Returns a one-line summary suitable for logging.
    pub fn summary_line(&self) -> String {
        format!(
            "prefill: {} tok @ {:.1} t/s, decode: {} tok @ {:.1} t/s",
            self.prefill_tokens,
            self.prefill_tps(),
            self.decode_tokens,
            self.decode_tps()
        )
    }
}

impl Default for GenerationStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    // #[test]
    // fn test_stats_disabled_by_default() {
    //     assert!(!GenerationStats::is_enabled());
    // }

    #[test]
    fn test_prefill_tps_calculation() {
        GenerationStats::enable();
        let mut stats = GenerationStats::new();

        stats.start_prefill(100);
        sleep(Duration::from_millis(100));
        stats.end_prefill();

        let tps = stats.prefill_tps();
        // Should be roughly 1000 t/s (100 tokens / 0.1s)
        assert!(tps > 500.0 && tps < 2000.0, "TPS was {}", tps);

        GenerationStats::disable();
    }
}