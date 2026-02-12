//! Generation statistics

use log::info;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

static STATS_ENABLED: AtomicBool = AtomicBool::new(false);

/// Tracks generation performance metrics.
#[derive(Debug)]
pub struct GenerationStats {
    prefill_start: Option<Instant>,
    prefill_tokens: usize,
    prefill_duration: Duration,
    decode_start: Option<Instant>,
    decode_tokens: usize,
    last_token_time: Option<Instant>,
}

impl GenerationStats {
    /// Creates a new stats tracker.
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

    /// Enables statistics collection globally.
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

    /// Marks the start of the prefill phase.
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

    /// Records a generated token. Call after each token is sampled.
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
    pub fn decode_tps(&self) -> f64 {
        if let (Some(start), Some(last)) = (self.decode_start, self.last_token_time) {
            let secs = last.duration_since(start).as_secs_f64();
            if secs > 0.0 && self.decode_tokens > 1 {
                return (self.decode_tokens - 1) as f64 / secs;
            }
        }
        0.0
    }

    /// Returns the total number of tokens generated.
    pub fn tokens_generated(&self) -> usize {
        self.decode_tokens
    }

    
    /// Prints a summary of generation statistics.
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

    #[test]
    fn test_prefill_tps() {
        GenerationStats::enable();
        let mut stats = GenerationStats::new();

        stats.start_prefill(100);
        sleep(Duration::from_millis(100));
        stats.end_prefill();

        let tps = stats.prefill_tps();
        assert!(tps > 500.0 && tps < 2000.0, "tps was {}", tps);

        GenerationStats::disable();
    }
}