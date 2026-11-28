#[cfg(feature = "alloc-stats")]
use std::alloc::{GlobalAlloc, Layout, System};
#[cfg(feature = "alloc-stats")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "alloc-stats")]
pub struct TracingAllocator;

#[cfg(feature = "alloc-stats")]
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "alloc-stats")]
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

/// How many times the allocator has been called.
///
/// Bytes and peak answer "how much memory"; this answers "how often did we go to
/// the allocator", which is the number that matters when chasing per-call
/// overhead. A forward pass that allocates nine times per layer pays nine
/// synchronisation points and nine chances to miss cache, regardless of size.
#[cfg(feature = "alloc-stats")]
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "alloc-stats")]
unsafe impl GlobalAlloc for TracingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            let ret = System.alloc(layout);
            if !ret.is_null() {
                ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
                let prev = ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
                let current = prev + layout.size();

                // Track Peak
                let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
                while current > peak {
                    match PEAK_BYTES.compare_exchange_weak(
                        peak,
                        current,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }

                if layout.size() > 50 * 1024 * 1024 {
                    log::trace!(
                        "Allocating {:.2} MB (Total: {:.2} MB)",
                        layout.size() as f64 / 1_048_576.0,
                        current as f64 / 1_048_576.0
                    );
                }
            }
            ret
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            System.dealloc(ptr, layout);
            ALLOCATED_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        }
    }
}

#[cfg(feature = "alloc-stats")]
pub fn get_current_ram_usage_mb() -> f64 {
    ALLOCATED_BYTES.load(Ordering::Relaxed) as f64 / 1_048_576.0
}

#[cfg(not(feature = "alloc-stats"))]
pub fn get_current_ram_usage_mb() -> f64 {
    0.0
}

#[cfg(feature = "alloc-stats")]
pub fn get_peak_ram_usage_mb() -> f64 {
    PEAK_BYTES.load(Ordering::Relaxed) as f64 / 1_048_576.0
}

#[cfg(not(feature = "alloc-stats"))]
pub fn get_peak_ram_usage_mb() -> f64 {
    0.0
}

/// Total allocator calls since the process started, or since the last reset.
#[cfg(feature = "alloc-stats")]
pub fn alloc_count() -> usize {
    ALLOC_COUNT.load(Ordering::Relaxed)
}

/// Allocation statistics are disabled unless the `alloc-stats` feature is enabled.
#[cfg(not(feature = "alloc-stats"))]
pub fn alloc_count() -> usize {
    0
}

/// Zeroes the allocation counter so a region can be measured on its own.
///
/// The counter is global and counts every thread, which is what you want for a
/// rayon-parallel forward pass: work handed to another thread still allocates.
#[cfg(feature = "alloc-stats")]
pub fn reset_alloc_count() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
}

/// No-op when allocation statistics are disabled.
#[cfg(not(feature = "alloc-stats"))]
pub fn reset_alloc_count() {}

#[cfg(all(test, feature = "alloc-stats"))]
mod tests {
    use super::{alloc_count, get_current_ram_usage_mb, reset_alloc_count};

    #[test]
    fn enabled_feature_counts_allocations() {
        reset_alloc_count();
        let before_mb = get_current_ram_usage_mb();
        let mut bytes = Vec::with_capacity(64 * 1024);
        bytes.extend_from_slice(&[1_u8; 64 * 1024]);
        let after_mb = get_current_ram_usage_mb();
        assert!(
            after_mb > before_mb,
            "expected tracked RAM usage to grow while allocation is alive (before={before_mb}, after={after_mb})"
        );
        assert!(alloc_count() > 0);
        assert_eq!(std::hint::black_box(&bytes).len(), 64 * 1024);
    }
}

#[cfg(all(test, not(feature = "alloc-stats")))]
mod tests {
    use super::{alloc_count, get_current_ram_usage_mb, get_peak_ram_usage_mb, reset_alloc_count};

    #[test]
    fn disabled_feature_is_a_noop() {
        reset_alloc_count();
        let _bytes = Box::new([0_u8; 32]);
        assert_eq!(alloc_count(), 0);
        assert_eq!(get_current_ram_usage_mb(), 0.0);
        assert_eq!(get_peak_ram_usage_mb(), 0.0);
    }
}
