use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TracingAllocator;

static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TracingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            let prev = ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            let current = prev + layout.size();
            
            // Track Peak
            let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
            while current > peak {
                match PEAK_BYTES.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }

            // Optional: Log allocations larger than 50MB
            if layout.size() > 50 * 1024 * 1024 {
                log::trace!("Allocating {:.2} MB (Total: {:.2} MB)", 
                    layout.size() as f64 / 1_048_576.0,
                    current as f64 / 1_048_576.0
                );
            }
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

pub fn get_current_ram_usage_mb() -> f64 {
    ALLOCATED_BYTES.load(Ordering::Relaxed) as f64 / 1_048_576.0
}

pub fn get_peak_ram_usage_mb() -> f64 {
    PEAK_BYTES.load(Ordering::Relaxed) as f64 / 1_048_576.0
}