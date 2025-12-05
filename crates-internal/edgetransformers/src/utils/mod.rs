//! Utility modules

pub mod linear_algebra;
pub mod masks;

pub use masks::*;

pub fn configure_threading() {
    // 1. Detect Hardware
    let physical_cores = num_cpus::get_physical();
    let logical_cores = num_cpus::get();
    
    // 2. Determine Optimal Count
    // If physical < logical, we have HyperThreading.
    // For AI (Matmul), HyperThreading is bad (cache contention).
    // We want 1 thread per Physical Core.
    let num_threads = if physical_cores > 0 && physical_cores < logical_cores {
        physical_cores
    } else {
        // Fallback or systems without HT (like Apple Silicon)
        logical_cores 
    };

    // 3. Initialize Rayon Global Pool
    // build_global() returns an error if Rayon is already initialized.
    // We ignore that error, respecting the host app's config if it exists.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global();
        
    log::info!(
        "Threading Configured: Physical Cores={}, Logical Cores={}, Threads Used={}", 
        physical_cores, logical_cores, num_threads
    );
}