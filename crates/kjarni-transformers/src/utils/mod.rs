//! Utility modules

pub mod linear_algebra;
pub mod masks;
pub mod levenshtein;
pub mod tensor_ops;
pub use masks::*;
pub use levenshtein::{find_similar, find_within_distance, distance, similarity};


#[cfg(target_os = "linux")]
fn set_thread_affinity(num_cores: usize) {
    use std::os::unix::thread::JoinHandleExt;

    // Use libc to set CPU affinity
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        for i in 0..num_cores {
            libc::CPU_SET(i, &mut cpuset);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
    }
}

pub fn configure_threading() {
    let physical_cores = num_cpus::get_physical();
    let logical_cores = num_cpus::get();

    let num_threads = if is_intel_hybrid() {
        let p_cores = get_p_core_count().unwrap_or(physical_cores / 2);

        // Pin main thread to P-cores
        #[cfg(target_os = "linux")]
        set_thread_affinity(p_cores);

        p_cores
    } else if physical_cores < logical_cores {
        physical_cores
    } else {
        logical_cores
    };

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global();

    log::info!("Threading: {} threads, hybrid={}", num_threads, is_intel_hybrid());
}

fn is_intel_hybrid() -> bool {
    // Check for Intel 12th gen+ hybrid architecture
    #[cfg(target_os = "linux")]
    {
        // Method 1: Check core_type sysfs (kernel 5.18+)
        if std::path::Path::new("/sys/devices/system/cpu/cpu0/topology/core_type").exists() {
            return true; // Hybrid system
        }

        // Method 2: Check CPU model name
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            let model = cpuinfo.lines()
                .find(|l| l.starts_with("model name"))
                .unwrap_or("");

            // Intel 12th, 13th, 14th gen are hybrid
            if model.contains("12th Gen Intel") ||
                model.contains("13th Gen Intel") ||
                model.contains("14th Gen Intel") ||
                model.contains("Core Ultra") {
                return true;
            }
        }
    }
    false
}

fn get_p_core_count() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        // Count cores where core_type == "Core" (P-core) vs "Atom" (E-core)
        let mut p_cores = 0;
        for i in 0..256 {
            let path = format!("/sys/devices/system/cpu/cpu{}/topology/core_type", i);
            match std::fs::read_to_string(&path) {
                Ok(core_type) => {
                    if core_type.trim() == "Core" {
                        p_cores += 1;
                    }
                }
                Err(_) => break,
            }
        }
        if p_cores > 0 {
            return Some(p_cores);
        }

        // Fallback: known configurations
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            let model = cpuinfo.lines()
                .find(|l| l.starts_with("model name"))
                .unwrap_or("");

            // Common hybrid configs (P-cores only, no HT count)
            if model.contains("i5-12") || model.contains("i5-13") { return Some(6); }
            if model.contains("i7-12") || model.contains("i7-13") { return Some(8); }
            if model.contains("i9-12") || model.contains("i9-13") { return Some(8); }
        }
    }
    None
}