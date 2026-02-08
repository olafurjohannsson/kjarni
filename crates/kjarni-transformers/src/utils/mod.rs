//! Utility modules

pub mod levenshtein;
pub mod linear_algebra;
pub mod masks;
pub mod alloc_stats;
pub mod tensor_ops;
pub use levenshtein::{distance, find_similar, find_within_distance, similarity};
pub use masks::*;

#[cfg(test)]
mod tests;

#[cfg(target_os = "linux")]
fn set_thread_affinity(num_cores: usize) {

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

    let (num_threads, is_hybrid) = if is_intel_hybrid() {
        (get_p_core_count().unwrap_or(physical_cores / 2), true)
    } else if physical_cores < logical_cores {
        (physical_cores, false) // This hits for your Xeon (sets 6 threads)
    } else {
        (logical_cores, false)
    };

    #[cfg(target_os = "linux")]
    set_thread_affinity(num_threads);

    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .start_handler(move |thread_index| {
            #[cfg(target_os = "linux")]
            unsafe {
                let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_SET(thread_index, &mut cpuset);
                libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset);
            }
        })
        .build_global();

    log::info!("Threading: {} threads, hybrid={}", num_threads, is_hybrid);
}

fn is_intel_hybrid() -> bool {
    // Check for Intel 12th gen+ hybrid architecture
    #[cfg(target_os = "linux")]
    {
        //Check core_type sysfs (kernel 5.18+)
        if std::path::Path::new("/sys/devices/system/cpu/cpu0/topology/core_type").exists() {
            return true; // Hybrid system
        }

        //  Check CPU model name
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            let model = cpuinfo
                .lines()
                .find(|l| l.starts_with("model name"))
                .unwrap_or("");

            // Intel 12th, 13th, 14th gen hybrid
            if model.contains("12th Gen Intel")
                || model.contains("13th Gen Intel")
                || model.contains("14th Gen Intel")
                || model.contains("Core Ultra")
            {
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
            let model = cpuinfo
                .lines()
                .find(|l| l.starts_with("model name"))
                .unwrap_or("");

            // Common hybrid configs (P-cores only, no HT count)
            if model.contains("i5-12") || model.contains("i5-13") {
                return Some(6);
            }
            if model.contains("i7-12") || model.contains("i7-13") {
                return Some(8);
            }
            if model.contains("i9-12") || model.contains("i9-13") {
                return Some(8);
            }
        }
    }
    None
}
