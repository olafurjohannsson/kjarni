//! Picks the chat model the installed daemon runs, from what this machine
//! has: a laptop with 8 GB and a desktop with a free 12 GB GPU should not
//! get the same model.

const GB: u64 = 1024 * 1024 * 1024;

/// What the choice is made from. `None` where it could not be found out.
#[derive(Debug, Clone, Copy, Default)]
pub struct Machine {
    pub ram: Option<u64>,
    pub free_vram: Option<u64>,
}

impl Machine {
    pub fn this() -> Self {
        Self {
            ram: total_ram(),
            free_vram: crate::commands::util::free_vram_bytes(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Pick {
    pub model: &'static str,
    pub gpu: bool,
    /// Why, in a few words, for the user.
    pub because: String,
}

fn gb(bytes: u64) -> String {
    format!("{:.0} GB", bytes as f64 / GB as f64)
}

/// The model for `machine`. `gpu` and `cpu` are the user's `--gpu` and
/// `--cpu`; with neither, a GPU is used when one has room.
///
/// The sizes are the Q4_K_M weights plus room for the KV cache: about
/// 6 GB for qwen3.5-9b, 3 GB for qwen3.5-4b, 1.5 GB for qwen3.5-2b. On the
/// CPU the model also shares the machine's memory with everything else, so
/// it gets a smaller share.
pub fn pick(machine: Machine, gpu: bool, cpu: bool) -> Pick {
    let vram = machine.free_vram.unwrap_or(0);
    if !cpu && (gpu || vram >= 4 * GB) {
        let model = match vram {
            v if v >= 10 * GB => "qwen3.5-9b",
            v if v >= 4 * GB => "qwen3.5-4b",
            // --gpu with too little free, or a GPU whose memory is unknown.
            v if v >= 2 * GB || machine.free_vram.is_none() => "qwen3.5-2b",
            _ => "qwen3.5-0.8b",
        };
        let because = match machine.free_vram {
            Some(v) => format!("{} of GPU memory free", gb(v)),
            None => "--gpu, with GPU memory unknown".into(),
        };
        return Pick {
            model,
            gpu: true,
            because,
        };
    }
    let (model, because) = match machine.ram {
        Some(r) if r >= 16 * GB => ("qwen3.5-4b", format!("{} of memory", gb(r))),
        Some(r) if r >= 8 * GB => ("qwen3.5-2b", format!("{} of memory", gb(r))),
        Some(r) => ("qwen3.5-0.8b", format!("{} of memory", gb(r))),
        None => ("qwen3.5-2b", "memory size unknown".into()),
    };
    Pick {
        model,
        gpu: false,
        because: if machine.free_vram.is_some() && !cpu {
            format!("{because}, too little free on the GPU")
        } else {
            because
        },
    }
}

/// The machine's physical memory in bytes.
#[cfg(target_os = "linux")]
fn total_ram() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

#[cfg(target_os = "macos")]
fn total_ram() -> Option<u64> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

#[cfg(windows)]
fn total_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: the struct is plain data, sized as the call requires.
    unsafe {
        let mut status: MEMORYSTATUSEX = std::mem::zeroed();
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        (GlobalMemoryStatusEx(&mut status) != 0).then_some(status.ullTotalPhys)
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn total_ram() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine(ram: u64, free_vram: Option<u64>) -> Machine {
        Machine {
            ram: Some(ram * GB),
            free_vram: free_vram.map(|v| v * GB),
        }
    }

    #[test]
    fn a_free_gpu_gets_the_largest_model_that_fits() {
        let p = pick(machine(32, Some(12)), false, false);
        assert_eq!((p.model, p.gpu), ("qwen3.5-9b", true));
        assert_eq!(p.because, "12 GB of GPU memory free");
        let p = pick(machine(32, Some(5)), false, false);
        assert_eq!((p.model, p.gpu), ("qwen3.5-4b", true));
    }

    #[test]
    fn a_busy_gpu_leaves_the_model_on_the_cpu() {
        let p = pick(machine(32, Some(1)), false, false);
        assert_eq!((p.model, p.gpu), ("qwen3.5-4b", false));
        assert_eq!(p.because, "32 GB of memory, too little free on the GPU");
    }

    #[test]
    fn without_a_gpu_memory_decides() {
        assert_eq!(pick(machine(16, None), false, false).model, "qwen3.5-4b");
        assert_eq!(pick(machine(8, None), false, false).model, "qwen3.5-2b");
        assert_eq!(pick(machine(4, None), false, false).model, "qwen3.5-0.8b");
        let unknown = pick(Machine::default(), false, false);
        assert_eq!((unknown.model, unknown.gpu), ("qwen3.5-2b", false));
    }

    #[test]
    fn the_users_switches_win() {
        let p = pick(machine(32, Some(12)), false, true);
        assert_eq!((p.model, p.gpu), ("qwen3.5-4b", false));
        assert_eq!(p.because, "32 GB of memory");
        // --gpu where nvidia-smi can't tell, such as an AMD card.
        let p = pick(machine(32, None), true, false);
        assert_eq!((p.model, p.gpu), ("qwen3.5-2b", true));
    }
}
