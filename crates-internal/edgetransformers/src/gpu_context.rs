// gpu_context.rs
use anyhow::{Result, anyhow};
use std::sync::atomic::{AtomicUsize, Ordering};
use wgpu::{
    Adapter, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits, PowerPreference,
    RequestAdapterOptions,
};

#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub max_buffer_size: u64,
    pub max_texture_dimension_2d: u32,
    pub max_storage_buffer_binding_size: u32,
    pub available_memory: Option<u64>, // Estimated available memory
    pub reserved_for_kv_cache: u64,
}

pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub memory_info: GpuMemoryInfo,
    allocated_memory: AtomicUsize,
}

impl WgpuContext {
    pub async fn new() -> Result<Self> {
        Self::with_config(GpuConfig::default()).await
    }

    pub async fn with_config(config: GpuConfig) -> Result<Self> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;

        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();

        log::info!(
            "GPU: {} ({})",
            adapter_info.name,
            adapter_info.backend.to_str()
        );

        // Calculate memory requirements
        let memory_info = Self::calculate_memory_info(&adapter, &config)?;

        // Request device with optimal limits
        let required_limits = Limits {
            max_buffer_size: memory_info.max_buffer_size,
            max_storage_buffer_binding_size: memory_info.max_storage_buffer_binding_size,
            max_texture_dimension_2d: memory_info.max_texture_dimension_2d,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroup_size_y: 1024,
            max_compute_invocations_per_workgroup: 1024,
            ..adapter_limits.clone()
        };

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("TransformerGPU"),
                required_features: Features::TIMESTAMP_QUERY
                    | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits,
                ..Default::default()
            })
            .await?;

        log::info!(
            "Device initialized: max_buffer={:.0}MB, max_binding={:.0}MB",
            memory_info.max_buffer_size as f32 / 1_048_576.0,
            memory_info.max_storage_buffer_binding_size as f32 / 1_048_576.0
        );

        Ok(Self {
            device,
            queue,
            memory_info,
            allocated_memory: AtomicUsize::new(0),
        })
    }

fn calculate_memory_info(adapter: &Adapter, config: &GpuConfig) -> Result<GpuMemoryInfo> {
    let limits = adapter.limits();
    let available_memory = Self::query_gpu_memory(adapter);
    let kv_cache_reservation = config.kv_cache_memory_mb * 1_048_576;

    // --- START CORRECTION ---
    // The max_buffer_size should reflect the hardware's actual capability.
    // Do not reduce it based on an application-level budget.
    let max_buffer_size = limits.max_buffer_size;
    // --- END CORRECTION ---

    Ok(GpuMemoryInfo {
        max_buffer_size,
        max_texture_dimension_2d: limits.max_texture_dimension_2d,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        available_memory,
        reserved_for_kv_cache: kv_cache_reservation,
    })
}

    #[cfg(target_os = "windows")]
    fn query_gpu_memory(adapter: &Adapter) -> Option<u64> {
        use windows::Win32::Graphics::Dxgi::*;
        use windows::core::Interface;

        unsafe {
            // Try to get DXGI adapter
            let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;

            let adapter_info = adapter.get_info();

            // Enumerate adapters to find matching one
            for i in 0.. {
                let dxgi_adapter = factory.EnumAdapters1(i).ok()?;
                let desc = dxgi_adapter.GetDesc1().ok()?;

                // Match by LUID if possible (this is a simplified check)
                // In production, you'd want to match more precisely

                // Query memory
                let mut video_memory_info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                if dxgi_adapter
                    .QueryVideoMemoryInfo(
                        0, // Node index
                        DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                        &mut video_memory_info,
                    )
                    .is_ok()
                {
                    return Some(video_memory_info.Budget);
                }
            }
        }

        None
    }

    #[cfg(target_os = "linux")]
    fn query_gpu_memory(adapter: &Adapter) -> Option<u64> {
        // Try NVIDIA first
        #[cfg(feature = "nvidia")]
        {
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                // Get first device for simplicity - in production, match properly
                if let Ok(device) = nvml.device_by_index(0) {
                    if let Ok(memory_info) = device.memory_info() {
                        return Some(memory_info.free);
                    }
                }
            }
        }

        // Try AMD ROCm sysfs
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("card"))
                    .unwrap_or(false)
                {
                    let mem_info_path = path.join("device/mem_info_vram_total");
                    if let Ok(contents) = std::fs::read_to_string(&mem_info_path) {
                        if let Ok(bytes) = contents.trim().parse::<u64>() {
                            return Some(bytes);
                        }
                    }
                }
            }
        }

        // Fallback: parse nvidia-smi if available
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(&["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            .output()
        {
            if let Ok(text) = String::from_utf8(output.stdout) {
                if let Ok(mb) = text.trim().parse::<u64>() {
                    return Some(mb * 1_048_576);
                }
            }
        }

        None
    }

    #[cfg(target_os = "macos")]
    fn query_gpu_memory(_adapter: &Adapter) -> Option<u64> {
        // On macOS with Apple Silicon, we have unified memory
        // Query total system memory and use a portion
        use sysinfo::{System, SystemExt};

        let mut sys = System::new_all();
        sys.refresh_memory();

        // On M1/M2/M3, assume we can use up to 75% of total RAM for GPU
        Some((sys.total_memory() * 1024 * 3) / 4)
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    fn query_gpu_memory(_adapter: &Adapter) -> Option<u64> {
        None
    }

    pub fn track_allocation(&self, bytes: usize) {
        self.allocated_memory.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn track_deallocation(&self, bytes: usize) {
        self.allocated_memory.fetch_sub(bytes, Ordering::Relaxed);
    }

    pub fn get_allocated_memory(&self) -> usize {
        self.allocated_memory.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub kv_cache_memory_mb: u64,
    pub prefer_texture_embeddings: bool,
    pub min_batch_size_for_gpu: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            kv_cache_memory_mb: 2048, // Reserve 2GB for KV cache
            prefer_texture_embeddings: true,
            min_batch_size_for_gpu: 128,
        }
    }
}
