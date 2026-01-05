use crate::gpu_ops::GpuTensorPool;
use anyhow::{Result};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use tokio::sync::Mutex;
use wgpu::{
    Adapter, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits, PowerPreference,
    RequestAdapterOptions,
};
use crate::gpu_ops::profiler::GpuProfiler;
use crate::gpu_ops::uniforms::GpuUniformBuffer;
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub max_buffer_size: u64,
    pub max_texture_dimension_2d: u32,
    pub max_storage_buffer_binding_size: u32,
    pub available_memory: Option<u64>, // Estimated available memory
    pub reserved_for_kv_cache: u64,
}

impl GpuMemoryInfo {
    /// Print a human-readable summary of GPU memory
    pub fn print_summary(&self) {
        println!("\n=== GPU Memory Info ===");

        if let Some(available) = self.available_memory {
            println!("Available VRAM:     {:.2} GB", available as f64 / 1_073_741_824.0);
        } else {
            println!("Available VRAM:     Unknown (could not query)");
        }

        println!("Max Buffer Size:    {:.2} GB", self.max_buffer_size as f64 / 1_073_741_824.0);
        println!("Max Binding Size:   {:.2} GB", self.max_storage_buffer_binding_size as f64 / 1_073_741_824.0);
        println!("KV Cache Reserved:  {:.2} GB", self.reserved_for_kv_cache as f64 / 1_073_741_824.0);
        println!("======================\n");
    }
}

pub struct WgpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter: Adapter,
    pub memory_info: GpuMemoryInfo,
    allocated_memory: AtomicUsize,

    // Lazy-initialized shared pool (best performance!)
    inference_pool: OnceLock<Arc<Mutex<GpuTensorPool>>>,
    pub profiler: GpuProfiler,

    pub uniform_arena: GpuUniformBuffer,

    // poll_stop: Arc<AtomicBool>,
    // poll_handle: Option<std::thread::JoinHandle<()>>,
}

impl WgpuContext {
    pub async fn new() -> Result<Arc<Self>> {
        Self::with_config(GpuConfig::default()).await
    }
    pub fn is_available() -> bool {
        true
    }
    pub async fn with_config(config: GpuConfig) -> Result<Arc<Self>> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });
        let required_features = Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;

        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();

        println!("\n=== GPU Adapter Info ===");
        println!("Name:     {}", adapter_info.name);
        println!("Backend:  {:?}", adapter_info.backend);
        println!("Vendor:   0x{:X}", adapter_info.vendor);
        println!("Device:   0x{:X}", adapter_info.device);
        println!("Type:     {:?}", adapter_info.device_type);
        println!("========================\n");

        log::info!(
            "GPU: {} ({})",
            adapter_info.name,
            adapter_info.backend.to_str()
        );

        // Calculate memory requirements
        let memory_info = Self::calculate_memory_info(&adapter, &config)?;

        // Print detailed memory info
        memory_info.print_summary();

        // Check if we have enough memory for the config
        if let Some(available) = memory_info.available_memory {
            let required_min = config.kv_cache_memory_mb * 1_048_576;
            if available < required_min {
                log::warn!(
                    "⚠️  Available VRAM ({:.2} GB) is less than KV cache reservation ({:.2} GB)",
                    available as f64 / 1_073_741_824.0,
                    required_min as f64 / 1_073_741_824.0
                );
                println!("⚠️  WARNING: You may run out of VRAM!");
            } else {
                log::info!(
                    "✓ Sufficient VRAM available ({:.2} GB >= {:.2} GB required)",
                    available as f64 / 1_073_741_824.0,
                    required_min as f64 / 1_073_741_824.0
                );
            }
        }

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
                required_features: required_features,
                required_limits,
                ..Default::default()
            })
            .await?;

        log::info!(
            "Device initialized: max_buffer={:.2} GB, max_binding={:.2} GB",
            memory_info.max_buffer_size as f32 / 1_073_741_824.0,
            memory_info.max_storage_buffer_binding_size as f32 / 1_073_741_824.0
        );
        let device_arc = Arc::new(device.clone());
        let queue_arc = Arc::new(queue);
        let profiler = GpuProfiler::new(&device, 4096);
        // Start polling thread
        // let stop_flag = Arc::new(AtomicBool::new(false));
        // let device_clone = device_arc.clone();
        // let thread_stop = stop_flag.clone();

        // let handle = std::thread::Builder::new()
        //     .name("wgpu-poller".into())
        //     .spawn(move || {
        //         log::debug!("WGPU polling thread started");
        //         while !thread_stop.load(Ordering::Relaxed) {
        //             device_clone.poll(wgpu::PollType::Poll);
        //             std::thread::sleep(Duration::from_micros(100));
        //         }
        //         log::debug!("WGPU polling thread stopped");
        //     })?;
        let uniform_arena = GpuUniformBuffer::new(&device, 1024 * 1024 * 4, "Global Uniforms"); // 4MB
        Ok(Arc::new(Self {
            device: device_arc,
            queue: queue_arc,
            adapter,
            memory_info,
            allocated_memory: AtomicUsize::new(0),
            inference_pool: OnceLock::new(),  // Empty initially
            profiler: profiler,
            uniform_arena,
            // poll_stop: stop_flag,
            // poll_handle: Some(handle),
        }))
    }
    pub fn get_inference_pool(self: &Arc<Self>) -> Arc<Mutex<GpuTensorPool>> {
        self.inference_pool
            .get_or_init(|| {
                log::debug!("Creating shared inference pool");
                Arc::new(Mutex::new(GpuTensorPool::new(self.clone())))
            })
            .clone()
    }

    fn calculate_memory_info(adapter: &Adapter, config: &GpuConfig) -> Result<GpuMemoryInfo> {
        let limits = adapter.limits();
        let available_memory = Self::query_gpu_memory(adapter);
        let kv_cache_reservation = config.kv_cache_memory_mb * 1_048_576;

        log::debug!("Adapter limits:");
        log::debug!("  max_buffer_size: {:.2} GB", limits.max_buffer_size as f64 / 1_073_741_824.0);
        log::debug!("  max_storage_buffer_binding_size: {:.2} GB",
                    limits.max_storage_buffer_binding_size as f64 / 1_073_741_824.0);

        if let Some(available) = available_memory {
            log::info!("Detected available VRAM: {:.2} GB", available as f64 / 1_073_741_824.0);
        } else {
            log::warn!("Could not query available VRAM - proceeding with adapter limits");
        }

        let max_buffer_size = limits.max_buffer_size;

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
            if let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() {
                // Enumerate adapters to find matching one
                for i in 0..16 {
                    if let Ok(dxgi_adapter) = factory.EnumAdapters1(i) {
                        if let Ok(desc) = dxgi_adapter.GetDesc1() {
                            log::debug!("Checking DXGI adapter {}: {:?}", i, desc.Description);
                        }

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
                            log::info!(
                                "DXGI Adapter {}: Budget={:.2} GB, CurrentUsage={:.2} GB, AvailableForReservation={:.2} GB",
                                i,
                                video_memory_info.Budget as f64 / 1_073_741_824.0,
                                video_memory_info.CurrentUsage as f64 / 1_073_741_824.0,
                                video_memory_info.AvailableForReservation as f64 / 1_073_741_824.0
                            );

                            // Return the budget (total available memory)
                            if video_memory_info.Budget > 0 {
                                return Some(video_memory_info.Budget);
                            }
                        }
                    } else {
                        break; // No more adapters
                    }
                }
            }
        }

        log::warn!("Failed to query GPU memory via DXGI");
        None
    }

    #[cfg(target_os = "linux")]
    fn query_gpu_memory(adapter: &Adapter) -> Option<u64> {
        // Try NVIDIA first
        #[cfg(feature = "nvidia")]
        {
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                if let Ok(device) = nvml.device_by_index(0) {
                    if let Ok(memory_info) = device.memory_info() {
                        log::info!(
                            "NVML: Total={:.2} GB, Free={:.2} GB, Used={:.2} GB",
                            memory_info.total as f64 / 1_073_741_824.0,
                            memory_info.free as f64 / 1_073_741_824.0,
                            memory_info.used as f64 / 1_073_741_824.0
                        );
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
                            log::info!("ROCm sysfs: Total VRAM={:.2} GB", bytes as f64 / 1_073_741_824.0);
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
                    let bytes = mb * 1_048_576;
                    log::info!("nvidia-smi: Free VRAM={:.2} GB", bytes as f64 / 1_073_741_824.0);
                    return Some(bytes);
                }
            }
        }

        log::warn!("Failed to query GPU memory on Linux");
        None
    }

    #[cfg(target_os = "macos")]
    fn query_gpu_memory(_adapter: &Adapter) -> Option<u64> {
        use sysinfo::{System, SystemExt};

        let mut sys = System::new_all();
        sys.refresh_memory();

        let total_memory = sys.total_memory() * 1024;
        let gpu_share = (total_memory * 3) / 4; // 75% of unified memory

        log::info!(
            "macOS unified memory: Total={:.2} GB, GPU share (75%)={:.2} GB",
            total_memory as f64 / 1_073_741_824.0,
            gpu_share as f64 / 1_073_741_824.0
        );

        Some(gpu_share)
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    fn query_gpu_memory(_adapter: &Adapter) -> Option<u64> {
        log::warn!("GPU memory query not implemented for this platform");
        None
    }

    pub fn track_allocation(&self, bytes: usize) {
        self.allocated_memory.fetch_add(bytes, Ordering::Relaxed);
        let total = self.allocated_memory.load(Ordering::Relaxed);
        log::debug!("Allocated {} bytes, total: {:.2} MB", bytes, total as f64 / 1_048_576.0);
    }

    pub fn track_deallocation(&self, bytes: usize) {
        self.allocated_memory.fetch_sub(bytes, Ordering::Relaxed);
        let total = self.allocated_memory.load(Ordering::Relaxed);
        log::debug!("Deallocated {} bytes, total: {:.2} MB", bytes, total as f64 / 1_048_576.0);
    }

    pub fn get_allocated_memory(&self) -> usize {
        self.allocated_memory.load(Ordering::Relaxed)
    }

    /// Print current memory usage
    pub fn print_memory_usage(&self) {
        let allocated = self.get_allocated_memory();
        println!("\n=== GPU Memory Usage ===");
        println!("Currently allocated: {:.2} MB ({:.2} GB)",
                 allocated as f64 / 1_048_576.0,
                 allocated as f64 / 1_073_741_824.0);

        if let Some(available) = self.memory_info.available_memory {
            let percent = (allocated as f64 / available as f64) * 100.0;
            println!("Usage: {:.1}% of available VRAM", percent);
        }
        println!("========================\n");
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
            kv_cache_memory_mb: 1024, // Reduced from 2048 to 1GB for 4GB cards
            prefer_texture_embeddings: true,
            min_batch_size_for_gpu: 128,
        }
    }
}

// impl Drop for WgpuContext {
//     fn drop(&mut self) {
//         log::debug!("Stopping WGPU polling thread");
//         self.poll_stop.store(true, Ordering::Relaxed);
//         if let Some(handle) = self.poll_handle.take() {
//             let _ = handle.join();
//         }
//     }
// }