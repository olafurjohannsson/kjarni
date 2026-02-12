use std::sync::atomic::{AtomicU64, Ordering};

/// Manages a large persistent buffer for uniforms to avoid per-frame allocation.
pub struct GpuUniformBuffer {
    buffer: wgpu::Buffer,
    capacity: u64,
    cursor: AtomicU64,
    alignment: u64,
    label: String,
}
impl GpuUniformBuffer {
    pub fn new(device: &wgpu::Device, size: u64, label: &str) -> Self {
        let limits = device.limits();
        let alignment = limits.min_uniform_buffer_offset_alignment as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            capacity: size,
            cursor: AtomicU64::new(0),
            alignment,
            label: label.to_string(),
        }
    }

    /// Resets the cursor. Call this at the start of a frame (before encoding).
    pub fn reset(&self) {
        self.cursor.store(0, Ordering::Relaxed);
    }

    /// Allocates space for data and writes to it. Returns the offset to use.
    pub fn write<T: bytemuck::Pod>(&self, queue: &wgpu::Queue, data: &T) -> u32 {
        let size = std::mem::size_of::<T>() as u64;
        let current = self.cursor.load(Ordering::Relaxed);
        let aligned_start = (current + self.alignment - 1) / self.alignment * self.alignment;
        let allocation_size = (size + self.alignment - 1) / self.alignment * self.alignment;
        let offset = self.cursor.fetch_add(allocation_size, Ordering::Relaxed);

        if offset + size > self.capacity {
            panic!(
                "GpuUniformBuffer '{}' overflow! Capacity: {}, Requested: {}",
                self.label,
                self.capacity,
                offset + size
            );
        }

        queue.write_buffer(&self.buffer, offset, bytemuck::bytes_of(data));
        offset as u32
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}
