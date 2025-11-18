use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensorPool; // Adjust path
use std::sync::Arc;
use tokio::sync::MutexGuard; // Use the async mutex guard

/// A guard that manages the resources for a single frame of GPU work.
pub struct GpuFrameContext<'a> {
    encoder: Option<wgpu::CommandEncoder>,
    pool_guard: MutexGuard<'a, GpuTensorPool>,
    context: &'a Arc<WgpuContext>,
    submitted: bool,
}

impl<'a> GpuFrameContext<'a> {
    pub fn new(context: &'a Arc<WgpuContext>, pool_guard: MutexGuard<'a, GpuTensorPool>) -> Self {
        let encoder = context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GpuFrameContext Encoder"),
        });
        Self {
            encoder: Some(encoder),
            pool_guard,
            context,
            submitted: false,
        }
    }

    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        self.encoder.as_mut().expect("Encoder has already been submitted.")
    }

    pub fn pool(&mut self) -> &mut GpuTensorPool {
        &mut *self.pool_guard
    }

    pub fn finish(mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.context.queue.submit(Some(encoder.finish()));
            self.pool_guard.next_frame();
            self.submitted = true;
        }
    }
}

impl Drop for GpuFrameContext<'_> {
    fn drop(&mut self) {
        if !self.submitted && !std::thread::panicking() {
            panic!("GpuFrameContext was dropped without calling .finish(). GPU work was not submitted and the memory pool was not advanced.");
        }
    }
}