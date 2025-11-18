use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensorPool; // Adjust path
use std::sync::Arc;
use tokio::sync::MutexGuard; // Use the async mutex guard
use wgpu::CommandEncoder;

/// A guard that manages the resources for a single frame of GPU work.
pub struct GpuFrameContext<'a> {
    pub encoder: CommandEncoder,
    pub pool_guard: MutexGuard<'a, GpuTensorPool>,
    context: &'a Arc<WgpuContext>,
    submitted: bool,
}

impl<'a> GpuFrameContext<'a> {
    pub fn new(context: &'a Arc<WgpuContext>, pool_guard: MutexGuard<'a, GpuTensorPool>) -> Self {
        let encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuFrameContext Encoder"),
            });
        Self {
            encoder: encoder,
            pool_guard,
            context,
            submitted: false,
        }
    }
    
    pub fn resources(&mut self) -> (&mut CommandEncoder, &mut GpuTensorPool) {
        (&mut self.encoder, &mut *self.pool_guard)
    }

    pub fn finish(mut self) {
        // To prevent the Drop panic, we need to manually take the encoder
        let encoder = std::mem::replace(
            &mut self.encoder,
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Dummy Encoder"),
                }),
        );

        self.context.queue.submit(Some(encoder.finish()));
        self.pool_guard.next_frame();
        self.submitted = true;
    }
}

impl Drop for GpuFrameContext<'_> {
    fn drop(&mut self) {
        if !self.submitted && !std::thread::panicking() {
            panic!(
                "GpuFrameContext was dropped without calling .finish(). GPU work was not submitted and the memory pool was not advanced."
            );
        }
    }
}
