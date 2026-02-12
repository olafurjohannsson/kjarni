use crate::gpu::GpuTensorPool;
use crate::WgpuContext;
use std::sync::Arc;
use tokio::sync::MutexGuard;
use wgpu::CommandEncoder;

/// A guard that manages the resources for a single frame of GPU work.
pub struct GpuFrameContext<'a> {
    pub encoder: Option<CommandEncoder>,
    pub pool_guard: MutexGuard<'a, GpuTensorPool>,
    pub context: &'a Arc<WgpuContext>,
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
            encoder: Some(encoder),
            pool_guard,
            context,
            submitted: false,
        }
    }

    pub fn resources(&mut self) -> (&mut CommandEncoder, &mut GpuTensorPool) {
        (self.encoder.as_mut().unwrap(), &mut *self.pool_guard)
    }

    pub fn finish(mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.context.queue.submit(Some(encoder.finish()));
        }
        self.pool_guard.next_frame();
        self.submitted = true;
    }
}

impl Drop for GpuFrameContext<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        if !self.submitted && self.encoder.is_some() {
            log::warn!("GpuFrameContext dropped without submission. Work discarded.");
        }
    }
}
