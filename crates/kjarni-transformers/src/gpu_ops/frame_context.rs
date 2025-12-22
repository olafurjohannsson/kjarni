use crate::gpu_ops::GpuTensorPool;
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
        // 1. If we are already panicking (e.g. from unwinding), DO NOT panic again.
        if std::thread::panicking() {
            return;
        }

        // 2. If we haven't submitted, but the encoder is still there, it means
        // we are returning early (likely due to an Err result).
        // Panicking here hides the actual error!
        if !self.submitted && self.encoder.is_some() {
            // OPTION A: Log a warning instead of panicking
            log::warn!("GpuFrameContext dropped without submission. Work discarded.");

            // OPTION B: Just silently let it drop. The command encoder will be destroyed.
            // This is actually safer for error handling patterns using `?`.
        }
    }
}
