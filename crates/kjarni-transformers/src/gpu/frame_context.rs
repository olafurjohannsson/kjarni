use crate::WgpuContext;
use crate::gpu::GpuTensorPool;
use std::sync::Arc;
use tokio::sync::MutexGuard;
use wgpu::CommandEncoder;

/// A command encoder that keeps one compute pass open across dispatches.
///
/// wgpu records every compute pass as a command buffer of its own, with its
/// resource transitions worked out when it ends. On an RTX A2000 that costs
/// about 10 us of GPU time and several of CPU per pass, more than most decode
/// kernels take. Dispatches recorded through [`pass`](Self::pass) share one
/// pass; wgpu still places a barrier between two dispatches that depend on
/// each other, so they run in order.
///
/// It dereferences to the `wgpu::CommandEncoder`, for copies and passes of
/// their own. Doing so first ends the shared pass, so commands keep the order
/// they were recorded in.
pub struct GpuEncoder {
    encoder: CommandEncoder,
    pass: Option<wgpu::ComputePass<'static>>,
}

impl GpuEncoder {
    pub fn new(device: &wgpu::Device, label: &str) -> Self {
        Self::from_encoder(
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) }),
        )
    }

    pub fn from_encoder(encoder: CommandEncoder) -> Self {
        Self {
            encoder,
            pass: None,
        }
    }

    /// The shared compute pass, begun if none is open.
    pub fn pass(&mut self) -> &mut wgpu::ComputePass<'static> {
        if self.pass.is_none() {
            let pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("shared compute pass"),
                    timestamp_writes: None,
                })
                .forget_lifetime();
            self.pass = Some(pass);
        }
        self.pass.as_mut().expect("just opened")
    }

    /// The wgpu encoder, after ending the shared pass.
    pub fn raw(&mut self) -> &mut CommandEncoder {
        self.pass = None;
        &mut self.encoder
    }

    pub fn finish(mut self) -> wgpu::CommandBuffer {
        self.pass = None;
        self.encoder.finish()
    }
}

impl std::ops::Deref for GpuEncoder {
    type Target = CommandEncoder;

    fn deref(&self) -> &CommandEncoder {
        &self.encoder
    }
}

impl std::ops::DerefMut for GpuEncoder {
    fn deref_mut(&mut self) -> &mut CommandEncoder {
        self.raw()
    }
}

/// Layers recorded between submissions in a decoder's forward pass.
pub const LAYERS_PER_SUBMIT: usize = 4;

/// Submits what `encoder` has recorded and leaves it a fresh encoder, so the
/// GPU starts on those commands while the CPU records the rest. The queue runs
/// submissions in order, so this changes nothing but when the work starts.
pub fn submit_recorded(context: &WgpuContext, encoder: &mut GpuEncoder) {
    let fresh = GpuEncoder::new(&context.device, "GpuFrameContext Encoder");
    let recorded = std::mem::replace(encoder, fresh);
    context.queue.submit(Some(recorded.finish()));
}

/// After layer `layer` of a pass ending at `end_layer`: submits every
/// LAYERS_PER_SUBMIT layers, except after the last, which the frame submits.
pub fn submit_between_layers(
    context: &WgpuContext,
    encoder: &mut GpuEncoder,
    layer: usize,
    end_layer: usize,
) {
    if submits_after(layer, end_layer) {
        submit_recorded(context, encoder);
    }
}

fn submits_after(layer: usize, end_layer: usize) -> bool {
    (layer + 1).is_multiple_of(LAYERS_PER_SUBMIT) && layer + 1 < end_layer
}

/// A guard that manages the resources for a single frame of GPU work.
pub struct GpuFrameContext<'a> {
    pub encoder: Option<GpuEncoder>,
    pub pool_guard: MutexGuard<'a, GpuTensorPool>,
    pub context: &'a Arc<WgpuContext>,
    submitted: bool,
}

impl<'a> GpuFrameContext<'a> {
    pub fn new(context: &'a Arc<WgpuContext>, pool_guard: MutexGuard<'a, GpuTensorPool>) -> Self {
        let encoder = GpuEncoder::new(&context.device, "GpuFrameContext Encoder");
        Self {
            encoder: Some(encoder),
            pool_guard,
            context,
            submitted: false,
        }
    }

    pub fn resources(&mut self) -> (&mut GpuEncoder, &mut GpuTensorPool) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_are_submitted_in_fours_and_the_last_is_left_to_the_frame() {
        let after: Vec<usize> = (0..12).filter(|&l| submits_after(l, 12)).collect();
        assert_eq!(after, vec![3, 7]);
        let after: Vec<usize> = (0..10).filter(|&l| submits_after(l, 10)).collect();
        assert_eq!(after, vec![3, 7]);
        // A pass over a later range of layers counts from the model's first.
        let after: Vec<usize> = (5..16).filter(|&l| submits_after(l, 16)).collect();
        assert_eq!(after, vec![7, 11]);
        assert!((0..3).all(|l| !submits_after(l, 3)));
    }
}
