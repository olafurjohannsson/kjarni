use crate::gpu::tensor::GpuTensor;
use wgpu::CommandEncoder;

pub trait Kernel: Send + Sync {
    /// Encodes the kernel's compute pass into the given command encoder
    fn encode(
        &self,
        encoder: &mut CommandEncoder,
        inputs: &[&GpuTensor],
        output: &GpuTensor,
    );
}