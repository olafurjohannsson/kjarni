use crate::gpu::tensor::GpuTensor;
use wgpu::CommandEncoder;

/// A trait for a reusable, pre-compiled GPU compute operation.
///
/// This represents a single, stateless GPU call (e.g., a matrix multiplication).
/// Kernels are composed by higher-level blocks (like `GpuLinear` or `GpuAttention`)
/// to build complex neural network layers.
pub trait Kernel: Send + Sync {
    /// Encodes the kernel's compute pass into the given command encoder.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder to record the compute pass.
    /// * `inputs` - A slice of input tensors required by the kernel.
    /// * `output` - The output tensor where the result will be written.
    fn encode(
        &self,
        encoder: &mut CommandEncoder,
        inputs: &[&GpuTensor],
        output: &GpuTensor,
    );
}