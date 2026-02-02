use crate::gpu::GpuTensor;
use crate::WgpuContext;
use std::sync::Arc;

/// A GPU kernel for applying the Tanh activation function element-wise.
pub struct GpuTanh {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    context: Arc<WgpuContext>,
}
impl GpuTanh {
    /// Creates a new `GpuTanh` kernel.
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let device = &context.device;
        let shader = device.create_shader_module(wgpu::include_wgsl!("tanh.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tanh Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, // In-place
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tanh Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tanh Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            context: context.clone(),
        }
    }

    /// Encodes the Tanh operation. This is an in-place operation.
    ///
    /// # Arguments
    /// * `encoder`: The command encoder to record the compute pass.
    /// * `tensor`: The tensor to apply the Tanh function to. Its contents will be modified.
    pub fn encode_inplace(&self, encoder: &mut wgpu::CommandEncoder, tensor: &GpuTensor) {
        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tanh Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor.buffer().as_entire_binding(),
                }],
            });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tanh Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let num_elements = tensor.num_elements() as u32;
        let workgroup_size = 256;
        let workgroups = (num_elements + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
}
