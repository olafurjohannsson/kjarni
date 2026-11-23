use crate::gpu::tensor::GpuTensor;

pub trait Kernel: Send + Sync {
    /// Encodes the kernel's compute pass into the given command encoder
    fn encode(
        &self,
        encoder: &mut crate::gpu::GpuEncoder,
        inputs: &[&GpuTensor],
        output: &GpuTensor,
    );
}

/// A compute pipeline and its bind group layout: a uniform at binding 0, then
/// one storage buffer per entry of `read_only`, in order. The entry point is
/// `main`.
pub struct ComputeKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Names the dispatches in the GPU profiler.
    label: String,
}

impl ComputeKernel {
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        shader: wgpu::ShaderModuleDescriptor<'_>,
        read_only: &[bool],
    ) -> Self {
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for (i, read_only) in read_only.iter().enumerate() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32 + 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: *read_only,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(shader);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout,
            label: label.to_string(),
        }
    }

    /// One dispatch of `threads` threads in workgroups of `workgroup_size`,
    /// laid out by [`dispatch_1d`](crate::gpu_ops::utils::dispatch_1d), over
    /// `tensors` at bindings 1 and up.
    pub fn encode(
        &self,
        context: &crate::WgpuContext,
        encoder: &mut crate::gpu::GpuEncoder,
        uniforms: &wgpu::Buffer,
        tensors: &[&GpuTensor],
        threads: usize,
        workgroup_size: usize,
    ) {
        let workgroups = threads.div_ceil(workgroup_size) as u32;
        let (x, y) = crate::gpu_ops::utils::grid_1d(workgroups);
        self.encode_workgroups(context, encoder, uniforms, tensors, [x, y, 1]);
    }

    /// One dispatch of the `workgroups` grid over `tensors` at bindings 1 and up.
    pub fn encode_workgroups(
        &self,
        context: &crate::WgpuContext,
        encoder: &mut crate::gpu::GpuEncoder,
        uniforms: &wgpu::Buffer,
        tensors: &[&GpuTensor],
        workgroups: [u32; 3],
    ) {
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniforms.as_entire_binding(),
        }];
        for (i, tensor) in tensors.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: tensor.buffer().as_entire_binding(),
            });
        }
        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &entries,
            });
        let [x, y, z] = workgroups;
        crate::gpu_profile!(
            context,
            encoder,
            &self.label,
            |pass: &mut wgpu::ComputePass<'_>| {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(x, y, z);
            }
        );
    }
}
