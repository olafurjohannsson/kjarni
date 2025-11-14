use crate::WgpuContext;
use crate::gpu_ops::GpuTensor;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, BindGroupLayout, Buffer, CommandEncoder, ComputePipeline};

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleUniforms {
    size: u32,
    scale: f32,
    _padding: [u32; 2],
}

/// A kernel for performing in-place element-wise scaling of a tensor.
pub struct GpuScale {
    in_place_pipeline: Arc<ComputePipeline>,
    in_place_layout: Arc<BindGroupLayout>,
    out_of_place_pipeline: Arc<ComputePipeline>,
    out_of_place_layout: Arc<BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl GpuScale {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let (in_place_pipeline, in_place_layout) = compile_in_place_pipeline(context);

        let (out_of_place_pipeline, out_of_place_layout) = compile_out_of_place_pipeline(context);

        Self {
            in_place_pipeline: Arc::new(in_place_pipeline),
            out_of_place_pipeline: Arc::new(out_of_place_pipeline),
            in_place_layout: Arc::new(in_place_layout),
            out_of_place_layout: Arc::new(out_of_place_layout),
            context: context.clone(),
        }
    }

    /// Encodes the scaling operation. This is an in-place operation.
    pub fn encode_in_place(&self, encoder: &mut CommandEncoder, tensor: &GpuTensor, scale: f32) {
        run_internal_scale(
            &self.context,
            encoder,
            &self.in_place_pipeline,
            &self.in_place_layout,
            tensor.buffer(),
            tensor.num_elements() as u32,
            scale,
        );
    }
    pub fn encode_out_of_place(
        &self,
        encoder: &mut CommandEncoder,
        input: &GpuTensor,
        output: &GpuTensor, // New parameter
        scale: f32,
    ) {
        let device = &self.context.device;
        let size = input.num_elements() as u32;

        let uniforms = ScaleUniforms {
            size,
            scale,
            _padding: [0; 2],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scale Out-of-Place Uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scale Out-of-Place Bind Group"),
            layout: &self.out_of_place_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    // Input buffer
                    binding: 1,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    // Output buffer
                    binding: 2,
                    resource: output.buffer().as_entire_binding(),
                },
            ],
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Scale Out-of-Place Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.out_of_place_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (size + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }
}

fn run_internal_scale(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group_layout: &BindGroupLayout,
    data: &Buffer,
    size: u32,
    scale: f32,
) {
    let device = &context.device;
    let uniforms = ScaleUniforms {
        size,
        scale,
        _padding: [0; 2],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scale Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Scale Bind Group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: data.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Scale Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);
    let workgroups = (size + 255) / 256;
    compute_pass.dispatch_workgroups(workgroups, 1, 1);
}

fn compile_out_of_place_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    // Load the new shader
    let shader = device.create_shader_module(wgpu::include_wgsl!("./out_of_place.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Scale Out-of-Place Bind Group Layout"),
        entries: &[
            // Uniforms (@binding(0))
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Input Data (@binding(1), read_only)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true }, // Important!
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Output Data (@binding(2), read_write)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, // Important!
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Out of Place Scale Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Out of Place  Scale Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, bind_group_layout)
}
fn compile_in_place_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(wgpu::include_wgsl!("./scale.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Scale Bind Group Layout"),
        entries: &[
            // Uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Data (read_write)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Scale Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Scale Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, bind_group_layout)
}
