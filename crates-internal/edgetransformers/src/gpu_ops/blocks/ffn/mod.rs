use crate::gpu_ops::{GpuTensor};
use crate::WgpuContext;
// Note: This is a "block", not a primitive "kernel", so it has a `forward` method.
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, BindGroupLayout, Buffer, CommandEncoder, ComputePipeline};

#[cfg(test)]
mod tests;

pub mod fc1;
pub mod fc2;

// Uniform struct remains internal to this module
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FfnUniforms {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

struct FfnPipelines {
    fc1: Arc<ComputePipeline>,
    fc1_layout: Arc<BindGroupLayout>,
    fc2: Arc<ComputePipeline>,
    fc2_layout: Arc<BindGroupLayout>,
}

pub struct GpuFeedForward {
    pipelines: FfnPipelines,
    fc1_weight: GpuTensor,
    fc1_bias: GpuTensor,
    fc2_weight: GpuTensor,
    fc2_bias: GpuTensor,
    context: Arc<WgpuContext>,
}

impl GpuFeedForward {
    pub fn new(
        context: &Arc<WgpuContext>,
        fc1_weight: GpuTensor,
        fc1_bias: GpuTensor,
        fc2_weight: GpuTensor,
        fc2_bias: GpuTensor,
    ) -> Self {
        let (fc1_pipeline, fc1_layout) = compile_fc1_pipeline(context);
        let (fc2_pipeline, fc2_layout) = compile_fc2_pipeline(context);
        let pipelines = FfnPipelines {
            fc1: Arc::new(fc1_pipeline),
            fc1_layout: Arc::new(fc1_layout),
            fc2: Arc::new(fc2_pipeline),
            fc2_layout: Arc::new(fc2_layout),
        };

        Self {
            pipelines,
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            context: context.clone(),
        }
    }

    pub fn forward(&self, encoder: &mut CommandEncoder, input: &GpuTensor, intermediate: &GpuTensor, output: &GpuTensor) {
        let input_shape = input.shape();
        let rows = (input_shape[0] * input_shape[1]) as u32;
        let hidden_size = input_shape[2] as u32;
        let intermediate_size = self.fc1_weight.shape()[1] as u32;

        let uniforms_fc1 = FfnUniforms { m: rows, k: hidden_size, n: intermediate_size, _padding: 0 };
        let uniform_buffer_fc1 = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFN FC1 Uniforms"),
            contents: bytemuck::cast_slice(&[uniforms_fc1]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let fc1_bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FC1 Bind Group"),
            layout: &self.pipelines.fc1_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer_fc1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.fc1_weight.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.fc1_bias.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: input.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: intermediate.buffer().as_entire_binding() },
            ],
        });
        
        let workgroups_fc1 = (rows * intermediate_size + 511) / 512;
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(&self.pipelines.fc1);
            compute_pass.set_bind_group(0, &fc1_bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups_fc1, 1, 1);
        }

        let uniforms_fc2 = FfnUniforms { m: rows, k: intermediate_size, n: hidden_size, _padding: 0 };
        let uniform_buffer_fc2 = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
             label: Some("FFN FC2 Uniforms"),
             contents: bytemuck::cast_slice(&[uniforms_fc2]),
             usage: wgpu::BufferUsages::UNIFORM,
        });

        let fc2_bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FC2 Bind Group"),
            layout: &self.pipelines.fc2_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer_fc2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.fc2_weight.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.fc2_bias.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: intermediate.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: output.buffer().as_entire_binding() },
            ],
        });

        let workgroups_fc2 = (rows * hidden_size + 511) / 512;
        {
            let mut compute_pass = encoder.begin_compute_pass(&Default::default());
            compute_pass.set_pipeline(&self.pipelines.fc2);
            compute_pass.set_bind_group(0, &fc2_bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups_fc2, 1, 1);
        }
    }
}

pub fn compile_fc1_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(wgpu::include_wgsl!("./fc1/fc1.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FC1 Bind Group Layout"),
        entries: &[
            // @binding(0) var<uniform> info: FfnUniforms;
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
            // @binding(1) var<storage, read> fc1_weight: array<f32>;
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(2) var<storage, read> fc1_bias: array<f32>;
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(3) var<storage, read> input: array<f32>;
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(4) var<storage, read_write> output: array<f32>;
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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
        label: Some("FC1 Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("FC1 Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    (pipeline, bind_group_layout)
}

pub fn compile_fc2_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(wgpu::include_wgsl!("./fc2/fc2.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("FC2 Bind Group Layout"),
        entries: &[
            // @binding(0) var<uniform> info: FfnUniforms;
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
            // @binding(1) var<storage, read> fc2_weight: array<f32>;
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(2) var<storage, read> fc2_bias: array<f32>;
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(3) var<storage, read> input: array<f32>;
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(4) var<storage, read_write> output: array<f32>;
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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
        label: Some("FC2 Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("FC2 Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    (pipeline, bind_group_layout)

}