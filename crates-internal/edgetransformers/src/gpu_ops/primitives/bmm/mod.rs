use crate::gpu_ops::{GpuTensor, Kernel};
use crate::WgpuContext;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, BindGroupLayout, Buffer, CommandEncoder, ComputePipeline};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulInfo {
    b: u32,
    m: u32,
    k: u32,
    n: u32,
    a_stride_batch: u32,
    b_stride_batch: u32,
    _padding1: u32,
    _padding2: u32,
}

pub struct GpuBatchedMatMul {
    pipeline: Arc<ComputePipeline>,
    bind_group_layout: Arc<BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl GpuBatchedMatMul {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let (pipeline, bind_group_layout) = compile_bmm_pipeline(context);
        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            context: context.clone(),
        }
    }
}

impl Kernel for GpuBatchedMatMul {
    fn encode(&self, encoder: &mut CommandEncoder, inputs: &[&GpuTensor], output: &GpuTensor) {
        let a = inputs[0];
        let b = inputs[1];

        assert!(a.rank() >= 2 && a.rank() <= 3, "Input A must be a 2D or 3D tensor");
        assert!(b.rank() >= 2 && b.rank() <= 3, "Input B must be a 2D or 3D tensor");
        let a_shape = if a.rank() == 2 { vec![1, a.shape()[0], a.shape()[1]] } else { a.shape().to_vec() };
        let b_shape = if b.rank() == 2 { vec![1, b.shape()[0], b.shape()[1]] } else { b.shape().to_vec() };
        
        let batch = a_shape[0].max(b_shape[0]);
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        assert_eq!(k, b_shape[1], "Inner dimensions of A and B do not match");
        assert_eq!(output.shape(), &[batch, m, n], "Output shape is incorrect");
        if a.rank() == 3 && b.rank() == 3 {
            assert!(a.shape()[0] == b.shape()[0] || a.shape()[0] == 1 || b.shape()[0] == 1, "Batch dimensions must match or one must be 1 for broadcasting");
        }

        let a_stride_batch = if a.shape().len() == 3 && a.shape()[0] > 1 { (m * k) as u32 } else { 0 };
        let b_stride_batch = if b.shape().len() == 3 && b.shape()[0] > 1 { (k * n) as u32 } else { 0 };

        run_internal_bmm(
            &self.context, encoder, &self.pipeline, &self.bind_group_layout,
            a.buffer(), b.buffer(), output.buffer(),
            batch as u32, m as u32, k as u32, n as u32,
            a_stride_batch, b_stride_batch,
        );
    }
}

fn run_internal_bmm(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group_layout: &BindGroupLayout,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
    a_stride_batch: u32, b_stride_batch: u32,
) {
    let device = &context.device;
    let uniforms = MatmulInfo { b: batch, m, k, n, a_stride_batch, b_stride_batch, _padding1:0, _padding2: 0 };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("BMM Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("BMM Bind Group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: c.as_entire_binding(),
            },
        ],
    });

    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("BMM Compute Pass"),
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, &bind_group, &[]);

    const TILE_DIM: u32 = 32;
    let workgroup_x = (n + TILE_DIM - 1) / TILE_DIM;
    let workgroup_y = (m + TILE_DIM - 1) / TILE_DIM;
    let workgroup_z = batch;

    compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z);
}

fn compile_bmm_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(wgpu::include_wgsl!("./bmm.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("BMM Bind Group Layout"),
        entries: &[
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
            wgpu::BindGroupLayoutEntry {
                binding: 3,
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
        label: Some("BMM Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("BMM Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, bind_group_layout)
}
#[cfg(test)]
mod tests;