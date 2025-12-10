use crate::gpu_ops::{GpuTensor, Kernel};
use crate::WgpuContext;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, BindGroupLayout, Buffer, CommandEncoder, ComputePipeline};


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxUniforms {
    rows: u32,
    cols: u32,
    valid_cols: u32,
    scale: f32,
}

pub struct GpuSoftmax {
    pipeline: Arc<ComputePipeline>,
    bind_group_layout: Arc<BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl GpuSoftmax {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let (pipeline, bind_group_layout) = compile_softmax_pipeline(context);
        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            context: context.clone(),
        }
    }

    /// A more specialized encode method for softmax which is in-place and needs extra params.
    pub fn encode(&self, encoder: &mut CommandEncoder, tensor: &GpuTensor, scale: f32) {
        // Softmax operates on the last dimension.
        // The "rows" are all preceding dimensions combined.
        let last_dim = tensor.shape().last().unwrap();
        let rows = tensor.num_elements() / last_dim;

        run_internal_softmax(
            &self.context,
            encoder,
            &self.pipeline,
            &self.bind_group_layout,
            tensor.buffer(),
            rows as u32,
            *last_dim as u32, // Both physical and logical are the same here
            *last_dim as u32,
            scale,
        );
    }

    /// The powerful version for padded/cached data.
    pub fn encode_padded(
        &self,
        encoder: &mut CommandEncoder,
        tensor: &GpuTensor,
        valid_cols: u32,
        scale: f32,
    ) {
        let last_dim_padded = *tensor.shape().last().unwrap();
        let rows = tensor.num_elements() / last_dim_padded;

        run_internal_softmax(
            &self.context,
            encoder,
            &self.pipeline,
            &self.bind_group_layout,
            tensor.buffer(),
            rows as u32,
            last_dim_padded as u32,
            valid_cols,
            scale,
        );
    }
}

fn run_internal_softmax(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group_layout: &BindGroupLayout,
    data: &Buffer,
    rows: u32,
    cols: u32,       // Physical width
    valid_cols: u32, // Logical width
    scale: f32,
) {
    let device = &context.device;
    let uniforms = SoftmaxUniforms {
        rows,
        cols,
        valid_cols,
        scale,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Softmax Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Softmax Bind Group"),
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

    // let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //     label: Some("Softmax Compute Pass"),
    //     timestamp_writes: None,
    // });
    let label = format!("Softmax");
    context.profiler.profile(encoder, &label, |compute_pass| {
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_x = (rows + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_x, 1, 1);
    });
}

fn compile_softmax_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(wgpu::include_wgsl!("./softmax.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Softmax Bind Group Layout"),
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
            // Note: The buffer is read-write, so we declare it as such.
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
        label: Some("Softmax Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Softmax Pipeline"),
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