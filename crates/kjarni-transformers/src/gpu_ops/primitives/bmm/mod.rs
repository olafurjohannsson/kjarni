use crate::WgpuContext;
use crate::gpu_ops::{GpuTensor, Kernel};
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
    b_stride_row: u32,
    b_stride_col: u32,
}

pub struct BStrides {
    pub row: u32,
    pub col: u32,
}

pub struct GpuBatchedMatMul {
    pipeline: Arc<ComputePipeline>,
    bind_group_layout: Arc<BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl Kernel for GpuBatchedMatMul {
    fn encode(&self, encoder: &mut CommandEncoder, inputs: &[&GpuTensor], output: &GpuTensor) {
        let a = inputs[0];
        let b = inputs[1];

        assert!(
            a.rank() >= 2 && a.rank() <= 3,
            "Input A must be a 2D or 3D tensor"
        );
        assert!(
            b.rank() >= 2 && b.rank() <= 3,
            "Input B must be a 2D or 3D tensor"
        );

        // Handle 2D tensors by treating them as 3D with batch=1
        let a_shape = if a.rank() == 2 {
            vec![1, a.shape()[0], a.shape()[1]]
        } else {
            a.shape().to_vec()
        };
        let b_shape = if b.rank() == 2 {
            vec![1, b.shape()[0], b.shape()[1]]
        } else {
            b.shape().to_vec()
        };

        let batch = a_shape[0].max(b_shape[0]);
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        assert_eq!(k, b_shape[1], "Inner dimensions of A and B do not match");

        // Validate output shape
        let expected_output_shape = if a.rank() == 2 && b.rank() == 2 {
            vec![m, n] // 2D output for 2D inputs
        } else {
            vec![batch, m, n] // 3D output otherwise
        };
        assert_eq!(
            output.shape(),
            &expected_output_shape,
            "Output shape {:?} doesn't match expected {:?}",
            output.shape(),
            expected_output_shape
        );

        // For standard matmul, B is [batch, K, N] in row-major
        // Row stride = N (to go down a row, skip N elements)
        // Col stride = 1 (to go across a column, skip 1 element)
        let b_strides = BStrides {
            row: n as u32,
            col: 1,
        };

        // Call the internal method with proper 3D shapes
        self.encode_internal(encoder, a, b, output, &a_shape, &b_shape, b_strides);
    }
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
    // Internal implementation that works with normalized 3D shapes
    fn encode_internal(
        &self,
        encoder: &mut CommandEncoder,
        a: &GpuTensor,
        b: &GpuTensor,
        output: &GpuTensor,
        a_shape: &[usize], // Normalized 3D shape [batch, m, k]
        b_shape: &[usize], // Normalized 3D shape [batch, k, n]
        b_strides: BStrides,
    ) {
        let batch = a_shape[0].max(b_shape[0]);
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        // Calculate batch strides for broadcasting
        let a_stride_batch = if a_shape[0] > 1 { (m * k) as u32 } else { 0 };

        // For B, we need the physical stride
        let b_stride_batch = if b_shape[0] > 1 {
            (b_shape[1] * b_shape[2]) as u32 // k * n
        } else {
            0
        };

        run_internal_bmm(
            &self.context,
            encoder,
            &self.pipeline,
            &self.bind_group_layout,
            a.buffer(),
            b.buffer(),
            output.buffer(),
            batch as u32,
            m as u32,
            k as u32,
            n as u32,
            a_stride_batch,
            b_stride_batch,
            b_strides.row,
            b_strides.col,
        );
    }

    /// Matmul with custom strides for B matrix (for KV cache optimization)
    pub fn encode_with_b_strides(
        &self,
        encoder: &mut CommandEncoder,
        a: &GpuTensor,
        b: &GpuTensor,
        output: &GpuTensor,
        b_strides: BStrides,
    ) {
        // This version requires 3D tensors
        assert_eq!(a.rank(), 3, "A must be 3D for strided matmul");
        assert_eq!(b.rank(), 3, "B must be 3D for strided matmul");
        assert_eq!(output.rank(), 3, "Output must be 3D for strided matmul");

        let a_shape = a.shape().to_vec();
        let b_shape = b.shape().to_vec(); // Physical shape of B

        let batch = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = output.shape()[2]; // Logical N from output

        assert_eq!(batch, b_shape[0], "Batch dimensions must match");
        assert_eq!(batch, output.shape()[0], "Batch dimensions must match");

        // Calculate batch strides
        let a_stride_batch = if batch > 1 { (m * k) as u32 } else { 0 };

        // For b_stride_batch, use the physical layout
        let b_stride_batch = if batch > 1 {
            (b_shape[1] * b_shape[2]) as u32
        } else {
            0
        };

        run_internal_bmm(
            &self.context,
            encoder,
            &self.pipeline,
            &self.bind_group_layout,
            a.buffer(),
            b.buffer(),
            output.buffer(),
            batch as u32,
            m as u32,
            k as u32,
            n as u32,
            a_stride_batch,
            b_stride_batch,
            b_strides.row,
            b_strides.col,
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
    a_stride_batch: u32,
    b_stride_batch: u32,
    b_stride_row: u32,
    b_stride_col: u32,
) {
    let device = &context.device;
    let uniforms = MatmulInfo {
        b: batch,
        m,
        k,
        n,
        a_stride_batch,
        b_stride_batch,
        b_stride_row,
        b_stride_col,
    };

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

    // let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //     label: Some("BMM Compute Pass"),
    //     timestamp_writes: None,
    // });
    let label = format!("BMM");
    context.profiler.profile(encoder, &label, |compute_pass| {
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        const TILE_DIM: u32 = 32;
        let workgroup_x = (n + TILE_DIM - 1) / TILE_DIM;
        let workgroup_y = (m + TILE_DIM - 1) / TILE_DIM;
        let workgroup_z = batch;

        compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, workgroup_z);
    });
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
