use crate::weights::DType; // Ensure DType is imported
use crate::WgpuContext;
use crate::gpu_ops::{GpuTensor, Kernel};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, BindGroupLayout, Buffer, CommandEncoder, ComputePipeline};

// The uniform struct matches both shaders (F32 and BF16)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulUniforms {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

pub struct GpuMatMul {
    // We hold a pipeline for each supported backend type
    pipeline_f32: Arc<ComputePipeline>,
    pipeline_bf16: Arc<ComputePipeline>,
    pipeline_gemv_bf16: Arc<ComputePipeline>,
    // In the future: pipeline_q4: Arc<ComputePipeline>, 
    
    bind_group_layout: Arc<BindGroupLayout>,
    context: Arc<WgpuContext>,
    uniform_buffer: wgpu::Buffer,
}

impl GpuMatMul {
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        // 1. Create Layout ONCE (Reuse for all types)
        let bind_group_layout = create_bind_group_layout(&context.device);

        // 2. Compile F32 Pipeline
        let shader_f32 = context.device.create_shader_module(wgpu::include_wgsl!("./matmul_tiled.wgsl"));
        let pipeline_f32 = create_pipeline(&context.device, &bind_group_layout, &shader_f32, "Matmul F32");
        let shader_gemv = context.device.create_shader_module(wgpu::include_wgsl!("./gemv_bf16.wgsl"));
        // 3. Compile BF16 Pipeline
        let shader_bf16 = context.device.create_shader_module(wgpu::include_wgsl!("./matmul_bf16.wgsl"));
        let pipeline_bf16 = create_pipeline(&context.device, &bind_group_layout, &shader_bf16, "Matmul BF16");
        let pipeline_gemv = create_pipeline(&context.device, &bind_group_layout, &shader_gemv, "GEMV BF16"); 

        let uniform_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MatMul Persistent Uniforms"),
            size: std::mem::size_of::<MatmulUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline_f32: Arc::new(pipeline_f32),
            pipeline_bf16: Arc::new(pipeline_bf16),
            pipeline_gemv_bf16: Arc::new(pipeline_gemv),
            bind_group_layout: Arc::new(bind_group_layout),
            context: context.clone(),
            uniform_buffer,
        }
    }
}

impl Kernel for GpuMatMul {
    fn encode(&self, encoder: &mut CommandEncoder, inputs: &[&GpuTensor], output: &GpuTensor) {
        let a = inputs[0];
        let b = inputs[1];

        assert_eq!(a.rank(), 2, "Input A must be a 2D tensor");
        assert_eq!(b.rank(), 2, "Input B must be a 2D tensor");

        // --- PIPELINE SELECTION & VALIDATION ---
        let (pipeline, m, k, n) = match b.dtype() {
            DType::F32 => {
                // Standard MatMul Logic: [M, K] @ [K, N] -> [M, N]
                assert_eq!(a.shape()[1], b.shape()[0], "F32: A cols must match B rows");
                let m = a.shape()[0] as u32;
                let k = a.shape()[1] as u32;
                let n = b.shape()[1] as u32;
                (&self.pipeline_f32, m, k, n)
            },
            DType::BF16 => {
                // Implicit Transpose Logic: [M, K] @ [N, K] -> [M, N]
                // B is physically [N, K] (Transposed in VRAM), logical is [K, N]
                assert_eq!(a.shape()[1], b.shape()[1], "BF16: A cols must match B cols (Implicit Transpose)");
                let m = a.shape()[0] as u32;
                let k = a.shape()[1] as u32;
                let n = b.shape()[0] as u32; // Note: N comes from dimension 0 here!

                // if m == 1 {
                //     pass.dispatch_workgroups(n, 1, 1);  // N workgroups
                // } else {
                //     // Tiled matmul
                //     let workgroup_x = (n + 127) / 128;
                //     let workgroup_y = (m + 127) / 128;
                //     pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
                // }

                if m == 1 {
                    // FAST PATH: GEMV
                    (&self.pipeline_gemv_bf16, m, k, n)
                } else {
                    // SLOW PATH: Tiled MatMul (Prefill phase)
                    (&self.pipeline_bf16, m, k, n)
                }
            },
            dtype => panic!("Unsupported MatMul weight type: {:?}", dtype),
        };

        // Validate Output
        assert_eq!(
            output.shape(),
            &[m as usize, n as usize],
            "Output shape is incorrect"
        );

        run_internal_matmul(
            &self.context,
            encoder,
            pipeline, // Pass selected pipeline
            &self.bind_group_layout,
            &self.uniform_buffer,
            a.buffer(),
            b.buffer(),
            output.buffer(),
            m,
            k,
            n,
        );
    }
}

// --- Helpers ---

fn create_bind_group_layout(device: &wgpu::Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Matmul Shared Layout"),
        entries: &[
            // Uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            },
            // Matrix A (ReadOnly)
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
            // Matrix B (ReadOnly)
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
            // Matrix C (ReadWrite)
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
    })
}

fn create_pipeline(
    device: &wgpu::Device, 
    layout: &BindGroupLayout, 
    module: &wgpu::ShaderModule,
    label: &str
) -> ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{} Layout", label)),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn run_internal_matmul(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group_layout: &BindGroupLayout,
    _uniform_buffer: &Buffer,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    m: u32,
    k: u32,
    n: u32,
) {
    let device = &context.device;
    let uniforms = MatmulUniforms { m, k, n, _padding: 0 };

    // Note: Creating a buffer every frame is slow! 
    // In production, use the StagingBelt or a RingBuffer for uniforms.
    // let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //     label: Some("Matmul Uniforms"),
    //     contents: bytemuck::cast_slice(&[uniforms]),
    //     usage: wgpu::BufferUsages::UNIFORM,
    // });

    // context.queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    let offset = context.uniform_arena.write(&context.queue, &uniforms);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Matmul Bind Group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: context.uniform_arena.buffer(),
                    offset: 0, // MUST BE 0. Dynamic offset adds to this.
                    size: Some(std::num::NonZeroU64::new(std::mem::size_of::<MatmulUniforms>() as u64).unwrap()),
                }),
            },
            wgpu::BindGroupEntry { binding: 1, resource: a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: c.as_entire_binding() },
        ],
    });
    let label = format!("MatMul [{}x{} @ {}]", m, n, k);
    
    context.profiler.profile(encoder, &label, |pass| {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[offset]);
        
        const TILE_DIM: u32 = 32;
        let workgroup_x = (n + TILE_DIM - 1) / TILE_DIM;
        let workgroup_y = (m + TILE_DIM - 1) / TILE_DIM;
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    });
    // let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //     label: Some("Matmul Compute Pass"),
    //     timestamp_writes: None,
    // });
    // compute_pass.set_pipeline(pipeline);
    // compute_pass.set_bind_group(0, &bind_group, &[]);

    // const TILE_DIM: u32 = 32;
    // let workgroup_x = (n + TILE_DIM - 1) / TILE_DIM;
    // let workgroup_y = (m + TILE_DIM - 1) / TILE_DIM;

    // compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
}

#[cfg(test)]
mod tests;