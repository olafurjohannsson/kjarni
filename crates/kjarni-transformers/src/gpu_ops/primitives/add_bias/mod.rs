use crate::gpu_ops::{GpuTensor, Kernel};
use crate::WgpuContext;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, BindGroupLayout, Buffer, CommandEncoder, ComputePipeline};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    size: u32,
    _padding: [u32; 3], // Padding to meet 16-byte alignment
}

/// A kernel for broadcasting a 1D bias vector and adding it to a larger tensor.
pub struct GpuAddBias {
    pipeline: Arc<ComputePipeline>,
    bind_group_layout: Arc<BindGroupLayout>,
    context: Arc<WgpuContext>,
}

impl GpuAddBias {
    /// Creates a new GpuAddBias kernel by compiling the shader.
    pub fn new(context: &Arc<WgpuContext>) -> Self {
        let (pipeline, bind_group_layout) = compile_add_bias_pipeline(context);
        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            context: context.clone(),
        }
    }
}

impl Kernel for GpuAddBias {
    /// Encodes the add-bias operation.
    ///
    /// # Arguments
    /// * `inputs` - A slice containing two tensors: `[&input, &bias]`.
    ///   - `input` can be any rank.
    ///   - `bias` must be a 1D tensor.
    /// * `output` - The tensor where the result is written. Must have the same shape as `input`.
    fn encode(
        &self,
        encoder: &mut CommandEncoder,
        inputs: &[&GpuTensor],
        output: &GpuTensor,
    ) {
        let input = inputs[0];
        let bias = inputs[1];

        // --- Assertions for correctness ---
        assert_eq!(inputs.len(), 2, "GpuAddBias kernel expects 2 inputs: [input, bias]");
        assert_eq!(input.shape(), output.shape(), "Input and output shapes must match for AddBias");
        assert_eq!(bias.rank(), 1, "Bias tensor must be 1D");
        assert_eq!(
            input.shape().last().unwrap(),
            &bias.shape()[0],
            "Last dimension of input tensor must match the size of the 1D bias vector"
        );

        run_internal_add_bias(
            &self.context,
            encoder,
            &self.pipeline,
            &self.bind_group_layout,
            input.buffer(),
            bias.buffer(),
            output.buffer(),
            input.num_elements() as u32,
        );
    }
}

/// The internal, private implementation that sets up and dispatches the shader.
fn run_internal_add_bias(
    context: &WgpuContext,
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bind_group_layout: &BindGroupLayout,
    input: &Buffer,
    bias: &Buffer,
    output: &Buffer,
    size: u32,
) {
    let device = &context.device;
    let uniforms = Uniforms { size, _padding: [0; 3] };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("AddBias Uniforms"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("AddBias Bind Group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: bias.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: output.as_entire_binding() },
        ],
    });

    // let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    //     label: Some("AddBias Compute Pass"),
    //     timestamp_writes: None,
    // });
    let label = format!("AddBias");
    context.profiler.profile(encoder, &label, |compute_pass| {
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        
        let workgroup_x = (size + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_x, 1, 1);
    });
}

/// The internal, private compilation function.
fn compile_add_bias_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(wgpu::include_wgsl!("./add_bias.wgsl"));

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("AddBias Bind Group Layout"),
        entries: &[
            // Uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            // Input Tensor
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            // Bias Vector
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            // Output Tensor
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("AddBias Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("AddBias Pipeline"),
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