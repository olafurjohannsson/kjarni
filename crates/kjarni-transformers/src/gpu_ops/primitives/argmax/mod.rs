use crate::gpu::GpuTensor;
use crate::WgpuContext;
use std::borrow::Cow;

pub struct GpuArgMax {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    context: std::sync::Arc<WgpuContext>,
}
impl GpuArgMax {
    pub fn new(context: &std::sync::Arc<WgpuContext>) -> Self {
        let shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ArgMax Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("argmax.wgsl"))),
            });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ArgMax Layout"),
                    entries: &[
                        // Binding 0: Input Logits (ReadOnly Storage)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Binding 1: Output Index (Storage)
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

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ArgMax Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ArgMax Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        Self {
            pipeline,
            bind_group_layout,
            context: context.clone(),
        }
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_logits: &GpuTensor,    // Shape: [Batch, Vocab]
        output_index: &wgpu::Buffer, // A 4-byte buffer for the result
    ) {
        let vocab_size = input_logits.shape()[1] as u32;

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ArgMax BindGroup"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_logits.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_index.as_entire_binding(),
                    },
                ],
            });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ArgMax Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch: Simplest approach is 1 workgroup per batch item.
        // This shader assumes 1D flattening within the shader for simplicity.
        pass.dispatch_workgroups(1, 1, 1);
    }
}


#[cfg(test)]
mod tests;