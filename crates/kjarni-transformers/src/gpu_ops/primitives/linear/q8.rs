//! Quantised GEMVs on int8 activations: `linear_q8.wgsl`.
//!
//! Used for decode and other calls of a few rows when the context allows it
//! ([`WgpuContext::int8_activations`]). Each call quantises its input to int8 in
//! blocks of 32 and then runs the GEMV, both under one bind group.

use crate::WgpuContext;
use crate::gpu::GpuTensor;
use std::sync::{Arc, Mutex};

use super::LinearInfo;

/// Workgroups in one dimension of a dispatch.
const MAX_DIM: u32 = 65535;

struct Scratch {
    act: wgpu::Buffer,
    aux: wgpu::Buffer,
    /// Activations the buffers hold.
    capacity: u64,
}

pub(super) struct Q8Gemv {
    context: Arc<WgpuContext>,
    layout: wgpu::BindGroupLayout,
    quantize: wgpu::ComputePipeline,
    gemv_q4k: wgpu::ComputePipeline,
    gemv_q4k_split: wgpu::ComputePipeline,
    /// The int8 activations and their scales. Calls recorded one after another
    /// reuse them: the GPU runs a call's quantize and GEMV before the next's.
    scratch: Mutex<Option<Scratch>>,
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl Q8Gemv {
    pub(super) fn new(context: &Arc<WgpuContext>) -> Self {
        let device = &context.device;
        let shader = device.create_shader_module(wgpu::include_wgsl!("./linear_q8.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Linear Q8 Layout"),
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
                storage(1, true),
                storage(2, true),
                storage(3, false),
                storage(4, false),
                storage(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Linear Q8 Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            context: context.clone(),
            quantize: make("quantize_q8"),
            gemv_q4k: make("gemv_q4k_q8"),
            gemv_q4k_split: make("gemv_q4k_q8_split"),
            layout,
            scratch: Mutex::new(None),
        }
    }

    /// The scratch buffers, grown to hold `elements` activations.
    fn scratch(&self, elements: u64) -> (wgpu::Buffer, wgpu::Buffer) {
        let mut slot = self.scratch.lock().unwrap();
        if slot.as_ref().is_none_or(|s| s.capacity < elements) {
            let capacity = elements.next_power_of_two().max(1 << 14);
            let make = |label: &str, size: u64| {
                self.context.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };
            *slot = Some(Scratch {
                // One byte an activation, and 16 bytes for each block of 32.
                act: make("Linear Q8 Activations", capacity),
                aux: make("Linear Q8 Scales", capacity / 2),
                capacity,
            });
        }
        let s = slot.as_ref().unwrap();
        (s.act.clone(), s.aux.clone())
    }

    /// Records `output = input @ weights^T` for Q4_K weights, `m` rows of `k`.
    pub(super) fn encode_q4k(
        &self,
        encoder: &mut crate::gpu::GpuEncoder,
        input: &GpuTensor,
        weights: &GpuTensor,
        output: &GpuTensor,
        info: LinearInfo,
        split: bool,
    ) {
        let LinearInfo { m, k, n } = info;
        let (act, aux) = self.scratch(m as u64 * k as u64);
        let uniform = self.context.uniform_buffer(&info);
        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Linear Q8 BindGroup"),
                layout: &self.layout,
                entries: &[
                    entry(0, &uniform),
                    entry(1, input.buffer()),
                    entry(2, weights.buffer()),
                    entry(3, &act),
                    entry(4, &aux),
                    entry(5, output.buffer()),
                ],
            });
        let quantize_groups = (m * k).div_ceil(256);
        let rows = n.div_ceil(8);
        let gemv = if split {
            &self.gemv_q4k_split
        } else {
            &self.gemv_q4k
        };
        let label = format!("Linear Q8 [{m}x{k} @ {n}]");
        crate::gpu_profile!(
            self.context,
            encoder,
            &label,
            |pass: &mut wgpu::ComputePass<'_>| {
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_pipeline(&self.quantize);
                dispatch(pass, quantize_groups, 1);
                pass.set_pipeline(gemv);
                dispatch(pass, rows, m);
            }
        );
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// `groups` workgroups in x, folded into y past the limit, and `z` in z.
fn dispatch(pass: &mut wgpu::ComputePass<'_>, groups: u32, z: u32) {
    if groups > MAX_DIM {
        pass.dispatch_workgroups(MAX_DIM, groups.div_ceil(MAX_DIM), z);
    } else {
        pass.dispatch_workgroups(groups, 1, z);
    }
}
