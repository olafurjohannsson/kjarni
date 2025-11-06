//! A GPU-accelerated Feed-Forward Network (FFN) block.
//!
//! This module defines a `GpuFeedForward` struct that encapsulates the logic for
//! a standard two-layer MLP found in Transformer architectures. It is designed for
//! performance, using fused kernels to combine matrix multiplication, bias addition,
//! and activation functions into single GPU passes.
//!
//! # Architecture
//!
//! 1.  **`GpuFeedForward` struct:** The main public-facing struct. It owns the compiled
//!     GPU pipelines and the weight/bias tensors for both linear layers.
//! 2.  **`forward` method:** The primary entry point. It orchestrates the two main
//!     compute passes (FC1 and FC2).
//! 3.  **`run_fc1`/`run_fc2` methods:** Helper methods that each encode a single,
//!     fused compute pass, which can be called individually for testing.
//! 4.  **Fused Kernels:**
//!     - `fc1.wgsl`: Performs `MatMul(input, W1) + Bias1` and applies the `GeLU` activation.
//!     - `fc2.wgsl`: Performs `MatMul(intermediate, W2) + Bias2`.
//!
//! # INVARIANT
//!
//! The constructor (`::new`) for this struct is "dumb." It expects that the weight tensors
//! it receives are **already in the correct layout** required by its internal shaders.
//! The shaders use an output-centric algorithm that is most efficient with a weight layout of
//! **`[out_features, in_features]`**.
//!
//! It is the responsibility of the higher-level model loading code (e.g., `GpuEncoder::new`)
//! to inspect the model's configuration (`transpose_ffn_weights` flag) and perform any
//! necessary transpositions *before* calling this constructor.

use crate::WgpuContext;
use crate::activations::Activation;
use crate::gpu_ops::GpuTensor;
use anyhow::{Result, anyhow};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, BindGroupLayout, CommandEncoder, ComputePipeline};

#[cfg(test)]
mod tests;

/// Uniform struct passed to both FC1 and FC2 shaders.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FfnUniforms {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

/// Holds the pre-compiled pipelines and their bind group layouts.
struct FfnPipelines {
    fc1: (Arc<ComputePipeline>, Arc<BindGroupLayout>),
    fc2: (Arc<ComputePipeline>, Arc<BindGroupLayout>),
}

pub struct GpuFeedForwardWeights {
    // Fields are now crate-public to be accessible by GpuFeedForward, but not user-constructible
    pub(crate) fc1_weight: GpuTensor,
    pub(crate) fc1_bias: GpuTensor,
    pub(crate) fc2_weight: GpuTensor,
    pub(crate) fc2_bias: GpuTensor,
}

impl GpuFeedForwardWeights {
    /// Creates a new `GpuFeedForwardWeights` container, validating tensor shapes.
    ///
    /// This is the new "gatekeeper" for weight integrity. It will panic if the
    /// provided tensors have mismatched or incorrect dimensions.
    pub fn new(
        fc1_weight: GpuTensor,
        fc1_bias: GpuTensor,
        fc2_weight: GpuTensor,
        fc2_bias: GpuTensor,
    ) -> Result<Self> {
        // --- ALL THE ASSERTIONS ARE MOVED HERE ---
        assert_eq!(fc1_weight.rank(), 2, "FC1 weight must be 2D");
        assert_eq!(fc2_weight.rank(), 2, "FC2 weight must be 2D");
        assert_eq!(fc1_bias.rank(), 1, "FC1 bias must be 1D");
        assert_eq!(fc2_bias.rank(), 1, "FC2 bias must be 1D");

        // Checks FC1: output dimension of weight must match bias size.
        assert_eq!(
            fc1_weight.shape()[1],
            fc1_bias.shape()[0],
            "FC1 weight's [in,OUT] must match bias [OUT]"
        );

        // Checks FC2: output dimension of weight must match bias size.
        assert_eq!(
            fc2_weight.shape()[1],
            fc2_bias.shape()[0],
            "FC2 weight's [in,OUT] must match bias [OUT]"
        );

        // Checks connection: FC1 output dimension must match FC2 input dimension.
        assert_eq!(
            fc1_weight.shape()[1],
            fc2_weight.shape()[0],
            "FC1's out_features must match FC2's in_features"
        );

        Ok(Self {
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        })
    }
}

/// A GPU-accelerated Feed-Forward Network block.
pub struct GpuFeedForward {
    pipelines: FfnPipelines,
    context: Arc<WgpuContext>,
}

impl GpuFeedForward {
    /// Creates a new `GpuFeedForward` block from pre-prepared weights.
    ///
    /// # INVARIANT
    ///
    /// The weight tensors provided to this constructor **MUST** be in the standard
    /// **[in_features, out_features]** layout. This is because the internal fused
    /// kernels are written to expect this format for compatibility with the CPU path.
    ///
    /// # Arguments
    /// * `context` - The shared WGPU context.
    /// * `activation` - The activation function to use.
    pub fn new(context: &Arc<WgpuContext>, activation: Activation) -> Result<Self> {
        match activation {
            Activation::Gelu => (), // Supported
            _ => {
                return Err(anyhow!(
                    "GpuFeedForward's fused kernel currently only supports GELU."
                ));
            }
        }

        let (fc1_pipeline, fc1_layout) = compile_fc1_pipeline(context);
        let (fc2_pipeline, fc2_layout) = compile_fc2_pipeline(context);

        Ok(Self {
            pipelines: FfnPipelines {
                fc1: (Arc::new(fc1_pipeline), Arc::new(fc1_layout)),
                fc2: (Arc::new(fc2_pipeline), Arc::new(fc2_layout)),
            },
            context: context.clone(),
        })
    }

    /// Encodes the complete FFN forward pass into the command encoder.
    pub fn forward(
        &self,
        encoder: &mut CommandEncoder,
        weights: &GpuFeedForwardWeights,
        input: &GpuTensor,
        intermediate: &GpuTensor,
        output: &GpuTensor,
    ) {
        self.run_fc1(encoder, weights, input, intermediate);
        self.run_fc2(encoder, weights, intermediate, output);
    }

    /// Encodes the first fused kernel pass: `MatMul(input, W1) + Bias1 + GeLU`.
    pub fn run_fc1(
        &self,
        encoder: &mut CommandEncoder,
        weights: &GpuFeedForwardWeights,
        input: &GpuTensor,
        output: &GpuTensor,
    ) {
        let input_shape = input.shape();
        let rows = (input_shape[0] * input_shape[1]) as u32;
        let hidden_size = input_shape[2] as u32;
        let intermediate_size = weights.fc1_weight.shape()[1] as u32;

        let uniforms = FfnUniforms {
            m: rows,
            k: hidden_size,
            n: intermediate_size,
            _padding: 0,
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("FFN FC1 Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("FC1 Bind Group"),
                layout: &self.pipelines.fc1.1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weights.fc1_weight.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weights.fc1_bias.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: input.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output.buffer().as_entire_binding(),
                    },
                ],
            });

        let workgroups = (rows * intermediate_size + 511) / 512;
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FC1 Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipelines.fc1.0);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// Encodes the second fused kernel pass: `MatMul(intermediate, W2) + Bias2`.
    pub fn run_fc2(
        &self,
        encoder: &mut CommandEncoder,
        weights: &GpuFeedForwardWeights,
        input: &GpuTensor,
        output: &GpuTensor,
    ) {
        let input_shape = input.shape();
        let rows = (input_shape[0] * input_shape[1]) as u32;
        let intermediate_size = input_shape[2] as u32;
        let hidden_size = weights.fc2_weight.shape()[1] as u32;

        let uniforms = FfnUniforms {
            m: rows,
            k: intermediate_size,
            n: hidden_size,
            _padding: 0,
        };
        let uniform_buffer =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("FFN FC2 Uniforms"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("FC2 Bind Group"),
                layout: &self.pipelines.fc2.1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weights.fc2_weight.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weights.fc2_bias.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: input.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: output.buffer().as_entire_binding(),
                    },
                ],
            });

        let workgroups = (rows * hidden_size + 511) / 512;
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FC2 Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipelines.fc2.0);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }
}

fn compile_fc1_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
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

fn compile_fc2_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
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
