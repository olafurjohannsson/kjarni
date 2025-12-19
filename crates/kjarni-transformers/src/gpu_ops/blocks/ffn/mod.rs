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

use crate::activations::Activation;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::GpuTensorPool;
use crate::WgpuContext;
use anyhow::Result;
use ndarray::{Array1, Array2};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, CommandEncoder, ComputePipeline};

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
        fc1_weight: GpuTensor, // Expects transposed [intermediate_size, hidden_size]
        fc1_bias: GpuTensor,
        fc2_weight: GpuTensor, // Expects transposed [hidden_size, intermediate_size]
        fc2_bias: GpuTensor,
    ) -> Result<Self> {
        assert_eq!(fc1_weight.rank(), 2, "FC1 weight must be 2D");
        assert_eq!(fc2_weight.rank(), 2, "FC2 weight must be 2D");
        assert_eq!(fc1_bias.rank(), 1, "FC1 bias must be 1D");
        assert_eq!(fc2_bias.rank(), 1, "FC2 bias must be 1D");

        // For FC1, transposed shape is [out, in] -> [intermediate_size, hidden_size].
        assert_eq!(fc1_weight.shape()[0], fc1_bias.shape()[0]);

        // For FC2, transposed shape is [out, in] -> [hidden_size, intermediate_size].
        assert_eq!(fc2_weight.shape()[0], fc2_bias.shape()[0]);

        // Connection check: The intermediate_size must match.
        assert_eq!(fc1_weight.shape()[0], fc2_weight.shape()[1]);

        Ok(Self {
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        })
    }
    pub fn from_config_names(
        context: &Arc<WgpuContext>,
        weights: &crate::weights_old::ModelWeights,
        names: &crate::traits::LayerFeedForwardNames,
    ) -> Result<Self> {
        // This assumes a standard FFN. You can add a match for SwiGLU here later.
        Ok(crate::gpu_ops::blocks::ffn::GpuFeedForwardWeights::new(
            GpuTensor::from_raw(context, &weights.get_raw(&names.intermediate_weight)?, "ff1_w")?,
            GpuTensor::from_raw(context, &weights.get_raw(&names.intermediate_bias)?, "ff1_b")?,
            GpuTensor::from_raw(context, &weights.get_raw(&names.output_weight)?, "ff2_w")?,
            GpuTensor::from_raw(context, &weights.get_raw(&names.output_bias)?, "ff2_b")?,
        )?)
    }
    /// Creates a new `GpuFeedForwardWeights` container from CPU ndarrays.
    /// This is the recommended "smart" constructor.
    pub fn from_ndarrays(
        context: &Arc<WgpuContext>,
        fc1_w_cpu: &Array2<f32>,
        fc1_b_cpu: &Array1<f32>,
        fc2_w_cpu: &Array2<f32>,
        fc2_b_cpu: &Array1<f32>,
    ) -> Result<Self> {
        assert_eq!(fc1_w_cpu.shape()[1], fc1_b_cpu.shape()[0]);
        assert_eq!(fc2_w_cpu.shape()[1], fc2_b_cpu.shape()[0]);
        assert_eq!(fc1_w_cpu.shape()[1], fc2_w_cpu.shape()[0]);

        let fc1_weight_transposed = GpuTensor::from_ndarray(context, &fc1_w_cpu.t().to_owned())?;
        let fc2_weight_transposed = GpuTensor::from_ndarray(context, &fc2_w_cpu.t().to_owned())?;
        let fc1_bias = GpuTensor::from_ndarray(context, fc1_b_cpu)?;
        let fc2_bias = GpuTensor::from_ndarray(context, fc2_b_cpu)?;

        Self::new(
            fc1_weight_transposed,
            fc1_bias,
            fc2_weight_transposed,
            fc2_bias,
        )
    }
}

/// A GPU-accelerated Feed-Forward Network block.
pub struct GpuFeedForward {
    pipelines: FfnPipelines,
    context: Arc<WgpuContext>,
}

impl GpuFeedForward {
    /// Creates a new `GpuFeedForward` block from pre-prepared weights.
    pub fn new(context: &Arc<WgpuContext>, activation: Activation) -> Result<Self> {
        match activation {
            Activation::Gelu => (),
            Activation::GeluNew => (),
            _ => {
                return Err(anyhow::anyhow!(
                    "GpuFeedForward's fused kernel currently only supports gelu and gelu_new."
                ));
            }
        }

        let (fc1_pipeline, fc1_layout) = compile_fc1_pipeline(context, activation);
        let (fc2_pipeline, fc2_layout) = compile_fc2_pipeline(context);

        Ok(Self {
            pipelines: FfnPipelines {
                fc1: (Arc::new(fc1_pipeline), Arc::new(fc1_layout)),
                fc2: (Arc::new(fc2_pipeline), Arc::new(fc2_layout)),
            },
            context: context.clone(),
        })
    }

    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        weights: &GpuFeedForwardWeights,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        // Get a temporary tensor for the intermediate result.
        let intermediate_size = weights.fc1_bias.shape()[0];
        let intermediate_shape = vec![
            input.shape()[0],
            input.shape()[1],
            intermediate_size,
        ];
        let intermediate = pool.get(intermediate_shape);

        // Get the final output tensor.
        let output = pool.get(input.shape().to_vec());

        // Run the two passes.
        self.run_fc1(encoder, weights, input, &intermediate);

        self.run_fc2(encoder, weights, &intermediate, &output);

        // Return the final output tensor.
        output
    }

    pub fn encode_2(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        weights: &GpuFeedForwardWeights,
        pool: &mut GpuTensorPool,
        output: &GpuTensor,
    ) {
        // Get a temporary tensor for the intermediate result.
        let intermediate_size = weights.fc1_bias.shape()[0];
        let intermediate_shape = vec![
            input.shape()[0],
            input.shape()[1],
            intermediate_size,
        ];
        let intermediate = pool.get(intermediate_shape);
        // Run the two passes.
        self.run_fc1(encoder, weights, input, &intermediate);
        self.run_fc2(encoder, weights, &intermediate, &output);
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
        let output_shape = output.shape();
        let rows = (input_shape[0] * input_shape[1]) as u32;
        let hidden_size = input_shape[2] as u32;
        let intermediate_size = output_shape[2] as u32;

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
        // let hidden_size = weights.fc2_weight.shape()[1] as u32;
        let output_shape = output.shape();
        let hidden_size = output_shape[2] as u32;

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

fn compile_fc1_pipeline(context: &WgpuContext, activation: Activation) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(wgpu::include_wgsl!("./fc1.wgsl"));

    let act_function = match activation {
        Activation::Gelu => 0.0,
        Activation::GeluNew => 1.0,
        _ => 0.0
    };
    let constants = [
        ("0", act_function)
    ];

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
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        },
        cache: None,
    });

    (pipeline, bind_group_layout)
}

fn compile_fc2_pipeline(context: &WgpuContext) -> (ComputePipeline, BindGroupLayout) {
    let device = &context.device;
    let shader = device.create_shader_module(wgpu::include_wgsl!("./fc2.wgsl"));

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

#[cfg(test)]
mod tests;