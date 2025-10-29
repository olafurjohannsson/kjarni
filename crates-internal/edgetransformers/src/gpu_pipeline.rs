//! A reusable orchestrator for running a generic transformer encoder pipeline on the GPU.

use anyhow::{Result};
use ndarray::{Array2, Array3};
use crate::cache::GpuKVCache;
use std::sync::Arc;
use wgpu::ComputePipeline;
use wgpu::util::DeviceExt;

use crate::gpu_ops::{
    blocks::attention::{
        AttentionConfig, AttentionPipelines, AttentionTempBuffers, AttentionWeights,
        run_attention_block,
    },
    blocks::ffn::{
        FFNConfig, FFNPipelines, FFNTempBuffers, FFNWeights, run_ffn_block,
    },
    primitives::{
        add::{compile_add_pipeline, run_gpu_add},
        layer_norm::{compile_layer_norm_pipeline, run_gpu_layer_norm},
    },
    utils::read_buffer_3d,
};
use crate::traits::{TransformerConfig};
use crate::gpu_context::WgpuContext;

pub struct TempBuffers {
    pub attention: AttentionTempBuffers,
    pub ffn: FFNTempBuffers,
}

impl TempBuffers {
    pub fn new(
        device: &wgpu::Device,
        config: &dyn TransformerConfig,
        batch_size: usize,
        seq_len: usize,
    ) -> Self {
        let hidden_size = config.hidden_size();
        let num_heads = config.num_attention_heads();
        let buffer_size = (batch_size * seq_len * hidden_size * 4) as u64;
        let scores_size = (batch_size * num_heads * seq_len * seq_len * 4) as u64;
        let intermediate_size = hidden_size * 4;
        let ffn_intermediate_size = (batch_size * seq_len * intermediate_size * 4) as u64;

        let create_buffer = |size: u64, label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };

        let attention = AttentionTempBuffers {
            q_proj: create_buffer(buffer_size, "Q Proj"),
            k_proj: create_buffer(buffer_size, "K Proj"),
            v_proj: create_buffer(buffer_size, "V Proj"),
            proj_biased: create_buffer(buffer_size, "Proj Biased"),
            q_permuted: create_buffer(buffer_size, "Q Permuted"),
            k_permuted_t: create_buffer(buffer_size, "K Permuted T"),
            v_permuted: create_buffer(buffer_size, "V Permuted"),
            scores: create_buffer(scores_size, "Attention Scores"),
            context_vectors: create_buffer(buffer_size, "Context Vectors"),
            ffn_intermediate: create_buffer(
                ffn_intermediate_size,
                "FFN Intermediate (in attention)",
            ),
        };

        let ffn = FFNTempBuffers {
            intermediate: create_buffer(ffn_intermediate_size, "FFN Intermediate"),
        };

        Self { attention, ffn }
    }
}

pub struct GpuTransformerLayer {
    pub attention_weights: AttentionWeights,
    pub ffn_weights: FFNWeights,
}

/// The reusable orchestrator for the GPU encoder pipeline.
pub struct GpuTransformerPipeline {
    context: Arc<WgpuContext>,
    attention_pipelines: AttentionPipelines,
    ffn_pipelines: FFNPipelines,
    add_pipeline: Arc<ComputePipeline>,
    layer_norm_pipeline: Arc<ComputePipeline>,
}

impl GpuTransformerPipeline {
    pub fn new(context: Arc<WgpuContext>) -> Result<Self> {
        Ok(Self {
            context: context.clone(),
            attention_pipelines: AttentionPipelines::new(&context),
            ffn_pipelines: FFNPipelines::new(&context),
            add_pipeline: Arc::new(compile_add_pipeline(&context)),
            layer_norm_pipeline: Arc::new(compile_layer_norm_pipeline(&context)),
        })
    }

    fn create_aligned_buffer(
        &self,
        device: &wgpu::Device,
        data: &[f32],
        label: &str,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let mut aligned_data = data.to_vec();

        // Ensure size is multiple of 4 (16 bytes)
        let remainder = aligned_data.len() % 4;
        if remainder != 0 {
            aligned_data.resize(aligned_data.len() + (4 - remainder), 0.0);
        }

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&aligned_data),
            usage,
        })
    }

    /// Forward pass with optional KV caching
    pub async fn forward_with_cache(
        &self,
        config: &dyn TransformerConfig,
        initial_embeddings: &Array3<f32>,
        attention_mask: &Array2<f32>,
        final_norm_weights: (&Arc<wgpu::Buffer>, &Arc<wgpu::Buffer>),
        layers: &[GpuTransformerLayer],
        cache: Option<&mut GpuKVCache>,
    ) -> Result<Array3<f32>> {
        // For now, just call forward without cache
        // TODO: Implement cache in GPU shaders
        self.forward(
            config,
            initial_embeddings,
            attention_mask,
            final_norm_weights,
            layers,
        ).await
    }

    /// Executes the full, end-to-end GPU forward pass for a transformer encoder.
    pub async fn forward(
        &self,
        config: &dyn TransformerConfig,
        initial_embeddings: &Array3<f32>,
        attention_mask: &Array2<f32>,
        norm_weights: (&Arc<wgpu::Buffer>, &Arc<wgpu::Buffer>),
        layers: &[GpuTransformerLayer],
    ) -> Result<Array3<f32>> {
        let (batch_size, seq_len, hidden_size) = initial_embeddings.dim();
        let device = &self.context.device;

        let buffer_size = (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>())
            as wgpu::BufferAddress;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        // Create main buffers
        let buffer_a = self.create_aligned_buffer(
            device,
            initial_embeddings.as_standard_layout().as_slice().unwrap(),
            "Pipeline Buffer A",
            usage,
        );

        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pipeline Buffer B"),
            size: buffer_size,
            usage,
            mapped_at_creation: false,
        });

        let residual_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Residual Buffer"),
            size: buffer_size,
            usage,
            mapped_at_creation: false,
        });

        let mask_buffer = self.create_aligned_buffer(
            device,
            attention_mask.as_slice().unwrap(),
            "Attention Mask Buffer",
            wgpu::BufferUsages::STORAGE,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder Pipeline"),
        });

        let temp_buffers = TempBuffers::new(&self.context.device, config, batch_size, seq_len);

        // Initial embedding layer norm (only for post-norm architectures like BERT)
        if !config.is_prenorm() {
            run_gpu_layer_norm(
                &self.context,
                &mut encoder,
                &self.layer_norm_pipeline,
                &buffer_a,
                &buffer_b,
                (batch_size * seq_len) as u32,
                hidden_size as u32,
                config.layer_norm_eps(),
                norm_weights.0,
                norm_weights.1,
            );
        } else {
            // For pre-norm, just copy input to buffer_b
            encoder.copy_buffer_to_buffer(&buffer_a, 0, &buffer_b, 0, buffer_size);
        }

        let mut current_state = &buffer_b;
        let mut intermediate_state = &buffer_a;

        // Process layers
        for layer in layers.iter() {
            let attention_config = AttentionConfig {
                batch_size,
                seq_len,
                num_heads: config.num_attention_heads(),
                head_dim: config.hidden_size() / config.num_attention_heads(),
                hidden_size: config.hidden_size(),
                is_causal: config.is_causal(),
            };

            let ffn_config = FFNConfig {
                batch_size,
                seq_len,
                hidden_size: config.hidden_size(),
                intermediate_size: config.hidden_size() * 4,
            };

            // === ATTENTION BLOCK ===
            encoder.copy_buffer_to_buffer(current_state, 0, &residual_buffer, 0, buffer_size);

            if config.is_prenorm() {
                // PRE-NORM: norm → attention → add
                run_gpu_layer_norm(
                    &self.context,
                    &mut encoder,
                    &self.layer_norm_pipeline,
                    current_state,
                    intermediate_state,
                    (batch_size * seq_len) as u32,
                    hidden_size as u32,
                    config.layer_norm_eps(),
                    &layer.attention_weights.norm_weight,
                    &layer.attention_weights.norm_bias,
                );

                run_attention_block(
                    &self.context,
                    &mut encoder,
                    &self.attention_pipelines,
                    intermediate_state,
                    intermediate_state,
                    &mask_buffer,
                    &attention_config,
                    &layer.attention_weights,
                    &temp_buffers.attention,
                );

                run_gpu_add(
                    &self.context,
                    &mut encoder,
                    &self.add_pipeline,
                    intermediate_state,
                    &residual_buffer,
                    current_state,
                    (buffer_size / 4) as u32,
                );
            } else {
                // POST-NORM: attention → add → norm
                run_attention_block(
                    &self.context,
                    &mut encoder,
                    &self.attention_pipelines,
                    current_state,
                    intermediate_state,
                    &mask_buffer,
                    &attention_config,
                    &layer.attention_weights,
                    &temp_buffers.attention,
                );

                run_gpu_add(
                    &self.context,
                    &mut encoder,
                    &self.add_pipeline,
                    intermediate_state,
                    &residual_buffer,
                    current_state,
                    (buffer_size / 4) as u32,
                );

                run_gpu_layer_norm(
                    &self.context,
                    &mut encoder,
                    &self.layer_norm_pipeline,
                    current_state,
                    intermediate_state,
                    (batch_size * seq_len) as u32,
                    hidden_size as u32,
                    config.layer_norm_eps(),
                    &layer.attention_weights.norm_weight,
                    &layer.attention_weights.norm_bias,
                );

                std::mem::swap(&mut current_state, &mut intermediate_state);
            }

            // === FFN BLOCK ===
            encoder.copy_buffer_to_buffer(current_state, 0, &residual_buffer, 0, buffer_size);

            if config.is_prenorm() {
                // PRE-NORM: norm → ffn → add
                run_gpu_layer_norm(
                    &self.context,
                    &mut encoder,
                    &self.layer_norm_pipeline,
                    current_state,
                    intermediate_state,
                    (batch_size * seq_len) as u32,
                    hidden_size as u32,
                    config.layer_norm_eps(),
                    &layer.ffn_weights.norm_weight,
                    &layer.ffn_weights.norm_bias,
                );

                run_ffn_block(
                    &self.context,
                    &mut encoder,
                    &self.ffn_pipelines,
                    intermediate_state,
                    intermediate_state,
                    &ffn_config,
                    &layer.ffn_weights,
                    &temp_buffers.ffn,
                );

                run_gpu_add(
                    &self.context,
                    &mut encoder,
                    &self.add_pipeline,
                    intermediate_state,
                    &residual_buffer,
                    current_state,
                    (buffer_size / 4) as u32,
                );
            } else {
                // POST-NORM: ffn → add → norm
                run_ffn_block(
                    &self.context,
                    &mut encoder,
                    &self.ffn_pipelines,
                    current_state,
                    intermediate_state,
                    &ffn_config,
                    &layer.ffn_weights,
                    &temp_buffers.ffn,
                );

                run_gpu_add(
                    &self.context,
                    &mut encoder,
                    &self.add_pipeline,
                    intermediate_state,
                    &residual_buffer,
                    current_state,
                    (buffer_size / 4) as u32,
                );

                run_gpu_layer_norm(
                    &self.context,
                    &mut encoder,
                    &self.layer_norm_pipeline,
                    current_state,
                    intermediate_state,
                    (batch_size * seq_len) as u32,
                    hidden_size as u32,
                    config.layer_norm_eps(),
                    &layer.ffn_weights.norm_weight,
                    &layer.ffn_weights.norm_bias,
                );

                std::mem::swap(&mut current_state, &mut intermediate_state);
            }
        }

        // Final layer norm (only for pre-norm architectures like GPT-2)
        if config.is_prenorm() {
            run_gpu_layer_norm(
                &self.context,
                &mut encoder,
                &self.layer_norm_pipeline,
                current_state,
                intermediate_state,
                (batch_size * seq_len) as u32,
                hidden_size as u32,
                config.layer_norm_eps(),
                norm_weights.0,
                norm_weights.1,
            );
            std::mem::swap(&mut current_state, &mut intermediate_state);
        }

        // Submit commands
        self.context.queue.submit(std::iter::once(encoder.finish()));
        self.context
            .device
            .poll(wgpu::PollType::wait_indefinitely());

        // Read back result
        let final_ndarray = read_buffer_3d(
            &self.context,
            current_state,
            (batch_size, seq_len, hidden_size),
        )
        .await?;

        Ok(final_ndarray)
    }
}
