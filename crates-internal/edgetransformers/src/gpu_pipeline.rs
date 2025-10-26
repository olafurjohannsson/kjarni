//! A reusable orchestrator for running a generic transformer encoder pipeline on the GPU.

use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, s};
use rand::seq;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::ComputePipeline;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, CommandEncoder};

use crate::gpu_ops::{
    primitives::{
        add::{compile_add_pipeline, run_gpu_add},
        add_bias::{compile_add_bias_pipeline, run_gpu_add_bias},
        apply_mask::{compile_apply_mask_pipeline, run_gpu_apply_mask},
        layer_norm::{compile_layer_norm_pipeline, run_gpu_layer_norm},
        matmul::{compile_bmm_pipeline, compile_matmul_pipeline, run_gpu_bmm, run_gpu_matmul},
        reshape::{compile_reshape_pipeline, compile_unreshape_pipeline, run_gpu_reshape, run_gpu_unreshape},
        softmax::{compile_softmax_pipeline, run_gpu_softmax}
    },
    
    blocks::ffn::{compile_fc1_pipeline, compile_fc2_pipeline, FFNWeights, FFNConfig, FFNPipelines, FFNTempBuffers, run_ffn_block},

    blocks::attention::{
        AttentionConfig, AttentionPipelines, AttentionTempBuffers, AttentionWeights, run_attention_block
    },

    utils::{
        read_buffer_3d
    }
};
use crate::traits::{TransformerConfig, EncoderArchitecture};
use crate::wgpu_context::WgpuContext;



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
            ffn_intermediate: create_buffer(ffn_intermediate_size, "FFN Intermediate (in attention)"),
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
pub struct GpuEncoderPipeline {
    context: Arc<WgpuContext>,
    attention_pipelines: AttentionPipelines,
    ffn_pipelines: FFNPipelines,
    add_pipeline: Arc<ComputePipeline>,
    layer_norm_pipeline: Arc<ComputePipeline>,
}


impl GpuEncoderPipeline {
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
    /// Executes the full, end-to-end GPU forward pass for a transformer encoder.
    pub async fn forward(
        &self,
        config: &dyn TransformerConfig,
        initial_embeddings: &Array3<f32>,
        attention_mask: &Array2<f32>,
        embedding_norm_weights: (&Buffer, &Buffer),
        layers: &[GpuTransformerLayer],
    ) -> Result<Array3<f32>>
     {
        use std::time::Instant;
        let total_start = Instant::now();

        let (batch_size, seq_len, hidden_size) = initial_embeddings.dim();
        println!("\n=== GPU Forward Pass ===");
        println!(
            "Batch: {}, SeqLen: {}, Hidden: {}",
            batch_size, seq_len, hidden_size
        );

        let device = &self.context.device;

        // Create timestamp query set for detailed profiling
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPU Timing"),
            ty: wgpu::QueryType::Timestamp,
            count: 100, // Enough for many operations
        });

        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Buffer"),
            size: 100 * 8, // 50 timestamps × 8 bytes
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let query_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Readback"),
            size: 100 * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer_size = (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>())
            as wgpu::BufferAddress;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        // === BUFFER CREATION ===
        let buffer_start = Instant::now();
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
        println!("Main buffers created: {:?}", buffer_start.elapsed());

        let encoder_start = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder Pipeline"),
        });
        println!("Command encoder created: {:?}", encoder_start.elapsed());

        // === TEMP BUFFERS ===
        let temp_start = Instant::now();
        let attention_temp_buffers = {
            let qkv_buffer_size = buffer_size as u64;
            let scores_buffer_size = (batch_size
                * config.num_attention_heads()
                * seq_len
                * seq_len
                * std::mem::size_of::<f32>()) as u64;
            let ffn_intermediate_size = (batch_size 
                * seq_len 
                * config.hidden_size() * 4  // intermediate_size
                * std::mem::size_of::<f32>()) as u64;

            AttentionTempBuffers {
                q_proj: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Q Proj"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                k_proj: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("K Proj"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                v_proj: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("V Proj"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                proj_biased: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Proj Biased Temp"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                q_permuted: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Q Permuted"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                k_permuted_t: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("K Permuted T"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                v_permuted: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("V Permuted"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                scores: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Scores"),
                    size: scores_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                context_vectors: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Context Vectors"),
                    size: qkv_buffer_size,
                    usage,
                    mapped_at_creation: false,
                }),
                ffn_intermediate: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("FFN Intermediate"),
                    size: ffn_intermediate_size,
                    usage,
                    mapped_at_creation: false,
                }),
            }
        };
        println!(
            "Temp buffers (9 buffers) created: {:?}",
            temp_start.elapsed()
        );

        // === GPU ENCODING WITH TIMESTAMPS ===
        let encode_start = Instant::now();
        
        let result_buffer = {
            let cache_start = Instant::now();
            println!("Mutex lock acquired: {:?}", cache_start.elapsed());

            let mut query_idx = 0u32;

            // Start timestamp
            encoder.write_timestamp(&query_set, query_idx);
            query_idx += 1;

            // Initial LayerNorm
            let norm_start = Instant::now();
            run_gpu_layer_norm(
                &self.context,
                &mut encoder,
                &self.layer_norm_pipeline,
                &buffer_a,
                &buffer_b,
                (batch_size * seq_len) as u32,
                hidden_size as u32,
                config.layer_norm_eps(),
                embedding_norm_weights.0,
                embedding_norm_weights.1,
            );
            encoder.write_timestamp(&query_set, query_idx);
            query_idx += 1;
            println!("Initial LayerNorm encoded: {:?}", norm_start.elapsed());


            let current_state = &buffer_b;
            let intermediate_state = &buffer_a;

            println!("Processing {} layers...", layers.len());
            for (idx, layer) in layers.iter().enumerate() {
                let layer_start = Instant::now();

                // -- Attention Block --
                encoder.copy_buffer_to_buffer(current_state, 0, &residual_buffer, 0, buffer_size);

                // Create attention config
                let attention_config = AttentionConfig {
                    batch_size,
                    seq_len,
                    num_heads: config.num_attention_heads(),
                    head_dim: config.hidden_size() / config.num_attention_heads(),
                    hidden_size: config.hidden_size(),
                    is_causal: false,  // Encoder = bidirectional
                };


                let temp_buffers = TempBuffers::new(
                    &self.context.device,
                    config as &dyn TransformerConfig,
                    batch_size,
                    seq_len,
                );

                // Timestamp before attention
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;
                
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


                // run_gpu_attention_block(
                //     &self.context,
                //     &mut encoder,
                //     &self.pipeline,
                //     current_state,
                //     intermediate_state,
                //     &layer.attention_weights,
                //     &mask_buffer,
                //     &temp_buffers,
                //     batch_size,
                //     seq_len,
                //     config.hidden_size(),
                //     config.num_attention_heads(),
                //     config.is_causal(),
                // );

                // Timestamp after attention
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                run_gpu_add(
                    &self.context,
                    &mut encoder,
                    // &self.pipeline.get(&Pipeline::Add).unwrap(),
                    &self.add_pipeline,
                    intermediate_state,
                    &residual_buffer,
                    current_state,
                    (buffer_size / 4) as u32,
                );

                // Timestamp after add
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                run_gpu_layer_norm(
                    &self.context,
                    &mut encoder,
                    // &self.pipeline.get(&Pipeline::LayerNorm).unwrap(),
                    &self.layer_norm_pipeline,
                    current_state,
                    intermediate_state,
                    (batch_size * seq_len) as u32,
                    hidden_size as u32,
                    config.layer_norm_eps(),
                    &layer.attention_weights.norm_weight,
                    &layer.attention_weights.norm_bias,
                );

                // Timestamp after norm
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                // -- FFN Block --
                encoder.copy_buffer_to_buffer(
                    intermediate_state,
                    0,
                    &residual_buffer,
                    0,
                    buffer_size,
                );

                // Timestamp before FFN
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;
                let ffn_config = FFNConfig {
                    batch_size,
                    seq_len,
                    hidden_size: config.hidden_size(),
                    intermediate_size: config.hidden_size() * 4,
                };
                run_ffn_block(
                    &self.context,
                    &mut encoder,
                    &self.ffn_pipelines,
                    intermediate_state,
                    current_state,
                    &ffn_config,  // ✅ Correct - FFNConfig struct
                    &layer.ffn_weights,
                    &temp_buffers.ffn,
                );
                // run_gpu_ffn(
                //     &self.context,
                //     &mut encoder,
                //     &self.pipeline.get(&Pipeline::FC1).unwrap(),
                //     &self.pipeline.get(&Pipeline::FC2).unwrap(),
                //     intermediate_state,               // ✅ INPUT - has data from attention
                //     &temp_buffers.ffn_intermediate,  // ✅ INTERMEDIATE - temp buffer
                //     current_state,                    // ✅ OUTPUT
                //     &layer.ffn_weights,
                //     (batch_size * seq_len) as u32,
                //     config.hidden_size() as u32,
                //     (config.hidden_size() * 4) as u32,
                // );

                // Timestamp after FFN
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                run_gpu_add(
                    &self.context,
                    &mut encoder,
                    // &self.pipeline.get(&Pipeline::Add).unwrap(),
                    &self.add_pipeline,
                    current_state,
                    &residual_buffer,
                    intermediate_state,
                    (buffer_size / 4) as u32,
                );

                // Timestamp after add
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                run_gpu_layer_norm(
                    &self.context,
                    &mut encoder,
                    // &self.pipeline.get(&Pipeline::LayerNorm).unwrap(),
                    &self.layer_norm_pipeline,
                    intermediate_state,
                    current_state,
                    (batch_size * seq_len) as u32,
                    hidden_size as u32,
                    config.layer_norm_eps(),
                    &layer.ffn_weights.norm_weight,
                    &layer.ffn_weights.norm_bias,
                );

                // Timestamp after final norm
                encoder.write_timestamp(&query_set, query_idx);
                query_idx += 1;

                println!("  Layer {} encoded: {:?}", idx, layer_start.elapsed());
                    // DEBUG: Read back after first layer

            }

            // Final timestamp
            encoder.write_timestamp(&query_set, query_idx);
            let final_query_idx = query_idx;

            // Resolve all timestamps
            encoder.resolve_query_set(&query_set, 0..final_query_idx + 1, &query_buffer, 0);
            encoder.copy_buffer_to_buffer(
                &query_buffer,
                0,
                &query_readback,
                0,
                (final_query_idx + 1) as u64 * 8,
            );

            let submit_start = Instant::now();
            self.context.queue.submit(std::iter::once(encoder.finish()));
            println!("Commands submitted to GPU: {:?}", submit_start.elapsed());

            current_state.clone()
        };
        println!("Total encoding time: {:?}", encode_start.elapsed());

        // === WAIT FOR GPU ===
        let gpu_wait_start = Instant::now();
        self.context
            .device
            .poll(wgpu::PollType::wait_indefinitely());
        println!("GPU completion wait: {:?}", gpu_wait_start.elapsed());

        // === READ AND ANALYZE TIMESTAMPS ===
        let timestamp_start = Instant::now();
        let slice = query_readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely());

        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        println!("\n=== GPU OPERATION BREAKDOWN ===");
        let total_gpu_time =
            (timestamps[timestamps.len() - 1] - timestamps[0]) as f64 / 1_000_000.0;
        println!("Total GPU time: {:.2}ms", total_gpu_time);

        let initial_layernorm = (timestamps[1] - timestamps[0]) as f64 / 1_000_000.0;
        println!("Initial LayerNorm: {:.2}ms", initial_layernorm);

        let mut idx = 2;
        let mut total_attn = 0.0;
        let mut total_ffn = 0.0;
        let mut total_adds = 0.0;
        let mut total_norms = 0.0;

        for layer_num in 0..layers.len() {
            let attn_time = (timestamps[idx + 1] - timestamps[idx]) as f64 / 1_000_000.0;
            let attn_add = (timestamps[idx + 2] - timestamps[idx + 1]) as f64 / 1_000_000.0;
            let attn_norm = (timestamps[idx + 3] - timestamps[idx + 2]) as f64 / 1_000_000.0;
            let ffn_time = (timestamps[idx + 5] - timestamps[idx + 4]) as f64 / 1_000_000.0;
            let ffn_add = (timestamps[idx + 6] - timestamps[idx + 5]) as f64 / 1_000_000.0;
            let ffn_norm = (timestamps[idx + 7] - timestamps[idx + 6]) as f64 / 1_000_000.0;

            total_attn += attn_time;
            total_ffn += ffn_time;
            total_adds += attn_add + ffn_add;
            total_norms += attn_norm + ffn_norm;

            println!(
                "Layer {}: Attn={:.2}ms, AttnAdd={:.2}ms, AttnNorm={:.2}ms, FFN={:.2}ms, FFNAdd={:.2}ms, FFNNorm={:.2}ms",
                layer_num, attn_time, attn_add, attn_norm, ffn_time, ffn_add, ffn_norm
            );

            idx += 8;
        }

        println!("\n=== TOTALS ===");
        println!(
            "Total Attention: {:.2}ms ({:.1}%)",
            total_attn,
            100.0 * total_attn / total_gpu_time
        );
        println!(
            "Total FFN: {:.2}ms ({:.1}%)",
            total_ffn,
            100.0 * total_ffn / total_gpu_time
        );
        println!(
            "Total Adds: {:.2}ms ({:.1}%)",
            total_adds,
            100.0 * total_adds / total_gpu_time
        );
        println!(
            "Total Norms: {:.2}ms ({:.1}%)",
            total_norms,
            100.0 * total_norms / total_gpu_time
        );

        drop(data);
        query_readback.unmap();
        println!("Timestamp analysis: {:?}", timestamp_start.elapsed());

        // === READBACK RESULT ===
        let read_start = Instant::now();
        let final_ndarray = read_buffer_3d(
            &self.context,
            &result_buffer,
            (batch_size, seq_len, hidden_size),
        )
        .await?;
        println!("Result readback only: {:?}", read_start.elapsed());

        println!("=== TOTAL FORWARD: {:?} ===\n", total_start.elapsed());
        Ok(final_ndarray)
    }
}

// fn run_gpu_attention_block(
//     context: &WgpuContext,
//     encoder: &mut CommandEncoder,
//     pipeline: &HashMap<Pipeline, Arc<ComputePipeline>>,
//     input: &Buffer,
//     output: &Buffer,
//     weights: &GpuAttentionWeights,
//     mask: &Buffer,
//     temp: &AttentionTempBuffers,
//     batch_size: usize,
//     seq_len: usize,
//     hidden_size: usize,
//     num_heads: usize,
//     is_causal: bool,
// ) {
//     // use std::time::Instant;

//     // let device = &context.device;
//     let head_dim = hidden_size / num_heads;

//     let m = (batch_size * seq_len) as u32;
//     let k = hidden_size as u32;

//     // Q = Input * Wq + Bq
//     // Q Path: input -> q_proj -> proj_biased -> q_permuted
//     // let start = Instant::now();
//     run_gpu_matmul(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::MatMul).unwrap(),
//         input,
//         &weights.q_weight,
//         &temp.q_proj,
//         m,
//         k,
//         k,
//     );
//     // println!("      Q matmul: {:?}", start.elapsed());

//     // let start = Instant::now();
//     run_gpu_add_bias(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::AddBias).unwrap(),
//         &temp.q_proj,
//         &weights.q_bias,
//         &temp.proj_biased,
//         m * k,
//     );
//     // println!("      Q bias: {:?}", start.elapsed());

//     // let start = Instant::now();
//     run_gpu_reshape(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::Reshape).unwrap(),
//         &temp.proj_biased,
//         &temp.q_permuted,
//         batch_size as u32,
//         seq_len as u32,
//         num_heads as u32,
//         head_dim as u32,
//         false,
//     );
//     // println!("      Q reshape: {:?}", start.elapsed());

//     // // K Path: input -> k_proj -> proj_biased -> k_permuted_t
//     // let start = Instant::now();
//     run_gpu_matmul(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::MatMul).unwrap(),
//         input,
//         &weights.k_weight,
//         &temp.k_proj,
//         m,
//         k,
//         k,
//     );
//     // println!("      K matmul: {:?}", start.elapsed());

//     // let start = Instant::now();
//     run_gpu_add_bias(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::AddBias).unwrap(),
//         &temp.k_proj,
//         &weights.k_bias,
//         &temp.proj_biased,
//         m * k,
//     );
//     // println!("      K bias: {:?}", start.elapsed());

//     // let start = Instant::now();
//     run_gpu_reshape(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::Reshape).unwrap(),
//         &temp.proj_biased,
//         &temp.k_permuted_t,
//         batch_size as u32,
//         seq_len as u32,
//         num_heads as u32,
//         head_dim as u32,
//         true,
//     );
//     // println!("      K reshape: {:?}", start.elapsed());

//     // // V Path: input -> v_proj -> proj_biased -> v_permuted
//     // let start = Instant::now();
//     run_gpu_matmul(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::MatMul).unwrap(),
//         input,
//         &weights.v_weight,
//         &temp.v_proj,
//         m,
//         k,
//         k,
//     );
//     // println!("      V matmul: {:?}", start.elapsed());

//     // let start = Instant::now();
//     run_gpu_add_bias(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::AddBias).unwrap(),
//         &temp.v_proj,
//         &weights.v_bias,
//         &temp.proj_biased,
//         m * k,
//     );
//     // println!("      V bias: {:?}", start.elapsed());

//     // let start = Instant::now();
//     run_gpu_reshape(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::Reshape).unwrap(),
//         &temp.proj_biased,
//         &temp.v_permuted,
//         batch_size as u32,
//         seq_len as u32,
//         num_heads as u32,
//         head_dim as u32,
//         false,
//     );
//     // println!("      V reshape: {:?}", start.elapsed());

//     // // --- 3. Attention Scores: Q @ K^T ---
//     // let start = Instant::now();
//     run_gpu_bmm(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::BMM).unwrap(),
//         &temp.q_permuted,
//         &temp.k_permuted_t,
//         &temp.scores,
//         (batch_size * num_heads) as u32,
//         seq_len as u32,
//         head_dim as u32,
//         seq_len as u32,
//     );
//     // println!("      Scores BMM (Q@K^T): {:?}", start.elapsed());

//     // // --- 4. Apply Mask ---
//     // let start = Instant::now();
//     run_gpu_apply_mask(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::ApplyMask).unwrap(),
//         &temp.scores,
//         mask,
//         batch_size as u32,
//         num_heads as u32,
//         seq_len as u32,
//         is_causal,
//     );
//     // println!("      Apply mask: {:?}", start.elapsed());

//     // // --- 5. Softmax (in-place on scores) ---
//     // let start = Instant::now();
//     let scale = 1.0 / (head_dim as f32).sqrt();
//     run_gpu_softmax(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::Softmax).unwrap(),
//         &temp.scores,
//         (batch_size * num_heads * seq_len) as u32,
//         seq_len as u32,
//         scale,
//     );
//     // println!("      Softmax: {:?}", start.elapsed());

//     // // --- 6. Apply Scores to V: Scores @ V ---
//     // let start = Instant::now();
//     run_gpu_bmm(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::BMM).unwrap(),
//         &temp.scores,
//         &temp.v_permuted,
//         &temp.context_vectors,
//         (batch_size * num_heads) as u32,
//         seq_len as u32,
//         seq_len as u32,
//         head_dim as u32,
//     );
//     // println!("      Context BMM (Scores@V): {:?}", start.elapsed());

//     // // --- 7. "Un-reshape" and Output Projection ---
//     // let start = Instant::now();
//     run_gpu_unreshape(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::Unreshape).unwrap(),
//         &temp.context_vectors,
//         &temp.proj_biased,
//         batch_size as u32,
//         seq_len as u32,
//         num_heads as u32,
//         head_dim as u32,
//     );
//     // println!("      Unreshape: {:?}", start.elapsed());

//     // // Use `temp.q_proj` as a temporary buffer for the matmul result before the final bias add.
//     // let start = Instant::now();
//     run_gpu_matmul(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::MatMul).unwrap(),
//         &temp.proj_biased,
//         &weights.output_weight,
//         &temp.q_proj,
//         m,
//         k,
//         k,
//     );
//     // println!("      Output matmul: {:?}", start.elapsed());

//     // // The final result is written to the main `output` buffer.
//     // let start = Instant::now();
//     run_gpu_add_bias(
//         context,
//         encoder,
//         pipeline.get(&Pipeline::AddBias).unwrap(),
//         &temp.q_proj,
//         &weights.output_bias,
//         output,
//         m * k,
//     );
//     // println!("      Output bias: {:?}", start.elapsed());
// }



