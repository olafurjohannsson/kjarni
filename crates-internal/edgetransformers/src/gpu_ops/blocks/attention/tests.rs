#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_ops::blocks::attention::{AttentionTempBuffers, AttentionConfig, AttentionPipelines, AttentionWeights, run_attention_block};
    use crate::gpu_ops::primitives::{
        add::{compile_add_pipeline, run_gpu_add},
        add_bias_old::{compile_add_bias_pipeline, run_gpu_add_bias},
        apply_mask::{compile_apply_mask_pipeline, run_gpu_apply_mask},
        layer_norm::{compile_layer_norm_pipeline, run_gpu_layer_norm},
        matmul_old::{compile_bmm_pipeline, compile_matmul_pipeline, run_gpu_bmm, run_gpu_matmul},
        reshape::{
            compile_reshape_pipeline, compile_unreshape_pipeline, run_gpu_reshape,
            run_gpu_unreshape,
        },
        softmax_old::{compile_softmax_pipeline, run_gpu_softmax},
    };
    use crate::gpu_ops::utils::{read_buffer_2d, read_buffer_3d};
    use crate::gpu_pipeline::TempBuffers;
    use crate::{FeedForward, LayerNorm, MultiHeadAttention, gpu_context::WgpuContext};
    use anyhow::Result;
    use ndarray::{Array, Array1, Array2, Array3, s};
    use ndarray_rand::RandomExt;
    use rand_distr::Uniform;
    use std::sync::Arc;
    use wgpu::util::DeviceExt;

    /// A helper function to get a WGPU context for testing.
    /// Panics if a GPU adapter cannot be found.
    async fn get_test_context() -> Arc<WgpuContext> {
        Arc::new(WgpuContext::new().await)
    }
    fn assert_vecs_are_close(vec1: &[f32], vec2: &[f32], tolerance: f32) {
        assert_eq!(vec1.len(), vec2.len(), "Vectors have different lengths");
        for (i, (a, b)) in vec1.iter().zip(vec2.iter()).enumerate() {
            if (a - b).abs() > tolerance {
                panic!(
                    "Mismatch at index {}: cpu = {}, gpu = {}. Difference: {}",
                    i,
                    a,
                    b,
                    (a - b).abs()
                );
            }
        }
    }

    #[tokio::test]
    async fn test_attention_q_projection_correctness() -> Result<()> {
        let context = get_test_context().await;
        let device = &context.device;

        // --- 1. Arrange ---
        let (batch_size, seq_len, hidden_size, num_heads) = (1, 16, 64, 4);
        let head_dim = hidden_size / num_heads;

        // Create random CPU data
        let input_cpu: Array3<f32> =
            Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
        let q_w_cpu: Array2<f32> =
            Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
        let q_b_cpu: Array1<f32> = Array::random(hidden_size, Uniform::new(-0.5, 0.5));

        // --- 2. Act (CPU Path - Ground Truth) ---

        // a) Perform the matrix multiplication: Input @ Wq^T
        // We use `.dot()` on a 2D view of the input.
        let input_2d = input_cpu
            .as_standard_layout()
            .into_shape((seq_len, hidden_size))?;
        let q_proj_cpu = input_2d.dot(&q_w_cpu); // Note: Here we don't transpose, so we compare against a non-transposed GPU weight later

        // b) Add the bias
        let q_biased_cpu = q_proj_cpu + &q_b_cpu;

        // c) Manually reshape [S, H*D] -> [H, S, D] to match the GPU kernel's output layout
        let mut cpu_q_permuted = Array3::<f32>::zeros((num_heads, seq_len, head_dim));
        for s_idx in 0..seq_len {
            for h_idx in 0..num_heads {
                for d_idx in 0..head_dim {
                    let val = q_biased_cpu[[s_idx, h_idx * head_dim + d_idx]];
                    cpu_q_permuted[[h_idx, s_idx, d_idx]] = val;
                }
            }
        }

        // --- 2. Act (GPU Path) ---

        // a) Upload data to GPU
        let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Q Proj Input"),
            contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // IMPORTANT: We upload the NON-TRANSPOSED weight to match the CPU calculation above.
        let q_weight_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Q Proj Weight"),
            contents: bytemuck::cast_slice(q_w_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let q_bias_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Q Proj Bias"),
            contents: bytemuck::cast_slice(q_b_cpu.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buffer_size = (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        // Create temporary buffers for the GPU calculation
        let q_proj_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Q Proj Matmul Out"),
            size: buffer_size,
            usage,
            mapped_at_creation: false,
        });
        let q_biased_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Q Proj Bias Out"),
            size: buffer_size,
            usage,
            mapped_at_creation: false,
        });
        let q_permuted_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Q Proj Permuted Out"),
            size: buffer_size,
            usage,
            mapped_at_creation: false,
        });

        // b) Record the GPU commands
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let matmul_pipeline = compile_matmul_pipeline(&context);
        let add_bias_pipeline = compile_add_bias_pipeline(&context);
        let reshape_pipeline = compile_reshape_pipeline(&context);

        run_gpu_matmul(
            &context,
            &mut encoder,
            &matmul_pipeline,
            &input_gpu,
            &q_weight_gpu,
            &q_proj_gpu,
            (batch_size * seq_len) as u32,
            hidden_size as u32,
            hidden_size as u32,
        );
        run_gpu_add_bias(
            &context,
            &mut encoder,
            &add_bias_pipeline,
            &q_proj_gpu,
            &q_bias_gpu,
            &q_biased_gpu,
            (batch_size * seq_len * hidden_size) as u32,
        );
        run_gpu_reshape(
            &context,
            &mut encoder,
            &reshape_pipeline,
            &q_biased_gpu,
            &q_permuted_gpu,
            batch_size as u32,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
            false,
        );
        context.queue.submit(std::iter::once(encoder.finish()));

        // c) Read back the final result of this sub-pipeline
        let gpu_q_permuted_array =
            read_buffer_3d(&context, &q_permuted_gpu, (num_heads, seq_len, head_dim)).await?;

        // --- 3. Assert ---
        println!("Verifying Attention Q-Projection GPU kernel against CPU implementation...");
        assert_vecs_are_close(
            cpu_q_permuted.as_slice().unwrap(),
            gpu_q_permuted_array.as_slice().unwrap(),
            1e-4,
        );
        println!("✅ Attention Q-Projection and Reshape are correct!");

        Ok(())
    }

    #[tokio::test]
    async fn test_attention_scores_correctness2() -> Result<()> {
        let context = get_test_context().await;
        let device = &context.device;

        // --- 1. Arrange ---
        let (batch_size, seq_len, hidden_size, num_heads) = (1, 8, 32, 4);
        let head_dim = hidden_size / num_heads;

        let input_cpu: Array3<f32> =
            Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
        let q_w_cpu: Array2<f32> =
            Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
        let q_b_cpu: Array1<f32> = Array::random(hidden_size, Uniform::new(-0.5, 0.5));
        let k_w_cpu: Array2<f32> =
            Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
        let k_b_cpu: Array1<f32> = Array::random(hidden_size, Uniform::new(-0.5, 0.5));

        // --- GPU Buffer Setup ---
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let qkv_buffer_size =
            (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64;

        let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scores Test Input"),
            contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
            usage,
        });
        let q_weight_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scores Test Q Weight"),
            contents: bytemuck::cast_slice(q_w_cpu.as_slice().unwrap()),
            usage,
        });
        let q_bias_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scores Test Q Bias"),
            contents: bytemuck::cast_slice(q_b_cpu.as_slice().unwrap()),
            usage,
        });
        let k_weight_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scores Test K Weight"),
            contents: bytemuck::cast_slice(k_w_cpu.as_slice().unwrap()),
            usage,
        });
        let k_bias_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scores Test K Bias"),
            contents: bytemuck::cast_slice(k_b_cpu.as_slice().unwrap()),
            usage,
        });

        let q_proj = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores Q Proj"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        });
        let k_proj = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores K Proj"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        });
        let proj_biased = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores Proj Biased"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        });
        let q_permuted = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores Q Permuted"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        });
        let k_permuted_t = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores K Permuted T"),
            size: qkv_buffer_size,
            usage,
            mapped_at_creation: false,
        });

        let scores_buffer_size =
            (batch_size * num_heads * seq_len * seq_len * std::mem::size_of::<f32>()) as u64;
        let scores_gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scores Output"),
            size: scores_buffer_size,
            usage,
            mapped_at_creation: false,
        });

        // --- 2. Act & Assert, Step-by-Step ---

        // == Step A: Verify Q-Permuted ==
        let input_2d = input_cpu.view().into_shape((seq_len, hidden_size))?;
        let q_biased_cpu = input_2d.dot(&q_w_cpu) + &q_b_cpu;
        let mut cpu_q_permuted = Array3::<f32>::zeros((num_heads, seq_len, head_dim));
        for s_idx in 0..seq_len {
            for h_idx in 0..num_heads {
                for d_idx in 0..head_dim {
                    let val = q_biased_cpu[[s_idx, h_idx * head_dim + d_idx]];
                    cpu_q_permuted[[h_idx, s_idx, d_idx]] = val;
                }
            }
        }

        let mut encoder_q =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let matmul_pipeline = compile_matmul_pipeline(&context);
        let add_bias_pipeline = compile_add_bias_pipeline(&context);
        let reshape_pipeline = compile_reshape_pipeline(&context);

        run_gpu_matmul(
            &context,
            &mut encoder_q,
            &matmul_pipeline,
            &input_gpu,
            &q_weight_gpu,
            &q_proj,
            (seq_len * batch_size) as u32,
            hidden_size as u32,
            hidden_size as u32,
        );
        run_gpu_add_bias(
            &context,
            &mut encoder_q,
            &add_bias_pipeline,
            &q_proj,
            &q_bias_gpu,
            &proj_biased,
            (seq_len * hidden_size * batch_size) as u32,
        );
        run_gpu_reshape(
            &context,
            &mut encoder_q,
            &reshape_pipeline,
            &proj_biased,
            &q_permuted,
            batch_size as u32,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
            false,
        );
        context.queue.submit(std::iter::once(encoder_q.finish()));
        let gpu_q_permuted_array =
            read_buffer_3d(&context, &q_permuted, (num_heads, seq_len, head_dim)).await?;

        println!("Verifying intermediate Q-Permuted...");
        assert_vecs_are_close(
            cpu_q_permuted.as_slice().unwrap(),
            gpu_q_permuted_array.as_slice().unwrap(),
            1e-4,
        );
        println!("✅ Intermediate Q-Permuted is correct.");

        // == Step B: Verify K-Permuted-Transposed ==
        let k_biased_cpu = input_2d.dot(&k_w_cpu) + &k_b_cpu;
        let mut cpu_k_permuted_t = Array3::<f32>::zeros((num_heads, head_dim, seq_len));
        // Manual reshape for K^T: [S, H*D] -> [H, D, S]
        for s_idx in 0..seq_len {
            for h_idx in 0..num_heads {
                for d_idx in 0..head_dim {
                    let val = k_biased_cpu[[s_idx, h_idx * head_dim + d_idx]];
                    cpu_k_permuted_t[[h_idx, d_idx, s_idx]] = val;
                }
            }
        }

        let mut encoder_k =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        run_gpu_matmul(
            &context,
            &mut encoder_k,
            &matmul_pipeline,
            &input_gpu,
            &k_weight_gpu,
            &k_proj,
            (seq_len * batch_size) as u32,
            hidden_size as u32,
            hidden_size as u32,
        );
        run_gpu_add_bias(
            &context,
            &mut encoder_k,
            &add_bias_pipeline,
            &k_proj,
            &k_bias_gpu,
            &proj_biased,
            (seq_len * hidden_size * batch_size) as u32,
        );
        run_gpu_reshape(
            &context,
            &mut encoder_k,
            &reshape_pipeline,
            &proj_biased,
            &k_permuted_t,
            batch_size as u32,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
            true,
        );
        context.queue.submit(std::iter::once(encoder_k.finish()));
        let gpu_k_permuted_t_array =
            read_buffer_3d(&context, &k_permuted_t, (num_heads, head_dim, seq_len)).await?;

        println!("Verifying intermediate K-Permuted-Transposed...");
        assert_vecs_are_close(
            cpu_k_permuted_t.as_slice().unwrap(),
            gpu_k_permuted_t_array.as_slice().unwrap(),
            1e-4,
        );
        println!("✅ Intermediate K-Permuted-Transposed is correct.");

        // == Step C: Verify Final Scores ==
        let mut scores_cpu = Array3::<f32>::zeros((num_heads, seq_len, seq_len));
        for i in 0..num_heads {
            let q_head = cpu_q_permuted.slice(s![i, .., ..]);
            let k_head = cpu_k_permuted_t.slice(s![i, .., ..]);
            scores_cpu
                .slice_mut(s![i, .., ..])
                .assign(&q_head.dot(&k_head));
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        scores_cpu *= scale;
        for i in 0..num_heads {
            for j in 0..seq_len {
                let mut row = scores_cpu.slice_mut(s![i, j, ..]);
                let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                row.mapv_inplace(|x| (x - max_val).exp());
                let sum = row.sum();
                if sum > 0.0 {
                    row /= sum;
                }
            }
        }
        let bmm_pipline = compile_bmm_pipeline(&context);
        let mut encoder_scores =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Use the new batched matmul for Q @ K^T
        run_gpu_bmm(
            &context,
            &mut encoder_scores,
            &bmm_pipline,
            &q_permuted,
            &k_permuted_t,
            &scores_gpu,
            (batch_size * num_heads) as u32, // B
            seq_len as u32,                  // M
            head_dim as u32,                 // K
            seq_len as u32,                  // N
        );
        let softmax_pipeline = compile_softmax_pipeline(&context);
        run_gpu_softmax(
            &context,
            &mut encoder_scores,
            &softmax_pipeline,
            &scores_gpu,
            (batch_size * num_heads * seq_len) as u32,
            seq_len as u32,
            scale,
        );
        context
            .queue
            .submit(std::iter::once(encoder_scores.finish()));
        let gpu_scores_array =
            read_buffer_3d(&context, &scores_gpu, (num_heads, seq_len, seq_len)).await?;

        println!("Verifying final Attention Scores against CPU implementation...");
        assert_vecs_are_close(
            scores_cpu.as_slice().unwrap(),
            gpu_scores_array.as_slice().unwrap(),
            1e-4,
        );
        println!("✅ Attention Score calculation is correct!");

        Ok(())
    }

    #[tokio::test]
async fn test_attention_block_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;

    // --- 1. Arrange ---
    let (batch_size, seq_len, hidden_size, num_heads) = (1, 16, 64, 4);
    let head_dim = hidden_size / num_heads;

    // Create random input data
    let input_cpu: Array3<f32> =
        Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
    let attention_mask_cpu: Array2<f32> = Array2::ones((batch_size, seq_len));

    // Create random weights
    let q_w_cpu = Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
    let q_b_cpu = Array::random(hidden_size, Uniform::new(-0.5, 0.5));
    let k_w_cpu = Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
    let k_b_cpu = Array::random(hidden_size, Uniform::new(-0.5, 0.5));
    let v_w_cpu = Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
    let v_b_cpu = Array::random(hidden_size, Uniform::new(-0.5, 0.5));
    let out_w_cpu = Array::random((hidden_size, hidden_size), Uniform::new(-0.5, 0.5));
    let out_b_cpu = Array::random(hidden_size, Uniform::new(-0.5, 0.5));

    println!("\n=== DEBUGGING ATTENTION BLOCK ===");
    println!(
        "Input stats: min={:.6}, max={:.6}, mean={:.6}",
        input_cpu.iter().cloned().fold(f32::INFINITY, f32::min),
        input_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        input_cpu.mean().unwrap()
    );

    // --- GPU Data Setup ---
    let upload_2d = |tensor: &Array2<f32>| -> Arc<wgpu::Buffer> {
        Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(tensor.as_standard_layout().as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        }))
    };
    let upload_1d = |tensor: &Array1<f32>| -> Arc<wgpu::Buffer> {
        Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(tensor.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        }))
    };

    let gpu_weights = AttentionWeights {
        q_weight: upload_2d(&q_w_cpu.t().to_owned()),
        q_bias: upload_1d(&q_b_cpu),
        k_weight: upload_2d(&k_w_cpu.t().to_owned()),
        k_bias: upload_1d(&k_b_cpu),
        v_weight: upload_2d(&v_w_cpu.t().to_owned()),
        v_bias: upload_1d(&v_b_cpu),
        output_weight: upload_2d(&out_w_cpu.t().to_owned()),
        output_bias: upload_1d(&out_b_cpu),
        norm_weight: upload_1d(&Array1::zeros(hidden_size)), // Dummy norm weights
        norm_bias: upload_1d(&Array1::zeros(hidden_size)),
    };

    let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Attention Input"),
        contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let output_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Attention Output"),
        size: (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // --- 2. Act ---

    // == Ground Truth (CPU Path) ==
    println!("\n--- CPU Computation ---");
    let cpu_attention = MultiHeadAttention::new(
        hidden_size,
        num_heads,
        q_w_cpu.t().to_owned(),
        q_b_cpu.clone(),
        k_w_cpu.t().to_owned(),
        k_b_cpu.clone(),
        v_w_cpu.t().to_owned(),
        v_b_cpu.clone(),
        out_w_cpu.t().to_owned(),
        out_b_cpu.clone(),
    );
    let cpu_result = cpu_attention.forward(&input_cpu, None, Some(&attention_mask_cpu))?;

    println!(
        "CPU output stats: min={:.6}, max={:.6}, mean={:.6}",
        cpu_result.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_result.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        cpu_result.mean().unwrap()
    );
    println!(
        "CPU first 10 values: {:?}",
        &cpu_result.as_slice().unwrap()[..10]
    );

    // == GPU Path ==
    println!("\n--- GPU Computation ---");
    let mask_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Attention Mask"),
        contents: bytemuck::cast_slice(attention_mask_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Attention Test Encoder"),
    });
    
    // Create the dedicated pipelines struct
    let pipelines = AttentionPipelines::new(&context);

    // Create temporary buffers
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let temp_buffers = {
        let qkv_buffer_size = (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64;
        let scores_buffer_size = (batch_size * num_heads * seq_len * seq_len * std::mem::size_of::<f32>()) as u64;
        let ffn_intermediate_size = (batch_size * seq_len * hidden_size * 4 * std::mem::size_of::<f32>()) as u64;

        AttentionTempBuffers {
            q_proj: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Test Q Proj"), size: qkv_buffer_size, usage, mapped_at_creation: false }),
            k_proj: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Test K Proj"), size: qkv_buffer_size, usage, mapped_at_creation: false }),
            v_proj: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Test V Proj"), size: qkv_buffer_size, usage, mapped_at_creation: false }),
            proj_biased: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Test Proj Biased"), size: qkv_buffer_size, usage, mapped_at_creation: false }),
            q_permuted: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Test Q Permuted"), size: qkv_buffer_size, usage, mapped_at_creation: false }),
            k_permuted_t: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Test K Permuted T"), size: qkv_buffer_size, usage, mapped_at_creation: false }),
            v_permuted: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Test V Permuted"), size: qkv_buffer_size, usage, mapped_at_creation: false }),
            scores: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Test Scores"), size: scores_buffer_size, usage, mapped_at_creation: false }),
            context_vectors: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Test Context Vectors"), size: qkv_buffer_size, usage, mapped_at_creation: false }),
            ffn_intermediate: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Test FFN Intermediate"), size: ffn_intermediate_size, usage, mapped_at_creation: false }),
        }
    };

    // Create the config struct
    let attention_config = AttentionConfig {
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        hidden_size,
        is_causal: false,
    };

    // Call the updated function
    run_attention_block(
        &context,
        &mut encoder,
        &pipelines,
        &input_gpu,
        &output_gpu,
        &mask_gpu,
        &attention_config,
        &gpu_weights,
        &temp_buffers,
    );
    context.queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely());

    let gpu_result_array =
        read_buffer_3d(&context, &output_gpu, (batch_size, seq_len, hidden_size)).await?;

    println!(
        "GPU output stats: min={:.6}, max={:.6}, mean={:.6}",
        gpu_result_array.iter().cloned().fold(f32::INFINITY, f32::min),
        gpu_result_array.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        gpu_result_array.mean().unwrap()
    );
    println!(
        "GPU first 10 values: {:?}",
        &gpu_result_array.as_slice().unwrap()[..10]
    );

    // --- 3. Assert with diagnostics ---
    println!("\n--- Comparison ---");
    let cpu_slice = cpu_result.as_slice().unwrap();
    let gpu_slice = gpu_result_array.as_slice().unwrap();

    assert_vecs_are_close(cpu_slice, gpu_slice, 1e-4);
    println!("✅ Attention Block GPU implementation is correct!");

    Ok(())
}
}
