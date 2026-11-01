use super::*;
use crate::WgpuContext;
use crate::activations::Activation;
use crate::feedforward::SwiGluFeedForward as CpuSwiGLUFFN;
use crate::gpu::{GpuTensor, GpuTensorPool};
use crate::linear_layer::LinearLayer;
use crate::tests::common::assert_tensors_are_close_2d as assert_tensors_are_close;
use anyhow::Result;
use ndarray::Array;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

#[ignore = "GPU required"]
#[tokio::test]
async fn test_gpu_swiglu_ffn_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (rows, hidden_size) = (128, 256);
    let intermediate_size = 512;

    let gpu_swiglu = GpuSwiGLUFFN::new(&context)?;
    let gate_cpu = Array::random((intermediate_size, hidden_size), Uniform::new(-1.0, 1.0));
    let up_cpu = Array::random((intermediate_size, hidden_size), Uniform::new(-1.0, 1.0));
    let down_cpu = Array::random((hidden_size, intermediate_size), Uniform::new(-1.0, 1.0));

    let weights_gpu = GpuSwiGLUFFNWeights::new(
        GpuTensor::from_ndarray(&context, &gate_cpu.as_standard_layout().to_owned())?,
        GpuTensor::from_ndarray(&context, &up_cpu.as_standard_layout().to_owned())?,
        GpuTensor::from_ndarray(&context, &down_cpu.as_standard_layout().to_owned())?,
    )?;
    let cpu_swiglu = CpuSwiGLUFFN::new(
        LinearLayer::from(gate_cpu),
        LinearLayer::from(up_cpu),
        LinearLayer::from(down_cpu),
        Activation::SilU,
    );

    let input_cpu = Array::random((rows, hidden_size), Uniform::new(-1.0, 1.0));
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let output_gpu = GpuTensor::uninitialized(
        &context,
        vec![rows, hidden_size],
        crate::tensor::DType::F32,
        "SwiGLU Output",
    );

    let expected_cpu = cpu_swiglu.forward_2d(&input_cpu)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(context.clone());

    gpu_swiglu.encode(
        &mut encoder,
        &weights_gpu,
        &input_gpu,
        &output_gpu,
        &mut pool,
    );
    context.queue.submit(Some(encoder.finish()));

    assert_tensors_are_close(&expected_cpu, &output_gpu, "SwiGLU FFN Output", 1e-2).await;
    Ok(())
}

/// The fused gate/up stage over packed Q4_K weights, against the CPU Q4_K matmul.
///
/// The linear primitive's Q4_K kernels are covered separately; this covers the
/// fused path, which reads two packed weight buffers at once and applies SiLU in
/// the shader. It is the only place the gate and up projections go, so an error
/// here shows up as text that starts plausibly and then drifts.
#[ignore = "GPU required"]
#[tokio::test]
async fn q4k_fused_gate_up_matches_cpu() -> Result<()> {
    use crate::cpu::kernels::q_common::BlockQ4_K;
    use crate::tensor::DType;
    use crate::weights::ModelWeights;

    let path = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf");
    if !path.exists() {
        println!("Skipping: model not present");
        return Ok(());
    }
    let weights = ModelWeights::new(&path)?;
    for name in ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"] {
        if weights.tensor_dtype(name)? != DType::Q4_K {
            println!("Skipping: {name} is not Q4_K");
            return Ok(());
        }
    }

    let context = WgpuContext::new().await?;
    let ffn = GpuSwiGLUFFN::new(&context)?;

    let gate_g =
        GpuTensor::from_model_weights(&context, &weights, "blk.0.ffn_gate.weight", None, "gate")?;
    let up_g =
        GpuTensor::from_model_weights(&context, &weights, "blk.0.ffn_up.weight", None, "up")?;
    let (n, k) = (gate_g.shape()[0], gate_g.shape()[1]);

    let gate_blocks: Vec<BlockQ4_K> = weights.with_raw_tensor("blk.0.ffn_gate.weight", |v| {
        Ok(bytemuck::cast_slice::<u8, BlockQ4_K>(&v.bytes).to_vec())
    })?;
    let up_blocks: Vec<BlockQ4_K> = weights.with_raw_tensor("blk.0.ffn_up.weight", |v| {
        Ok(bytemuck::cast_slice::<u8, BlockQ4_K>(&v.bytes).to_vec())
    })?;

    for rows in [1usize, 2, 4, 10, 11] {
        let input = Array::random((rows, k), Uniform::new(-1.0f32, 1.0));
        let gpu_in = GpuTensor::from_ndarray(&context, &input.as_standard_layout().to_owned())?;
        let gpu_out = GpuTensor::uninitialized(&context, vec![rows, n], DType::F32, "fused_out");

        // `down` is unused by the fused stage but the weights struct needs one.
        let w = GpuSwiGLUFFNWeights::new(gate_g.clone(), up_g.clone(), gate_g.clone())?;
        let mut enc = context.device.create_command_encoder(&Default::default());
        ffn.encode_fused_gate_up(&mut enc, &gpu_in, &w, &gpu_out);
        context.queue.submit(std::iter::once(enc.finish()));

        let (v, shape) = crate::tests::common::read_gpu_tensor_to_vec::<f32>(&gpu_out).await?;
        let gpu = ndarray::Array2::from_shape_vec((shape[0], shape[1]), v)?;

        // The exact f32 kernels, not `matmul_2d_cpu_q4_k`: that path quantises its
        // activations to Q8_K, and the shader does not, so comparing against it would
        // measure int8 loss rather than this kernel's unpacking.
        let exact = |w: &[crate::cpu::kernels::q_common::BlockQ4_K]| {
            let mut out = ndarray::Array2::<f32>::zeros((rows, n));
            for r in 0..rows {
                let row = input.row(r).to_owned();
                unsafe {
                    crate::cpu::kernels::x86::q4_k::matmul_vec_q4_k_avx2(
                        out.row_mut(r).as_slice_mut().unwrap(),
                        row.as_ptr(),
                        w,
                        k,
                    )
                }
            }
            out
        };
        let g = exact(&gate_blocks);
        let u = exact(&up_blocks);
        let expect = g.mapv(|x| x / (1.0 + (-x).exp())) * &u;

        let mut worst = 0.0f32;
        let mut scale = 0.0f32;
        for (a, b) in gpu.iter().zip(expect.iter()) {
            worst = worst.max((a - b).abs());
            scale = scale.max(b.abs());
        }
        println!("  rows={rows}: max abs diff {worst:.3e}, |expected| up to {scale:.3}");
        assert!(
            worst < 1e-2 * scale.max(1.0),
            "fused Q4_K gate/up disagrees with CPU at rows={rows}: {worst:.3e}"
        );
    }
    Ok(())
}

/// The fused kernel when the output rows do not fill its last workgroup. Every tensor in
/// the file has a row count divisible by 4, so these use the first rows of two.
#[ignore = "GPU required"]
#[tokio::test]
async fn q4k_fused_rows_short_of_a_workgroup() -> Result<()> {
    use crate::cpu::kernels::q_common::BlockQ4_K;
    use crate::tensor::DType;
    use crate::tensor::raw_tensor::TensorView;
    use crate::weights::ModelWeights;

    let path = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf");
    if !path.exists() {
        println!("Skipping: model not present");
        return Ok(());
    }
    let weights = ModelWeights::new(&path)?;
    let (gate, up) = ("blk.0.ffn_gate.weight", "blk.0.ffn_up.weight");
    if weights.tensor_dtype(gate)? != DType::Q4_K || weights.tensor_dtype(up)? != DType::Q4_K {
        println!("Skipping: gate or up is not Q4_K");
        return Ok(());
    }
    let read = |name: &str| -> Result<Vec<BlockQ4_K>> {
        weights.with_raw_tensor(name, |v| {
            Ok(bytemuck::cast_slice::<u8, BlockQ4_K>(&v.bytes).to_vec())
        })
    };
    let (gate_blocks, up_blocks) = (read(gate)?, read(up)?);
    let k = 3072;

    let context = WgpuContext::new().await?;
    let ffn = GpuSwiGLUFFN::new(&context)?;
    for (rows, n) in [(1usize, 1usize), (1, 3), (1, 7), (3, 5), (10, 6)] {
        let blocks = n * k / 256;
        let upload = |name: &str, b: &[BlockQ4_K]| {
            let view = TensorView {
                name: name.to_string(),
                bytes: std::borrow::Cow::Borrowed(bytemuck::cast_slice(b)),
                shape: vec![n, k],
                dtype: DType::Q4_K,
            };
            GpuTensor::from_raw(&context, &view, name)
        };
        let gate_g = upload(gate, &gate_blocks[..blocks])?;
        let up_g = upload(up, &up_blocks[..blocks])?;
        let w = GpuSwiGLUFFNWeights::new(gate_g.clone(), up_g, gate_g)?;

        let input = ndarray::Array2::from_shape_fn((rows, k), |(r, c)| {
            ((r * 31 + c * 7) % 97) as f32 / 48.0 - 1.0
        });
        let gpu_in = GpuTensor::from_ndarray(&context, &input)?;
        let out = GpuTensor::uninitialized(&context, vec![rows, n], DType::F32, "o");
        let mut enc = context.device.create_command_encoder(&Default::default());
        ffn.encode_fused_gate_up(&mut enc, &gpu_in, &w, &out);
        context.queue.submit(std::iter::once(enc.finish()));
        let (got, _) = crate::tests::common::read_gpu_tensor_to_vec::<f32>(&out).await?;

        let exact = |b: &[BlockQ4_K]| {
            let mut o = ndarray::Array2::<f32>::zeros((rows, n));
            for r in 0..rows {
                let row = input.row(r).to_owned();
                unsafe {
                    crate::cpu::kernels::x86::q4_k::matmul_vec_q4_k_avx2(
                        o.row_mut(r).as_slice_mut().unwrap(),
                        row.as_ptr(),
                        b,
                        k,
                    )
                }
            }
            o
        };
        let g = exact(&gate_blocks[..blocks]);
        let u = exact(&up_blocks[..blocks]);
        let expected = g.mapv(|x| x / (1.0 + (-x).exp())) * &u;
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "rows={rows} n={n}: {a} against {b}");
        }
    }
    Ok(())
}

/// Achieved bandwidth of the fused gate and up stage on packed Q4_K. Most of a decoder
/// layer's weights go through this kernel, so it sets the decode speed of a GGUF model.
#[ignore = "GPU required"]
#[tokio::test]
async fn q4k_fused_bandwidth() -> Result<()> {
    use crate::tensor::DType;
    use crate::weights::ModelWeights;

    let path = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf");
    if !path.exists() {
        println!("Skipping: model not present");
        return Ok(());
    }
    let weights = ModelWeights::new(&path)?;
    let (gate, up) = ("blk.0.ffn_gate.weight", "blk.0.ffn_up.weight");
    if weights.tensor_dtype(gate)? != DType::Q4_K || weights.tensor_dtype(up)? != DType::Q4_K {
        println!("Skipping: gate or up is not Q4_K");
        return Ok(());
    }

    let context = WgpuContext::new().await?;
    let ffn = GpuSwiGLUFFN::new(&context)?;
    let gate_g = GpuTensor::from_model_weights(&context, &weights, gate, None, "gate")?;
    let up_g = GpuTensor::from_model_weights(&context, &weights, up, None, "up")?;
    let (n, k) = (gate_g.shape()[0], gate_g.shape()[1]);
    let w = GpuSwiGLUFFNWeights::new(gate_g.clone(), up_g.clone(), gate_g.clone())?;

    let input = ndarray::Array2::<f32>::zeros((1, k));
    let gpu_in = GpuTensor::from_ndarray(&context, &input)?;
    let out = GpuTensor::uninitialized(&context, vec![1, n], DType::F32, "o");

    // 50 dispatches per submit so launch overhead is amortised away.
    let reps = 50;
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        let mut enc = context.device.create_command_encoder(&Default::default());
        for _ in 0..reps {
            ffn.encode_fused_gate_up(&mut enc, &gpu_in, &w, &out);
        }
        context.queue.submit(std::iter::once(enc.finish()));
        context.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    let per_call = best / reps as f64;
    let bytes = 2 * n * k * 144 / 256;
    let gbps = bytes as f64 / per_call / 1e9;
    println!(
        "  fused gate+up [{n}x{k}] Q4_K: {:>7.3} ms  {gbps:>6.1} GB/s  ({:.0}% of 288)",
        per_call * 1e3,
        100.0 * gbps / 288.0
    );
    Ok(())
}
