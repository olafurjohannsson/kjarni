use crate::WgpuContext;
use crate::cpu::normalization::rms_norm::RMSNorm;
use crate::gpu::GpuTensor;
use crate::gpu::normalization::{GpuRMSNorm, GpuRMSNormWeights};
use crate::tests::common::assert_tensors_are_close;
use anyhow::Result;
use ndarray::{Array, Array1, Array3, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

#[tokio::test]
#[ignore = "GPU required"]
async fn test_gpu_rmsnorm_parity_with_cpu_impl() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (b, s, h) = (4, 64, 256); // Batch, SeqLen, HiddenSize
    let eps = 1e-5;
    let gpu_rmsnorm = GpuRMSNorm::new(&context, eps);
    let input_cpu = Array::random((b, s, h), Uniform::new(-5.0, 5.0));
    let gamma_cpu = Array::random(h, Uniform::new(0.5, 1.5));
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let weights_gpu = GpuRMSNormWeights::new(GpuTensor::from_ndarray(&context, &gamma_cpu)?)?;
    let output_gpu =
        GpuTensor::uninitialized(&context, vec![b, s, h], input_gpu.dtype(), "RMSNorm Output");
    let cpu_rmsnorm = RMSNorm::new(gamma_cpu.clone(), eps);
    let expected_cpu = cpu_rmsnorm.forward_3d(&input_cpu);
    let mut encoder = crate::gpu::GpuEncoder::from_encoder(
        context.device.create_command_encoder(&Default::default()),
    );
    gpu_rmsnorm.encode(&mut encoder, &weights_gpu, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));
    assert_tensors_are_close(&expected_cpu, &output_gpu, "RMSNorm Output", 1e-4).await;
    Ok(())
}
#[tokio::test]
#[ignore = "GPU required"]
async fn test_gpu_rmsnorm_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (b, s, h) = (4, 64, 256); // Batch, SeqLen, HiddenSize
    let eps = 1e-5;
    let gpu_rmsnorm = GpuRMSNorm::new(&context, eps);
    let input_cpu = Array3::from_shape_fn((b, s, h), |(i, j, k)| (i + j + k) as f32 * 0.01 - 5.0);
    let gamma_cpu = Array1::from_shape_fn(h, |i| 1.0 + i as f32 * 0.005);

    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let weights_gpu = GpuRMSNormWeights::new(GpuTensor::from_ndarray(&context, &gamma_cpu)?)?;
    let output_gpu =
        GpuTensor::uninitialized(&context, vec![b, s, h], input_gpu.dtype(), "RMSNorm Output");
    let variance = input_cpu
        .mapv(|x| x.powi(2))
        .mean_axis(Axis(2))
        .unwrap()
        .insert_axis(Axis(2));
    let inv_rms = 1.0 / (variance + eps).mapv(f32::sqrt);
    let normalized_cpu = &input_cpu * &inv_rms;
    let expected_cpu = &normalized_cpu * &gamma_cpu;
    let mut encoder = crate::gpu::GpuEncoder::from_encoder(
        context.device.create_command_encoder(&Default::default()),
    );
    gpu_rmsnorm.encode(&mut encoder, &weights_gpu, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));
    assert_tensors_are_close(&expected_cpu, &output_gpu, "RMSNorm Output", 1e-4).await;
    Ok(())
}

/// `encode_add` against an add and then `encode`: the same sum and the same
/// normalised output, bit for bit, since the fused kernel adds as the add kernel
/// does and normalises as `encode` does. Rows narrower and wider than the 256
/// threads, widths that are not a multiple of them, and F32 and BF16 gamma.
#[tokio::test]
#[ignore = "GPU required"]
async fn add_then_norm_fused_matches_add_and_norm() -> Result<()> {
    use crate::gpu::Kernel;
    use crate::gpu_ops::primitives::add::GpuAdd;
    use crate::tensor::DType;
    use crate::tensor::raw_tensor::TensorView;
    use crate::tests::common::read_gpu_tensor_to_vec;
    use std::borrow::Cow;

    let context = WgpuContext::new().await?;
    let norm = GpuRMSNorm::new(&context, 1e-6);
    let add = GpuAdd::new(&context);

    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / 8_388_608.0 - 1.0) * 3.0
    };

    for (rows, h) in [
        (1usize, 3usize),
        (1, 896),
        (3, 2560),
        (7, 257),
        (5, 258),
        (2, 4096),
    ] {
        let a = Array::from_shape_fn((1, rows, h), |_| next());
        let b = Array::from_shape_fn((1, rows, h), |_| next());
        let gamma = Array1::from_shape_fn(h, |_| 1.0 + next() * 0.2);
        let gamma_bf16: Vec<half::bf16> = gamma.iter().map(|&g| half::bf16::from_f32(g)).collect();
        let mut gammas = vec![GpuRMSNormWeights::new(GpuTensor::from_ndarray(
            &context, &gamma,
        )?)?];
        // BF16 tensors pack two values a word, so they take even lengths.
        if h % 2 == 0 {
            gammas.push(GpuRMSNormWeights::new(GpuTensor::from_raw(
                &context,
                &TensorView {
                    name: "gamma_bf16".into(),
                    bytes: Cow::Borrowed(bytemuck::cast_slice(&gamma_bf16)),
                    shape: vec![h],
                    dtype: DType::BF16,
                },
                "gamma_bf16",
            )?)?);
        }
        let a_gpu = GpuTensor::from_ndarray(&context, &a)?;
        let b_gpu = GpuTensor::from_ndarray(&context, &b)?;
        let shape = vec![1, rows, h];
        for weights in &gammas {
            let out = |label| GpuTensor::uninitialized(&context, shape.clone(), DType::F32, label);
            let (sum_ref, normed_ref, sum, normed) = (out("sr"), out("nr"), out("s"), out("n"));

            let mut encoder = crate::gpu::GpuEncoder::from_encoder(
                context.device.create_command_encoder(&Default::default()),
            );
            add.encode(&mut encoder, &[&a_gpu, &b_gpu], &sum_ref);
            norm.encode(&mut encoder, weights, &sum_ref, &normed_ref);
            norm.encode_add(&mut encoder, weights, &a_gpu, &b_gpu, &sum, &normed);
            context.queue.submit(Some(encoder.finish()));

            let read = async |t: &GpuTensor| -> Result<Vec<u32>> {
                let (v, _) = read_gpu_tensor_to_vec::<f32>(t).await?;
                Ok(v.iter().map(|x| x.to_bits()).collect())
            };
            let what = format!("[{rows}x{h}] {:?} gamma", weights.gamma.dtype());
            assert_eq!(read(&sum).await?, read(&sum_ref).await?, "sum {what}");
            assert_eq!(
                read(&normed).await?,
                read(&normed_ref).await?,
                "normed {what}"
            );

            let expected_sum = &a + &b;
            let (s, _) = read_gpu_tensor_to_vec::<f32>(&sum).await?;
            assert_eq!(
                s,
                expected_sum.iter().copied().collect::<Vec<_>>(),
                "sum vs CPU {what}"
            );
        }
    }
    Ok(())
}
