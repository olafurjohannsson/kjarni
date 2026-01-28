use anyhow::Result;
use ndarray::{s, Array1, Array4};

use crate::gpu_ops::blocks::cache::reorder::GpuReorderCache;
use crate::gpu_ops::GpuTensor;
use crate::WgpuContext;

async fn read_gpu_tensor(tensor: &GpuTensor) -> Result<Array4<f32>> {
    let shape = tensor.shape().to_vec();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array4::from_shape_vec(
        (shape[0], shape[1], shape[2], shape[3]),
        data_slice.to_vec(),
    )?)
}

#[tokio::test]
async fn test_gpu_reorder_cache_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let reorder_kernel = GpuReorderCache::new(&context);

    let (num_beams, num_heads, seq_len, head_dim) = (4, 2, 5, 8);

    let source_cpu =
        Array4::from_shape_fn((num_beams, num_heads, seq_len, head_dim), |(b, _, _, _)| {
            (b as f32 + 1.0) * 100.0
        });
    let source_gpu = GpuTensor::from_ndarray(&context, &source_cpu)?;

    let parent_indices_cpu = Array1::from(vec![2u32, 0, 2, 1]);
    let indices_gpu = GpuTensor::from_ndarray(&context, &parent_indices_cpu)?;

    let mut expected_cpu = Array4::zeros(source_cpu.dim());
    for i in 0..num_beams {
        let parent_idx = parent_indices_cpu[i] as usize;
        let mut dest_slice = expected_cpu.slice_mut(s![i, .., .., ..]);
        let src_slice = source_cpu.slice(s![parent_idx, .., .., ..]);
        dest_slice.assign(&src_slice);
    }

    let output_gpu = GpuTensor::uninitialized(
        &context,
        source_cpu.shape().to_vec(),
        source_gpu.dtype(),
        "reorder_dst",
    );

    let mut encoder = context.device.create_command_encoder(&Default::default());
    reorder_kernel.encode(
        &mut encoder,
        &source_gpu,
        &output_gpu,
        &indices_gpu,
        seq_len,
    );
    context.queue.submit(Some(encoder.finish()));

    let actual_gpu_result = read_gpu_tensor(&output_gpu).await?;

    assert_eq!(
        expected_cpu, actual_gpu_result,
        "gpu reorder result does not match cpu ground truth"
    );

    Ok(())
}

#[tokio::test]
async fn test_reorder_at_step_2() -> Result<()> {
    let context = WgpuContext::new().await?;
    let reorder_kernel = GpuReorderCache::new(&context);

    const CURRENT_SEQ_LEN: usize = 2;
    let (num_beams, num_heads, capacity, head_dim) = (4, 16, 142, 64);

    let source_cpu = Array4::from_shape_fn(
        (num_beams, num_heads, capacity, head_dim),
        |(b, _, s, _)| {
            if s < CURRENT_SEQ_LEN {
                (b + 1) as f32
            } else {
                0.0
            }
        },
    );
    let source_gpu = GpuTensor::from_ndarray(&context, &source_cpu)?;

    let parent_indices_cpu = Array1::from(vec![0u32, 0, 1, 0]);
    let indices_gpu = GpuTensor::from_ndarray(&context, &parent_indices_cpu)?;

    let mut expected_cpu = Array4::zeros(source_cpu.dim());
    for new_beam_idx in 0..num_beams {
        let parent_beam_idx = parent_indices_cpu[new_beam_idx] as usize;
        let mut dest_slice = expected_cpu.slice_mut(s![new_beam_idx, .., .., ..]);
        let src_slice = source_cpu.slice(s![parent_beam_idx, .., .., ..]);
        dest_slice.assign(&src_slice);
    }

    let output_gpu = GpuTensor::uninitialized(
        &context,
        source_cpu.shape().to_vec(),
        source_gpu.dtype(),
        "reorder_dst",
    );
    let mut encoder = context.device.create_command_encoder(&Default::default());
    reorder_kernel.encode(
        &mut encoder,
        &source_gpu,
        &output_gpu,
        &indices_gpu,
        CURRENT_SEQ_LEN,
    );
    context.queue.submit(Some(encoder.finish()));

    let actual_gpu_result = read_gpu_tensor(&output_gpu).await?;
    assert_eq!(
        expected_cpu, actual_gpu_result,
        "gpu reorder result does not match cpu ground truth"
    );

    let new_beam_0_history_val = expected_cpu[[0, 0, 0, 0]];
    assert_eq!(new_beam_0_history_val, 1.0);

    let new_beam_2_history_val = expected_cpu[[2, 0, 0, 0]];
    assert_eq!(new_beam_2_history_val, 2.0);

    Ok(())
}