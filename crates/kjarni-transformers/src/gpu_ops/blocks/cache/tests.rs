use std::sync::Arc;

use anyhow::Result;
use ndarray::{Array, Array3, Array4};

use super::GpuUpdateCache;
use crate::gpu_ops::{DType, GpuTensor};
use crate::WgpuContext;

async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}

async fn read_gpu_tensor_4d(tensor: &GpuTensor) -> Result<Array4<f32>> {
    let shape = tensor.shape();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(
        (shape[0], shape[1], shape[2], shape[3]),
        data_slice.to_vec(),
    )?)
}

#[tokio::test]
async fn test_update_cache_kernel() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuUpdateCache::new(&context);

    let (b, s_new, h, d) = (1, 1, 4, 4);
    let hidden_size = h * d;
    let capacity = 8;
    let position_offset = 3;

    let new_k_cpu = Array3::from_shape_fn((b, s_new, hidden_size), |(_, _, k)| (k + 1) as f32);
    let new_k_gpu = GpuTensor::from_ndarray(&context, &new_k_cpu)?;

    let dummy_v_gpu = GpuTensor::from_ndarray(&context, &new_k_cpu)?;

    let cache_shape = vec![b, h, capacity, d];
    let cache_k_gpu = GpuTensor::uninitialized(&context, cache_shape, DType::F32, "test_k_cache");
    let dummy_cache_v_gpu = cache_k_gpu.clone();

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(
        &mut encoder,
        &new_k_gpu,
        &dummy_v_gpu,
        &cache_k_gpu,
        &dummy_cache_v_gpu,
        position_offset,
    );
    context.queue.submit(Some(encoder.finish()));

    let mut expected_cpu = Array4::<f32>::zeros((b, h, capacity, d));
    for head_idx in 0..h {
        for head_dim_idx in 0..d {
            let original_idx = head_idx * d + head_dim_idx;
            expected_cpu[[0, head_idx, position_offset, head_dim_idx]] = (original_idx + 1) as f32;
        }
    }

    let result_gpu = read_gpu_tensor_4d(&cache_k_gpu).await?;

    assert_eq!(
        expected_cpu.as_slice(),
        result_gpu.as_slice(),
        "cache update kernel did not write data to correct location"
    );

    Ok(())
}