use super::GpuUpdateCache;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;
use anyhow::Result;
use ndarray::{Array, Array3, Array4, Ix4, s};
use std::sync::Arc;

// Helper to get a test context.
async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}

// Helper to read a 4D GPU tensor back to the CPU for comparison.
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
    println!("\n--- Testing GpuUpdateCache Kernel Logic ---");
    let context = get_test_context().await;
    let kernel = GpuUpdateCache::new(&context);

    // 1. ARRANGE
    let (b, s_new, h, d) = (1, 1, 4, 4); // Batch, New Tokens, NumHeads, HeadDim
    let hidden_size = h * d; // 16
    let capacity = 8;
    let position_offset = 3; // We are writing the 4th token (index 3)

    // Create the 3D "new key" tensor with predictable data
    let new_k_cpu = Array3::from_shape_fn((b, s_new, hidden_size), |(_, _, k)| (k + 1) as f32);
    let new_k_gpu = GpuTensor::from_ndarray(&context, &new_k_cpu)?;
    // A dummy tensor for the 'new_v' argument, as we only need to test 'k'
    let dummy_v_gpu = GpuTensor::from_ndarray(&context, &new_k_cpu)?;

    // Create the 4D cache tensor, initialized to zeros
    let cache_shape = vec![b, h, capacity, d];
    let cache_k_gpu = GpuTensor::uninitialized(
        &context,
        cache_shape,
        crate::gpu_ops::DType::F32,
        "Test K Cache",
    );
    let dummy_cache_v_gpu = cache_k_gpu.clone(); // Dummy for the kernel call

    // 2. ACT: Run the kernel to write `new_k_gpu` into `cache_k_gpu` at the offset
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

    // 3. ASSERT
    // Manually construct the expected result on the CPU.
    let mut expected_cpu = Array4::<f32>::zeros((b, h, capacity, d));
    // The kernel should have split the heads of `new_k_cpu` and written them
    // into the slice at `position_offset`.
    for head_idx in 0..h {
        for head_dim_idx in 0..d {
            let original_idx = head_idx * d + head_dim_idx;
            expected_cpu[[0, head_idx, position_offset, head_dim_idx]] = (original_idx + 1) as f32;
        }
    }

    // Read the actual result back from the GPU.
    let result_gpu = read_gpu_tensor_4d(&cache_k_gpu).await?;

    // Compare. Since it's a direct copy, they should be bit-for-bit identical.
    assert_eq!(
        expected_cpu.as_slice(),
        result_gpu.as_slice(),
        "The GpuUpdateCache kernel did not write the data to the correct location."
    );

    println!("✅ GpuUpdateCache kernel test passed!");
    Ok(())
}
