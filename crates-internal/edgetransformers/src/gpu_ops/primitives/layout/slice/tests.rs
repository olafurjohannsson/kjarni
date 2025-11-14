use crate::gpu_context::WgpuContext;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::gpu_ops::GpuTensor;
use anyhow::Result;
use ndarray::{s, Array, Array4};

use std::sync::Arc;

#[path = "../../../../tests/common.rs"]
mod common;

use common::{read_gpu_tensor_to_vec};
// Helper to read a GpuTensor back to the CPU for comparison.
async fn read_gpu_tensor<D: ndarray::Dimension>(tensor: &GpuTensor) -> Result<Array<f32, D>> {
    let rank = tensor.rank();
    let shape = tensor.shape().to_vec();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(shape, data_slice.to_vec())?.into_dimensionality::<D>()?)
}
/// A dedicated test to verify that the GpuSlice kernel works correctly.
/// It compares the GPU slice output against the ground truth from ndarray's slice.
#[tokio::test]
async fn test_gpu_slice_parity() -> Result<()> {
    let context = Arc::new(WgpuContext::new().await?);
    let slice_kernel = GpuSlice::new(&context);

    // 1. --- SETUP ---
    // Create a large, predictable source tensor on both CPU and GPU.
    let (b, h, s, d) = (2, 4, 8, 16);
    let source_cpu = Array4::from_shape_fn((b, h, s, d), |(i, j, k, l)| {
        (i * 1000 + j * 100 + k * 10 + l) as f32
    });
    let source_gpu = GpuTensor::from_ndarray(&context, &source_cpu)?;

    // 2. --- DEFINE THE SLICE ---
    // Define the slice we want to extract. For example, from the second batch item,
    // the third head, starting at the fourth sequence element.
    let offset = [1, 2, 3, 0];
    let shape = [1, 1, 4, d]; // Take 4 sequence elements.

    // 3. --- CPU GROUND TRUTH ---
    // Use ndarray's slicing to get the correct result.
    let expected_slice_cpu = source_cpu
        .slice(s![
            offset[0]..offset[0] + shape[0],
            offset[1]..offset[1] + shape[1],
            offset[2]..offset[2] + shape[2],
            offset[3]..offset[3] + shape[3]
        ])
        .to_owned();

    // 4. --- GPU EXECUTION ---
    // Use our new kernel-based slice method on GpuTensor.
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let actual_slice_gpu =
        source_gpu.slice(&mut encoder, &slice_kernel, &offset, &shape)?;
    context.queue.submit(Some(encoder.finish()));

    // 5. --- COMPARE RESULTS ---
    // Read the GPU result back to the CPU.
    let actual_slice_cpu: Array4<f32> = read_gpu_tensor(&actual_slice_gpu).await?;

    // For a direct data copy operation like slice, the results should be EXACTLY equal.
    // We don't need a tolerance here.
    assert_eq!(
        expected_slice_cpu, actual_slice_cpu,
        "GPU slice result does not match CPU ground truth."
    );

    println!("✅ GpuSlice passed parity test!");
    Ok(())
}