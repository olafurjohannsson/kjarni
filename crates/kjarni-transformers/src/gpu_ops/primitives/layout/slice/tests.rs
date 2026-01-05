use crate::WgpuContext;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use anyhow::Result;
use common::read_gpu_tensor;
use ndarray::{Array4, s};

#[path = "../../../../tests/common.rs"]
mod common;

#[tokio::test]
async fn test_gpu_slice_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let slice_kernel = GpuSlice::new(&context);
    let (b, h, s, d) = (2, 4, 8, 16);
    let source_cpu = Array4::from_shape_fn((b, h, s, d), |(i, j, k, l)| {
        (i * 1000 + j * 100 + k * 10 + l) as f32
    });
    let source_gpu = GpuTensor::from_ndarray(&context, &source_cpu)?;
    let offset = [1, 2, 3, 0];
    let shape = [1, 1, 4, d]; // Take 4 sequence elements.
    let expected_slice_cpu = source_cpu
        .slice(s![
            offset[0]..offset[0] + shape[0],
            offset[1]..offset[1] + shape[1],
            offset[2]..offset[2] + shape[2],
            offset[3]..offset[3] + shape[3]
        ])
        .to_owned();
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let actual_slice_gpu = source_gpu.slice(&mut encoder, &slice_kernel, &offset, &shape)?;
    context.queue.submit(Some(encoder.finish()));
    let actual_slice_cpu: Array4<f32> = read_gpu_tensor(&actual_slice_gpu).await?;
    assert_eq!(
        expected_slice_cpu, actual_slice_cpu,
        "GPU slice result does not match CPU ground truth."
    );
    Ok(())
}
