use super::*;
use crate::WgpuContext;
use anyhow::Result;
use ndarray::{arr2, arr3, Array2, Array3};
use ndarray_rand::RandomExt;
use wgpu::util::DeviceExt;

#[path = "../../../tests/common.rs"]
mod common;


#[tokio::test]
async fn test_gpu_lookup() -> Result<()> {
    let context = WgpuContext::new().await?;
    let embedding_table_cpu = arr2(&[
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0],
    ]);
    let input_ids_cpu: Array2<u32> = arr2(&[[0, 2], [3, 1]]);

    let table_gpu = GpuTensor::from_ndarray(&context, &embedding_table_cpu)?;
    let ids_gpu = GpuTensor::from_ndarray(&context, &input_ids_cpu)?;
    let output_gpu = GpuTensor::zeros(&context, vec![2, 2, 3], crate::gpu::DType::F32, "f32")?;
    let lookup_kernel = GpuLookup::new(&context);
    let mut encoder = context.device.create_command_encoder(&Default::default());
    lookup_kernel.encode(&mut encoder, &table_gpu, &ids_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));


    // 4. Verification
    let output_cpu: Array3<f32> = output_gpu.to_ndarray_3d().await?;

    // Manually compute the expected output
    let expected_output = arr3(&[
        [
            [0.0, 1.0, 2.0], // ID 0
            [6.0, 7.0, 8.0],
        ], // ID 2
        [
            [9.0, 10.0, 11.0], // ID 3
            [3.0, 4.0, 5.0],
        ], // ID 1
    ]);

    assert_eq!(output_cpu, expected_output);
    Ok(())
}
