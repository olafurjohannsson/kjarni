// ============================================================================
// Tests for GpuBroadcast (broadcast.rs)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::DType;
    use crate::gpu_ops::primitives::broadcast::BroadcastUniforms;
    use crate::{gpu::GpuTensor, gpu_ops::primitives::broadcast::GpuBroadcast};
    use crate::WgpuContext;
    use anyhow::Result;
    use ndarray::{Array2, Array3};
    use std::sync::Arc;

    async fn get_test_context() -> Arc<WgpuContext> {
       WgpuContext::new().await.unwrap()
    }

    // ========================================================================
    //  Construction Tests
    // ========================================================================

    #[tokio::test]
    async fn test_broadcast_new() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;
        
        // Should create without error
        let _ = kernel;
        Ok(())
    }

    // ========================================================================
    //  Basic Broadcast Tests (2D -> 3D)
    // ========================================================================

    #[tokio::test]
    async fn test_broadcast_2d_to_3d_axis_0() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        // Source: [2, 4] -> Dest: [3, 2, 4] (broadcast along axis 0)
        let src_data = Array2::from_shape_vec((2, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ])?;
        
        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![3, 2, 4], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 0);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

        // Each slice along axis 0 should be the same as src
        for i in 0..3 {
            for j in 0..2 {
                for k in 0..4 {
                    assert_eq!(result[[i, j, k]], src_data[[j, k]]);
                }
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_broadcast_2d_to_3d_axis_1() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        // Source: [2, 4] -> Dest: [2, 3, 4] (broadcast along axis 1)
        let src_data = Array2::from_shape_vec((2, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ])?;
        
        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![2, 3, 4], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 1);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

        // Each slice along axis 1 should repeat the corresponding row
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(result[[i, j, k]], src_data[[i, k]]);
                }
            }
        }

        Ok(())
    }

    // ========================================================================
    //  3D -> 4D Broadcast Tests
    // ========================================================================

    #[tokio::test]
    async fn test_broadcast_3d_to_4d_axis_0() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        // Source: [2, 3, 4] -> Dest: [5, 2, 3, 4] (broadcast along axis 0)
        let src_data = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| {
            (i * 12 + j * 4 + k) as f32
        });
        
        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![5, 2, 3, 4], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 0);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: ndarray::Array4<f32> = gpu_dst.to_ndarray_4d().await?;

        // Each slice along axis 0 should equal src
        for b in 0..5 {
            for i in 0..2 {
                for j in 0..3 {
                    for k in 0..4 {
                        assert_eq!(result[[b, i, j, k]], src_data[[i, j, k]]);
                    }
                }
            }
        }

        Ok(())
    }

    // ========================================================================
    //  Edge Cases
    // ========================================================================

    #[tokio::test]
    async fn test_broadcast_single_element() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        // Source: [1, 1] -> Dest: [4, 1, 1]
        let src_data = Array2::from_shape_vec((1, 1), vec![42.0])?;
        
        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![4, 1, 1], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 0);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

        for i in 0..4 {
            assert_eq!(result[[i, 0, 0]], 42.0);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_broadcast_single_repeat() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        // Source: [2, 4] -> Dest: [1, 2, 4] (only 1 repeat)
        let src_data = Array2::from_shape_vec((2, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ])?;
        
        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![1, 2, 4], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 0);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

        for j in 0..2 {
            for k in 0..4 {
                assert_eq!(result[[0, j, k]], src_data[[j, k]]);
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_broadcast_large_repeat() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        // Source: [1, 64] -> Dest: [32, 1, 64] (many repeats)
        let src_data = Array2::from_shape_fn((1, 64), |(_, j)| j as f32);
        
        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![32, 1, 64], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 0);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

        for i in 0..32 {
            for k in 0..64 {
                assert_eq!(result[[i, 0, k]], k as f32);
            }
        }

        Ok(())
    }

    // ========================================================================
    //  Transformer-like Shapes
    // ========================================================================

    #[tokio::test]
    async fn test_broadcast_attention_mask_shape() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        // Common pattern: [batch, seq] -> [batch, heads, seq, seq]
        // For simplicity, test [2, 8] -> [2, 4, 8] (broadcast heads)
        let src_data = Array2::from_shape_fn((2, 8), |(i, j)| (i * 8 + j) as f32);
        
        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![2, 4, 8], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 1);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

        // Each head should have the same mask
        for b in 0..2 {
            for h in 0..4 {
                for s in 0..8 {
                    assert_eq!(result[[b, h, s]], src_data[[b, s]]);
                }
            }
        }

        Ok(())
    }

    // ========================================================================
    //  Kernel Reuse
    // ========================================================================

    #[tokio::test]
    async fn test_broadcast_kernel_reuse() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        // Use kernel multiple times
        for repeat_count in [2, 4, 8] {
            let src_data = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0])?;
            
            let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
            let gpu_dst = GpuTensor::zeros(&context, vec![repeat_count, 1, 4], DType::F32, "")?;

            let mut encoder = context.device.create_command_encoder(&Default::default());
            kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 0);
            context.queue.submit(std::iter::once(encoder.finish()));

            let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

            for i in 0..repeat_count {
                for k in 0..4 {
                    assert_eq!(result[[i, 0, k]], (k + 1) as f32);
                }
            }
        }

        Ok(())
    }

    // ========================================================================
    //  Uniform Values Verification
    // ========================================================================

    #[test]
    fn test_broadcast_uniforms_struct() {
        let uniforms = BroadcastUniforms {
            src_stride: 16,
            dst_stride: 64,
        };
        
        assert_eq!(uniforms.src_stride, 16);
        assert_eq!(uniforms.dst_stride, 64);
        
        // Verify Pod/Zeroable derive works
        let bytes = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 8); // 2 x u32
    }
}