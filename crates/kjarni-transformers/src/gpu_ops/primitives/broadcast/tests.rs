// ============================================================================
// Tests for GpuBroadcast (broadcast.rs)
// ============================================================================

#[cfg(test)]
mod tests {
    
    use crate::gpu::DType;
    use crate::gpu_ops::primitives::broadcast::BroadcastUniforms;
    use crate::{gpu::GpuTensor, gpu_ops::primitives::broadcast::GpuBroadcast};
    use crate::WgpuContext;
    use anyhow::Result;
    use ndarray::Array3;
    use std::sync::Arc;

    async fn get_test_context() -> Arc<WgpuContext> {
        WgpuContext::new().await.unwrap()
    }

    // ========================================================================
    // Construction Tests
    // ========================================================================

    #[tokio::test]
    async fn test_broadcast_new() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;
        let _ = kernel;
        Ok(())
    }

    #[tokio::test]
    async fn test_broadcast_encoder_states_for_beam_search() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        // [1, seq, hidden] -> [4, seq, hidden]
        let src = Array3::from_shape_fn((1, 8, 64), |(_, s, h)| (s * 64 + h) as f32);
        let gpu_src = GpuTensor::from_ndarray(&context, &src)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![4, 8, 64], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 0);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

        // All 4 beams should be identical copies
        for beam in 0..4 {
            assert_eq!(
                result.slice(ndarray::s![beam, .., ..]),
                src.slice(ndarray::s![0, .., ..])
            );
        }
        Ok(())
    }


    // ========================================================================
    // Actual use case: Broadcast encoder states for beam search
    // src: [1, seq, hidden] -> dst: [num_beams, seq, hidden]
    // ========================================================================

    #[tokio::test]
    async fn test_broadcast_encoder_states_4_beams() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        // Simulate encoder output: [1, seq=4, hidden=8]
        let seq = 4;
        let hidden = 8;
        let num_beams = 4;

        let src_data = Array3::from_shape_fn((1, seq, hidden), |(_, s, h)| {
            (s * hidden + h) as f32
        });

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![num_beams, seq, hidden], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 0);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

        // Each beam should have identical copy of encoder output
        for beam in 0..num_beams {
            for s in 0..seq {
                for h in 0..hidden {
                    assert_eq!(
                        result[[beam, s, h]],
                        src_data[[0, s, h]],
                        "Mismatch at beam={}, seq={}, hidden={}",
                        beam, s, h
                    );
                }
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_broadcast_encoder_states_single_beam() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        // Edge case: 1 beam (should just copy)
        let src_data = Array3::from_shape_fn((1, 2, 4), |(_, s, h)| (s * 10 + h) as f32);

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![1, 2, 4], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 0);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

        assert_eq!(result, src_data);

        Ok(())
    }

    #[tokio::test]
    async fn test_broadcast_encoder_states_large() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        // Realistic size: [1, 128, 768] -> [4, 128, 768]
        let seq = 128;
        let hidden = 768;
        let num_beams = 4;

        let src_data = Array3::from_shape_fn((1, seq, hidden), |(_, s, h)| {
            ((s * hidden + h) % 1000) as f32 * 0.001
        });

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![num_beams, seq, hidden], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 0);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

        // Spot check a few values
        for beam in 0..num_beams {
            assert_eq!(result[[beam, 0, 0]], src_data[[0, 0, 0]]);
            assert_eq!(result[[beam, 64, 384]], src_data[[0, 64, 384]]);
            assert_eq!(result[[beam, 127, 767]], src_data[[0, 127, 767]]);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_broadcast_encoder_states_8_beams() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        let seq = 16;
        let hidden = 64;
        let num_beams = 8;

        let src_data = Array3::from_shape_fn((1, seq, hidden), |(_, s, h)| {
            (s as f32) * 100.0 + (h as f32)
        });

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![num_beams, seq, hidden], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 0);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

        for beam in 0..num_beams {
            for s in 0..seq {
                for h in 0..hidden {
                    assert_eq!(result[[beam, s, h]], src_data[[0, s, h]]);
                }
            }
        }

        Ok(())
    }

    // ========================================================================
    // Kernel reuse
    // ========================================================================

    #[tokio::test]
    async fn test_broadcast_kernel_reuse() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuBroadcast::new(&context)?;

        for num_beams in [2, 4, 8] {
            let src_data = Array3::from_shape_fn((1, 4, 8), |(_, s, h)| (s * 8 + h) as f32);

            let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
            let gpu_dst = GpuTensor::zeros(&context, vec![num_beams, 4, 8], DType::F32, "")?;

            let mut encoder = context.device.create_command_encoder(&Default::default());
            kernel.encode(&mut encoder, &gpu_src, &gpu_dst, 0);
            context.queue.submit(std::iter::once(encoder.finish()));

            let result: Array3<f32> = gpu_dst.to_ndarray_3d().await?;

            for beam in 0..num_beams {
                assert_eq!(
                    result.slice(ndarray::s![beam, .., ..]),
                    src_data.slice(ndarray::s![0, .., ..])
                );
            }
        }

        Ok(())
    }

    // ========================================================================
    // Uniforms struct
    // ========================================================================

    #[test]
    fn test_broadcast_uniforms_struct() {
        let uniforms = BroadcastUniforms {
            src_stride: 16,
            dst_stride: 64,
        };

        assert_eq!(uniforms.src_stride, 16);
        assert_eq!(uniforms.dst_stride, 64);

        let bytes = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 8);
    }
}