#[cfg(test)]
mod tests {
    
    use crate::gpu::DType;
    use crate::gpu_ops::primitives::layout::clsslice::GpuClsSlice;
    use crate::{gpu::GpuTensor, gpu_ops::primitives::layout::clsslice::ClsSliceUniforms};
    use crate::WgpuContext;
    use anyhow::Result;
    use ndarray::{Array2, Array3};
    use std::sync::Arc;

    async fn get_test_context() -> Arc<WgpuContext> {
       WgpuContext::new().await.unwrap()
    }

    #[tokio::test]
    async fn test_cls_slice_new() {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);
        
        // Should create without error
        let _ = kernel;
    }

    #[tokio::test]
    async fn test_cls_slice_simple() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);

        // [1, 4, 8] -> [1, 8]
        // CLS token is at position 0
        let mut src_data = Array3::<f32>::zeros((1, 4, 8));
        // Fill CLS token (position 0) with distinct values
        for h in 0..8 {
            src_data[[0, 0, h]] = (h + 1) as f32 * 10.0;
        }
        // Fill other positions with different values
        for s in 1..4 {
            for h in 0..8 {
                src_data[[0, s, h]] = -1.0; // Should not appear in output
            }
        }

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![1, 8], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array2<f32> = gpu_dst.to_ndarray_2d().await?;

        for h in 0..8 {
            assert_eq!(result[[0, h]], (h + 1) as f32 * 10.0);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cls_slice_batch_size_1() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);

        // [1, 10, 64] -> [1, 64]
        let src_data = Array3::from_shape_fn((1, 10, 64), |(_, s, h)| {
            if s == 0 {
                h as f32 // CLS token
            } else {
                -999.0 // Other tokens
            }
        });

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![1, 64], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array2<f32> = gpu_dst.to_ndarray_2d().await?;

        for h in 0..64 {
            assert_eq!(result[[0, h]], h as f32);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cls_slice_batched() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);

        // [4, 8, 16] -> [4, 16]
        let src_data = Array3::from_shape_fn((4, 8, 16), |(b, s, h)| {
            if s == 0 {
                (b * 100 + h) as f32 // CLS token unique per batch
            } else {
                -1.0
            }
        });

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![4, 16], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array2<f32> = gpu_dst.to_ndarray_2d().await?;

        for b in 0..4 {
            for h in 0..16 {
                assert_eq!(result[[b, h]], (b * 100 + h) as f32);
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cls_slice_large_batch() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);

        // [32, 16, 64] -> [32, 64]
        let src_data = Array3::from_shape_fn((32, 16, 64), |(b, s, h)| {
            if s == 0 {
                (b * 1000 + h) as f32
            } else {
                0.0
            }
        });

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![32, 64], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array2<f32> = gpu_dst.to_ndarray_2d().await?;

        for b in 0..32 {
            for h in 0..64 {
                assert_eq!(result[[b, h]], (b * 1000 + h) as f32);
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cls_slice_bert_base_hidden() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);

        // BERT-base: [2, 512, 768] -> [2, 768]
        let hidden_size = 768;
        let seq_len = 512;
        let batch_size = 2;

        let src_data = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(b, s, h)| {
            if s == 0 {
                (b * hidden_size + h) as f32
            } else {
                -999.0
            }
        });

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![batch_size, hidden_size], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array2<f32> = gpu_dst.to_ndarray_2d().await?;

        for b in 0..batch_size {
            for h in 0..hidden_size {
                assert_eq!(result[[b, h]], (b * hidden_size + h) as f32);
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cls_slice_minilm_hidden() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);

        // MiniLM: [4, 128, 384] -> [4, 384]
        let hidden_size = 384;
        let seq_len = 128;
        let batch_size = 4;

        let src_data = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(b, s, h)| {
            if s == 0 {
                h as f32 + b as f32 * 0.1
            } else {
                0.0
            }
        });

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![batch_size, hidden_size], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array2<f32> = gpu_dst.to_ndarray_2d().await?;

        for b in 0..batch_size {
            for h in 0..hidden_size {
                let expected = h as f32 + b as f32 * 0.1;
                assert!((result[[b, h]] - expected).abs() < 1e-5);
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cls_slice_seq_len_1() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);

        // Only CLS token: [1, 1, 64] -> [1, 64]
        let src_data = Array3::from_shape_fn((1, 1, 64), |(_, _, h)| h as f32);

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![1, 64], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array2<f32> = gpu_dst.to_ndarray_2d().await?;

        for h in 0..64 {
            assert_eq!(result[[0, h]], h as f32);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cls_slice_hidden_size_1() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);

        // Minimal hidden: [2, 4, 1] -> [2, 1]
        let src_data = Array3::from_shape_fn((2, 4, 1), |(b, s, _)| {
            if s == 0 {
                (b + 1) as f32 * 100.0
            } else {
                -1.0
            }
        });

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![2, 1], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array2<f32> = gpu_dst.to_ndarray_2d().await?;

        assert_eq!(result[[0, 0]], 100.0);
        assert_eq!(result[[1, 0]], 200.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_cls_slice_single_element() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);

        // Absolute minimum: [1, 1, 1] -> [1, 1]
        let src_data = Array3::from_shape_vec((1, 1, 1), vec![42.0])?;

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![1, 1], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array2<f32> = gpu_dst.to_ndarray_2d().await?;

        assert_eq!(result[[0, 0]], 42.0);

        Ok(())
    }
    #[tokio::test]
    async fn test_cls_slice_negative_values() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);

        let src_data = Array3::from_shape_fn((1, 4, 8), |(_, s, h)| {
            if s == 0 {
                -(h as f32) - 1.0
            } else {
                1000.0
            }
        });

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![1, 8], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array2<f32> = gpu_dst.to_ndarray_2d().await?;

        for h in 0..8 {
            assert_eq!(result[[0, h]], -(h as f32) - 1.0);
        }

        Ok(())
    }

  
    fn cpu_cls_slice(src: &Array3<f32>) -> Array2<f32> {
        let batch_size = src.shape()[0];
        let hidden_size = src.shape()[2];
        
        let mut result = Array2::<f32>::zeros((batch_size, hidden_size));
        for b in 0..batch_size {
            for h in 0..hidden_size {
                result[[b, h]] = src[[b, 0, h]];
            }
        }
        result
    }

    #[tokio::test]
    async fn test_cls_slice_matches_cpu() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);

        // Random-ish values
        let src_data = Array3::from_shape_fn((8, 32, 128), |(b, s, h)| {
            ((b * 17 + s * 13 + h * 7) % 1000) as f32 / 10.0 - 50.0
        });

        let expected = cpu_cls_slice(&src_data);

        let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
        let gpu_dst = GpuTensor::zeros(&context, vec![8, 128], DType::F32, "")?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_src, &gpu_dst);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result: Array2<f32> = gpu_dst.to_ndarray_2d().await?;

        assert_eq!(result.shape(), expected.shape());
        for b in 0..8 {
            for h in 0..128 {
                assert!((result[[b, h]] - expected[[b, h]]).abs() < 1e-5);
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cls_slice_kernel_reuse() -> Result<()> {
        let context = get_test_context().await;
        let kernel = GpuClsSlice::new(&context);

        // Run multiple times with different shapes
        let configs = vec![
            (1, 4, 8),
            (2, 16, 32),
            (4, 8, 64),
        ];

        for (batch, seq, hidden) in configs {
            let src_data = Array3::from_shape_fn((batch, seq, hidden), |(b, s, h)| {
                if s == 0 { (b * hidden + h) as f32 } else { -1.0 }
            });

            let gpu_src = GpuTensor::from_ndarray(&context, &src_data)?;
            let gpu_dst = GpuTensor::zeros(&context, vec![batch, hidden], DType::F32, "")?;

            let mut encoder = context.device.create_command_encoder(&Default::default());
            kernel.encode(&mut encoder, &gpu_src, &gpu_dst);
            context.queue.submit(std::iter::once(encoder.finish()));

            let result: Array2<f32> = gpu_dst.to_ndarray_2d().await?;

            for b in 0..batch {
                for h in 0..hidden {
                    assert_eq!(result[[b, h]], (b * hidden + h) as f32);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_cls_slice_uniforms_struct() {
        let uniforms = ClsSliceUniforms {
            batch_size: 4,
            seq_len: 512,
            hidden_size: 768,
            _padding: 0,
        };

        assert_eq!(uniforms.batch_size, 4);
        assert_eq!(uniforms.seq_len, 512);
        assert_eq!(uniforms.hidden_size, 768);

        // Verify Pod/Zeroable works
        let bytes = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 16); // 4 x u32
    }
}