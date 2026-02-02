use super::{Cache, CpuKVCache, GpuKVCache};

#[cfg(test)]
mod cache_tests {
    use super::*;

    use anyhow::Result;
    use ndarray::{s, Array, Array1, Array3, Array4};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    use crate::cache::{CpuBeamKVCache, GpuBeamKVCache};
    use crate::gpu::GpuTensor;
    use crate::WgpuContext;

    async fn read_gpu_tensor<D: ndarray::Dimension>(tensor: &GpuTensor) -> Result<Array<f32, D>> {
        let shape = tensor.shape().to_vec();
        let raw_data = tensor.read_raw_data().await?;
        let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
        Ok(Array::from_shape_vec(shape, data_slice.to_vec())?
            .into_dimensionality::<D>()
            .unwrap())
    }

    fn assert_all_close_3d(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32, context: &str) {
        if a.shape() != b.shape() {
            panic!(
                "[{}] shape mismatch: {:?} vs {:?}",
                context,
                a.shape(),
                b.shape()
            );
        }
        let mut max_diff = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let diff = (a_val - b_val).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            let tolerance = atol + rtol * b_val.abs();
            if diff > tolerance {
                panic!(
                    "[{}] arrays not close. max diff {} > tolerance {} at values a={}, b={}",
                    context, diff, tolerance, a_val, b_val
                );
            }
        }
    }

    #[test]
    fn test_cpu_cache_initialization() {
        let cache = CpuKVCache::new(16, 1, 100, 512);
        assert_eq!(cache.get_seq_length(), 0);
        assert_eq!(cache.layers().len(), 16);

        let _ = cache.as_any();
        let mut mut_cache = cache;
        let _ = mut_cache.as_any_mut();
    }

    #[test]
    fn test_cpu_cache_update_and_grow() {
        let mut cache = CpuKVCache::new(2, 1, 100, 512);
        let k1 = Array3::ones((1, 11, 512));
        let v1 = Array3::ones((1, 11, 512));

        cache.update(0, &k1, &v1).unwrap();
        cache.increment_len(11);

        assert_eq!(cache.get_seq_length(), 11);

        cache.set_seq_length(5);
        assert_eq!(cache.get_seq_length(), 5);

        cache.clear();
        assert_eq!(cache.get_seq_length(), 0);
    }

    #[test]
    fn test_cpu_cache_clone_box() {
        let mut cache = CpuKVCache::new(1, 1, 10, 10);
        let k = Array3::ones((1, 5, 10));
        let v = Array3::ones((1, 5, 10));
        cache.update(0, &k, &v).unwrap();
        cache.increment_len(5);

        let boxed_clone = cache.clone_box();
        assert_eq!(boxed_clone.get_seq_length(), 5);

        let cpu_clone = boxed_clone.as_any().downcast_ref::<CpuKVCache>().unwrap();
        let (ck, _) = cpu_clone.get(0).unwrap();
        assert_eq!(ck[[0, 0, 0]], 1.0);
    }

    #[tokio::test]
    async fn test_gpu_kv_cache_edge_cases() -> Result<()> {
        let context = WgpuContext::new().await?;

        let res = GpuKVCache::new(&context, 1, 1, 1, 1, 0);
        assert!(res.is_err());

        let cache = GpuKVCache::new(&context, 1, 1, 1, 1, 10)?;
        let res = cache.get(99);
        assert!(res.is_none());

        let boxed = cache.clone_box();
        assert_eq!(boxed.get_seq_length(), 0);

        let _ = cache.as_any();

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_kv_cache_overflow_panic() -> Result<()> {
        let context = WgpuContext::new().await?;
        let mut cache = GpuKVCache::new(&context, 1, 1, 1, 1, 10)?;

        cache.increment_len(10);
        assert_eq!(cache.get_seq_length(), 10);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache.increment_len(1);
        }));
        assert!(result.is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_beam_cache_lifecycle() -> Result<()> {
        let context = WgpuContext::new().await?;

        let res = GpuBeamKVCache::new(&context, 1, 1, 1, 1, 0);
        assert!(res.is_err(), "zero capacity should fail");

        let mut cache = GpuBeamKVCache::new(&context, 1, 2, 1, 4, 10)?;

        assert_eq!(cache.get_seq_length(), 0);
        cache.set_seq_length(5);
        assert_eq!(cache.get_seq_length(), 5);
        cache.clear();
        assert_eq!(cache.get_seq_length(), 0);

        let boxed = cache.clone_box();
        let downcasted = boxed.as_any().downcast_ref::<GpuBeamKVCache>();
        assert!(downcasted.is_some());

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_beam_cache_reorder_logic() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (layers, beams, heads, dim, cap) = (1, 4, 1, 1, 10);
        let mut cache = GpuBeamKVCache::new(&context, layers, beams, heads, dim, cap)?;

        let k_data = Array3::from_shape_fn((4, 1, 1), |(b, _, _)| b as f32);
        let v_data = k_data.clone();

        let k_gpu = GpuTensor::from_ndarray(&context, &k_data)?;
        let v_gpu = GpuTensor::from_ndarray(&context, &v_data)?;

        let mut enc = context.device.create_command_encoder(&Default::default());
        cache.update(&mut enc, 0, &k_gpu, &v_gpu)?;

        cache.increment_len(1);

        let indices = Array1::from_vec(vec![3u32, 1, 2, 0]);
        let ind_gpu = GpuTensor::from_ndarray(&context, &indices)?;

        cache.reorder(&mut enc, &ind_gpu);
        context.queue.submit(Some(enc.finish()));

        let (k_out, _) = cache.get_layer_tensors(0).unwrap();
        let k_cpu = k_out.to_ndarray_4d::<f32>().await?;

        assert_eq!(k_cpu[[0, 0, 0, 0]], 3.0);
        assert_eq!(k_cpu[[3, 0, 0, 0]], 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_symmetry_standard() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (layers, batch, max_len, hidden) = (1, 1, 8, 16);
        let heads = 4;
        let head_dim = 4;

        let mut cpu = CpuKVCache::new(layers, batch, max_len, hidden);
        let mut gpu = GpuKVCache::new(&context, layers, batch, heads, head_dim, max_len)?;

        let k1 = Array3::from_elem((1, 2, 16), 1.0f32);
        let v1 = Array3::from_elem((1, 2, 16), 2.0f32);

        cpu.update(0, &k1, &v1)?;
        cpu.increment_len(2);

        let k1_g = GpuTensor::from_ndarray(&context, &k1)?;
        let v1_g = GpuTensor::from_ndarray(&context, &v1)?;

        let mut enc = context.device.create_command_encoder(&Default::default());
        gpu.update(&mut enc, 0, &k1_g, &v1_g, 0)?;
        context.queue.submit(Some(enc.finish()));
        gpu.increment_len(2);

        assert_eq!(cpu.get_seq_length(), gpu.get_seq_length());

        let (ck, _) = cpu.get(0).unwrap();
        let (gk, _) = gpu.get(0).unwrap();
        let gk_cpu = gk.to_ndarray_4d::<f32>().await?;

        assert_eq!(gk_cpu[[0, 0, 0, 0]], 1.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_symmetry() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (num_layers, batch_size, max_len, hidden_size) = (1, 1, 8, 16);
        let layer_idx = 0;

        let mut cpu_cache = CpuKVCache::new(num_layers, batch_size, max_len, hidden_size);
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;
        let mut gpu_cache =
            GpuKVCache::new(&context, num_layers, batch_size, num_heads, head_dim, max_len)?;

        let prompt_len = 3;
        let new_k_cpu_1 =
            Array3::<f32>::from_shape_fn((batch_size, prompt_len, hidden_size), |(b, s, h)| {
                (b * 100 + s * 10 + h) as f32
            });
        let new_v_cpu_1 =
            Array3::<f32>::from_shape_fn((batch_size, prompt_len, hidden_size), |(b, s, h)| {
                (b * 100 + s * 10 + h) as f32 * 10.0
            });

        cpu_cache.update(layer_idx, &new_k_cpu_1, &new_v_cpu_1)?;
        cpu_cache.increment_len(prompt_len);

        let new_k_gpu_1 = GpuTensor::from_ndarray(&context, &new_k_cpu_1)?;
        let new_v_gpu_1 = GpuTensor::from_ndarray(&context, &new_v_cpu_1)?;
        let mut encoder1 = context.device.create_command_encoder(&Default::default());
        gpu_cache.update(
            &mut encoder1,
            layer_idx,
            &new_k_gpu_1,
            &new_v_gpu_1,
            gpu_cache.get_seq_length(),
        )?;
        context.queue.submit(Some(encoder1.finish()));
        gpu_cache.increment_len(prompt_len);

        let gen_len = 1;
        let new_k_cpu_2 =
            Array3::<f32>::from_shape_fn((batch_size, gen_len, hidden_size), |(b, s, h)| {
                (b * 100 + (s + prompt_len) * 10 + h) as f32
            });
        let new_v_cpu_2 =
            Array3::<f32>::from_shape_fn((batch_size, gen_len, hidden_size), |(b, s, h)| {
                (b * 100 + (s + prompt_len) * 10 + h) as f32 * 10.0
            });

        cpu_cache.update(layer_idx, &new_k_cpu_2, &new_v_cpu_2)?;
        cpu_cache.increment_len(gen_len);

        let new_k_gpu_2 = GpuTensor::from_ndarray(&context, &new_k_cpu_2)?;
        let new_v_gpu_2 = GpuTensor::from_ndarray(&context, &new_v_cpu_2)?;
        let mut encoder2 = context.device.create_command_encoder(&Default::default());

        gpu_cache.update(
            &mut encoder2,
            layer_idx,
            &new_k_gpu_2,
            &new_v_gpu_2,
            gpu_cache.get_seq_length(),
        )?;
        context.queue.submit(Some(encoder2.finish()));
        gpu_cache.increment_len(gen_len);

        let final_len = prompt_len + gen_len;
        assert_eq!(cpu_cache.get_seq_length(), final_len);
        assert_eq!(gpu_cache.get_seq_length(), final_len);

        let (cpu_k_final_view, cpu_v_final_view) = cpu_cache.get(layer_idx).unwrap();
        let (gpu_k_full_buffer, gpu_v_full_buffer) = gpu_cache.get(layer_idx).unwrap();

        let gpu_k_full_cpu: Array4<f32> = read_gpu_tensor(&gpu_k_full_buffer).await?;
        let gpu_v_full_cpu: Array4<f32> = read_gpu_tensor(&gpu_v_full_buffer).await?;

        let gpu_k_active_view = gpu_k_full_cpu.slice(s![.., .., 0..final_len, ..]);
        let gpu_v_active_view = gpu_v_full_cpu.slice(s![.., .., 0..final_len, ..]);

        let cpu_k_reshaped = cpu_k_final_view
            .to_owned()
            .into_shape_with_order((batch_size, final_len, num_heads, head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let cpu_v_reshaped = cpu_v_final_view
            .to_owned()
            .into_shape_with_order((batch_size, final_len, num_heads, head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let cpu_k_standard = cpu_k_reshaped.as_standard_layout();
        let gpu_k_standard = gpu_k_active_view.as_standard_layout();

        assert_eq!(
            cpu_k_standard.as_slice(),
            gpu_k_standard.as_slice(),
            "K-cache states do not match"
        );

        let cpu_v_standard = cpu_v_reshaped.as_standard_layout();
        let gpu_v_standard = gpu_v_active_view.as_standard_layout();

        assert_eq!(
            cpu_v_standard.as_slice(),
            gpu_v_standard.as_slice(),
            "V-cache states do not match"
        );

        Ok(())
    }

    #[test]
    fn test_cache_initialization() {
        let cache = CpuKVCache::new(16, 1, 100, 512);

        assert_eq!(cache.get_seq_length(), 0);
        assert_eq!(cache.layers().len(), 16);

        for i in 0..16 {
            let (k, v) = cache.get(i).unwrap();
            assert_eq!(k.shape(), &[1, 0, 512]);
            assert_eq!(v.shape(), &[1, 0, 512]);
        }
    }

    #[test]
    fn test_cache_update_and_grow() {
        let mut cache = CpuKVCache::new(2, 1, 100, 512);

        let k1 = Array3::ones((1, 11, 512));
        let v1 = Array3::ones((1, 11, 512));

        cache.update(0, &k1, &v1).unwrap();
        cache.increment_len(11);

        assert_eq!(cache.get_seq_length(), 11);
        let (cached_k, cached_v) = cache.get(0).unwrap();
        assert_eq!(cached_k.shape(), &[1, 11, 512]);
        assert_eq!(cached_v.shape(), &[1, 11, 512]);

        let k2 = Array3::ones((1, 1, 512)) * 2.0;
        let v2 = Array3::ones((1, 1, 512)) * 2.0;

        cache.update(0, &k2, &v2).unwrap();
        cache.increment_len(1);

        assert_eq!(cache.get_seq_length(), 12);
        let (cached_k, cached_v) = cache.get(0).unwrap();
        assert_eq!(cached_k.shape(), &[1, 12, 512]);
        assert_eq!(cached_v.shape(), &[1, 12, 512]);

        assert_eq!(cached_k[[0, 0, 0]], 1.0);
        assert_eq!(cached_k[[0, 11, 0]], 2.0);
    }

    #[test]
    fn test_cache_multiple_layers() {
        let mut cache = CpuKVCache::new(3, 1, 100, 512);

        for layer in 0..3 {
            let k = Array3::ones((1, 5, 512)) * (layer as f32 + 1.0);
            let v = Array3::ones((1, 5, 512)) * (layer as f32 + 1.0);
            cache.update(layer, &k, &v).unwrap();
        }
        cache.increment_len(5);

        for layer in 0..3 {
            let (k, v) = cache.get(layer).unwrap();
            assert_eq!(k.shape(), &[1, 5, 512]);
            assert_eq!(k[[0, 0, 0]], (layer as f32 + 1.0));
            assert_eq!(v[[0, 0, 0]], (layer as f32 + 1.0));
        }
    }

    #[tokio::test]
    async fn test_gpu_kv_cache_update_and_readback() -> Result<()> {
        let context = WgpuContext::new().await?;
        let num_layers = 2;
        let batch_size = 1;
        let num_heads = 4;
        let head_dim = 32;
        let capacity = 16;

        let mut cache =
            GpuKVCache::new(&context, num_layers, batch_size, num_heads, head_dim, capacity)?;

        let new_seq_len = 3;
        let position_offset = 5;
        let layer_idx_to_test = 1;

        let new_k_cpu = Array::random(
            (batch_size, new_seq_len, num_heads * head_dim),
            Uniform::new(-1.0, 1.0),
        );
        let new_v_cpu = Array::random(
            (batch_size, new_seq_len, num_heads * head_dim),
            Uniform::new(-1.0, 1.0),
        );

        let new_k_gpu = GpuTensor::from_ndarray(&context, &new_k_cpu)?;
        let new_v_gpu = GpuTensor::from_ndarray(&context, &new_v_cpu)?;
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        cache.update(
            &mut encoder,
            layer_idx_to_test,
            &new_k_gpu,
            &new_v_gpu,
            position_offset,
        )?;
        context.queue.submit(Some(encoder.finish()));

        let (k_cache_gpu, _) = cache.get(layer_idx_to_test).unwrap();
        let k_cache_cpu_result: Array4<f32> = k_cache_gpu.to_ndarray_4d().await?;

        let new_k_cpu_reshaped = new_k_cpu
            .into_shape_with_order((batch_size, new_seq_len, num_heads, head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let updated_slice = k_cache_cpu_result.slice(s![
            ..,
            ..,
            position_offset..position_offset + new_seq_len,
            ..
        ]);
        assert_eq!(updated_slice, new_k_cpu_reshaped.as_standard_layout());

        let prefix_slice = k_cache_cpu_result.slice(s![.., .., 0..position_offset, ..]);
        assert!(prefix_slice.iter().all(|&x| x == 0.0));

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_cache_stateful_update_simulation() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (num_layers, batch_size, num_heads, head_dim, capacity) = (1, 1, 2, 4, 10);
        let layer_idx = 0;

        let mut gpu_cache =
            GpuKVCache::new(&context, num_layers, batch_size, num_heads, head_dim, capacity)?;

        {
            let prompt_len = 3;
            let new_k_cpu_1 =
                Array3::from_elem((batch_size, prompt_len, num_heads * head_dim), 1.0);
            let new_k_gpu_1 = GpuTensor::from_ndarray(&context, &new_k_cpu_1)?;
            let new_v_gpu_1 = GpuTensor::from_ndarray(&context, &new_k_cpu_1)?;

            let position_offset = gpu_cache.get_seq_length();
            assert_eq!(position_offset, 0);

            let mut encoder = context.device.create_command_encoder(&Default::default());
            gpu_cache.update(
                &mut encoder,
                layer_idx,
                &new_k_gpu_1,
                &new_v_gpu_1,
                position_offset,
            )?;
            context.queue.submit(Some(encoder.finish()));

            gpu_cache.set_seq_length(position_offset + prompt_len);
            assert_eq!(gpu_cache.get_seq_length(), 3);
        }

        {
            let gen_len = 1;
            let new_k_cpu_2 = Array3::from_elem((batch_size, gen_len, num_heads * head_dim), 99.0);
            let new_k_gpu_2 = GpuTensor::from_ndarray(&context, &new_k_cpu_2)?;
            let new_v_gpu_2 = GpuTensor::from_ndarray(&context, &new_k_cpu_2)?;

            let position_offset = gpu_cache.get_seq_length();
            assert_eq!(position_offset, 3);

            let mut encoder = context.device.create_command_encoder(&Default::default());
            gpu_cache.update(
                &mut encoder,
                layer_idx,
                &new_k_gpu_2,
                &new_v_gpu_2,
                position_offset,
            )?;
            context.queue.submit(Some(encoder.finish()));

            gpu_cache.set_seq_length(position_offset + gen_len);
            assert_eq!(gpu_cache.get_seq_length(), 4);
        }

        let (k_cache_gpu, _) = gpu_cache.get(layer_idx).unwrap();
        let k_cache_cpu: Array4<f32> = read_gpu_tensor(&k_cache_gpu).await?;

        let slice1 = k_cache_cpu.slice(s![0, 0, 0..3, ..]);
        assert!(slice1.iter().all(|&x| x == 1.0));

        let slice2 = k_cache_cpu.slice(s![0, 0, 3..4, ..]);
        assert!(slice2.iter().all(|&x| x == 99.0));

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_reorder_parity() -> Result<()> {
        const NUM_LAYERS: usize = 2;
        const NUM_BEAMS: usize = 4;
        const NUM_HEADS: usize = 8;
        const HEAD_DIM: usize = 64;
        const HIDDEN_SIZE: usize = NUM_HEADS * HEAD_DIM;
        const CAPACITY: usize = 10;
        const NUM_STEPS_TO_POPULATE: usize = 3;

        let context = WgpuContext::new().await?;

        let mut cpu_cache = CpuBeamKVCache::new(NUM_LAYERS, NUM_BEAMS, CAPACITY, HIDDEN_SIZE);
        let mut gpu_cache =
            GpuBeamKVCache::new(&context, NUM_LAYERS, NUM_BEAMS, NUM_HEADS, HEAD_DIM, CAPACITY)?;

        for step in 0..NUM_STEPS_TO_POPULATE {
            let new_k_cpu = Array3::from_shape_fn((NUM_BEAMS, 1, HIDDEN_SIZE), |(b, _, h)| {
                (step as f32 * 1000.0) + (b as f32 * 100.0) + h as f32
            });
            let new_v_cpu = &new_k_cpu + 5000.0;

            for layer in 0..NUM_LAYERS {
                cpu_cache.update(layer, &new_k_cpu, &new_v_cpu)?;
            }
            cpu_cache.increment_len(1);

            let new_k_gpu = GpuTensor::from_ndarray(&context, &new_k_cpu)?;
            let new_v_gpu = GpuTensor::from_ndarray(&context, &new_v_cpu)?;
            let mut encoder = context.device.create_command_encoder(&Default::default());
            for layer in 0..NUM_LAYERS {
                gpu_cache.update(&mut encoder, layer, &new_k_gpu, &new_v_gpu)?;
            }
            context.queue.submit(Some(encoder.finish()));
            gpu_cache.increment_len(1);
        }

        assert_eq!(cpu_cache.get_seq_length(), NUM_STEPS_TO_POPULATE);
        assert_eq!(gpu_cache.get_seq_length(), NUM_STEPS_TO_POPULATE);

        let reorder_indices = vec![2, 0, 2, 1];

        cpu_cache.reorder(&reorder_indices);

        let indices_gpu = GpuTensor::from_ndarray(
            &context,
            &Array1::from_vec(reorder_indices.iter().map(|&i| i as u32).collect()),
        )?;
        let mut encoder = context.device.create_command_encoder(&Default::default());
        gpu_cache.reorder(&mut encoder, &indices_gpu);
        context.queue.submit(Some(encoder.finish()));

        for layer_idx in 0..NUM_LAYERS {
            let (cpu_k, cpu_v) = cpu_cache.get(layer_idx).unwrap();

            let (gpu_k_tensor, gpu_v_tensor) = gpu_cache.get_layer_tensors(layer_idx).unwrap();
            let gpu_k_4d: Array4<f32> = gpu_k_tensor.to_ndarray_4d().await?;
            let gpu_v_4d: Array4<f32> = gpu_v_tensor.to_ndarray_4d().await?;

            let p1 = gpu_k_4d.permuted_axes([0, 2, 1, 3]);
            let p2 = gpu_v_4d.permuted_axes([0, 2, 1, 3]);

            let gpu_k_reshaped =
                p1.as_standard_layout()
                    .into_shape_with_order((NUM_BEAMS, CAPACITY, HIDDEN_SIZE))?;
            let gpu_v_reshaped =
                p2.as_standard_layout()
                    .into_shape_with_order((NUM_BEAMS, CAPACITY, HIDDEN_SIZE))?;

            let active_slice = s![.., 0..NUM_STEPS_TO_POPULATE, ..];
            let cpu_k_active = cpu_k.slice(active_slice);
            let gpu_k_active = gpu_k_reshaped.slice(active_slice);
            let cpu_v_active = cpu_v.slice(active_slice);
            let gpu_v_active = gpu_v_reshaped.slice(active_slice);

            let context_k = format!("layer {} key cache", layer_idx);
            assert_all_close_3d(
                &cpu_k_active.to_owned(),
                &gpu_k_active.to_owned(),
                1e-5,
                1e-5,
                &context_k,
            );

            let context_v = format!("layer {} value cache", layer_idx);
            assert_all_close_3d(
                &cpu_v_active.to_owned(),
                &gpu_v_active.to_owned(),
                1e-5,
                1e-5,
                &context_v,
            );
        }

        Ok(())
    }
}