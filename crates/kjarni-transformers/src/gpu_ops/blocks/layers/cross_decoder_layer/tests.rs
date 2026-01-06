use super::*;
use crate::encoder_decoder::decoder_cross_attn::DecoderCrossAttention;
use crate::encoder_decoder::decoder_cross_attn_layer::CrossDecoderLayer as CpuDecoderLayer;
use crate::encoder_decoder::decoder_self_attn::DecoderSelfAttention;
use crate::feedforward::{FeedForward as CpuFf, LegacyFeedForward as CpuStdFf};
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::{
    GpuFeedForward, GpuFeedForwardStd, GpuFeedForwardWeights, GpuFeedForwardWeightsStd,
    GpuLayerNorm, GpuLayerNormWeights, GpuNormalization, GpuNormalizationWeights,
};
use crate::gpu_ops::{GpuTensor, GpuTensorPool, Kernel};
use crate::normalization::LayerNorm as CpuLayerNorm;
use crate::{Normalization, WgpuContext};
use anyhow::Result;
use ndarray::{Array, Array1, Array2, Array3};
use std::sync::Arc;

use crate::linear_layer::LinearLayer;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn assert_all_close(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32, context: &str) {
    if a.shape() != b.shape() {
        panic!(
            "[{}] Shape mismatch: {:?} vs {:?}",
            context,
            a.shape(),
            b.shape()
        );
    }

    let mut max_abs_diff = 0.0;
    let mut max_rel_diff = 0.0;

    for (a_val, b_val) in a.iter().zip(b.iter()) {
        let abs_diff = (a_val - b_val).abs();
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }

        // The check: absolute difference must be within the combined tolerance
        let tolerance = atol + rtol * b_val.abs();
        if abs_diff > tolerance {
            panic!(
                "[{}] Arrays are not close. Failed at values a={}, b={}. \
                 Absolute difference {} is greater than tolerance {}",
                context, a_val, b_val, abs_diff, tolerance
            );
        }

        if b_val.abs() > 1e-8 {
            // Avoid division by zero
            let rel_diff = abs_diff / b_val.abs();
            if rel_diff > max_rel_diff {
                max_rel_diff = rel_diff;
            }
        }
    }
    println!(
        "[{}] Check passed. Max absolute difference: {:.6e}, Max relative difference: {:.6e}",
        context, max_abs_diff, max_rel_diff
    );
}

/// Creates a mock CPU DecoderCrossAttentionLayer with deterministic weights.
fn create_mock_cpu_layer(
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
) -> CpuDecoderLayer {
    // Use a simple pattern for weights to make them deterministic
    let gen_weight = |shape, scale| Array2::from_shape_fn(shape, |(i, j)| ((i + j) as f32 * scale));
    let gen_bias = |size, val| Array1::from_elem(size, val);

    let self_attn = DecoderSelfAttention::new(
        hidden_size,
        num_heads,
        LinearLayer::from(gen_weight((hidden_size, hidden_size), 0.001)),
        LinearLayer::from(gen_weight((hidden_size, hidden_size), 0.002)),
        LinearLayer::from(gen_weight((hidden_size, hidden_size), 0.003)),
        LinearLayer::from(gen_weight((hidden_size, hidden_size), -0.001)),
    );
    let self_attn_layer_norm = Normalization::LayerNorm(CpuLayerNorm::new(
        gen_bias(hidden_size, 1.0),
        gen_bias(hidden_size, 0.0),
        1e-5,
    ));

    let cross_attn = DecoderCrossAttention::new(
        hidden_size,
        num_heads,
        LinearLayer::from(gen_weight((hidden_size, hidden_size), 0.004)),
        LinearLayer::from(gen_weight((hidden_size, hidden_size), 0.005)),
        LinearLayer::from(gen_weight((hidden_size, hidden_size), 0.006)),
        LinearLayer::from(gen_weight((hidden_size, hidden_size), -0.002)),
    );
    let cross_attn_layer_norm = Normalization::LayerNorm(CpuLayerNorm::new(
        gen_bias(hidden_size, 1.0),
        gen_bias(hidden_size, 0.0),
        1e-5,
    ));

    let feedforward = CpuFf::Legacy(CpuStdFf::new(
        gen_weight((hidden_size, intermediate_size), 0.01),
        gen_bias(intermediate_size, 0.0),
        gen_weight((intermediate_size, hidden_size), -0.01),
        gen_bias(hidden_size, 0.0),
        crate::activations::Activation::Gelu,
    ));
    let ffn_layer_norm = Normalization::LayerNorm(CpuLayerNorm::new(
        gen_bias(hidden_size, 1.0),
        gen_bias(hidden_size, 0.0),
        1e-5,
    ));

    CpuDecoderLayer {
        self_attn,
        self_attn_layer_norm,
        cross_attn,
        cross_attn_layer_norm,
        feedforward,
        ffn_layer_norm,
        pre_norm: false,
    }
}

/// Creates a GPU GpuCrossDecoderLayer from a CPU layer's weights.
fn create_gpu_layer_from_cpu(
    context: &Arc<WgpuContext>,
    cpu_layer: &CpuDecoderLayer, // Assuming this now uses your new DecoderSelfAttention struct
    hidden_size: u32,
    num_heads: u32,
) -> Result<GpuCrossDecoderLayer> {
    // --- HELPER: Extracts Weight and Bias from LinearLayer for GPU ---
    let load_linear = |layer: &LinearLayer| -> Result<(GpuTensor, GpuTensor)> {
        // 1. Weight: Use your existing to_gpu() method which handles BF16->F32 and Transposing
        let w_gpu = layer.to_gpu(context)?;

        // 2. Bias: Handle the Option. If None, create a Zero tensor.
        let b_gpu = if let Some(bias) = &layer.bias {
            GpuTensor::from_ndarray(context, bias)?
        } else {
            // Create a zero-filled bias if the layer doesn't have one
            let out_features = layer.out_features();
            let zeros = ndarray::Array1::<f32>::zeros(out_features);
            GpuTensor::from_ndarray(context, &zeros)?
        };

        Ok((w_gpu, b_gpu))
    };

    // --- 1. Self-Attention ---
    // Note: Accessing .q_proj, .k_proj etc instead of .q_weight
    let (qw, qb) = load_linear(&cpu_layer.self_attn.q_proj)?;
    let (kw, kb) = load_linear(&cpu_layer.self_attn.k_proj)?;
    let (vw, vb) = load_linear(&cpu_layer.self_attn.v_proj)?;
    let (ow, ob) = load_linear(&cpu_layer.self_attn.o_proj)?;

    let self_attn_weights =
        GpuAttentionWeights::new(qw, Some(qb), kw, Some(kb), vw, Some(vb), ow, Some(ob))?;

    // (Assuming you haven't refactored LayerNorm yet, keep this as is)
    let self_attn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
        GpuTensor::from_ndarray(
            context,
            &cpu_layer
                .self_attn_layer_norm
                .as_layer_norm()
                .unwrap()
                .weight,
        )?,
        GpuTensor::from_ndarray(
            context,
            &cpu_layer.self_attn_layer_norm.as_layer_norm().unwrap().bias,
        )?,
    )?);

    // --- 2. Cross-Attention ---
    let (xqw, xqb) = load_linear(&cpu_layer.cross_attn.q_proj)?;
    let (xkw, xkb) = load_linear(&cpu_layer.cross_attn.k_proj)?;
    let (xvw, xvb) = load_linear(&cpu_layer.cross_attn.v_proj)?;
    let (xow, xob) = load_linear(&cpu_layer.cross_attn.o_proj)?;

    let cross_attn_weights = GpuAttentionWeights::new(
        xqw,
        Some(xqb),
        xkw,
        Some(xkb),
        xvw,
        Some(xvb),
        xow,
        Some(xob),
    )?;

    let cross_attn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
        GpuTensor::from_ndarray(
            context,
            &cpu_layer
                .cross_attn_layer_norm
                .as_layer_norm()
                .unwrap()
                .weight,
        )?,
        GpuTensor::from_ndarray(
            context,
            &cpu_layer
                .cross_attn_layer_norm
                .as_layer_norm()
                .unwrap()
                .bias,
        )?,
    )?);

    // --- 3. Feed-Forward ---
    let ff_weights = if let crate::feedforward::FeedForward::Legacy(ff) = &cpu_layer.feedforward {
        // Assuming Standard FFN doesn't use LinearLayer yet (based on your snippet),
        // or if it does, genericize logic.
        // Based on your snippet, keeping the smart constructor:
        let weights_std = GpuFeedForwardWeightsStd::from_ndarrays(
            context,
            &ff.dense1_weight,
            &ff.dense1_bias,
            &ff.dense2_weight,
            &ff.dense2_bias,
        )?;
        GpuFeedForwardWeights::Standard(weights_std)
    } else {
        panic!("Expected standard feedforward layer");
    };

    let ffn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
        GpuTensor::from_ndarray(
            context,
            &cpu_layer.ffn_layer_norm.as_layer_norm().unwrap().weight,
        )?,
        GpuTensor::from_ndarray(
            context,
            &cpu_layer.ffn_layer_norm.as_layer_norm().unwrap().bias,
        )?,
    )?);

    // Assemble the GPU layer (Unchanged)
    Ok(GpuCrossDecoderLayer {
        self_attn: GpuDecoderSelfAttention::new(context, hidden_size, num_heads),
        self_attn_weights,
        self_attn_norm: GpuNormalization::LayerNorm(GpuLayerNorm::new(context, 1e-5)),
        self_attn_norm_weights,
        cross_attn: GpuCrossAttention::new(context, hidden_size, num_heads),
        cross_attn_weights,
        cross_attn_norm: GpuNormalization::LayerNorm(GpuLayerNorm::new(context, 1e-5)),
        cross_attn_norm_weights,
        feedforward: GpuFeedForward::Standard(GpuFeedForwardStd::new(
            context,
            crate::activations::Activation::Gelu,
        )?),
        ff_weights,
        ffn_norm: GpuNormalization::LayerNorm(GpuLayerNorm::new(context, 1e-5)),
        ffn_norm_weights,
        add: crate::gpu_ops::primitives::add::GpuAdd::new(context),
    })
}

#[tokio::test]
async fn test_gpu_cpu_layer_consistency() -> Result<()> {
    // 1. SETUP
    let context = WgpuContext::new().await?;
    let (batch, dec_len, enc_len, hidden, inter, heads) = (1, 1, 93, 1024, 4096, 16);

    // 2. CREATE MODULES with identical weights
    let cpu_layer = create_mock_cpu_layer(hidden, inter, heads);
    let gpu_layer = create_gpu_layer_from_cpu(&context, &cpu_layer, hidden as u32, heads as u32)?;

    // 3. CREATE IDENTICAL RANDOM INPUTS
    let cpu_decoder_hs = Array::random((batch, dec_len, hidden), Uniform::new(-1.0, 1.0));
    let cpu_encoder_hs = Array::random((batch, enc_len, hidden), Uniform::new(-1.0, 1.0));
    let cpu_decoder_mask = Array2::ones((batch, dec_len));
    let cpu_encoder_mask = Array2::ones((batch, enc_len));

    // 4. RUN CPU FORWARD PASS
    // First precompute cross KV on CPU
    let cpu_cross_kv = cpu_layer.precompute_cross_kv(&cpu_encoder_hs)?;

    let (cpu_output, (cpu_k, cpu_v)) = cpu_layer.forward(
        &cpu_decoder_hs,
        &cpu_encoder_hs,
        Some(&cpu_decoder_mask),
        Some(&cpu_encoder_mask),
        None,                // No self-attn cache
        Some(&cpu_cross_kv), // Pass precomputed cross KV
        None,
    )?;

    // 5. RUN GPU FORWARD PASS
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(context.clone());

    let gpu_decoder_hs = GpuTensor::from_ndarray(&context, &cpu_decoder_hs)?;
    let gpu_encoder_hs = GpuTensor::from_ndarray(&context, &cpu_encoder_hs)?;
    let gpu_decoder_mask = GpuTensor::from_ndarray(&context, &cpu_decoder_mask)?;
    let gpu_encoder_mask = GpuTensor::from_ndarray(&context, &cpu_encoder_mask)?;

    // Precompute cross KV on GPU (matches what we do in real inference)
    let gpu_cross_kv = gpu_layer.precompute_cross_kv(&mut encoder, &gpu_encoder_hs, &mut pool);

    // Forward pass with precomputed cross KV
    let (gpu_output_t, gpu_k_t, gpu_v_t) = gpu_layer.forward(
        &mut encoder,
        &gpu_decoder_hs,
        &gpu_cross_kv, // Required: precomputed cross KV
        &gpu_decoder_mask,
        Some(&gpu_encoder_mask), // Optional encoder mask
        None,                    // No self-attn cache
        0,                       // cache_len = 0
        &mut pool,
    )?;

    // Submit and sync
    context.queue.submit(Some(encoder.finish()));
    pool.next_frame();

    let gpu_output = gpu_output_t.to_ndarray_3d().await?;
    let gpu_k = gpu_k_t.to_ndarray_3d().await?;
    let gpu_v = gpu_v_t.to_ndarray_3d().await?;

    // 6. COMPARE RESULTS
    let rtol = 1e-3;
    let atol = 1e-4;
    assert_all_close(&cpu_output, &gpu_output, rtol, atol, "Final Output");
    assert_all_close(&cpu_k, &gpu_k, rtol, atol, "New K Value");
    assert_all_close(&cpu_v, &gpu_v, rtol, atol, "New V Value");

    println!("✅ GPU and CPU decoder layer outputs are consistent!");
    Ok(())
}
#[tokio::test]
async fn test_layer_subcomponent_parity() -> Result<()> {
    // 1. SETUP
    let context = WgpuContext::new().await?;
    let (batch, dec_len, enc_len, hidden, inter, heads) = (1, 1, 93, 1024, 4096, 16);

    // 2. CREATE MODULES (Identical Weights)
    let cpu_layer = create_mock_cpu_layer(hidden, inter, heads);
    let gpu_layer = create_gpu_layer_from_cpu(&context, &cpu_layer, hidden as u32, heads as u32)?;

    // 3. CREATE INPUTS
    // Use random inputs to ensure we aren't hitting "lucky" zeros
    let cpu_hidden = Array::random((batch, dec_len, hidden), Uniform::new(-1.0, 1.0));
    let cpu_encoder_hs = Array::random((batch, enc_len, hidden), Uniform::new(-1.0, 1.0));

    // Masks (Standard generation case: 1x1 decoder mask, 1x93 encoder mask)
    let cpu_dec_mask = Array2::ones((batch, dec_len));
    let cpu_enc_mask = Array2::ones((batch, enc_len));

    // Upload to GPU
    let gpu_hidden = GpuTensor::from_ndarray(&context, &cpu_hidden)?;
    let gpu_encoder_hs = GpuTensor::from_ndarray(&context, &cpu_encoder_hs)?;
    let gpu_dec_mask = GpuTensor::from_ndarray(&context, &cpu_dec_mask)?;
    let gpu_enc_mask = GpuTensor::from_ndarray(&context, &cpu_enc_mask)?;

    // Resources
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(context.clone());

    // ========================================================================
    // STEP 1: SELF-ATTENTION BLOCK PARITY
    // ========================================================================
    println!("--- Testing Self-Attention Block ---");

    let (attn_out, new_k, new_v) =
        cpu_layer
            .self_attn
            .forward(&cpu_hidden, Some(&cpu_dec_mask), None, None)?;
    let hidden_states_after_add = &cpu_hidden + &attn_out;
    let final_output = cpu_layer
        .self_attn_layer_norm
        .forward(&hidden_states_after_add);
    let (cpu_sa_out, (cpu_k, cpu_v)) = (final_output, (new_k, new_v));

    let o = gpu_layer.self_attn.forward(
        &mut encoder,
        &gpu_hidden,
        &gpu_layer.self_attn_weights,
        &gpu_dec_mask,
        None, // No cache
        0,    // Cache len
        &mut pool,
    )?;
    //(gpu_sa_attn_out, gpu_k, gpu_v)
    let gpu_sa_attn_out = o.hidden_states;
    let gpu_k = o.new_k;
    let gpu_v = o.new_v;

    // 2. Add (Residual)
    let gpu_sa_add = pool.get(gpu_hidden.shape().to_vec());
    gpu_layer
        .add
        .encode(&mut encoder, &[&gpu_hidden, &gpu_sa_attn_out], &gpu_sa_add);

    // 3. Norm
    let gpu_sa_out = pool.get(gpu_sa_add.shape().to_vec());
    gpu_layer.self_attn_norm.encode(
        &mut encoder,
        &gpu_layer.self_attn_norm_weights,
        &gpu_sa_add,
        &gpu_sa_out,
    );

    // Submit & Compare Step 1
    context.queue.submit(Some(encoder.finish()));
    let gpu_sa_out_cpu = gpu_sa_out.to_ndarray_3d().await?;

    assert_all_close(
        &cpu_sa_out,
        &gpu_sa_out_cpu,
        1e-3,
        1e-4,
        "Self-Attention Block",
    );
    println!("✅ Self-Attention Matches");

    // ========================================================================
    // STEP 2: CROSS-ATTENTION BLOCK PARITY
    // ========================================================================
    println!("--- Testing Cross-Attention Block ---");

    // IMPORTANT: Use the *CPU* output from Step 1 as input to Step 2.
    // This isolates the error to this specific block.
    let input_for_step_2 = cpu_sa_out.clone();
    let gpu_input_for_step_2 = GpuTensor::from_ndarray(&context, &input_for_step_2)?;

    encoder = context.device.create_command_encoder(&Default::default());
    pool.next_frame(); // Reset pool for cleanliness

    let (k, v) = cpu_layer
        .cross_attn
        .precompute_encoder_kv(&cpu_encoder_hs)?;
    let cross_attn_output =
        cpu_layer
            .cross_attn
            .forward(&input_for_step_2, &k, &v, Some(&cpu_enc_mask));
    let hidden_states_after_add = &input_for_step_2 + &cross_attn_output?;
    let final_output = cpu_layer
        .cross_attn_layer_norm
        .forward(&hidden_states_after_add);
    let cpu_ca_out = final_output;

    let gpu_cross_kv = gpu_layer.precompute_cross_kv(&mut encoder, &gpu_encoder_hs, &mut pool);
    let gpu_ca_attn_out = gpu_layer.cross_attn.forward(
        &mut encoder,
        &gpu_input_for_step_2, // Query
        &gpu_cross_kv,
        // &gpu_encoder_hs,       // Key/Value
        &gpu_layer.cross_attn_weights,
        Some(&gpu_enc_mask),
        &mut pool,
    );

    // 2. Add (Residual)
    let gpu_ca_add = pool.get(gpu_input_for_step_2.shape().to_vec());
    gpu_layer.add.encode(
        &mut encoder,
        &[&gpu_input_for_step_2, &gpu_ca_attn_out],
        &gpu_ca_add,
    );

    // 3. Norm
    let gpu_ca_out = pool.get(gpu_ca_add.shape().to_vec());
    gpu_layer.cross_attn_norm.encode(
        &mut encoder,
        &gpu_layer.cross_attn_norm_weights,
        &gpu_ca_add,
        &gpu_ca_out,
    );

    // Submit & Compare Step 2
    context.queue.submit(Some(encoder.finish()));
    let gpu_ca_out_cpu = gpu_ca_out.to_ndarray_3d().await?;

    assert_all_close(
        &cpu_ca_out,
        &gpu_ca_out_cpu,
        1e-3,
        1e-4,
        "Cross-Attention Block",
    );
    println!("✅ Cross-Attention Matches");

    // ========================================================================
    // STEP 3: FEED-FORWARD BLOCK PARITY
    // ========================================================================
    println!("--- Testing Feed-Forward Block ---");

    // Use CPU output from Step 2
    let input_for_step_3 = cpu_ca_out.clone();
    let gpu_input_for_step_3 = GpuTensor::from_ndarray(&context, &input_for_step_3)?;

    encoder = context.device.create_command_encoder(&Default::default());
    pool.next_frame();

    // CPU Execution
    let cpu_ffn_out = cpu_layer.feed_forward(&input_for_step_3)?;

    // GPU Execution
    // 1. FFN
    let gpu_ffn_inner_out = pool.get(gpu_input_for_step_3.shape().to_vec());
    gpu_layer.feedforward.encode(
        &mut encoder,
        &gpu_layer.ff_weights,
        &gpu_input_for_step_3,
        &gpu_ffn_inner_out,
        &mut pool,
    );

    // 2. Add (Residual)
    let gpu_ffn_add = pool.get(gpu_input_for_step_3.shape().to_vec());
    gpu_layer.add.encode(
        &mut encoder,
        &[&gpu_input_for_step_3, &gpu_ffn_inner_out],
        &gpu_ffn_add,
    );

    // 3. Norm
    let gpu_ffn_out = pool.get(gpu_ffn_add.shape().to_vec());
    gpu_layer.ffn_norm.encode(
        &mut encoder,
        &gpu_layer.ffn_norm_weights,
        &gpu_ffn_add,
        &gpu_ffn_out,
    );

    // Submit & Compare Step 3
    context.queue.submit(Some(encoder.finish()));
    let gpu_ffn_out_cpu = gpu_ffn_out.to_ndarray_3d().await?;

    assert_all_close(
        &cpu_ffn_out,
        &gpu_ffn_out_cpu,
        1e-3,
        1e-4,
        "Feed-Forward Block",
    );
    println!("✅ Feed-Forward Matches");

    Ok(())
}
