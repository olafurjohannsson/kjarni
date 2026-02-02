use std::sync::Arc;

use approx::{assert_abs_diff_eq, assert_relative_eq};
use half::bf16;
use ndarray::{Array1, Array2, arr1, arr2};

use crate::{
    linear_layer::{F32MatmulStrategy, LinearData, LinearLayer},
    tensor::{DType, QuantizedMatrix},
    WgpuContext,
};

// =============================================================================
// Helper Functions
// =============================================================================

fn create_f32_layer() -> LinearLayer {
    let weights = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    LinearLayer::new_f32(weights, None)
}

fn make_f32_layer(out: usize, inp: usize) -> LinearLayer {
    let weights = Array2::from_shape_fn((out, inp), |(i, j)| {
        ((i * 17 + j * 13) % 1000) as f32 * 0.001 - 0.5
    });
    LinearLayer::new_f32(weights, None)
}

fn make_f32_layer_with_bias(out: usize, inp: usize) -> LinearLayer {
    let weights = Array2::from_shape_fn((out, inp), |(i, j)| {
        ((i * 17 + j * 13) % 1000) as f32 * 0.001 - 0.5
    });
    let bias = Array1::from_shape_fn(out, |i| (i % 100) as f32 * 0.01);
    LinearLayer::new_f32(weights, Some(bias))
}

fn make_bf16_layer(out: usize, inp: usize) -> LinearLayer {
    let weights = Array2::from_shape_fn((out, inp), |(i, j)| {
        bf16::from_f32(((i * 17 + j * 13) % 1000) as f32 * 0.001 - 0.5)
    });
    LinearLayer::new_bf16(weights, None)
}

fn make_bf16_layer_with_bias(out: usize, inp: usize) -> LinearLayer {
    let weights = Array2::from_shape_fn((out, inp), |(i, j)| {
        bf16::from_f32(((i * 17 + j * 13) % 1000) as f32 * 0.001 - 0.5)
    });
    let bias = Array1::from_shape_fn(out, |i| (i % 100) as f32 * 0.01);
    LinearLayer::new_bf16(weights, Some(bias))
}

fn make_input(batch: usize, inp: usize) -> Array2<f32> {
    Array2::from_shape_fn((batch, inp), |(b, i)| {
        ((b * 31 + i * 7) % 1000) as f32 * 0.001 - 0.5
    })
}

// =============================================================================
// Original Tests
// =============================================================================

#[test]
fn test_matmul_f32_basic() {
    let layer = create_f32_layer();
    let input = arr2(&[[1.0, 1.0]]);
    let output = layer.matmul(&input.view());
    assert_eq!(output, arr2(&[[3.0, 7.0]]));
}

#[test]
fn test_matmul_strategies() {
    let rows = 4;
    let cols = 4;
    let weights = Array2::<f32>::eye(rows);

    let l1 = LinearLayer::new_f32(weights.clone(), None);
    let input = Array2::<f32>::ones((1, cols));
    let out1 = l1.matmul(&input.view());
    assert_eq!(out1, input);

    let mut l2 = LinearLayer::new_f32(weights.t().to_owned(), None);
    l2.f32_strategy = F32MatmulStrategy::Faer;
    let out2 = l2.matmul(&input.view());
    assert_eq!(out2, input);
}

#[test]
fn test_matmul_bf16() {
    let weights = Array2::<f32>::eye(4).mapv(half::bf16::from_f32);
    let layer = LinearLayer::new_bf16(weights, None);

    let input = Array2::<f32>::ones((1, 4));
    let output = layer.matmul(&input.view());

    assert!((output[[0, 0]] - 1.0).abs() < 1e-2);
}

#[test]
fn test_matmul_f32_with_bias() {
    let weights = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let bias = arr1(&[10.0, 20.0]);
    let layer = LinearLayer::new_f32(weights, Some(bias));

    let input = arr2(&[[1.0, 1.0]]);
    let output = layer.matmul(&input.view());

    assert_eq!(output, arr2(&[[13.0, 27.0]]));
}

#[test]
fn test_matmul_f32_batch() {
    let layer = create_f32_layer();
    let input = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let output = layer.matmul(&input.view());
    let expected = arr2(&[[1.0, 3.0], [2.0, 4.0]]);
    assert_eq!(output, expected);
}

#[test]
fn test_matmul_bf16_2() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let weights_bf16 =
        Array2::from_shape_vec((2, 2), data.iter().map(|&x| bf16::from_f32(x)).collect()).unwrap();

    let layer = LinearLayer::new_bf16(weights_bf16, None);
    let input = arr2(&[[1.0, 1.0]]);
    let output = layer.matmul(&input.view());

    assert_eq!(output, arr2(&[[3.0, 7.0]]));
}

#[test]
fn test_f32_strategies_consistency() {
    let weights_standard = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let weights_transposed = weights_standard.t().to_owned();

    let layer_simd = LinearLayer {
        data: LinearData::F32(Arc::new(weights_standard)),
        bias: None,
        f32_strategy: F32MatmulStrategy::CustomSimd,
    };

    let layer_faer = LinearLayer {
        data: LinearData::F32(Arc::new(weights_transposed)),
        bias: None,
        f32_strategy: F32MatmulStrategy::Faer,
    };

    let input = arr2(&[[0.5, 2.0]]);

    let out_simd: Array2<f32> = layer_simd.matmul(&input.view());
    let out_faer: Array2<f32> = layer_faer.matmul(&input.view());

    for ((i, j), &val_simd) in out_simd.indexed_iter() {
        let val_faer = out_faer[(i, j)];
        assert_abs_diff_eq!(val_simd, val_faer, epsilon = 1e-5);
    }
}

#[test]
fn test_quantization_conversion() {
    let rows = 32;
    let cols = 32;
    let weights = Array2::<f32>::zeros((rows, cols));
    let layer = LinearLayer::new_f32(weights, None);

    let q_layer = layer
        .to_quantized(DType::Q8_0)
        .expect("Quantization failed");

    assert_eq!(q_layer.dtype(), DType::Q8_0);
    match q_layer.data {
        LinearData::Q8_0(m) => {
            assert_eq!(m.shape, [rows, cols]);
            assert_eq!(m.blocks.len(), rows * (cols / 32));
        }
        _ => panic!("Wrong data variant after conversion"),
    }
}

#[test]
fn test_quantization_discards_bias() {
    let weights = Array2::<f32>::zeros((32, 32));
    let bias = Array1::<f32>::zeros(32);
    let layer = LinearLayer::new_f32(weights, Some(bias));

    assert!(layer.has_bias());

    let q_layer = layer.to_quantized(DType::Q8_0).unwrap();

    assert!(
        !q_layer.has_bias(),
        "Bias should be discarded in quantized layer"
    );
}

#[test]
fn test_shape_metadata() {
    let layer = LinearLayer::new(10, 20, DType::F32);
    assert_eq!(layer.out_features(), 10);
    assert_eq!(layer.in_features(), 20);
    assert_eq!(layer.shape(), [10, 20]);
}

// =============================================================================
// LinearData::dtype() Tests
// =============================================================================

#[test]
fn test_linear_data_dtype_f32() {
    let data = LinearData::F32(Arc::new(Array2::<f32>::zeros((4, 4))));
    assert_eq!(data.dtype(), DType::F32);
}

#[test]
fn test_linear_data_dtype_bf16() {
    let data = LinearData::BF16(Arc::new(Array2::<bf16>::zeros((4, 4))));
    assert_eq!(data.dtype(), DType::BF16);
}

#[test]
fn test_linear_data_dtype_f16() {
    let data = LinearData::F16(Arc::new(Array2::<half::f16>::zeros((4, 4))));
    assert_eq!(data.dtype(), DType::F16);
}

#[test]
fn test_linear_data_dtype_q8_0() {
    let layer = make_f32_layer(64, 64).to_quantized(DType::Q8_0).unwrap();
    assert_eq!(layer.dtype(), DType::Q8_0);
}

// =============================================================================
// LinearLayer::new() Tests
// =============================================================================

#[test]
fn test_new_f32() {
    let layer = LinearLayer::new(128, 64, DType::F32);
    assert_eq!(layer.out_features(), 128);
    assert_eq!(layer.in_features(), 64);
    assert_eq!(layer.dtype(), DType::F32);
}

#[test]
fn test_new_bf16() {
    let layer = LinearLayer::new(128, 64, DType::BF16);
    assert_eq!(layer.out_features(), 128);
    assert_eq!(layer.in_features(), 64);
    assert_eq!(layer.dtype(), DType::BF16);
}

#[test]
#[should_panic(expected = "Unsupported dtype")]
fn test_new_quantized_panics() {
    let _ = LinearLayer::new(128, 64, DType::Q8_0);
}

// =============================================================================
// matmul_noalloc() Tests
// =============================================================================

#[test]
fn test_matmul_noalloc_f32_small_batch() {
    let layer = make_f32_layer_with_bias(64, 32);
    let input = make_input(4, 32);
    let mut output = Array2::<f32>::zeros((4, 64));

    layer.matmul_noalloc(&input.view(), &mut output);

    let expected = layer.matmul(&input.view());
    for (a, e) in output.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-5);
    }
}

#[test]
fn test_matmul_noalloc_f32_large_batch() {
    let layer = make_f32_layer(64, 32);
    let input = make_input(1500, 32);
    let mut output = Array2::<f32>::zeros((1500, 64));

    layer.matmul_noalloc(&input.view(), &mut output);

    let expected = layer.matmul(&input.view());
    for (a, e) in output.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-4);
    }
}

#[test]
fn test_matmul_noalloc_bf16_fallback() {
    let layer = make_bf16_layer(64, 32);
    let input = make_input(4, 32);
    let mut output = Array2::<f32>::zeros((4, 64));

    layer.matmul_noalloc(&input.view(), &mut output);

    let expected = layer.matmul(&input.view());
    for (a, e) in output.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-3);
    }
}

#[test]
fn test_matmul_noalloc_q8_0_fallback() {
    let layer = make_f32_layer(64, 32).to_quantized(DType::Q8_0).unwrap();
    let input = make_input(4, 32);
    let mut output = Array2::<f32>::zeros((4, 64));

    layer.matmul_noalloc(&input.view(), &mut output);

    let expected = layer.matmul(&input.view());
    for (a, e) in output.iter().zip(expected.iter()) {
        assert!((a - e).abs() < 1e-4);
    }
}

#[test]
fn test_matmul_noalloc_faer_fallback() {
    let weights = Array2::from_shape_fn((32, 64), |(i, j)| {
        ((i * 17 + j * 13) % 1000) as f32 * 0.001
    });
    let layer = LinearLayer::new_f32_with_strategy(weights, None, F32MatmulStrategy::Faer);

    let input = make_input(4, 32);
    let mut output = Array2::<f32>::zeros((4, 64));

    layer.matmul_noalloc(&input.view(), &mut output);

    assert!(output.iter().any(|&x| x != 0.0));
}

// =============================================================================
// matmul() Strategy Tests
// =============================================================================

#[test]
fn test_matmul_f32_decode_path() {
    let layer = make_f32_layer_with_bias(64, 32);
    let input = make_input(1, 32);

    let output = layer.matmul(&input.view());

    assert_eq!(output.shape(), &[1, 64]);
    assert!(output.iter().any(|&x| x != 0.0));
}

#[test]
fn test_matmul_f32_batch_path() {
    let layer = make_f32_layer_with_bias(64, 32);
    let input = make_input(8, 32);

    let output = layer.matmul(&input.view());

    assert_eq!(output.shape(), &[8, 64]);
}

#[test]
fn test_matmul_faer_strategy() {
    let weights = Array2::from_shape_fn((32, 64), |(i, j)| {
        ((i * 17 + j * 13) % 1000) as f32 * 0.001
    });
    let bias = Array1::from_shape_fn(64, |i| i as f32 * 0.01);
    let layer = LinearLayer::new_f32_with_strategy(weights, Some(bias), F32MatmulStrategy::Faer);

    let input = make_input(4, 32);
    let output = layer.matmul(&input.view());

    assert_eq!(output.shape(), &[4, 64]);
}

#[test]
fn test_matmul_faer_out_in_strategy() {
    let weights = Array2::from_shape_fn((64, 32), |(i, j)| {
        ((i * 17 + j * 13) % 1000) as f32 * 0.001
    });
    let bias = Array1::from_shape_fn(64, |i| i as f32 * 0.01);
    let layer =
        LinearLayer::new_f32_with_strategy(weights, Some(bias), F32MatmulStrategy::FaerOutIn);

    let input = make_input(4, 32);
    let output = layer.matmul(&input.view());

    assert_eq!(output.shape(), &[4, 64]);
}

#[test]
fn test_matmul_bf16_with_bias() {
    let layer = make_bf16_layer_with_bias(64, 32);
    let input = make_input(4, 32);

    let output = layer.matmul(&input.view());

    assert_eq!(output.shape(), &[4, 64]);
    assert!(output.iter().any(|&x| x != 0.0));
}

#[test]
fn test_matmul_q8_0_with_bias() {
    let mut layer = make_f32_layer(64, 32).to_quantized(DType::Q8_0).unwrap();
    layer.bias = Some(Array1::from_shape_fn(64, |i| i as f32 * 0.01));

    let input = make_input(4, 32);
    let output = layer.matmul(&input.view());

    assert_eq!(output.shape(), &[4, 64]);
}

#[test]
#[should_panic(expected = "not implemented")]
fn test_matmul_f16_panics() {
    let weights = Array2::<half::f16>::zeros((64, 32));
    let layer = LinearLayer {
        data: LinearData::F16(Arc::new(weights)),
        bias: None,
        f32_strategy: F32MatmulStrategy::CustomSimd,
    };
    let input = make_input(1, 32);
    let _ = layer.matmul(&input.view());
}

// =============================================================================
// weights_view() / weights_slice() Tests
// =============================================================================

#[test]
fn test_weights_view_f32() {
    let layer = make_f32_layer(64, 32);
    let view = layer.weights_view();
    assert_eq!(view.shape(), &[64, 32]);
}

#[test]
#[should_panic(expected = "Only f32")]
fn test_weights_view_bf16_panics() {
    let layer = make_bf16_layer(64, 32);
    let _ = layer.weights_view();
}

#[test]
fn test_weights_slice_f32() {
    let layer = make_f32_layer(64, 32);
    let slice = layer.weights_slice();
    assert_eq!(slice.len(), 64 * 32);
}

#[test]
#[should_panic(expected = "Only f32")]
fn test_weights_slice_bf16_panics() {
    let layer = make_bf16_layer(64, 32);
    let _ = layer.weights_slice();
}

#[test]
fn test_weights_slice_bf16_method() {
    let layer = make_bf16_layer(64, 32);
    let slice = layer.weights_slice_bf16();
    assert!(slice.is_some());
    assert_eq!(slice.unwrap().len(), 64 * 32);
}

#[test]
fn test_weights_slice_bf16_returns_none_for_f32() {
    let layer = make_f32_layer(64, 32);
    assert!(layer.weights_slice_bf16().is_none());
}

// =============================================================================
// to_quantized() Tests
// =============================================================================

#[test]
fn test_to_quantized_f32_to_q8_0() {
    let layer = make_f32_layer(64, 32);
    let quantized = layer.to_quantized(DType::Q8_0).unwrap();

    assert_eq!(quantized.dtype(), DType::Q8_0);
    assert_eq!(quantized.out_features(), 64);
    assert_eq!(quantized.in_features(), 32);
    assert!(quantized.bias.is_none());
}

#[test]
fn test_to_quantized_bf16_to_q8_0() {
    let layer = make_bf16_layer(64, 32);
    let quantized = layer.to_quantized(DType::Q8_0).unwrap();

    assert_eq!(quantized.dtype(), DType::Q8_0);
}

#[test]
fn test_to_quantized_with_bias_logs_warning() {
    let layer = make_f32_layer_with_bias(64, 32);
    let quantized = layer.to_quantized(DType::Q8_0).unwrap();

    assert!(quantized.bias.is_none());
}

#[test]
fn test_to_quantized_unsupported_returns_error() {
    let layer = make_f32_layer(64, 32);
    let result = layer.to_quantized(DType::Q4_K);

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Unsupported"));
}

#[test]
fn test_to_quantized_q8_0_to_q8_0_fails() {
    let layer = make_f32_layer(64, 32).to_quantized(DType::Q8_0).unwrap();
    let result = layer.to_quantized(DType::Q8_0);

    assert!(result.is_err());
}

// =============================================================================
// Dimension Tests with Different Strategies
// =============================================================================

#[test]
fn test_out_features_faer_strategy() {
    let weights = Array2::<f32>::zeros((32, 64));
    let layer = LinearLayer::new_f32_with_strategy(weights, None, F32MatmulStrategy::Faer);

    assert_eq!(layer.out_features(), 64);
    assert_eq!(layer.in_features(), 32);
}

#[test]
fn test_out_features_custom_simd_strategy() {
    let weights = Array2::<f32>::zeros((64, 32));
    let layer = LinearLayer::new_f32_with_strategy(weights, None, F32MatmulStrategy::CustomSimd);

    assert_eq!(layer.out_features(), 64);
    assert_eq!(layer.in_features(), 32);
}

#[test]
fn test_dimensions_f16() {
    let weights = Array2::<half::f16>::zeros((64, 32));
    let layer = LinearLayer {
        data: LinearData::F16(Arc::new(weights)),
        bias: None,
        f32_strategy: F32MatmulStrategy::CustomSimd,
    };

    assert_eq!(layer.out_features(), 64);
    assert_eq!(layer.in_features(), 32);
    assert_eq!(layer.dtype(), DType::F16);
}

#[test]
fn test_shape_method() {
    let layer = make_f32_layer(64, 32);
    assert_eq!(layer.shape(), [64, 32]);
}

#[test]
fn test_has_bias() {
    let no_bias = make_f32_layer(64, 32);
    let with_bias = make_f32_layer_with_bias(64, 32);

    assert!(!no_bias.has_bias());
    assert!(with_bias.has_bias());
}

// =============================================================================
// From Implementations
// =============================================================================

#[test]
fn test_from_f32_array() {
    let weights = Array2::<f32>::zeros((64, 32));
    let layer: LinearLayer = weights.into();

    assert_eq!(layer.dtype(), DType::F32);
    assert!(!layer.has_bias());
}

#[test]
fn test_from_f32_array_with_bias() {
    let weights = Array2::<f32>::zeros((64, 32));
    let bias = Array1::<f32>::zeros(64);
    let layer: LinearLayer = (weights, bias).into();

    assert_eq!(layer.dtype(), DType::F32);
    assert!(layer.has_bias());
}

#[test]
fn test_from_bf16_array() {
    let weights = Array2::<bf16>::zeros((64, 32));
    let layer: LinearLayer = weights.into();

    assert_eq!(layer.dtype(), DType::BF16);
    assert!(!layer.has_bias());
}

#[test]
fn test_from_bf16_array_with_bias() {
    let weights = Array2::<bf16>::zeros((64, 32));
    let bias = Array1::<f32>::zeros(64);
    let layer: LinearLayer = (weights, bias).into();

    assert_eq!(layer.dtype(), DType::BF16);
    assert!(layer.has_bias());
}

// =============================================================================
// Arc Constructors
// =============================================================================

#[test]
fn test_from_arc_f32() {
    let weights = Arc::new(Array2::<f32>::zeros((64, 32)));
    let layer = LinearLayer::from_arc_f32(weights.clone(), None);

    assert_eq!(layer.dtype(), DType::F32);
    assert!(Arc::ptr_eq(
        &weights,
        match &layer.data {
            LinearData::F32(w) => w,
            _ => panic!("Wrong type"),
        }
    ));
}

#[test]
fn test_from_arc_bf16() {
    let weights = Arc::new(Array2::<bf16>::zeros((64, 32)));
    let layer = LinearLayer::from_arc_bf16(weights.clone(), None);

    assert_eq!(layer.dtype(), DType::BF16);
}

#[test]
fn test_from_arc_q8_0() {
    let f32_layer = make_f32_layer(64, 32);
    let q8_layer = f32_layer.to_quantized(DType::Q8_0).unwrap();

    let arc = match &q8_layer.data {
        LinearData::Q8_0(a) => a.clone(),
        _ => panic!("Wrong type"),
    };

    let layer = LinearLayer::from_arc_q8_0(arc.clone(), None);
    assert_eq!(layer.dtype(), DType::Q8_0);
}

// =============================================================================
// Quantized Matmul Accuracy Tests
// =============================================================================

#[test]
fn test_q8_0_matmul_accuracy() {
    let f32_layer = make_f32_layer(64, 32);
    let q8_layer = f32_layer.to_quantized(DType::Q8_0).unwrap();

    let input = make_input(4, 32);

    let f32_output = f32_layer.matmul(&input.view());
    let q8_output = q8_layer.matmul(&input.view());

    let mut max_rel_error = 0.0f32;
    for (f, q) in f32_output.iter().zip(q8_output.iter()) {
        if f.abs() > 1e-6 {
            let rel_error = (f - q).abs() / f.abs();
            max_rel_error = max_rel_error.max(rel_error);
        }
    }
    assert!(max_rel_error < 0.05, "Max relative error: {}", max_rel_error);
}

// =============================================================================
// Clone Test
// =============================================================================

#[test]
fn test_linear_layer_clone() {
    let layer = make_f32_layer_with_bias(64, 32);
    let cloned = layer.clone();

    assert_eq!(layer.dtype(), cloned.dtype());
    assert_eq!(layer.out_features(), cloned.out_features());
    assert_eq!(layer.has_bias(), cloned.has_bias());
}

// =============================================================================
// GPU Tests
// =============================================================================

pub async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}

#[tokio::test]
async fn test_to_gpu_f32() {
    let ctx = get_test_context().await;
    let layer = make_f32_layer(64, 32);
    let gpu_tensor = layer.to_gpu(&ctx).unwrap();

    assert_eq!(gpu_tensor.shape(), &[64, 32]);
}

#[tokio::test]
async fn test_to_gpu_bf16() {
    let ctx = get_test_context().await;
    let layer = make_bf16_layer(64, 32);
    let gpu_tensor = layer.to_gpu(&ctx).unwrap();

    assert_eq!(gpu_tensor.shape(), &[64, 32]);
}

#[tokio::test]
async fn test_to_gpu_f16() {
    let ctx = get_test_context().await;
    let weights = Array2::<half::f16>::zeros((64, 32));
    let layer = LinearLayer {
        data: LinearData::F16(Arc::new(weights)),
        bias: None,
        f32_strategy: F32MatmulStrategy::CustomSimd,
    };
    let gpu_tensor = layer.to_gpu(&ctx).unwrap();

    assert_eq!(gpu_tensor.shape(), &[64, 32]);
}

#[tokio::test]
async fn test_to_gpu_q8_0() {
    let ctx = get_test_context().await;
    let layer = make_f32_layer(64, 32).to_quantized(DType::Q8_0).unwrap();
    let gpu_tensor = layer.to_gpu(&ctx).unwrap();

    assert_eq!(gpu_tensor.shape(), &[64, 32]);
}

#[tokio::test]
async fn test_bias_to_gpu() {
    let ctx = get_test_context().await;
    let layer = make_f32_layer_with_bias(64, 32);
    let gpu_bias = layer.bias_to_gpu(&ctx).unwrap();

    assert!(gpu_bias.is_some());
    assert_eq!(gpu_bias.unwrap().shape(), &[64]);
}

#[tokio::test]
async fn test_bias_to_gpu_none() {
    let ctx = get_test_context().await;
    let layer = make_f32_layer(64, 32);
    let gpu_bias = layer.bias_to_gpu(&ctx).unwrap();

    assert!(gpu_bias.is_none());
}

#[tokio::test]
async fn test_to_gpu_tensor_with_label() {
    let ctx = get_test_context().await;
    let layer = make_f32_layer(64, 32);
    let gpu_tensor = layer.to_gpu_tensor(&ctx, "test_weights").unwrap();

    assert_eq!(gpu_tensor.shape(), &[64, 32]);
}