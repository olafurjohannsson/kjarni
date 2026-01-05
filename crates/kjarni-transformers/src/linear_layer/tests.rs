
use approx::{assert_abs_diff_eq, assert_relative_eq};
use half::bf16;
use ndarray::{Array1, Array2, arr1, arr2};

use crate::{linear_layer::{F32MatmulStrategy, LinearData, LinearLayer}, tensor::DType};

// Helper to create a standard test layer
fn create_f32_layer() -> LinearLayer {
    // Weights: 2x2 matrix
    // [[1.0, 2.0],
    //  [3.0, 4.0]]
    let weights = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    LinearLayer::new_f32(weights, None)
}

#[test]
fn test_matmul_f32_basic() {
    let layer = create_f32_layer();
    // Input: [1, 2] vector (batch size 1)
    // [[1.0, 1.0]]
    let input = arr2(&[[1.0, 1.0]]);

    // Expected: Input @ Weights^T
    // [1*1 + 1*2, 1*3 + 1*4] = [3.0, 7.0]
    let output = layer.matmul(&input.view());

    assert_eq!(output, arr2(&[[3.0, 7.0]]));
}

#[test]
fn test_matmul_f32_with_bias() {
    let weights = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let bias = arr1(&[10.0, 20.0]); // Bias for each output feature
    let layer = LinearLayer::new_f32(weights, Some(bias));

    let input = arr2(&[[1.0, 1.0]]);
    let output = layer.matmul(&input.view());

    // Expected: [3.0 + 10.0, 7.0 + 20.0] = [13.0, 27.0]
    assert_eq!(output, arr2(&[[13.0, 27.0]]));
}

#[test]
fn test_matmul_f32_batch() {
    let layer = create_f32_layer();
    // Batch size 2
    let input = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

    let output = layer.matmul(&input.view());

    // Row 0: [1*1 + 0*2, 1*3 + 0*4] = [1.0, 3.0]
    // Row 1: [0*1 + 1*2, 0*3 + 1*4] = [2.0, 4.0]
    let expected = arr2(&[[1.0, 3.0], [2.0, 4.0]]);

    assert_eq!(output, expected);
}

#[test]
fn test_matmul_bf16() {
    // Create BF16 weights
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let weights_bf16 =
        Array2::from_shape_vec((2, 2), data.iter().map(|&x| bf16::from_f32(x)).collect()).unwrap();

    let layer = LinearLayer::new_bf16(weights_bf16, None);
    let input = arr2(&[[1.0, 1.0]]);

    let output = layer.matmul(&input.view());

    // BF16 matmul upcasts to F32 for computation, so results should match F32 exactly for small integers
    assert_eq!(output, arr2(&[[3.0, 7.0]]));
}

#[test]
fn test_f32_strategies_consistency() {
    // Verify that CustomSimd (Standard Layout) and Faer (Transposed Layout) produce identical results
    let weights_standard = arr2(&[[1.0, 2.0], [3.0, 4.0]]); // [Out, In]
    let weights_transposed = weights_standard.t().to_owned(); // [In, Out]

    let layer_simd = LinearLayer {
        data: LinearData::F32(weights_standard),
        bias: None,
        f32_strategy: F32MatmulStrategy::CustomSimd,
    };

    let layer_faer = LinearLayer {
        data: LinearData::F32(weights_transposed),
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
    // Q8_0 requires block size alignment (usually 32)
    let rows = 32;
    let cols = 32;
    let weights = Array2::<f32>::zeros((rows, cols));

    let layer = LinearLayer::new_f32(weights, None);

    // Convert to Q8_0
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
