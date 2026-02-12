use approx::assert_abs_diff_eq;
use half::bf16;
use ndarray::{Array2, Array3, Array4, arr2};

use crate::activations::softmax_4d_inplace;
use crate::utils::linear_algebra::{
    apply_attention_mask, matmul_2d, matmul_2d_f32_notranspose, matmul_2d_mixed_bf16_new,
    matmul_2d_transposed, matmul_3d_2d_transposed, matmul_4d, matmul_4d_context_gqa,
    matmul_4d_decode_gqa,
};

use super::*;

#[test]
fn test_matmul_2d_basic() {
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

    let result = matmul_2d(&a.view(), &b.view());
    assert_eq!(result, a);
}

#[test]
fn test_matmul_2d_transposed() {
    let a = arr2(&[[1.0, 2.0]]);
    let b_transposed = arr2(&[[3.0, 4.0], [5.0, 6.0]]);

    let result = matmul_2d_transposed(&a.view(), &b_transposed.view());
    assert_eq!(result, arr2(&[[11.0, 17.0]]));
}

#[test]
fn test_matmul_3d_2d_transposed() {
    let a = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

    let result = matmul_3d_2d_transposed(&a, &b);

    assert_eq!(result.shape(), &[1, 2, 2]);
    assert_eq!(result.into_shape((2, 2)).unwrap(), arr2(&[[1.0, 2.0], [3.0, 4.0]]));
}

#[test]
fn test_matmul_2d_mixed_bf16() {
    let m = 2;
    let k = 64;
    let n = 32;

    let a = Array2::from_elem((m, k), 0.5f32);
    let b_f32 = Array2::from_elem((n, k), 2.0f32);
    let b_bf16 = b_f32.mapv(bf16::from_f32);

    let result = matmul_2d_mixed_bf16_new(&a.view(), &b_bf16.view());

    assert_eq!(result.shape(), &[m, n]);
    assert!(result.iter().all(|&x| (x - 64.0).abs() < 1.0));
}

#[test]
fn test_matmul_2d_f32_notranspose() {
    let m = 2;
    let k = 64;
    let n = 32;

    let a = Array2::from_elem((m, k), 1.0f32);
    let b = Array2::from_elem((n, k), 1.0f32);

    let result = matmul_2d_f32_notranspose(&a.view(), &b.view());

    assert_eq!(result.shape(), &[m, n]);
    assert_eq!(result[[0, 0]], 64.0);
}

#[test]
fn test_matmul_4d_broadcast() {
    let a = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let b = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let result = matmul_4d(&a, &b);
    assert_eq!(result, b);
}

#[test]
fn test_matmul_4d_decode_gqa() {
    let q = Array4::from_elem((1, 4, 1, 2), 1.0f32);
    let k = Array4::from_elem((1, 2, 2, 3), 1.0f32);

    let result = matmul_4d_decode_gqa(&q, &k.view(), 2);

    assert_eq!(result.shape(), &[1, 4, 1, 3]);
    assert_eq!(result[[0, 0, 0, 0]], 2.0);
    assert_eq!(result[[0, 3, 0, 2]], 2.0);
}

#[test]
fn test_matmul_4d_context_gqa() {
    let scores = Array4::from_elem((1, 4, 1, 3), 1.0f32);
    let v = Array4::from_elem((1, 2, 3, 2), 1.0f32);

    let result = matmul_4d_context_gqa(&scores, &v.view(), 2);

    assert_eq!(result.shape(), &[1, 4, 1, 2]);
    assert_eq!(result[[0, 0, 0, 0]], 3.0);
}

#[test]
fn test_apply_attention_mask() {
    let scores = Array4::from_elem((1, 1, 1, 4), 10.0f32);
    let mask = arr2(&[[1.0, 1.0, 0.0, 0.0]]);

    let result = apply_attention_mask(scores, &mask);

    assert_eq!(result[[0, 0, 0, 0]], 10.0);
    assert_eq!(result[[0, 0, 0, 1]], 10.0);
    assert_eq!(result[[0, 0, 0, 2]], MASK_VALUE);
    assert_eq!(result[[0, 0, 0, 3]], MASK_VALUE);
}

#[test]
fn test_softmax_4d_inplace() {
    let mut scores = Array4::from_shape_vec((1, 1, 1, 2), vec![0.0f32, 10.0]).unwrap();

    softmax_4d_inplace(&mut scores);

    assert!(scores[[0, 0, 0, 0]] < 0.001);
    assert!(scores[[0, 0, 0, 1]] > 0.999);
    assert_abs_diff_eq!(scores.sum(), 1.0, epsilon = 1e-5);
}

#[test]
fn test_apply_attention_mask_mismatch() {
    let scores = Array4::zeros((1, 1, 1, 4));
    let mask = Array2::zeros((1, 3));

    let result = apply_attention_mask(scores.clone(), &mask);
    assert_eq!(result, scores);
}