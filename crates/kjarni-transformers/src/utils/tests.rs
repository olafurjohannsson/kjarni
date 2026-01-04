use crate::utils::linear_algebra::{apply_attention_mask, matmul_2d, matmul_2d_f32_notranspose, matmul_2d_mixed_bf16_new, matmul_2d_transposed, matmul_3d_2d_transposed, matmul_4d, matmul_4d_context_gqa, matmul_4d_decode_gqa, softmax_inplace};

use super::*;
use ndarray::{Array2, Array3, Array4, ArrayView2, arr2};
use approx::assert_abs_diff_eq;
use half::bf16;

// ========================================================================
//  Helper: Simple Scalar Reference
// ========================================================================

fn reference_matmul_2d(a: &ArrayView2<f32>, b: &ArrayView2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    assert_eq!(k, k2);
    
    let mut c = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for x in 0..k {
                sum += a[[i, x]] * b[[x, j]];
            }
            c[[i, j]] = sum;
        }
    }
    c
}

// ========================================================================
//  Standard Matmul Tests
// ========================================================================

#[test]
fn test_matmul_2d_basic() {
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]); // 2x2
    let b = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // Identity 2x2
    
    let result = matmul_2d(&a.view(), &b.view());
    assert_eq!(result, a);
}

#[test]
fn test_matmul_2d_transposed() {
    let a = arr2(&[[1.0, 2.0]]); // 1x2
    let b_transposed = arr2(&[[3.0, 4.0], [5.0, 6.0]]); // [Out=2, In=2]
    
    // Expected: [1*3 + 2*4, 1*5 + 2*6] = [11.0, 17.0]
    let result = matmul_2d_transposed(&a.view(), &b_transposed.view());
    
    assert_eq!(result, arr2(&[[11.0, 17.0]]));
}

#[test]
fn test_matmul_3d_2d_transposed() {
    // A: [Batch=1, Seq=2, In=2]
    let a = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    // B: [Out=2, In=2]
    let b = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // Identity
    
    let result = matmul_3d_2d_transposed(&a, &b);
    
    assert_eq!(result.shape(), &[1, 2, 2]);
    assert_eq!(result.into_shape((2, 2)).unwrap(), arr2(&[[1.0, 2.0], [3.0, 4.0]]));
}

// ========================================================================
//  Hardware Kernel Tests (BF16 / F32)
// ========================================================================

#[test]
fn test_matmul_2d_mixed_bf16_correctness() {
    let m = 2;
    let k = 64; // Multiple of 32 for AVX2
    let n = 32;
    
    let a = Array2::from_elem((m, k), 0.5f32);
    
    // Create BF16 weights [Out, In] -> [N, K]
    // Note: The function expects `b_weights` to be [N, K] layout
    let b_f32 = Array2::from_elem((n, k), 2.0f32);
    let b_bf16 = b_f32.mapv(bf16::from_f32);
    
    // Expected: 0.5 * 2.0 * 64 = 64.0 per element
    let result = matmul_2d_mixed_bf16_new(&a.view(), &b_bf16.view());
    
    assert_eq!(result.shape(), &[m, n]);
    // Allow small error due to BF16 precision
    assert!(result.iter().all(|&x| (x - 64.0).abs() < 1.0));
}

#[test]
fn test_matmul_2d_f32_notranspose_correctness() {
    let m = 2;
    let k = 64;
    let n = 32;
    
    let a = Array2::from_elem((m, k), 1.0f32);
    let b = Array2::from_elem((n, k), 1.0f32); // [Out, In]
    
    let result = matmul_2d_f32_notranspose(&a.view(), &b.view());
    
    // Expected: 1.0 * 1.0 * 64 = 64.0
    assert_eq!(result.shape(), &[m, n]);
    assert_eq!(result[[0,0]], 64.0);
}

// ========================================================================
//  Attention / 4D Tests
// ========================================================================

#[test]
fn test_matmul_4d_broadcast() {
    // Batch=1, Heads=1, Seq1=2, Dim=2
    let a = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    // Seq2=2 -> Output [1, 1, 2, 2]
    let b = Array4::from_shape_vec((1, 1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let result = matmul_4d(&a, &b);
    
    // Identity * Matrix = Matrix
    assert_eq!(result, b);
}

#[test]
fn test_matmul_4d_decode_gqa() {
    // Q: [B=1, H=4, 1, D=2]
    let q = Array4::from_elem((1, 4, 1, 2), 1.0f32);
    
    // K: [B=1, KV_H=2, D=2, S=3] (Transposed layout)
    // 2 KV heads, so n_rep = 2 (4 query heads / 2 kv heads)
    let k = Array4::from_elem((1, 2, 2, 3), 1.0f32);
    
    let result = matmul_4d_decode_gqa(&q, &k.view(), 2);
    
    // Output should be [1, 4, 1, 3]
    assert_eq!(result.shape(), &[1, 4, 1, 3]);
    
    // Dot product: [1, 1] . [1, 1] = 2.0
    assert_eq!(result[[0, 0, 0, 0]], 2.0);
    // Verify GQA mapping: Head 0 and 1 should map to KV Head 0
    // Head 2 and 3 should map to KV Head 1
    // Since all values are identical, results should be identical
    assert_eq!(result[[0, 3, 0, 2]], 2.0);
}

#[test]
fn test_matmul_4d_context_gqa() {
    // Scores: [B=1, H=4, 1, S=3]
    let scores = Array4::from_elem((1, 4, 1, 3), 1.0f32);
    
    // V: [B=1, KV_H=2, S=3, D=2]
    let v = Array4::from_elem((1, 2, 3, 2), 1.0f32);
    
    let result = matmul_4d_context_gqa(&scores, &v.view(), 2);
    
    // Output: [1, 4, 1, 2]
    assert_eq!(result.shape(), &[1, 4, 1, 2]);
    // Dot product: [1, 1, 1] . [1, 1, 1] = 3.0
    assert_eq!(result[[0, 0, 0, 0]], 3.0);
}

// ========================================================================
//  Masking & Softmax Tests
// ========================================================================

#[test]
fn test_apply_attention_mask() {
    // Scores: [1, 1, 1, 4]
    let scores = Array4::from_elem((1, 1, 1, 4), 10.0f32);
    
    // Mask: [1, 4] -> 1 1 0 0
    let mask = arr2(&[[1.0, 1.0, 0.0, 0.0]]);
    
    let result = apply_attention_mask(scores, &mask);
    
    // Valid positions remain 10.0
    assert_eq!(result[[0, 0, 0, 0]], 10.0);
    assert_eq!(result[[0, 0, 0, 1]], 10.0);
    
    // Masked positions become MASK_VALUE
    assert_eq!(result[[0, 0, 0, 2]], MASK_VALUE);
    assert_eq!(result[[0, 0, 0, 3]], MASK_VALUE);
}

#[test]
fn test_softmax_inplace() {
    // [1, 1, 1, 2] -> [0.0, 10.0]
    let mut scores = Array4::from_shape_vec((1, 1, 1, 2), vec![0.0f32, 10.0]).unwrap();
    
    softmax_inplace(&mut scores);
    
    // exp(0) vs exp(10). The 10 should dominate.
    // e^0 / (e^0 + e^10) approx 0
    // e^10 / (e^0 + e^10) approx 1
    assert!(scores[[0, 0, 0, 0]] < 0.001);
    assert!(scores[[0, 0, 0, 1]] > 0.999);
    
    // Sum to 1 check
    assert_abs_diff_eq!(scores.sum(), 1.0, epsilon = 1e-5);
}

#[test]
fn test_apply_attention_mask_mismatch() {
    let scores = Array4::zeros((1, 1, 1, 4));
    let mask = Array2::zeros((1, 3)); // Wrong length
    
    // Should return scores unmodified
    let result = apply_attention_mask(scores.clone(), &mask);
    assert_eq!(result, scores);
}