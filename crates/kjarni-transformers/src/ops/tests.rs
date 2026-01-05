use crate::kernels::quantize::quantize_matrix_q8_0;
use crate::kernels::q_common::{BlockQ4_K, BlockQ6_K, QK_K};
use crate::ops::matmul::{matmul_2d_cpu_f32, matmul_2d_cpu_bf16, matmul_2d_cpu_q4_k, matmul_2d_cpu_q6_k, matmul_2d_cpu_q8_0};
use half::bf16;
use ndarray::{Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ========================================================================
//  Helpers & Ground Truth
// ========================================================================

/// Generates a random F32 matrix with values in [-1.0, 1.0].
fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    // StdRng is deterministic and seedable, just like ChaCha8Rng
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-1.0..1.0))
}

/// A high-precision (F64) reference implementation of MatMul: C = A @ B^T
/// Used as Ground Truth to verify the optimized kernels.
fn ground_truth_matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (n, k2) = b.dim();
    assert_eq!(k, k2, "Ground truth dimension mismatch");

    let mut c = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64; // Accumulate in f64 for max precision
            for x in 0..k {
                sum += (a[[i, x]] as f64) * (b[[j, x]] as f64);
            }
            c[[i, j]] = sum as f32;
        }
    }
    c
}

/// Asserts that two matrices are close within a given tolerance (RMSE).
fn assert_matrices_close(actual: &Array2<f32>, expected: &Array2<f32>, tolerance: f32, label: &str) {
    assert_eq!(actual.shape(), expected.shape(), "Shape mismatch in {}", label);

    let diff = actual - expected;
    let mse = diff.mapv(|x| x.powi(2)).sum() / diff.len() as f32;
    let rmse = mse.sqrt();

    if rmse > tolerance {
        println!("Failure in {}: RMSE {} > Tolerance {}", label, rmse, tolerance);
        // Print sample of failures
        let mut count = 0;
        for ((idx, val_a), val_b) in actual.indexed_iter().zip(expected.iter()) {
            if (val_a - val_b).abs() > tolerance * 2.0 {
                println!("  [{:?}] Actual: {:.5}, Expected: {:.5}, Diff: {:.5}", idx, val_a, val_b, (val_a - val_b).abs());
                count += 1;
                if count > 10 { break; }
            }
        }
        panic!("Matrices not close enough");
    }
}

// ========================================================================
//  F32 Tests (The Gold Standard)
// ========================================================================

#[test]
fn test_matmul_f32_decode_path() {
    // Batch size 1 triggers the "Decode" path (parallelize over output features)
    let k = 256;
    let n = 512;
    let a = random_matrix(1, k, 42);
    let b = random_matrix(n, k, 43); // Weight matrix [out, in]

    let expected = ground_truth_matmul(&a, &b);
    let actual = matmul_2d_cpu_f32(&a.view(), &b.view());

    // F32 to F32 should be extremely precise
    assert_matrices_close(&actual, &expected, 1e-5, "F32 Decode Path");
}

#[test]
fn test_matmul_f32_prefill_path() {
    // Batch size > 1 triggers the "Prefill" path (parallelize over batch)
    let m = 8;
    let k = 128;
    let n = 256;
    let a = random_matrix(m, k, 44);
    let b = random_matrix(n, k, 45);

    let expected = ground_truth_matmul(&a, &b);
    let actual = matmul_2d_cpu_f32(&a.view(), &b.view());

    assert_matrices_close(&actual, &expected, 1e-5, "F32 Prefill Path");
}

#[test]
fn test_matmul_f32_odd_dimensions() {
    // Test weird shapes to ensure remainder loops in kernels handle things correctly
    let m = 3;
    let k = 127; // Prime/Odd
    let n = 33;
    let a = random_matrix(m, k, 46);
    let b = random_matrix(n, k, 47);

    let expected = ground_truth_matmul(&a, &b);
    let actual = matmul_2d_cpu_f32(&a.view(), &b.view());

    assert_matrices_close(&actual, &expected, 1e-5, "F32 Odd Dims");
}

// ========================================================================
//  BF16 Tests
// ========================================================================

#[test]
fn test_matmul_bf16_correctness() {
    let m = 4;
    let k = 256;
    let n = 128;

    let a = random_matrix(m, k, 100);
    let b_f32 = random_matrix(n, k, 101);

    // Convert weights to BF16
    let b_bf16_vec: Vec<bf16> = b_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let b_bf16 = Array2::from_shape_vec((n, k), b_bf16_vec).unwrap();

    // Actual Calculation
    let actual = matmul_2d_cpu_bf16(&a.view(), &b_bf16.view());

    // Since BF16 truncates mantissa, we expect some precision loss compared to F32 math.
    // However, the kernel converts BF16->F32 before multiply, so it should be reasonably close
    // to the F32 result of the truncated weights.
    
    // To generate a fair ground truth, we convert BF16 back to F32 and run ground truth on that.
    let b_reconstructed_vec: Vec<f32> = b_bf16.iter().map(|&x| x.to_f32()).collect();
    let b_reconstructed = Array2::from_shape_vec((n, k), b_reconstructed_vec).unwrap();
    
    let expected = ground_truth_matmul(&a, &b_reconstructed);

    // Tolerance is higher due to BF16 precision
    assert_matrices_close(&actual, &expected, 1e-3, "BF16 Correctness");
}

// ========================================================================
//  Q8_0 Tests (Quantized)
// ========================================================================

#[test]
fn test_matmul_q8_0_full_pipeline() {
    // This tests the full flow: F32 Weights -> Quantize -> Matmul -> Verify
    let m = 2; // Prefill path
    let k = 512; // Must be multiple of 32
    let n = 128;

    let a = random_matrix(m, k, 200);
    let b_f32 = random_matrix(n, k, 201);

    // 1. Quantize weights
    let b_q8_0 = quantize_matrix_q8_0(&b_f32).expect("Quantization failed");
    
    // 2. Run Quantized Matmul
    let actual = matmul_2d_cpu_q8_0(&a.view(), &b_q8_0);

    // 3. Compare against F32 Ground Truth
    // Q8_0 is lossy. We are checking that the result is "directionally correct".
    let expected = ground_truth_matmul(&a, &b_f32);

    // Q8_0 usually has an error around 0.01 - 0.05 depending on distribution
    assert_matrices_close(&actual, &expected, 0.05, "Q8_0 Full Pipeline");
}

#[test]
fn test_matmul_q8_0_decode_path() {
    let m = 1; // Decode path
    let k = 256;
    let n = 64;

    let a = random_matrix(m, k, 202);
    let b_f32 = random_matrix(n, k, 203);
    let b_q8_0 = quantize_matrix_q8_0(&b_f32).unwrap();

    let actual = matmul_2d_cpu_q8_0(&a.view(), &b_q8_0);
    let expected = ground_truth_matmul(&a, &b_f32);

    assert_matrices_close(&actual, &expected, 0.05, "Q8_0 Decode Path");
}

#[test]
#[should_panic(expected = "Input features must be a multiple of the block size")]
fn test_matmul_q8_0_alignment_panic() {
    let k = 33; // Not divisible by 32
    let a = random_matrix(1, k, 0);
    
    // Create dummy blocks (size doesn't matter as it panics before reading)
    let dummy_block = crate::kernels::q_common::BlockQ8_0 { 
        d: half::f16::from_f32(1.0), 
        qs: [0; 32] 
    };
    let b_weights = vec![dummy_block; 10]; // Arbitrary length

    matmul_2d_cpu_q8_0(&a.view(), &b_weights);
}

// ========================================================================
//  K-Quant Plumbing Tests (Q4_K / Q6_K)
// ========================================================================
// Note: Since we don't have a `quantize_matrix_q4_k` or `q6_k` implementation
// in the provided code (only Q8_0), we cannot strictly test numerical accuracy 
// against a random matrix. 
// Instead, we test that the plumbing works (no crashes) using zeroed blocks.

#[test]
fn test_matmul_q4_k_plumbing() {
    let m = 1;
    let k = 256; // QK_K
    let n = 10;
    
    let a = random_matrix(m, k, 300);
    
    // Create dummy Q4_K blocks
    let num_blocks = (n * k) / QK_K;
    let dummy_block = BlockQ4_K {
        d: half::f16::from_f32(1.0),
        dmin: half::f16::from_f32(0.0),
        scales: [0; 12],
        qs: [0; QK_K / 2],
    };
    let b_weights = vec![dummy_block; num_blocks];

    // Should run without panic
    let output = matmul_2d_cpu_q4_k(&a.view(), &b_weights);
    
    assert_eq!(output.shape(), &[m, n]);
    // Since weights are zero/dummy, result should be effectively meaningless but finite
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_matmul_q6_k_plumbing() {
    let m = 4; // Prefill path
    let k = 512; // 2 * QK_K
    let n = 10;
    
    let a = random_matrix(m, k, 301);
    
    // Create dummy Q6_K blocks
    let num_blocks = (n * k) / QK_K;
    let dummy_block = BlockQ6_K {
        ql: [0; 128],
        qh: [0; 64],
        scales: [0; 16],
        d: half::f16::from_f32(1.0),
    };
    let b_weights = vec![dummy_block; num_blocks];

    // Should run without panic
    let output = matmul_2d_cpu_q6_k(&a.view(), &b_weights);
    
    assert_eq!(output.shape(), &[m, n]);
    assert!(output.iter().all(|x| x.is_finite()));
}

// ========================================================================
//  Integration / Sanity Checks
// ========================================================================

#[test]
fn test_dimension_mismatches() {
    let a = random_matrix(1, 10, 0);
    let b = random_matrix(1, 11, 0); // K=11 != 10

    let result = std::panic::catch_unwind(|| {
        matmul_2d_cpu_f32(&a.view(), &b.view())
    });
    assert!(result.is_err(), "Should panic on mismatched K dimension");
}

#[test]
fn test_zero_matrix() {
    // A * 0 = 0
    let a = random_matrix(2, 32, 50);
    let b = Array2::<f32>::zeros((4, 32));
    
    let actual = matmul_2d_cpu_f32(&a.view(), &b.view());
    assert!(actual.iter().all(|&x| x == 0.0), "Multiplication by zero matrix failed");
}