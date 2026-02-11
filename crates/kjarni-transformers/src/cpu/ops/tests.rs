use crate::cpu::kernels::q_common::{BlockQ4_K, BlockQ6_K, QK_K};
use crate::cpu::kernels::quantize::quantize_matrix_q8_0;
use crate::cpu::ops::matmul::{
    matmul_2d_cpu_bf16, matmul_2d_cpu_f32, matmul_2d_cpu_q4_k, matmul_2d_cpu_q6_k,
    matmul_2d_cpu_q8_0,
};
use half::bf16;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-1.0..1.0))
}
fn ground_truth_matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let (m, k) = a.dim();
    let (n, k2) = b.dim();
    assert_eq!(k, k2, "Ground truth dimension mismatch");

    let mut c = Array2::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64; 
            for x in 0..k {
                sum += (a[[i, x]] as f64) * (b[[j, x]] as f64);
            }
            c[[i, j]] = sum as f32;
        }
    }
    c
}
fn assert_matrices_close(
    actual: &Array2<f32>,
    expected: &Array2<f32>,
    tolerance: f32,
    label: &str,
) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "Shape mismatch in {}",
        label
    );

    let diff = actual - expected;
    let mse = diff.mapv(|x| x.powi(2)).sum() / diff.len() as f32;
    let rmse = mse.sqrt();

    if rmse > tolerance {
        println!(
            "Failure in {}: RMSE {} > Tolerance {}",
            label, rmse, tolerance
        );
        // Print sample of failures
        let mut count = 0;
        for ((idx, val_a), val_b) in actual.indexed_iter().zip(expected.iter()) {
            if (val_a - val_b).abs() > tolerance * 2.0 {
                println!(
                    "  [{:?}] Actual: {:.5}, Expected: {:.5}, Diff: {:.5}",
                    idx,
                    val_a,
                    val_b,
                    (val_a - val_b).abs()
                );
                count += 1;
                if count > 10 {
                    break;
                }
            }
        }
        panic!("Matrices not close enough");
    }
}

#[test]
fn test_matmul_f32_decode_path() {
    let k = 256;
    let n = 512;
    let a = random_matrix(1, k, 42);
    let b = random_matrix(n, k, 43); 

    let expected = ground_truth_matmul(&a, &b);
    let actual = matmul_2d_cpu_f32(&a.view(), &b.view());

    assert_matrices_close(&actual, &expected, 1e-5, "F32 Decode Path");
}

#[test]
fn test_matmul_f32_prefill_path() {
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
    let m = 3;
    let k = 127; 
    let n = 33;
    let a = random_matrix(m, k, 46);
    let b = random_matrix(n, k, 47);

    let expected = ground_truth_matmul(&a, &b);
    let actual = matmul_2d_cpu_f32(&a.view(), &b.view());

    assert_matrices_close(&actual, &expected, 1e-5, "F32 Odd Dims");
}
#[test]
fn test_matmul_bf16_correctness() {
    let m = 4;
    let k = 256;
    let n = 128;

    let a = random_matrix(m, k, 100);
    let b_f32 = random_matrix(n, k, 101);
    let b_bf16_vec: Vec<bf16> = b_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let b_bf16 = Array2::from_shape_vec((n, k), b_bf16_vec).unwrap();
    let actual = matmul_2d_cpu_bf16(&a.view(), &b_bf16.view());
    let b_reconstructed_vec: Vec<f32> = b_bf16.iter().map(|&x| x.to_f32()).collect();
    let b_reconstructed = Array2::from_shape_vec((n, k), b_reconstructed_vec).unwrap();

    let expected = ground_truth_matmul(&a, &b_reconstructed);
    assert_matrices_close(&actual, &expected, 1e-3, "BF16 Correctness");
}

#[test]
fn test_matmul_q8_0_full_pipeline() {
    let m = 2; 
    let k = 512; // Must be multiple of 32
    let n = 128;

    let a = random_matrix(m, k, 200);
    let b_f32 = random_matrix(n, k, 201);
    let b_q8_0 = quantize_matrix_q8_0(&b_f32).expect("Quantization failed");
    let actual = matmul_2d_cpu_q8_0(&a.view(), &b_q8_0);
    let expected = ground_truth_matmul(&a, &b_f32);
    assert_matrices_close(&actual, &expected, 0.05, "Q8_0 Full Pipeline");
}

#[test]
fn test_matmul_q8_0_decode_path() {
    let m = 1;
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
    let k = 33; 
    let a = random_matrix(1, k, 0);
    let dummy_block = crate::cpu::kernels::q_common::BlockQ8_0 {
        d: half::f16::from_f32(1.0),
        qs: [0; 32],
    };
    let b_weights = vec![dummy_block; 10];

    matmul_2d_cpu_q8_0(&a.view(), &b_weights);
}


#[test]
fn test_matmul_q4_k_plumbing() {
    let m = 1;
    let k = 256; // QK_K
    let n = 10;

    let a = random_matrix(m, k, 300);
    let num_blocks = (n * k) / QK_K;
    let dummy_block = BlockQ4_K {
        d: half::f16::from_f32(1.0),
        dmin: half::f16::from_f32(0.0),
        scales: [0; 12],
        qs: [0; QK_K / 2],
    };
    let b_weights = vec![dummy_block; num_blocks];
    let output = matmul_2d_cpu_q4_k(&a.view(), &b_weights);
    assert_eq!(output.shape(), &[m, n]);
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_matmul_q6_k_plumbing() {
    let m = 4; // Prefill path
    let k = 512; // 2 * QK_K
    let n = 10;

    let a = random_matrix(m, k, 301);
    let num_blocks = (n * k) / QK_K;
    let dummy_block = BlockQ6_K {
        ql: [0; 128],
        qh: [0; 64],
        scales: [0; 16],
        d: half::f16::from_f32(1.0),
    };
    let b_weights = vec![dummy_block; num_blocks];
    let output = matmul_2d_cpu_q6_k(&a.view(), &b_weights);

    assert_eq!(output.shape(), &[m, n]);
    assert!(output.iter().all(|x| x.is_finite()));
}
#[test]
fn test_dimension_mismatches() {
    let a = random_matrix(1, 10, 0);
    let b = random_matrix(1, 11, 0); // K=11 != 10

    let result = std::panic::catch_unwind(|| matmul_2d_cpu_f32(&a.view(), &b.view()));
    assert!(result.is_err(), "Should panic on mismatched K dimension");
}

#[test]
fn test_zero_matrix() {
    let a = random_matrix(2, 32, 50);
    let b = Array2::<f32>::zeros((4, 32));

    let actual = matmul_2d_cpu_f32(&a.view(), &b.view());
    assert!(
        actual.iter().all(|&x| x == 0.0),
        "Multiplication by zero matrix failed"
    );
}
