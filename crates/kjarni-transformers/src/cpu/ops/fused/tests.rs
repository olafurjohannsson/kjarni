//! Unit tests for fused gate+up+silu operations.
//!
//! These tests verify that fused kernels produce identical results
//! to the reference implementation using separate matmuls.

use super::gate_up_silu::fused_gate_up_silu;
use crate::cpu::kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0};
use crate::cpu::ops::fused::simd_x86::fused_gate_up_silu_f32_parallel;
use crate::linear_layer::LinearLayer;
use crate::tensor::{DType, QuantizedMatrix};
use half::bf16;
use ndarray::{Array2, ArrayView2};
use std::sync::Arc;

// =============================================================================
// Test Helpers
// =============================================================================

/// SiLU activation for reference implementation
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Reference implementation: separate matmuls + silu + multiply
fn reference_gate_up_silu(
    gate: &LinearLayer,
    up: &LinearLayer,
    input: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
    tokens: usize,
) -> Vec<f32> {
    let input_2d = ArrayView2::from_shape((tokens, hidden_dim), input).unwrap();
    let gate_out = gate.matmul(&input_2d);
    let up_out = up.matmul(&input_2d);

    let mut result = vec![0.0f32; tokens * intermediate_dim];
    for t in 0..tokens {
        for i in 0..intermediate_dim {
            let g = gate_out[[t, i]];
            let u = up_out[[t, i]];
            result[t * intermediate_dim + i] = silu(g) * u;
        }
    }
    result
}

/// Compare two slices with relative tolerance
fn assert_close(expected: &[f32], actual: &[f32], tolerance: f32, msg: &str) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{}: length mismatch {} vs {}",
        msg,
        expected.len(),
        actual.len()
    );

    for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        let diff = (e - a).abs();
        let rel_diff = if e.abs() > 1e-6 { diff / e.abs() } else { diff };

        assert!(
            rel_diff < tolerance || diff < 1e-6,
            "{}: mismatch at index {}: expected {}, got {}, diff={}, rel_diff={}",
            msg,
            i,
            e,
            a,
            diff,
            rel_diff
        );
    }
}

/// Generate deterministic pseudo-random weights
fn generate_weights(rows: usize, cols: usize, seed: usize) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let idx = i * cols + j + seed;
        ((idx * 17 + 13) % 1000) as f32 * 0.002 - 1.0
    })
}

/// Generate deterministic input
fn generate_input(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let idx = i + seed;
            ((idx * 31 + 7) % 1000) as f32 * 0.002 - 1.0
        })
        .collect()
}

// =============================================================================
// F32 Tests
// =============================================================================
fn reference_gate_up_silu2(input: &[f32], gate: &[f32], up: &[f32], k: usize) -> Vec<f32> {
        let n = gate.len() / k;
        let mut output = vec![0.0f32; n];

        for i in 0..n {
            let offset = i * k;
            let mut gate_sum = 0.0f32;
            let mut up_sum = 0.0f32;

            for j in 0..k {
                let val = input[j];
                gate_sum += val * gate[offset + j];
                up_sum += val * up[offset + j];
            }

            output[i] = silu(gate_sum) * up_sum;
        }
        output
    }
#[test]
fn test_fused_f32_avx2_matches_reference() {
    let k = 256;
    let n = 512;

    // Generate deterministic test data
    let input: Vec<f32> = (0..k)
        .map(|i| ((i * 31 + 7) % 1000) as f32 * 0.002 - 1.0)
        .collect();
    let gate: Vec<f32> = (0..n * k)
        .map(|i| ((i * 17 + 13) % 1000) as f32 * 0.002 - 1.0)
        .collect();
    let up: Vec<f32> = (0..n * k)
        .map(|i| ((i * 19 + 11) % 1000) as f32 * 0.002 - 1.0)
        .collect();

    let expected = reference_gate_up_silu2(&input, &gate, &up, k);

    let mut output = vec![0.0f32; n];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                use crate::cpu::ops::fused::simd_x86::fused_gate_up_silu_f32_avx2;

                fused_gate_up_silu_f32_avx2(
                    &mut output,
                    input.as_ptr(),
                    gate.as_ptr(),
                    up.as_ptr(),
                    k,
                );
            }

            for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
                let diff = (exp - got).abs();
                assert!(
                    diff < 1e-4,
                    "Mismatch at {}: expected {}, got {}, diff={}",
                    i,
                    exp,
                    got,
                    diff
                );
            }
        }
    }
}



#[test]
fn test_fused_f32_parallel() {
    let k = 3072;
    let n = 8192;

    let input: Vec<f32> = (0..k).map(|i| (i as f32 * 0.001) - 1.5).collect();
    let gate: Vec<f32> = (0..n * k)
        .map(|i| ((i % 1000) as f32 * 0.001) - 0.5)
        .collect();
    let up: Vec<f32> = (0..n * k)
        .map(|i| ((i % 997) as f32 * 0.001) - 0.5)
        .collect();

    let expected = reference_gate_up_silu2(&input, &gate, &up, k);

    let mut output = vec![0.0f32; n];
    fused_gate_up_silu_f32_parallel(&mut output, &input, &gate, &up, k);

    for (i, (&exp, &got)) in expected.iter().zip(output.iter()).enumerate() {
        let diff = (exp - got).abs();
        assert!(
            diff < 1e-3,
            "Mismatch at {}: expected {}, got {}, diff={}",
            i,
            exp,
            got,
            diff
        );
    }
}

#[test]
fn test_fused_f32_decode_matches_reference() {
    let hidden = 256;
    let intermediate = 512;
    let tokens = 1;

    let gate_w = generate_weights(intermediate, hidden, 42);
    let up_w = generate_weights(intermediate, hidden, 123);

    let gate = LinearLayer::new_f32(gate_w, None);
    let up = LinearLayer::new_f32(up_w, None);

    let input = generate_input(tokens * hidden, 0);

    // Reference
    let expected = reference_gate_up_silu(&gate, &up, &input, hidden, intermediate, tokens);

    // Fused
    let mut output = vec![0.0f32; tokens * intermediate];
    fused_gate_up_silu(&gate, &up, &input, &mut output, tokens).unwrap();

    assert_close(&expected, &output, 1e-5, "F32 decode");
}

#[test]
fn test_fused_f32_prefill_matches_reference() {
    let hidden = 256;
    let intermediate = 512;
    let tokens = 16;

    let gate_w = generate_weights(intermediate, hidden, 42);
    let up_w = generate_weights(intermediate, hidden, 123);

    let gate = LinearLayer::new_f32(gate_w, None);
    let up = LinearLayer::new_f32(up_w, None);

    let input = generate_input(tokens * hidden, 0);

    // Reference
    let expected = reference_gate_up_silu(&gate, &up, &input, hidden, intermediate, tokens);

    // Fused
    let mut output = vec![0.0f32; tokens * intermediate];
    fused_gate_up_silu(&gate, &up, &input, &mut output, tokens).unwrap();

    assert_close(&expected, &output, 1e-4, "F32 prefill");
}

#[test]
fn test_fused_f32_large_hidden_dim() {
    // Test with realistic Llama-like dimensions
    let hidden = 3072;
    let intermediate = 8192;
    let tokens = 1;

    let gate_w = generate_weights(intermediate, hidden, 42);
    let up_w = generate_weights(intermediate, hidden, 123);

    let gate = LinearLayer::new_f32(gate_w, None);
    let up = LinearLayer::new_f32(up_w, None);

    let input = generate_input(tokens * hidden, 0);

    let expected = reference_gate_up_silu(&gate, &up, &input, hidden, intermediate, tokens);

    let mut output = vec![0.0f32; tokens * intermediate];
    fused_gate_up_silu(&gate, &up, &input, &mut output, tokens).unwrap();

    assert_close(&expected, &output, 1e-3, "F32 large dimensions");
}

// =============================================================================
// BF16 Tests
// =============================================================================

#[test]
fn test_fused_bf16_decode_matches_reference() {
    let hidden = 256;
    let intermediate = 512;
    let tokens = 1;

    // Create F32 reference
    let gate_w_f32 = generate_weights(intermediate, hidden, 42);
    let up_w_f32 = generate_weights(intermediate, hidden, 123);

    // Convert to BF16
    let gate_w_bf16 = gate_w_f32.mapv(bf16::from_f32);
    let up_w_bf16 = up_w_f32.mapv(bf16::from_f32);

    let gate_bf16 = LinearLayer::new_bf16(gate_w_bf16, None);
    let up_bf16 = LinearLayer::new_bf16(up_w_bf16, None);

    let input = generate_input(tokens * hidden, 0);

    // Reference using BF16 matmul
    let expected =
        reference_gate_up_silu(&gate_bf16, &up_bf16, &input, hidden, intermediate, tokens);

    // Fused
    let mut output = vec![0.0f32; tokens * intermediate];
    fused_gate_up_silu(&gate_bf16, &up_bf16, &input, &mut output, tokens).unwrap();

    // BF16 has lower precision
    assert_close(&expected, &output, 1e-2, "BF16 decode");
}

#[test]
fn test_fused_bf16_prefill_matches_reference() {
    let hidden = 256;
    let intermediate = 512;
    let tokens = 8;

    let gate_w_bf16 = generate_weights(intermediate, hidden, 42).mapv(bf16::from_f32);
    let up_w_bf16 = generate_weights(intermediate, hidden, 123).mapv(bf16::from_f32);

    let gate = LinearLayer::new_bf16(gate_w_bf16, None);
    let up = LinearLayer::new_bf16(up_w_bf16, None);

    let input = generate_input(tokens * hidden, 0);

    let expected = reference_gate_up_silu(&gate, &up, &input, hidden, intermediate, tokens);

    let mut output = vec![0.0f32; tokens * intermediate];
    fused_gate_up_silu(&gate, &up, &input, &mut output, tokens).unwrap();

    assert_close(&expected, &output, 1e-2, "BF16 prefill");
}

// =============================================================================
// Q8_0 Tests
// =============================================================================

#[test]
fn test_fused_q8_0_decode_matches_reference() {
    let hidden = 256; // Must be multiple of 32
    let intermediate = 512;
    let tokens = 1;

    let gate_w = generate_weights(intermediate, hidden, 42);
    let up_w = generate_weights(intermediate, hidden, 123);

    let gate_f32 = LinearLayer::new_f32(gate_w.clone(), None);
    let up_f32 = LinearLayer::new_f32(up_w.clone(), None);

    // Quantize to Q8_0
    let gate_q8 = gate_f32.to_quantized(DType::Q8_0).unwrap();
    let up_q8 = up_f32.to_quantized(DType::Q8_0).unwrap();

    let input = generate_input(tokens * hidden, 0);

    // Reference using Q8_0 matmul path
    let expected = reference_gate_up_silu(&gate_q8, &up_q8, &input, hidden, intermediate, tokens);

    // Fused
    let mut output = vec![0.0f32; tokens * intermediate];
    fused_gate_up_silu(&gate_q8, &up_q8, &input, &mut output, tokens).unwrap();

    assert_close(&expected, &output, 1e-4, "Q8_0 decode");
}

#[test]
fn test_fused_q8_0_prefill_matches_reference() {
    let hidden = 256;
    let intermediate = 512;
    let tokens = 8;

    let gate_f32 = LinearLayer::new_f32(generate_weights(intermediate, hidden, 42), None);
    let up_f32 = LinearLayer::new_f32(generate_weights(intermediate, hidden, 123), None);

    let gate_q8 = gate_f32.to_quantized(DType::Q8_0).unwrap();
    let up_q8 = up_f32.to_quantized(DType::Q8_0).unwrap();

    let input = generate_input(tokens * hidden, 0);

    let expected = reference_gate_up_silu(&gate_q8, &up_q8, &input, hidden, intermediate, tokens);

    let mut output = vec![0.0f32; tokens * intermediate];
    fused_gate_up_silu(&gate_q8, &up_q8, &input, &mut output, tokens).unwrap();

    assert_close(&expected, &output, 1e-3, "Q8_0 prefill");
}

// =============================================================================
// Q4_K Tests (if you have Q4_K quantization)
// =============================================================================

// Note: These tests require Q4_K quantization support
// Uncomment when LinearLayer::to_quantized supports Q4_K

/*
#[test]
fn test_fused_q4_k_decode_matches_reference() {
    let hidden = 256; // Must be multiple of 256
    let intermediate = 512;
    let tokens = 1;

    let gate_f32 = LinearLayer::new_f32(generate_weights(intermediate, hidden, 42), None);
    let up_f32 = LinearLayer::new_f32(generate_weights(intermediate, hidden, 123), None);

    let gate_q4 = gate_f32.to_quantized(DType::Q4_K).unwrap();
    let up_q4 = up_f32.to_quantized(DType::Q4_K).unwrap();

    let input = generate_input(tokens * hidden, 0);

    let expected = reference_gate_up_silu(&gate_q4, &up_q4, &input, hidden, intermediate, tokens);

    let mut output = vec![0.0f32; tokens * intermediate];
    fused_gate_up_silu(&gate_q4, &up_q4, &input, &mut output, tokens).unwrap();

    // Q4_K has lower precision due to 4-bit quantization
    assert_close(&expected, &output, 1e-2, "Q4_K decode");
}
*/

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_fused_dtype_mismatch_error() {
    let hidden = 256;
    let intermediate = 512;

    let gate = LinearLayer::new_f32(generate_weights(intermediate, hidden, 42), None);
    let up = LinearLayer::new_bf16(
        generate_weights(intermediate, hidden, 123).mapv(bf16::from_f32),
        None,
    );

    let input = generate_input(hidden, 0);
    let mut output = vec![0.0f32; intermediate];

    let result = fused_gate_up_silu(&gate, &up, &input, &mut output, 1);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("same dtype"));
}

#[test]
fn test_fused_dimension_mismatch_error() {
    let hidden = 256;
    let intermediate = 512;

    // Gate and up have different output dimensions
    let gate = LinearLayer::new_f32(generate_weights(intermediate, hidden, 42), None);
    let up = LinearLayer::new_f32(generate_weights(intermediate + 1, hidden, 123), None);

    let input = generate_input(hidden, 0);
    let mut output = vec![0.0f32; intermediate];

    let result = fused_gate_up_silu(&gate, &up, &input, &mut output, 1);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("dimension mismatch")
    );
}

#[test]
fn test_fused_input_length_error() {
    let hidden = 256;
    let intermediate = 512;

    let gate = LinearLayer::new_f32(generate_weights(intermediate, hidden, 42), None);
    let up = LinearLayer::new_f32(generate_weights(intermediate, hidden, 123), None);

    // Wrong input length
    let input = generate_input(hidden - 1, 0);
    let mut output = vec![0.0f32; intermediate];

    let result = fused_gate_up_silu(&gate, &up, &input, &mut output, 1);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("input length"));
}

#[test]
fn test_fused_output_length_error() {
    let hidden = 256;
    let intermediate = 512;

    let gate = LinearLayer::new_f32(generate_weights(intermediate, hidden, 42), None);
    let up = LinearLayer::new_f32(generate_weights(intermediate, hidden, 123), None);

    let input = generate_input(hidden, 0);
    // Wrong output length
    let mut output = vec![0.0f32; intermediate - 1];

    let result = fused_gate_up_silu(&gate, &up, &input, &mut output, 1);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("output length"));
}

// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_fused_zeros() {
    let hidden = 256;
    let intermediate = 512;

    let gate = LinearLayer::new_f32(Array2::zeros((intermediate, hidden)), None);
    let up = LinearLayer::new_f32(Array2::zeros((intermediate, hidden)), None);

    let input = vec![0.0f32; hidden];
    let mut output = vec![1.0f32; intermediate]; // Pre-fill with 1s to verify overwrite

    fused_gate_up_silu(&gate, &up, &input, &mut output, 1).unwrap();

    // silu(0) * 0 = 0
    for &v in &output {
        assert!(v.abs() < 1e-10, "Expected zero, got {}", v);
    }
}

#[test]
fn test_fused_identity_like() {
    // Test with specific values where we know the expected output
    let hidden = 32;
    let intermediate = 32;

    // Identity-like weights (scaled for numerical stability)
    let mut gate_w = Array2::zeros((intermediate, hidden));
    let mut up_w = Array2::zeros((intermediate, hidden));
    for i in 0..intermediate.min(hidden) {
        gate_w[[i, i]] = 1.0;
        up_w[[i, i]] = 1.0;
    }

    let gate = LinearLayer::new_f32(gate_w, None);
    let up = LinearLayer::new_f32(up_w, None);

    let input: Vec<f32> = (0..hidden).map(|i| i as f32 * 0.1).collect();

    let expected = reference_gate_up_silu(&gate, &up, &input, hidden, intermediate, 1);

    let mut output = vec![0.0f32; intermediate];
    fused_gate_up_silu(&gate, &up, &input, &mut output, 1).unwrap();

    assert_close(&expected, &output, 1e-5, "Identity-like");
}

// =============================================================================
// Benchmarking Helpers (for manual testing)
// =============================================================================

#[test]
#[ignore] // Run with: cargo test --release -- --ignored --nocapture
fn bench_fused_vs_separate_f32() {
    use std::time::Instant;

    let hidden = 3072;
    let intermediate = 8192;
    let tokens = 1;
    let iterations = 100;

    let gate = LinearLayer::new_f32(generate_weights(intermediate, hidden, 42), None);
    let up = LinearLayer::new_f32(generate_weights(intermediate, hidden, 123), None);
    let input = generate_input(tokens * hidden, 0);

    // Warmup
    let mut output = vec![0.0f32; tokens * intermediate];
    for _ in 0..10 {
        fused_gate_up_silu(&gate, &up, &input, &mut output, tokens).unwrap();
    }

    // Benchmark fused
    let start = Instant::now();
    for _ in 0..iterations {
        fused_gate_up_silu(&gate, &up, &input, &mut output, tokens).unwrap();
    }
    let fused_time = start.elapsed();

    // Benchmark separate
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = reference_gate_up_silu(&gate, &up, &input, hidden, intermediate, tokens);
    }
    let separate_time = start.elapsed();

    println!("\n=== F32 Decode Benchmark ({} iterations) ===", iterations);
    println!("Hidden: {}, Intermediate: {}", hidden, intermediate);
    println!(
        "Fused:    {:?} ({:.2} µs/iter)",
        fused_time,
        fused_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "Separate: {:?} ({:.2} µs/iter)",
        separate_time,
        separate_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "Speedup:  {:.2}x",
        separate_time.as_nanos() as f64 / fused_time.as_nanos() as f64
    );
}

#[test]
#[ignore]
fn bench_fused_vs_separate_bf16() {
    use std::time::Instant;

    let hidden = 3072;
    let intermediate = 8192;
    let tokens = 1;
    let iterations = 100;

    let gate = LinearLayer::new_bf16(
        generate_weights(intermediate, hidden, 42).mapv(bf16::from_f32),
        None,
    );
    let up = LinearLayer::new_bf16(
        generate_weights(intermediate, hidden, 123).mapv(bf16::from_f32),
        None,
    );
    let input = generate_input(tokens * hidden, 0);

    let mut output = vec![0.0f32; tokens * intermediate];

    // Warmup
    for _ in 0..10 {
        fused_gate_up_silu(&gate, &up, &input, &mut output, tokens).unwrap();
    }

    // Benchmark fused
    let start = Instant::now();
    for _ in 0..iterations {
        fused_gate_up_silu(&gate, &up, &input, &mut output, tokens).unwrap();
    }
    let fused_time = start.elapsed();

    // Benchmark separate
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = reference_gate_up_silu(&gate, &up, &input, hidden, intermediate, tokens);
    }
    let separate_time = start.elapsed();

    println!(
        "\n=== BF16 Decode Benchmark ({} iterations) ===",
        iterations
    );
    println!("Hidden: {}, Intermediate: {}", hidden, intermediate);
    println!(
        "Fused:    {:?} ({:.2} µs/iter)",
        fused_time,
        fused_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "Separate: {:?} ({:.2} µs/iter)",
        separate_time,
        separate_time.as_micros() as f64 / iterations as f64
    );
    println!(
        "Speedup:  {:.2}x",
        separate_time.as_nanos() as f64 / fused_time.as_nanos() as f64
    );
}
