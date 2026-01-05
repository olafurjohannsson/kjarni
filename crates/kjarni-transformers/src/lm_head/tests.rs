
use ndarray::{Array2, Array3};

// ============================================================================
// LMHeadConfig Tests
// ============================================================================

#[test]
fn test_lm_head_config_new() {
    let config = LMHeadConfig::new("lm_head.weight", 32000, 2048);

    assert_eq!(config.weight_name, "lm_head.weight");
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.hidden_size, 2048);
}

#[test]
fn test_lm_head_config_from_string() {
    let config = LMHeadConfig::new(String::from("model.lm_head"), 50257, 768);

    assert_eq!(config.weight_name, "model.lm_head");
    assert_eq!(config.vocab_size, 50257);
    assert_eq!(config.hidden_size, 768);
}

// ============================================================================
// LoadedLMHead CPU Tests (No GPU Required)
// ============================================================================

mod cpu_tests {
    use super::*;
    use crate::linear_layer::LinearLayer;

    /// Helper to create a LoadedLMHead with synthetic CPU weights
    fn create_cpu_lm_head(vocab_size: usize, hidden_size: usize) -> LoadedLMHead {
        // Create simple identity-like weights for testing
        // Weight shape: [vocab_size, hidden_size]
        let weights = Array2::<f32>::zeros((vocab_size, hidden_size));
        let linear = LinearLayer::from(weights);

        LoadedLMHead {
            cpu_weights: Some(linear),
            gpu_weights: None,
            gpu_kernel: None,
            vocab_size,
            hidden_size,
            context: None,
        }
    }

    #[test]
    fn test_loaded_lm_head_cpu_only() {
        let head = create_cpu_lm_head(1000, 256);

        assert!(head.has_cpu());
        assert!(!head.has_gpu());
        assert_eq!(head.vocab_size(), 1000);
        assert_eq!(head.hidden_size(), 256);
    }

    #[test]
    fn test_loaded_lm_head_forward_cpu_shape() {
        let vocab_size = 1000;
        let hidden_size = 256;
        let head = create_cpu_lm_head(vocab_size, hidden_size);

        // Input: [batch=2, seq=5, hidden=256]
        let hidden_states = Array3::<f32>::zeros((2, 5, hidden_size));

        let logits = head.forward_cpu(&hidden_states).unwrap();

        // Output: [batch=2, seq=5, vocab=1000]
        assert_eq!(logits.dim(), (2, 5, vocab_size));
    }

    #[test]
    fn test_loaded_lm_head_forward_cpu_single_token() {
        let vocab_size = 500;
        let hidden_size = 128;
        let head = create_cpu_lm_head(vocab_size, hidden_size);

        // Single token decode: [batch=1, seq=1, hidden=128]
        let hidden_states = Array3::<f32>::zeros((1, 1, hidden_size));

        let logits = head.forward_cpu(&hidden_states).unwrap();

        assert_eq!(logits.dim(), (1, 1, vocab_size));
    }

    #[test]
    fn test_loaded_lm_head_forward_cpu_last_token() {
        let vocab_size = 500;
        let hidden_size = 128;
        let head = create_cpu_lm_head(vocab_size, hidden_size);

        // Sequence: [batch=1, seq=10, hidden=128]
        let hidden_states = Array3::<f32>::zeros((1, 10, hidden_size));

        let logits = head.forward_cpu_last_token(&hidden_states).unwrap();

        // Should return [vocab_size] for last token only
        assert_eq!(logits.dim(), vocab_size);
    }

    #[test]
    fn test_loaded_lm_head_forward_cpu_values() {
        let vocab_size = 4;
        let hidden_size = 3;

        // Create weights where each vocab token responds to one hidden dim
        // Weight[0, :] = [1, 0, 0] -> responds to hidden[0]
        // Weight[1, :] = [0, 1, 0] -> responds to hidden[1]
        // etc.
        let mut weights = Array2::<f32>::zeros((vocab_size, hidden_size));
        weights[[0, 0]] = 1.0;
        weights[[1, 1]] = 1.0;
        weights[[2, 2]] = 1.0;
        weights[[3, 0]] = 0.5;
        weights[[3, 1]] = 0.5;

        let linear = LinearLayer::from(weights);
        let head = LoadedLMHead {
            cpu_weights: Some(linear),
            gpu_weights: None,
            gpu_kernel: None,
            vocab_size,
            hidden_size,
            context: None,
        };

        // Input hidden state: [1, 0, 0] should give high logit for vocab[0]
        let mut hidden = Array3::<f32>::zeros((1, 1, hidden_size));
        hidden[[0, 0, 0]] = 1.0;

        let logits = head.forward_cpu(&hidden).unwrap();

        assert!(
            (logits[[0, 0, 0]] - 1.0).abs() < 1e-6,
            "Vocab 0 should be 1.0"
        );
        assert!(
            (logits[[0, 0, 1]] - 0.0).abs() < 1e-6,
            "Vocab 1 should be 0.0"
        );
        assert!(
            (logits[[0, 0, 2]] - 0.0).abs() < 1e-6,
            "Vocab 2 should be 0.0"
        );
        assert!(
            (logits[[0, 0, 3]] - 0.5).abs() < 1e-6,
            "Vocab 3 should be 0.5"
        );
    }

    #[test]
    fn test_loaded_lm_head_forward_cpu_no_weights_error() {
        let head = LoadedLMHead {
            cpu_weights: None,
            gpu_weights: None,
            gpu_kernel: None,
            vocab_size: 1000,
            hidden_size: 256,
            context: None,
        };

        let hidden_states = Array3::<f32>::zeros((1, 1, 256));

        let result = head.forward_cpu(&hidden_states);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("CPU LM head not loaded")
        );
    }

    #[test]
    fn test_loaded_lm_head_batched() {
        let vocab_size = 100;
        let hidden_size = 64;
        let head = create_cpu_lm_head(vocab_size, hidden_size);

        // Batched input: [batch=4, seq=8, hidden=64]
        let hidden_states = Array3::<f32>::ones((4, 8, hidden_size));

        let logits = head.forward_cpu(&hidden_states).unwrap();

        assert_eq!(logits.dim(), (4, 8, vocab_size));
    }
}

// ============================================================================
// LoadedLMHead GPU Tests
// ============================================================================

use super::*;
use crate::WgpuContext;
use crate::gpu_ops::{GpuTensor, GpuTensorPool};
use crate::linear_layer::LinearLayer;
use crate::tensor::DType;
use std::sync::Arc;

async fn setup_gpu_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}

/// Helper to create a LoadedLMHead with synthetic GPU weights
async fn create_gpu_lm_head(
    ctx: &Arc<WgpuContext>,
    vocab_size: usize,
    hidden_size: usize,
) -> LoadedLMHead {
    use crate::gpu_ops::primitives::linear::GpuLinearLayer;

    // Create weights on GPU
    let weights_cpu = Array2::<f32>::zeros((vocab_size, hidden_size));
    let gpu_weights = GpuTensor::from_ndarray(ctx, &weights_cpu).unwrap();
    let gpu_kernel = GpuLinearLayer::new(ctx);

    LoadedLMHead {
        cpu_weights: None,
        gpu_weights: Some(gpu_weights),
        gpu_kernel: Some(gpu_kernel),
        vocab_size,
        hidden_size,
        context: Some(ctx.clone()),
    }
}

#[tokio::test]
async fn test_loaded_lm_head_gpu_only() {
    let ctx = setup_gpu_context().await;
    let head = create_gpu_lm_head(&ctx, 1000, 256).await;

    assert!(!head.has_cpu());
    assert!(head.has_gpu());
    assert_eq!(head.vocab_size(), 1000);
    assert_eq!(head.hidden_size(), 256);
}

#[tokio::test]
async fn test_loaded_lm_head_forward_gpu_shape() {
    let ctx = setup_gpu_context().await;
    let vocab_size = 1000;
    let hidden_size = 256;
    let head = create_gpu_lm_head(&ctx, vocab_size, hidden_size).await;

    // Create GPU input
    let hidden_cpu = Array3::<f32>::zeros((2, 5, hidden_size));
    let hidden_gpu = GpuTensor::from_ndarray(&ctx, &hidden_cpu).unwrap();

    // Create encoder and pool
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(ctx.clone());

    let logits = head
        .forward_gpu(&mut encoder, &mut pool, &hidden_gpu)
        .unwrap();

    // Submit and verify shape
    ctx.queue.submit(Some(encoder.finish()));

    assert_eq!(logits.shape(), &[2, 5, vocab_size]);
}

#[tokio::test]
async fn test_loaded_lm_head_forward_gpu_values() {
    let ctx = setup_gpu_context().await;
    let vocab_size = 4;
    let hidden_size = 3;

    // Create weights
    let mut weights_cpu = Array2::<f32>::zeros((vocab_size, hidden_size));
    weights_cpu[[0, 0]] = 1.0;
    weights_cpu[[1, 1]] = 1.0;
    weights_cpu[[2, 2]] = 1.0;

    let gpu_weights = GpuTensor::from_ndarray(&ctx, &weights_cpu).unwrap();
    let gpu_kernel = crate::gpu_ops::primitives::linear::GpuLinearLayer::new(&ctx);

    let head = LoadedLMHead {
        cpu_weights: None,
        gpu_weights: Some(gpu_weights),
        gpu_kernel: Some(gpu_kernel),
        vocab_size,
        hidden_size,
        context: Some(ctx.clone()),
    };

    // Input: [1, 0, 0]
    let mut hidden_cpu = Array3::<f32>::zeros((1, 1, hidden_size));
    hidden_cpu[[0, 0, 0]] = 1.0;
    let hidden_gpu = GpuTensor::from_ndarray(&ctx, &hidden_cpu).unwrap();

    // Forward
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(ctx.clone());

    let logits_gpu = head
        .forward_gpu(&mut encoder, &mut pool, &hidden_gpu)
        .unwrap();

    ctx.queue.submit(Some(encoder.finish()));

    // Download and verify
    let logits_cpu = logits_gpu.to_ndarray_3d::<f32>().await.unwrap();

    assert!((logits_cpu[[0, 0, 0]] - 1.0).abs() < 1e-5);
    assert!((logits_cpu[[0, 0, 1]] - 0.0).abs() < 1e-5);
    assert!((logits_cpu[[0, 0, 2]] - 0.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_loaded_lm_head_forward_gpu_no_weights_error() {
    let ctx = setup_gpu_context().await;

    let head = LoadedLMHead {
        cpu_weights: None,
        gpu_weights: None,
        gpu_kernel: None,
        vocab_size: 1000,
        hidden_size: 256,
        context: Some(ctx.clone()),
    };

    let hidden_cpu = Array3::<f32>::zeros((1, 1, 256));
    let hidden_gpu = GpuTensor::from_ndarray(&ctx, &hidden_cpu).unwrap();

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(ctx);

    let result = head.forward_gpu(&mut encoder, &mut pool, &hidden_gpu);

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("GPU LM head not loaded")
    );
}

// kjarni-transformers/src/lm_head/tests.rs

#[tokio::test]
async fn test_loaded_lm_head_cpu_gpu_parity() {
    let ctx = setup_gpu_context().await;
    let vocab_size = 100;
    let hidden_size = 64;

    // Create identical weights for CPU and GPU
    let weights_cpu = Array2::<f32>::from_shape_fn((vocab_size, hidden_size), |(i, j)| {
        ((i * hidden_size + j) as f32) * 0.01
    });

    let linear = LinearLayer::from(weights_cpu.clone());
    
    // Ensure GPU weights are F32
    let gpu_weights = GpuTensor::from_ndarray(&ctx, &weights_cpu).unwrap();
    assert_eq!(gpu_weights.dtype(), DType::F32, "GPU weights should be F32");
    
    let gpu_kernel = crate::gpu_ops::primitives::linear::GpuLinearLayer::new(&ctx);

    let head = LoadedLMHead {
        cpu_weights: Some(linear),
        gpu_weights: Some(gpu_weights),
        gpu_kernel: Some(gpu_kernel),
        vocab_size,
        hidden_size,
        context: Some(ctx.clone()),
    };

    // Create input - use smaller values to reduce accumulation error
    let hidden_cpu = Array3::<f32>::from_shape_fn((1, 4, hidden_size), |(_, _, k)| {
        (k as f32) * 0.01  // Smaller values
    });
    let hidden_gpu = GpuTensor::from_ndarray(&ctx, &hidden_cpu).unwrap();
    assert_eq!(hidden_gpu.dtype(), DType::F32, "Input should be F32");

    // CPU forward
    let logits_cpu = head.forward_cpu(&hidden_cpu).unwrap();

    // GPU forward
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(ctx.clone());
    let logits_gpu_tensor = head
        .forward_gpu(&mut encoder, &mut pool, &hidden_gpu)
        .unwrap();
    ctx.queue.submit(Some(encoder.finish()));
    let logits_gpu = logits_gpu_tensor.to_ndarray_3d::<f32>().await.unwrap();

    // Debug: Print first few values
    println!("CPU logits first 5: {:?}", &logits_cpu.as_slice().unwrap()[..5]);
    println!("GPU logits first 5: {:?}", &logits_gpu.as_slice().unwrap()[..5]);
    println!("CPU shape: {:?}, GPU shape: {:?}", logits_cpu.shape(), logits_gpu.shape());

    // Compare with detailed error reporting
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = (0, 0, 0);
    let mut max_diff_values = (0.0f32, 0.0f32);
    
    for ((idx, &cpu_val), &gpu_val) in logits_cpu.indexed_iter().zip(logits_gpu.iter()) {
        let diff = (cpu_val - gpu_val).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = idx;
            max_diff_values = (cpu_val, gpu_val);
        }
    }

    println!(
        "Max diff: {} at {:?} (CPU: {}, GPU: {})",
        max_diff, max_diff_idx, max_diff_values.0, max_diff_values.1
    );

    // F32 matmul can have small differences due to operation ordering
    // 1e-3 is reasonable for large reductions
    assert!(
        max_diff < 1e-3,
        "CPU/GPU parity failed: max diff = {} at {:?} (CPU: {}, GPU: {})",
        max_diff, max_diff_idx, max_diff_values.0, max_diff_values.1
    );
}

// ============================================================================
// Integration Tests with Real Weights (Optional)
// ============================================================================

use super::*;
use crate::weights::ModelWeights;
use std::path::PathBuf;

fn get_test_model_path() -> Option<PathBuf> {
    std::env::var("KJARNI_TEST_MODEL_PATH")
        .ok()
        .map(PathBuf::from)
}

#[test]
fn test_loaded_lm_head_from_real_weights() {
    let Some(model_path) = get_test_model_path() else {
        eprintln!("Skipping: KJARNI_TEST_MODEL_PATH not set");
        return;
    };

    let weights = ModelWeights::new(&model_path).unwrap();

    let config = LMHeadConfig::new("lm_head.weight", 32000, 2048);

    let head = LoadedLMHead::new(
        None, // No GPU
        &weights, config, true,  // Load CPU
        false, // Don't load GPU
        None,  // Default dtype
    )
    .unwrap();

    assert!(head.has_cpu());
    assert!(!head.has_gpu());

    // Test forward
    let hidden = Array3::<f32>::ones((1, 1, 2048));
    let logits = head.forward_cpu(&hidden).unwrap();

    assert_eq!(logits.dim(), (1, 1, 32000));
}
