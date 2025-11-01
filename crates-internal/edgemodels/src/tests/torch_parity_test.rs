use crate::sentence_encoder::SentenceEncoder;
use crate::cross_encoder::CrossEncoder;
use edgetransformers::models::ModelType;
use edgetransformers::traits::Device;
use edgetransformers::gpu_context::WgpuContext;
use std::process::Command;
use std::sync::Arc;
use serde_json;

// A tolerance for float comparisons. Even with identical logic, tiny differences
// in GELU implementations or compiler math can lead to microscopic variations.
const FLOAT_TOLERANCE: f32 = 1e-5;

// ===================================================================
//                        HELPER FUNCTIONS
// ===================================================================

/// A helper function for comparing vectors of floats with a tolerance.
/// Using `assert_eq!` directly on floats is bad practice.
fn assert_vecs_are_close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len(), "Vectors have different lengths (Rust: {}, Python: {})", a.len(), b.len());
    for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (val_a - val_b).abs() < tolerance,
            "Mismatch at index {}: Rust={:.6}, Python={:.6}, Diff={:.6}", i, val_a, val_b, (val_a - val_b).abs()
        );
    }
}

/// Executes a Python script as a subprocess and captures its stdout.
fn get_python_output(script_path: &str, args: &[&str]) -> String {
    let output = Command::new("python3")
        .arg(script_path)
        .args(args)
        .output()
        .expect(&format!("Failed to execute Python script: {}", script_path));

    assert!(output.status.success(), "Python script failed with error: {}", String::from_utf8_lossy(&output.stderr));
    
    String::from_utf8(output.stdout)
        .expect("Python output was not valid UTF-8")
        .trim()
        .to_string()
}

// ===================================================================
//                        SENTENCE EMBEDDING TESTS
// ===================================================================

async fn run_embedding_test(device: Device, ctx: Option<Arc<WgpuContext>>) {
    // --- 1. SETUP ---
    let model_type = ModelType::MiniLML6V2;
    // The model name expected by the `sentence-transformers` library
    let python_model_name = "all-MiniLM-L6-v2";
    let sentences = vec![
        "The cat sits on the mat",
        "A feline rests on a rug",
        "Dogs are playing in the park",
    ];

    // --- 2. GET PYTORCH (EXPECTED) OUTPUT ---
    let python_output = get_python_output(
        "src/tests/python/generate_embeddings.py",
        &["--model", python_model_name, "--sentences", sentences[0], sentences[1], sentences[2]],
    );
    let python_embeddings: Vec<Vec<f32>> = serde_json::from_str(&python_output).expect("Failed to parse Python JSON output for embeddings");

    // --- 3. GET RUST (ACTUAL) OUTPUT ---
    let encoder = SentenceEncoder::from_registry(model_type, None, device, ctx).await.unwrap();
    let rust_embeddings = encoder.encode_batch(&sentences).await.unwrap();

    // --- 4. ASSERT PARITY ---
    assert_eq!(rust_embeddings.len(), python_embeddings.len(), "Number of embeddings does not match");
    for i in 0..rust_embeddings.len() {
        assert_vecs_are_close(&rust_embeddings[i], &python_embeddings[i], FLOAT_TOLERANCE);
    }
}
#[tokio::test]
async fn test_cls_pooling_parity_cpu() {
    println!("--- LITMUS TEST: Testing CLS Pooling on CPU ---");
    let model_type = ModelType::MiniLML6V2;
    let python_model_name = "all-MiniLM-L6-v2";
    let sentence = "This is a test sentence.";

    // --- Get PyTorch CLS embedding ---
    let python_output = get_python_output(
        "src/tests/python/generate_cls_embedding.py",
        &["--model", python_model_name, "--sentence", sentence],
    );
    let python_embedding: Vec<f32> = serde_json::from_str(&python_output).unwrap();

    // --- Get Rust CLS embedding ---
    let encoder = SentenceEncoder::from_registry(model_type, None, Device::Cpu, None).await.unwrap();
    // Explicitly call encode with "cls" pooling
    let rust_embedding = encoder.encode2(sentence, "cls").await.unwrap();

    // --- Assert Parity ---
    assert_vecs_are_close(&rust_embedding, &python_embedding, FLOAT_TOLERANCE);
    println!("--- CLS POOLING TEST PASSED ---");
}
#[tokio::test]
async fn test_embedding_cpu() {
    println!("--- Testing Sentence Embeddings on CPU ---");
    run_embedding_test(Device::Cpu, None).await;
}

#[tokio::test]
#[cfg(feature = "gpu")] // Optional: run GPU tests only if a "gpu" feature is enabled in Cargo.toml
async fn test_embedding_gpu() {
    println!("--- Testing Sentence Embeddings on GPU ---");
    let ctx = Arc::new(WgpuContext::new().await.unwrap());
    run_embedding_test(Device::Wgpu, Some(ctx)).await;
}


// ===================================================================
//                        CROSS-ENCODER SCORING TESTS
// ===================================================================

async fn run_cross_encoder_score_test(device: Device, ctx: Option<Arc<WgpuContext>>) {
    // --- 1. SETUP ---
    let model_type = ModelType::MiniLML6V2CrossEncoder;
    // The model name expected by the `sentence-transformers` library
    let python_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2";
    let query = "How do I train a neural network?";
    let document = "Neural networks are trained using backpropagation and gradient descent.";

    // --- 2. GET PYTORCH (EXPECTED) OUTPUT ---
    let python_output = get_python_output(
        "src/tests/python/generate_cross_score.py",
        &["--model", python_model_name, "--query", query, "--document", document],
    );
    let python_score: f32 = python_output.parse().expect("Failed to parse Python score");

    // --- 3. GET RUST (ACTUAL) OUTPUT ---
    let cross_encoder = CrossEncoder::from_registry(model_type, None, device, ctx).await.unwrap();
    let rust_score = cross_encoder.predict(query, document).await.unwrap();

    // --- 4. ASSERT PARITY ---
    assert!(
        (rust_score - python_score).abs() < FLOAT_TOLERANCE,
        "Score mismatch: Rust={:.6}, Python={:.6}", rust_score, python_score
    );
}

#[tokio::test]
async fn test_cross_encoder_score_cpu() {
    println!("--- Testing Cross-Encoder Scoring on CPU ---");
    run_cross_encoder_score_test(Device::Cpu, None).await;
}

#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_cross_encoder_score_gpu() {
    println!("--- Testing Cross-Encoder Scoring on GPU ---");
    let ctx = Arc::new(WgpuContext::new().await.unwrap());
    run_cross_encoder_score_test(Device::Wgpu, Some(ctx)).await;
}


// ===================================================================
//                        CROSS-ENCODER RERANKING TESTS
// ===================================================================

async fn run_cross_encoder_rerank_test(device: Device, ctx: Option<Arc<WgpuContext>>) {
    // --- 1. SETUP ---
    let model_type = ModelType::MiniLML6V2CrossEncoder;
    let python_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2";
    let query = "machine learning algorithms";
    let documents = vec![
        "Machine learning algorithms include decision trees, neural networks, and SVMs.", // Should be 1st
        "The weather forecast predicts rain tomorrow.",                                    // Should be last
        "Deep learning is a subset of machine learning using neural networks.",           // Should be 2nd
        "Cooking recipes for Italian pasta dishes.",                                      // Should be 3rd
    ];

    // --- 2. GET PYTORCH (EXPECTED) OUTPUT ---
    let python_output = get_python_output(
        "src/tests/python/generate_rerank_indices.py",
        &["--model", python_model_name, "--query", query, "--documents", documents[0], documents[1], documents[2], documents[3]],
    );
    let python_indices: Vec<usize> = serde_json::from_str(&python_output).expect("Failed to parse Python JSON output for reranking");

    // --- 3. GET RUST (ACTUAL) OUTPUT ---
    let cross_encoder = CrossEncoder::from_registry(model_type, None, device, ctx).await.unwrap();
    let rust_indices = cross_encoder.rerank(query, &documents).await.unwrap();

    // --- 4. ASSERT PARITY ---
    // For reranking, the exact order is what matters. No tolerance needed.
    assert_eq!(rust_indices, python_indices, "Reranked order does not match");
}

#[tokio::test]
async fn test_cross_encoder_rerank_cpu() {
    println!("--- Testing Cross-Encoder Reranking on CPU ---");
    run_cross_encoder_rerank_test(Device::Cpu, None).await;
}

#[tokio::test]
#[cfg(feature = "gpu")]
async fn test_cross_encoder_rerank_gpu() {
    println!("--- Testing Cross-Encoder Reranking on GPU ---");
    let ctx = Arc::new(WgpuContext::new().await.unwrap());
    run_cross_encoder_rerank_test(Device::Wgpu, Some(ctx)).await;
}