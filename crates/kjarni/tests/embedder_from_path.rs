//! `Embedder::from_path` loads from a directory.
//!
//! It used to be unusable: `from_path` sets the model name to "custom" and
//! `from_builder` resolved that name against the registry without ever looking
//! at `model_path`, so every call failed with UnknownModel("custom"). Classifier
//! and Reranker check the path first; the Embedder now does too.

use kjarni::Embedder;

#[tokio::test]
#[ignore = "needs minilm in the local model cache"]
async fn embedder_from_path_loads() {
    let dir = kjarni_transformers::models::get_default_cache_dir()
        .join("sentence-transformers_all-MiniLM-L6-v2");
    let e = Embedder::from_path(&dir).build().await.expect("from_path");
    let v = e.embed("Hello world").await.expect("embed");
    println!("  dim {} first {:.6}", v.len(), v[0]);

    let r = Embedder::new("minilm-l6-v2").await.expect("registry");
    let v2 = r.embed("Hello world").await.expect("embed");
    let d = v
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  worst difference against the registry build: {d:.3e}");
    assert!(d < 1e-6, "from_path and registry disagree: {d}");
}
