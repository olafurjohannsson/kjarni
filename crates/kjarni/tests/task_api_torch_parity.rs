//! The task-level API against PyTorch.
//!
//! This is the layer the FFI boundary calls, so it is what every binding
//! ultimately runs: C#, Python, Go and C++ all reach `Embedder::embed`,
//! `embed_batch`, `embed_batch_flat`, `similarity`, `Reranker::rerank`,
//! `rerank_top_k` and `Classifier::classify`. Tests one layer down exercise the
//! same engine but not the same entry points, and these methods add pooling,
//! normalisation and result shaping of their own.
//!
//! `#[ignore]`d: needs models from the local cache and the reference files, so it
//! runs under `test-everything.sh` (both test stages pass `--include-ignored`)
//! and not in CI. Regenerate the references with:
//!
//!     cd bench && .venv/bin/python encoder_parity.py && .venv/bin/python head_parity.py

use kjarni::{Classifier, Embedder, Reranker};

fn cache() -> std::path::PathBuf {
    kjarni_transformers::models::get_default_cache_dir()
}

fn model_dir(name: &str) -> std::path::PathBuf {
    let dir = cache().join(name);
    assert!(
        dir.join("model.safetensors").exists(),
        "{name} is not in the model cache; this test only runs deliberately"
    );
    dir
}

fn reference(file: &str) -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench")
        .join(file);
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Regenerate it, see the module docs",
            path.display()
        )
    });
    serde_json::from_str(&raw).expect("parse reference")
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("float") as f32)
        .collect()
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[tokio::test]
#[ignore = "needs minilm in the local model cache"]
async fn embedder_matches_torch() {
    let reference = reference("torch_encoder_parity.json");
    let expected = &reference["models"]["sentence-transformers_all-MiniLM-L6-v2"];

    // Built from the registry rather than `Embedder::from_path`, which cannot
    // work: `from_path` sets the model name to "custom" and `from_builder` then
    // resolves that name without ever looking at `model_path`, so it fails with
    // UnknownModel("custom"). Classifier and Reranker check `model_path` first
    // and load from disk; the Embedder does not. Tracked separately.
    model_dir("sentence-transformers_all-MiniLM-L6-v2");
    let embedder = Embedder::new("minilm-l6-v2")
        .await
        .expect("build embedder from the registry");

    // embed: one string, the FFI's kjarni_embedder_embed.
    let short = reference["short"].as_str().expect("short");
    let got = embedder.embed(short).await.expect("embed");
    let d = worst(&got, &floats(&expected["short"]));
    eprintln!("  Embedder::embed            worst {d:.3e}");
    assert!(d < 1e-4, "embed differs from torch by {d:.3e}");

    let docs: Vec<String> = reference["docs"]
        .as_array()
        .expect("docs")
        .iter()
        .map(|x| x.as_str().expect("doc").to_string())
        .collect();
    let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let expected_batch = expected["batch"].as_array().expect("batch");

    // embed_batch: the FFI's batch entry point.
    let got = embedder.embed_batch(&refs).await.expect("embed_batch");
    let mut worst_all = 0.0f32;
    for (i, row) in got.iter().enumerate() {
        worst_all = worst_all.max(worst(row, &floats(&expected_batch[i])));
    }
    eprintln!("  Embedder::embed_batch      worst {worst_all:.3e}");
    assert!(worst_all < 1e-4, "embed_batch differs by {worst_all:.3e}");

    // embed_batch_flat: same vectors, flattened. The FFI uses this one to avoid
    // allocating a Vec per document across the boundary.
    let (flat, rows, dim) = embedder
        .embed_batch_flat(&refs)
        .await
        .expect("embed_batch_flat");
    assert_eq!(rows, refs.len(), "row count");
    assert_eq!(dim, got[0].len(), "dimension");
    let mut worst_flat = 0.0f32;
    for i in 0..rows {
        worst_flat = worst_flat.max(worst(&flat[i * dim..(i + 1) * dim], &got[i]));
    }
    eprintln!("  embed_batch_flat vs batch  worst {worst_flat:.3e}");
    assert!(
        worst_flat < 1e-6,
        "embed_batch_flat disagrees with embed_batch by {worst_flat:.3e}"
    );

    // similarity: cosine of two embeddings, computed inside the task layer.
    let a = floats(&expected_batch[0]);
    let b = floats(&expected_batch[1]);
    let expected_cos: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let got_cos = embedder
        .similarity(&docs[0], &docs[1])
        .await
        .expect("similarity");
    eprintln!("  Embedder::similarity       {got_cos:.6} against {expected_cos:.6}");
    assert!(
        (got_cos - expected_cos).abs() < 1e-4,
        "similarity differs by {:.3e}",
        (got_cos - expected_cos).abs()
    );
}

#[tokio::test]
#[ignore = "needs the ms-marco cross-encoder in the local model cache"]
async fn reranker_matches_torch() {
    let reference = reference("torch_head_parity.json");
    let model = "cross-encoder_ms-marco-MiniLM-L-6-v2";
    let expected: Vec<f32> = floats(&reference["cross_encoders"][model]["scores"]);

    let reranker = Reranker::from_path(model_dir(model))
        .build()
        .await
        .expect("build reranker");

    let query = reference["query"].as_str().expect("query");
    let docs: Vec<&str> = reference["docs"]
        .as_array()
        .expect("docs")
        .iter()
        .map(|d| d.as_str().expect("doc"))
        .collect();

    let ranked = reranker.rerank(query, &docs).await.expect("rerank");
    assert_eq!(ranked.len(), docs.len());
    let mut d = 0.0f32;
    for r in &ranked {
        d = d.max((r.score - expected[r.index]).abs());
    }
    eprintln!("  Reranker::rerank           worst {d:.3e}");
    assert!(d < 1e-2, "rerank differs from torch by {d:.3e}");

    // rerank_top_k must be the same ordering, truncated.
    let top2 = reranker.rerank_top_k(query, &docs, 2).await.expect("top_k");
    assert_eq!(top2.len(), 2, "top_k must return k results");
    assert_eq!(
        top2[0].index, ranked[0].index,
        "top_k disagrees with rerank on the best document"
    );
}

#[tokio::test]
#[ignore = "needs distilbert sst-2 in the local model cache"]
async fn classifier_matches_torch() {
    let reference = reference("torch_head_parity.json");
    let model = "distilbert_distilbert-base-uncased-finetuned-sst-2-english";
    let expected = &reference["classifiers"][model];
    let labels: Vec<&str> = expected["labels"]
        .as_array()
        .expect("labels")
        .iter()
        .map(|l| l.as_str().expect("label"))
        .collect();

    let classifier = Classifier::from_path(model_dir(model))
        .build()
        .await
        .expect("build classifier");

    for (i, text) in reference["texts"]
        .as_array()
        .expect("texts")
        .iter()
        .enumerate()
    {
        let text = text.as_str().expect("text");
        let got = classifier.classify(text).await.expect("classify");

        let logits = floats(&expected["logits"][i]);
        let top = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
            .expect("non-empty")
            .0;

        assert_eq!(
            got.label, labels[top],
            "different label for {text:?}: torch says {}",
            labels[top]
        );
    }
    eprintln!("  Classifier::classify       labels match torch on every text");
}
