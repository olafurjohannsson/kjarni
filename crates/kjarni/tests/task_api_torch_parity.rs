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

    // Built from the registry. `Embedder::from_path` is covered separately in
    // `embedder_from_path`, which also checks the two agree.
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
    reranker_matches_torch_on(Dev::Cpu).await;
}

#[tokio::test]
#[ignore = "needs a GPU and the ms-marco cross-encoder in the local model cache"]
async fn reranker_matches_torch_on_gpu() {
    reranker_matches_torch_on(Dev::Gpu).await;
}

async fn reranker_matches_torch_on(dev: Dev) {
    let reference = reference("torch_head_parity.json");
    let model = "cross-encoder_ms-marco-MiniLM-L-6-v2";
    let expected: Vec<f32> = floats(&reference["cross_encoders"][model]["scores"]);

    let builder = Reranker::from_path(model_dir(model)).quiet(true);
    let builder = match dev {
        Dev::Cpu => builder.cpu(),
        Dev::Gpu => builder.gpu(),
    };
    let reranker = builder.build().await.expect("build reranker");

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
    eprintln!("  {dev:?} Reranker::rerank       worst {d:.3e}");
    assert!(d < 1e-4, "{dev:?} rerank differs from torch by {d:.3e}");

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

// ── Every reference model, on each device ─────────────────────────────────
//
// The checks above pin one model per task in detail. These run every model in
// the reference files through the same entry points, once on the CPU and once
// on the GPU, so a backend that drifts on one architecture (RoBERTa's offset
// position ids, MPNet's relative attention bias, XLM-R's long window) fails
// here instead of in a user's results.

#[derive(Clone, Copy, Debug)]
enum Dev {
    Cpu,
    Gpu,
}

/// Every embedder in the reference against torch: mean pooling, normalised,
/// which is how the references were computed.
async fn embedders_match_torch(dev: Dev, tolerance: f32) {
    let reference = reference("torch_encoder_parity.json");
    let short = reference["short"].as_str().expect("short");
    let docs: Vec<&str> = reference["docs"]
        .as_array()
        .expect("docs")
        .iter()
        .map(|x| x.as_str().expect("doc"))
        .collect();

    let mut failures = Vec::new();
    for (name, expected) in reference["models"].as_object().expect("models") {
        let mut builder = Embedder::from_path(model_dir(name))
            .pooling(kjarni::PoolingStrategy::Mean)
            .normalize(true)
            .quiet(true);
        builder = match dev {
            Dev::Cpu => builder.cpu(),
            Dev::Gpu => builder.gpu(),
        };
        let embedder = builder.build().await.expect("build embedder");

        let got = embedder.embed(short).await.expect("embed");
        let mut d = worst(&got, &floats(&expected["short"]));
        let batch = embedder.embed_batch(&docs).await.expect("embed_batch");
        for (i, row) in batch.iter().enumerate() {
            d = d.max(worst(row, &floats(&expected["batch"][i])));
        }
        eprintln!("  {dev:?} {name:<62} worst {d:.3e}");
        if d >= tolerance {
            failures.push(format!("{name}: {d:.3e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{dev:?} embeddings differ from torch by {tolerance:.0e} or more: {failures:?}"
    );
}

/// Every classifier in the reference against torch, logit by logit.
async fn classifiers_match_torch(dev: Dev, tolerance: f32) {
    let reference = reference("torch_head_parity.json");
    let texts: Vec<&str> = reference["texts"]
        .as_array()
        .expect("texts")
        .iter()
        .map(|t| t.as_str().expect("text"))
        .collect();

    let mut failures = Vec::new();
    for (name, expected) in reference["classifiers"].as_object().expect("classifiers") {
        let labels: Vec<&str> = expected["labels"]
            .as_array()
            .expect("labels")
            .iter()
            .map(|l| l.as_str().expect("label"))
            .collect();
        let mut builder = Classifier::from_path(model_dir(name))
            .return_logits(true)
            .quiet(true);
        builder = match dev {
            Dev::Cpu => builder.cpu(),
            Dev::Gpu => builder.gpu(),
        };
        let classifier = builder.build().await.expect("build classifier");

        let mut d = 0.0f32;
        for (i, text) in texts.iter().enumerate() {
            let got = classifier.classify(text).await.expect("classify");
            let logits = floats(&expected["logits"][i]);
            for (label, score) in &got.all_scores {
                let j = labels
                    .iter()
                    .position(|l| l == label)
                    .unwrap_or_else(|| panic!("{name}: unknown label {label}"));
                d = d.max((score - logits[j]).abs());
            }
        }
        eprintln!("  {dev:?} {name:<62} worst logit {d:.3e}");
        if d >= tolerance {
            failures.push(format!("{name}: {d:.3e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{dev:?} logits differ from torch by {tolerance:.0e} or more: {failures:?}"
    );
}

#[tokio::test]
#[ignore = "needs the reference models in the local model cache"]
async fn every_embedder_matches_torch_on_cpu() {
    embedders_match_torch(Dev::Cpu, 1e-4).await;
}

#[tokio::test]
#[ignore = "needs a GPU and the reference models in the local model cache"]
async fn every_embedder_matches_torch_on_gpu() {
    embedders_match_torch(Dev::Gpu, 1e-4).await;
}

#[tokio::test]
#[ignore = "needs the reference models in the local model cache"]
async fn every_classifier_matches_torch_on_cpu() {
    classifiers_match_torch(Dev::Cpu, 1e-3).await;
}

#[tokio::test]
#[ignore = "needs a GPU and the reference models in the local model cache"]
async fn every_classifier_matches_torch_on_gpu() {
    classifiers_match_torch(Dev::Gpu, 1e-3).await;
}

/// nomic-bert has no torch reference here (its modelling code is remote), so
/// its GPU path is held to the CPU one. It is the encoder with a SwiGLU
/// feed-forward and rotary positions, which no other reference model has.
#[tokio::test]
#[ignore = "needs a GPU and nomic-embed-text in the local model cache"]
async fn nomic_gpu_matches_cpu() {
    let reference = reference("torch_encoder_parity.json");
    let docs: Vec<&str> = reference["docs"]
        .as_array()
        .expect("docs")
        .iter()
        .map(|x| x.as_str().expect("doc"))
        .collect();
    let dir = model_dir("nomic-ai_nomic-embed-text-v1.5");

    let cpu = Embedder::from_path(&dir)
        .cpu()
        .quiet(true)
        .build()
        .await
        .expect("cpu");
    let gpu = Embedder::from_path(&dir)
        .gpu()
        .quiet(true)
        .build()
        .await
        .expect("gpu");
    let a = cpu.embed_batch(&docs).await.expect("cpu embed_batch");
    let b = gpu.embed_batch(&docs).await.expect("gpu embed_batch");

    let mut d = 0.0f32;
    for (x, y) in a.iter().zip(&b) {
        d = d.max(worst(x, y));
    }
    eprintln!("  nomic gpu vs cpu           worst {d:.3e}");
    assert!(d < 1e-4, "nomic GPU differs from CPU by {d:.3e}");
}
