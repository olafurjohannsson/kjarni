//! Encoder output against PyTorch, for the models the local cache holds.
//!
//! `#[ignore]`d: needs the weights and the reference file, both local. Both of
//! `test-everything.sh`'s test stages pass `--include-ignored`, so it runs there.
//!
//! Regenerate the reference with:
//!
//!     cd bench && .venv/bin/python encoder_parity.py
//!
//! The corpus is deliberately long enough that a 16 document batch crosses the
//! 1000 token mark, because that is where the encoder switches to the buffered
//! path. RoBERTa used to crash there and nothing noticed.

use kjarni_models::SentenceEncoder;
use kjarni_transformers::models::get_default_cache_dir;
use kjarni_transformers::traits::Device;

fn reference() -> serde_json::Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/torch_encoder_parity.json");
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Regenerate with: cd bench && .venv/bin/python encoder_parity.py",
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

/// Worst absolute element difference, and the cosine, between two vectors.
fn compare(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "dimension mismatch");
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let cos: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    (worst, cos)
}

async fn parity_for(model: &str) {
    let reference = reference();
    let Some(expected) = reference["models"].get(model) else {
        panic!("{model} is not in the reference file; regenerate it");
    };

    let dir = get_default_cache_dir().join(model);
    assert!(
        dir.join("model.safetensors").exists(),
        "{model} is not in the model cache. This test is ignored by default and \
         only runs with --include-ignored, so a missing model is a broken run."
    );
    let encoder =
        SentenceEncoder::from_pretrained(&dir, Device::Cpu, None, None, None).expect("load");

    // One short string, through encode(): its own code path.
    let short = reference["short"].as_str().expect("short");
    let got = encoder.encode(short).await.expect("encode");
    let (worst, cos) = compare(&got, &floats(&expected["short"]));
    eprintln!("  {model} encode()      cosine {cos:.6}, worst {worst:.3e}");
    let short_worst = worst;

    // A batch long enough to take the buffered path.
    let docs: Vec<String> = reference["docs"]
        .as_array()
        .expect("docs")
        .iter()
        .map(|d| d.as_str().expect("doc").to_string())
        .collect();
    let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
    let got = encoder.encode_batch(&refs).await.expect("encode_batch");

    let expected_batch = expected["batch"].as_array().expect("batch");
    let mut worst_all = 0.0f32;
    for (i, row) in got.iter().enumerate() {
        let (worst, cos) = compare(row, &floats(&expected_batch[i]));
        worst_all = worst_all.max(worst);
        assert!(
            cos > 0.999,
            "document {i} cosine {cos:.6} against torch, worst {worst:.3e}"
        );
    }
    eprintln!(
        "  {model} encode_batch  worst {worst_all:.3e} across {} documents",
        got.len()
    );
    assert!(
        short_worst < 1e-4 && worst_all < 1e-4,
        "differs from torch: encode {short_worst:.3e}, encode_batch {worst_all:.3e}"
    );
}

#[tokio::test]
#[ignore = "needs minilm and the torch reference in bench/"]
async fn minilm_matches_torch() {
    parity_for("sentence-transformers_all-MiniLM-L6-v2").await;
}

#[tokio::test]
#[ignore = "needs roberta-base and the torch reference in bench/"]
async fn roberta_matches_torch() {
    parity_for("SamLowe_roberta-base-go_emotions").await;
}

/// mpnet and distilbert both crashed from C# with "end <= axis_len" before their
/// configs stopped inheriting `intermediate_size` = 0 from the trait default.
#[tokio::test]
#[ignore = "needs mpnet-base-v2 and the torch reference"]
async fn mpnet_matches_torch() {
    parity_for("sentence-transformers_all-mpnet-base-v2").await;
}

#[tokio::test]
#[ignore = "needs distilbert and the torch reference"]
async fn distilbert_matches_torch() {
    parity_for("distilbert_distilbert-base-uncased-finetuned-sst-2-english").await;
}
