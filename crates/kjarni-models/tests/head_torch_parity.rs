//! Classification and cross-encoder heads against PyTorch.
//!
//! `encoder_torch_parity.rs` loads `AutoModel` on the torch side, which drops the
//! classification head, so it validates the encoder body and nothing above it.
//! These compare what the heads actually output: class probabilities and
//! reranking scores.
//!
//! Both model families had `intermediate_size: 0` hardcoded in their metadata,
//! which crashed any batch large enough to take the buffered encoder path, and
//! nothing here noticed because nothing here existed.
//!
//! `#[ignore]`d: needs local models and the reference file. Regenerate with:
//!
//!     cd bench && .venv/bin/python head_parity.py

use kjarni_models::{CrossEncoder, SequenceClassifier};
use kjarni_transformers::models::get_default_cache_dir;
use kjarni_transformers::traits::Device;

fn reference() -> serde_json::Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/torch_head_parity.json");
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Regenerate with: cd bench && .venv/bin/python head_parity.py",
            path.display()
        )
    });
    serde_json::from_str(&raw).expect("parse reference")
}

fn model_dir(name: &str) -> std::path::PathBuf {
    let dir = get_default_cache_dir().join(name);
    assert!(
        dir.join("model.safetensors").exists(),
        "{name} is not in the model cache; this test only runs deliberately"
    );
    dir
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect()
}

async fn classifier_parity(model: &str) {
    let reference = reference();
    let expected = reference["classifiers"]
        .get(model)
        .unwrap_or_else(|| panic!("{model} missing from the reference; regenerate it"));

    let classifier =
        SequenceClassifier::from_pretrained(&model_dir(model), Device::Cpu, None, None, None)
            .expect("load classifier");

    let texts: Vec<&str> = reference["texts"]
        .as_array()
        .expect("texts")
        .iter()
        .map(|t| t.as_str().expect("text"))
        .collect();

    let mut worst = 0.0f32;
    for (i, text) in texts.iter().enumerate() {
        let got = classifier.classify_scores(text).await.expect("classify");

        let logits: Vec<f32> = expected["logits"][i]
            .as_array()
            .expect("logits")
            .iter()
            .map(|x| x.as_f64().expect("float") as f32)
            .collect();
        let want = softmax(&logits);

        assert_eq!(got.len(), want.len(), "label count differs for {text:?}");
        let delta = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(delta);

        // The top label has to match, not merely be close.
        let got_top = got
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
            .expect("non-empty")
            .0;
        let want_top = want
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
            .expect("non-empty")
            .0;
        assert_eq!(got_top, want_top, "different top label for {text:?}");
    }

    eprintln!("  {model}: worst probability difference {worst:.3e}");
    assert!(worst < 1e-3, "{model} differs from torch by {worst:.3e}");
}

#[tokio::test]
#[ignore = "needs distilbert sst-2 and the torch reference"]
async fn distilbert_classifier_matches_torch() {
    classifier_parity("distilbert_distilbert-base-uncased-finetuned-sst-2-english").await;
}

#[tokio::test]
#[ignore = "needs roberta go-emotions and the torch reference"]
async fn roberta_classifier_matches_torch() {
    classifier_parity("SamLowe_roberta-base-go_emotions").await;
}

#[tokio::test]
#[ignore = "needs the ms-marco cross-encoder and the torch reference"]
async fn cross_encoder_matches_torch() {
    let model = "cross-encoder_ms-marco-MiniLM-L-6-v2";
    let reference = reference();
    let expected: Vec<f32> = reference["cross_encoders"][model]["scores"]
        .as_array()
        .unwrap_or_else(|| panic!("{model} missing from the reference; regenerate it"))
        .iter()
        .map(|x| x.as_f64().expect("float") as f32)
        .collect();

    let reranker = CrossEncoder::from_pretrained(&model_dir(model), Device::Cpu, None, None, None)
        .expect("load cross encoder");

    let query = reference["query"].as_str().expect("query");
    let docs: Vec<&str> = reference["docs"]
        .as_array()
        .expect("docs")
        .iter()
        .map(|d| d.as_str().expect("doc"))
        .collect();

    // rerank returns (original index, score), sorted by score.
    let ranked = reranker.rerank(query, &docs).await.expect("rerank");
    assert_eq!(ranked.len(), docs.len());

    let mut worst = 0.0f32;
    for (index, score) in &ranked {
        let delta = (score - expected[*index]).abs();
        worst = worst.max(delta);
    }

    eprintln!("  {model}: worst score difference {worst:.3e}");
    // Raw logits, so the tolerance is relative to their magnitude (around 10).
    assert!(worst < 1e-2, "{model} differs from torch by {worst:.3e}");
}
