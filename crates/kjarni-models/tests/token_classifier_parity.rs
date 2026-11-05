//! Token classification matches PyTorch: the logits of every token, and the
//! entities transformers' `pipeline("ner", aggregation_strategy="simple")`
//! returns.
//!
//! The pinned tests load real NER models from the Hugging Face cache, without
//! a tokenizer.json, and skip when they are not cached. Their entities are from
//! transformers 4.57 on the CPU.
//!
//! The logits check runs against a reference file a script writes, when
//! `KJARNI_NER_REFERENCE` names it:
//!
//!     KJARNI_NER_REFERENCE=reference.json \
//!         cargo test --release -p kjarni-models --test token_classifier_parity -- --nocapture

use std::path::Path;

use kjarni_models::{Aggregation, TokenClassifier};
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::tensor::DType;
use kjarni_transformers::traits::Device;
use serde_json::Value;

fn f32_config() -> Option<ModelLoadConfig> {
    Some(ModelLoadConfig {
        target_dtype: Some(DType::F32),
        ..Default::default()
    })
}

fn load(dir: &Path) -> TokenClassifier {
    TokenClassifier::from_pretrained(dir, Device::Cpu, None, f32_config()).expect("load")
}

/// A cached Hugging Face model's files, without tokenizer.json, in a folder
/// of their own. The weights are linked, not copied, where the system allows.
fn cached(repo: &str) -> Option<tempfile::TempDir> {
    let home = std::env::var_os("HF_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".cache/huggingface")))?;
    let snapshots = home
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    let snapshot = std::fs::read_dir(snapshots).ok()?.flatten().next()?.path();
    if !snapshot.join("model.safetensors").exists() {
        return None;
    }
    let dir = tempfile::tempdir().unwrap();
    for f in [
        "config.json",
        "model.safetensors",
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ] {
        let from = snapshot.join(f);
        if from.exists() {
            let to = dir.path().join(f);
            std::fs::hard_link(&from, &to)
                .or_else(|_| std::fs::copy(&from, &to).map(|_| ()))
                .unwrap();
        }
    }
    Some(dir)
}

/// (label, start, end, score) of each entity.
async fn spans(
    repo: &str,
    text: &str,
    aggregation: Aggregation,
) -> Option<Vec<(String, usize, usize, f32)>> {
    let Some(dir) = cached(repo) else {
        eprintln!("skipping: {repo} not in the Hugging Face cache");
        return None;
    };
    let classifier = load(dir.path());
    let entities = classifier
        .entities(&[text], aggregation)
        .await
        .unwrap()
        .remove(0);
    Some(
        entities
            .into_iter()
            .map(|e| (e.label, e.start, e.end, e.score))
            .collect(),
    )
}

fn assert_spans(got: &[(String, usize, usize, f32)], want: &[(&str, usize, usize, f32)]) {
    let got_spans: Vec<(&str, usize, usize)> = got
        .iter()
        .map(|(l, s, e, _)| (l.as_str(), *s, *e))
        .collect();
    let want_spans: Vec<(&str, usize, usize)> =
        want.iter().map(|(l, s, e, _)| (*l, *s, *e)).collect();
    assert_eq!(got_spans, want_spans);
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g.3 - w.3).abs() < 1e-4,
            "{}: score {} vs {}",
            g.0,
            g.3,
            w.3
        );
    }
}

const MIDEIND: &str = "Miðeind ehf. var stofnuð árið 2013 og er í Reykjavík.";

#[tokio::test]
async fn icebert_ner_keeps_a_word_whole_with_first() {
    let Some(got) = spans("vesteinn/IceBERT-ner", MIDEIND, Aggregation::First).await else {
        return;
    };
    // "Miðeind ehf." is one organization; transformers' word heuristic for
    // byte-level BPE keeps the full stop with "Reykjavík".
    assert_spans(
        &got,
        &[
            ("Organization", 0, 12, 0.99651),
            ("Date", 30, 34, 0.99902),
            ("Location", 43, 53, 0.99871),
        ],
    );
}

#[tokio::test]
async fn icebert_ner_with_simple_splits_where_the_tokens_disagree() {
    let Some(got) = spans("vesteinn/IceBERT-ner", MIDEIND, Aggregation::Simple).await else {
        return;
    };
    assert_spans(
        &got,
        &[
            ("Organization", 0, 3, 0.99678),
            ("Organization", 3, 12, 0.99645),
            ("Date", 30, 34, 0.99902),
            ("Location", 43, 52, 0.99871),
            ("Location", 52, 53, 0.99499),
        ],
    );
}

#[tokio::test]
async fn bert_base_ner_finds_people_companies_and_places() {
    let text = "Sarah Connor met Apple's CEO in San Francisco on Tuesday.";
    let Some(got) = spans("dslim/bert-base-NER", text, Aggregation::First).await else {
        return;
    };
    assert_spans(
        &got,
        &[
            ("PER", 0, 12, 0.99966),
            ("ORG", 17, 22, 0.99906),
            ("LOC", 32, 45, 0.99944),
        ],
    );
}

#[tokio::test]
async fn multilingual_ner_reads_icelandic_names() {
    let text = "Þórður Jónsson býr í Reykjavík.";
    let repo = "Davlan/distilbert-base-multilingual-cased-ner-hrl";
    let Some(got) = spans(repo, text, Aggregation::First).await else {
        return;
    };
    assert_spans(&got, &[("PER", 0, 14, 0.99987), ("LOC", 21, 30, 0.99937)]);
    let Some(got) = spans(repo, text, Aggregation::Simple).await else {
        return;
    };
    assert_spans(&got, &[("PER", 0, 14, 0.99552), ("LOC", 21, 30, 0.99937)]);
}

#[test]
fn a_sequence_classifier_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"architectures": ["BertForSequenceClassification"], "model_type": "bert"}"#,
    )
    .unwrap();
    let err = TokenClassifier::from_pretrained(dir.path(), Device::Cpu, None, None)
        .err()
        .expect("must fail")
        .to_string();
    assert!(err.contains("BertForSequenceClassification"), "{err}");
    assert!(err.contains("not a token classification model"), "{err}");
}

#[tokio::test]
async fn logits_and_entities_match_the_reference_file() {
    let Ok(path) = std::env::var("KJARNI_NER_REFERENCE") else {
        eprintln!("skipping: KJARNI_NER_REFERENCE not set");
        return;
    };
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for (name, model) in reference.as_object().unwrap() {
        let classifier = load(Path::new(model["dir"].as_str().unwrap()));
        let n_labels = classifier.labels().len();
        for case in model["cases"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();

            let enc = classifier.tokenizer().encode(text, true).unwrap();
            let want_ids: Vec<u32> = case["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(
                enc.get_ids(),
                want_ids.as_slice(),
                "{name}: token ids for {text:?}"
            );

            // Tokens skip the special ones, so line the reference rows up by
            // position: row i + 1 is the i-th ordinary token.
            let tokens = &classifier.tokens(&[text]).await.unwrap()[0];
            let rows = case["logits"].as_array().unwrap();
            assert_eq!(
                tokens.len() + 2,
                rows.len(),
                "{name}: token count for {text:?}"
            );
            for (i, t) in tokens.iter().enumerate() {
                let row: Vec<f32> = rows[i + 1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect();
                assert_eq!(row.len(), n_labels);
                let max = row.iter().cloned().fold(f32::MIN, f32::max);
                let sum: f32 = row.iter().map(|x| (x - max).exp()).sum();
                let best = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                assert_eq!(
                    t.label,
                    classifier.labels()[best],
                    "{name}: label of {:?} in {text:?}",
                    t.token
                );
                assert!(
                    (t.score - 1.0 / sum).abs() < 1e-4,
                    "{name}: score of {:?}: {} vs {}",
                    t.token,
                    t.score,
                    1.0 / sum
                );
            }

            for (key, aggregation) in [
                ("simple", Aggregation::Simple),
                ("first", Aggregation::First),
                ("max", Aggregation::Max),
                ("average", Aggregation::Average),
            ] {
                let got = &classifier.entities(&[text], aggregation).await.unwrap()[0];
                let want = case["entities"][key].as_array().unwrap();
                let got_spans: Vec<(String, usize, usize)> = got
                    .iter()
                    .map(|e| (e.label.clone(), e.start, e.end))
                    .collect();
                let want_spans: Vec<(String, usize, usize)> = want
                    .iter()
                    .map(|e| {
                        (
                            e["label"].as_str().unwrap().into(),
                            e["start"].as_u64().unwrap() as usize,
                            e["end"].as_u64().unwrap() as usize,
                        )
                    })
                    .collect();
                assert_eq!(got_spans, want_spans, "{name} {key}: entities of {text:?}");
                for (g, w) in got.iter().zip(want) {
                    assert!(
                        (g.score - w["score"].as_f64().unwrap() as f32).abs() < 1e-4,
                        "{name} {key}: score of {:?}",
                        g.text
                    );
                }
            }
            eprintln!("{name:28} ok  {text:?}");
        }
    }
}
