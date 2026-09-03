//! Encoder models that live in the local model cache rather than in CI.
//!
//! `#[ignore]`d, so `cargo test` skips these and `test-everything.sh` runs them:
//! both of its test stages pass `--include-ignored`.
//!
//! These exist because of a bug that survived a long time. RoBERTa's and the
//! MiniLM cross-encoder's `metadata()` hardcoded `intermediate_size: 0` while
//! deserialising the real value from `config.json`, so the encoder's
//! `ffn_intermediate` buffer was allocated with zero width and the feed forward
//! sliced past the end of it. It only fired when the buffered path was taken,
//! which a token-count threshold made rare, and no test covered either model at a
//! size that crossed it.
//!
//! A missing model fails rather than skips: these only run when asked for, and a
//! quiet skip is how the gap lasted.

use kjarni_models::{CrossEncoder, SentenceEncoder};
use kjarni_transformers::models::get_default_cache_dir;
use kjarni_transformers::traits::Device;

fn model_dir(name: &str) -> std::path::PathBuf {
    let dir = get_default_cache_dir().join(name);
    assert!(
        dir.join("model.safetensors").exists(),
        "{name} is not in the model cache at {}. These tests are ignored by \
         default and only run with --include-ignored, so a missing model is a \
         broken run rather than something to skip past.",
        dir.display()
    );
    dir
}

/// Long enough to cross the old 1000-token threshold, which is where the
/// buffered path starts and where the crash used to be.
fn long_corpus(docs: usize) -> Vec<String> {
    (0..docs)
        .map(|i| {
            "The history of Iceland begins with settlement in the ninth century. ".repeat(12)
                + &format!("Document {i}.")
        })
        .collect()
}

#[tokio::test]
#[ignore = "needs roberta-base in the local model cache"]
async fn roberta_encodes_past_the_buffered_threshold() {
    let dir = model_dir("SamLowe_roberta-base-go_emotions");
    let encoder =
        SentenceEncoder::from_pretrained(&dir, Device::Cpu, None, None, None).expect("load");

    let corpus = long_corpus(16);
    let refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
    let out = encoder.encode_batch(&refs).await.expect("encode_batch");

    assert_eq!(out.len(), refs.len());
    assert_eq!(out[0].len(), 768, "roberta-base is 768 dimensions");
    for (i, v) in out.iter().enumerate() {
        assert!(
            v.iter().all(|x| x.is_finite()),
            "document {i} produced a non-finite value"
        );
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "document {i} is not unit length: {norm}"
        );
    }
}

/// `encode` and `encode_batch` are separate code with separate buffer decisions.
/// They must agree, or one of them is wrong.
#[tokio::test]
#[ignore = "needs roberta-base in the local model cache"]
async fn roberta_single_and_batch_agree() {
    let dir = model_dir("SamLowe_roberta-base-go_emotions");
    let encoder =
        SentenceEncoder::from_pretrained(&dir, Device::Cpu, None, None, None).expect("load");

    let text = "The capital of Iceland is Reykjavik, which sits on the south west coast.";
    let single = encoder.encode(text).await.expect("encode");
    let batch = encoder.encode_batch(&[text]).await.expect("encode_batch");

    let worst = single
        .iter()
        .zip(batch[0].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-5,
        "encode and encode_batch disagree by {worst:.3e}"
    );
}

#[tokio::test]
#[ignore = "needs the ms-marco cross-encoder in the local model cache"]
async fn cross_encoder_reranks_past_the_buffered_threshold() {
    let dir = model_dir("cross-encoder_ms-marco-MiniLM-L-6-v2");
    let reranker =
        CrossEncoder::from_pretrained(&dir, Device::Cpu, None, None, None).expect("load");

    let corpus = long_corpus(16);
    let refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
    let ranked = reranker
        .rerank("when was Iceland settled", &refs)
        .await
        .expect("rerank");

    assert_eq!(ranked.len(), refs.len(), "every document must be scored");
    assert!(
        ranked.iter().all(|(_, s)| s.is_finite()),
        "a score was not finite"
    );
    assert!(
        ranked.windows(2).all(|w| w[0].1 >= w[1].1),
        "results must come back sorted by score"
    );
}
