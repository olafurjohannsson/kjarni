//! Translation and summarization with models loaded from Hugging Face folders,
//! against transformers 4.57 with the same folders in float32 on the CPU: the
//! output under each model's own generation config, and greedy.
//!
//! The folders cover Marian with weights in pytorch_model.bin and its
//! tokenizer in source.spm and vocab.json (one-direction models, and
//! en-ROMANCE, which takes the target as a `>>code<<` token), BART in
//! pytorch_model.bin with vocab.json and merges.txt, and T5 in safetensors.
//!
//! Runs when `KJARNI_SEQ2SEQ_REFERENCE` names the file the reference script
//! writes:
//!
//!     KJARNI_SEQ2SEQ_REFERENCE=ref.json \
//!         cargo test --release -p kjarni --test seq2seq_from_folder -- --nocapture

use kjarni::summarizer::Summarizer;
use kjarni::translator::Translator;
use serde_json::Value;

fn reference() -> Option<Value> {
    let Ok(path) = std::env::var("KJARNI_SEQ2SEQ_REFERENCE") else {
        eprintln!("skipping: KJARNI_SEQ2SEQ_REFERENCE not set");
        return None;
    };
    Some(serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap())
}

fn models<'a>(reference: &'a Value, kind: &'a str) -> impl Iterator<Item = &'a Value> {
    reference["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(move |m| m["kind"] == kind)
}

#[tokio::test]
async fn translation_from_a_folder_matches_transformers() {
    let Some(reference) = reference() else { return };
    let mut failures = Vec::new();
    for model in models(&reference, "translate") {
        let repo = model["repo"].as_str().unwrap();
        let dir = model["dir"].as_str().unwrap();
        for greedy in [false, true] {
            let mut builder = Translator::builder("")
                .model_path(dir)
                .cpu()
                .with_load_config(|b| b.f32())
                .quiet();
            if greedy {
                builder = builder.greedy();
            }
            let translator = builder.build().await.unwrap();
            assert_eq!(translator.model_type(), None);
            let tokenizer = translator.generator().encoder_decoder().model.tokenizer();
            for case in model["cases"].as_array().unwrap() {
                let prompt = case["prompt"].as_str().unwrap();
                let ids = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();
                let want_ids: Vec<u32> =
                    serde_json::from_value(case["prompt_ids"].clone()).unwrap();
                if ids != want_ids {
                    failures.push(format!(
                        "{repo}: ids of {prompt:?}\n  kjarni: {ids:?}\n  hf:     {want_ids:?}"
                    ));
                }
                let (text, from, to) = (
                    case["input"].as_str().unwrap(),
                    case["from"].as_str().unwrap(),
                    case["to"].as_str().unwrap(),
                );
                let got = translator.translate(text, from, to).await.unwrap();
                let mode = if greedy { "greedy" } else { "default" };
                let want = case[mode].as_str().unwrap();
                eprintln!("{repo} {mode}: {got:?}");
                if got.trim() != want.trim() {
                    failures.push(format!(
                        "{repo} {mode}: {text:?} {from}->{to}\n  kjarni: {got:?}\n  hf:     {want:?}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[tokio::test]
async fn summarization_from_a_folder_matches_transformers() {
    let Some(reference) = reference() else { return };
    let mut failures = Vec::new();
    for model in models(&reference, "summarize") {
        let repo = model["repo"].as_str().unwrap();
        let dir = model["dir"].as_str().unwrap();
        for greedy in [false, true] {
            let mut builder = Summarizer::builder("")
                .model_path(dir)
                .cpu()
                .with_load_config(|b| b.f32())
                .quiet();
            if greedy {
                builder = builder.greedy();
            }
            let summarizer = builder.build().await.unwrap();
            assert_eq!(summarizer.model_type(), None);
            for case in model["cases"].as_array().unwrap() {
                let text = case["input"].as_str().unwrap();
                let got = summarizer.summarize(text).await.unwrap();
                let mode = if greedy { "greedy" } else { "default" };
                let want = case[mode].as_str().unwrap();
                eprintln!("{repo} {mode}: {got:?}");
                if got.trim() != want.trim() {
                    failures.push(format!(
                        "{repo} {mode}\n  kjarni: {got:?}\n  hf:     {want:?}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Refusals that need no model: a folder whose family cannot do the task.
#[tokio::test]
async fn folders_of_the_wrong_family_are_refused() {
    let Some(reference) = reference() else { return };
    let marian = models(&reference, "translate").next().unwrap()["dir"]
        .as_str()
        .unwrap()
        .to_string();
    let bart = models(&reference, "summarize").next().unwrap()["dir"]
        .as_str()
        .unwrap()
        .to_string();

    let err = Summarizer::builder("")
        .model_path(&marian)
        .cpu()
        .quiet()
        .build()
        .await;
    let err = err.expect_err("Marian cannot summarize").to_string();
    assert!(err.contains("Marian"), "{err}");

    let err = Translator::builder("")
        .model_path(&bart)
        .cpu()
        .quiet()
        .build()
        .await;
    let err = err.expect_err("BART does not translate").to_string();
    assert!(err.contains("summarization"), "{err}");

    let translator = Translator::builder("")
        .model_path(&marian)
        .cpu()
        .quiet()
        .build()
        .await
        .unwrap();
    let err = translator
        .translate("Hallo", "de", "en")
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("only translates English to German"), "{err}");
}
