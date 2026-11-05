//! Chat with a model loaded from a folder: the prompt its own chat template
//! renders, the ids of that prompt and the greedy reply, against transformers
//! 4.57 with the same folder in float32 on the CPU.
//!
//! Runs when `KJARNI_CHAT_REFERENCE` names the file the reference script
//! writes:
//!
//!     KJARNI_CHAT_REFERENCE=ref.json \
//!         cargo test --release -p kjarni --test chat_from_folder -- --nocapture

use kjarni::chat::Chat;
use kjarni_transformers::{Conversation, Message};
use serde_json::Value;

fn conversation(messages: &Value) -> Conversation {
    let mut c = Conversation::new();
    for m in messages.as_array().unwrap() {
        let content = m["content"].as_str().unwrap();
        c.push(match m["role"].as_str().unwrap() {
            "system" => Message::system(content),
            "user" => Message::user(content),
            _ => Message::assistant(content),
        });
    }
    c
}

#[tokio::test]
async fn chat_from_a_folder_matches_transformers() {
    let Ok(path) = std::env::var("KJARNI_CHAT_REFERENCE") else {
        eprintln!("skipping: KJARNI_CHAT_REFERENCE not set");
        return;
    };
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut failures = Vec::new();
    for (name, model) in reference["models"].as_object().unwrap() {
        let mut builder = Chat::builder("custom")
            .model_path(model["dir"].as_str().unwrap())
            .cpu()
            .greedy()
            .max_tokens(24)
            .with_load_config(|b| b.f32())
            .quiet();
        if let Some(p) = model.get("repetition_penalty").and_then(Value::as_f64) {
            builder = builder.repetition_penalty(p as f32);
        }
        let chat = builder.build().await;
        let chat = match chat {
            Ok(c) => c,
            Err(e) => {
                failures.push(format!("{name}: {e:#}"));
                continue;
            }
        };
        assert_eq!(chat.model_type(), None);
        for case in model["cases"].as_array().unwrap() {
            let convo = conversation(&case["messages"]);
            let prompt = chat.format_prompt(&convo);
            let want_prompt = case["prompt"].as_str().unwrap();
            if prompt != want_prompt {
                failures.push(format!(
                    "{name}: prompt\n  kjarni: {prompt:?}\n  hf:     {want_prompt:?}"
                ));
                continue;
            }
            let generator = chat.generator();
            let ids = generator
                .decoder()
                .encode(&prompt, generator.generation_config().as_ref())
                .unwrap();
            let want_ids: Vec<u32> = serde_json::from_value(case["prompt_ids"].clone()).unwrap();
            if ids != want_ids {
                failures.push(format!(
                    "{name}: prompt ids\n  kjarni: {ids:?}\n  hf:     {want_ids:?}"
                ));
                continue;
            }
            let reply = chat.generate(&prompt, &Default::default()).await.unwrap();
            let want = case["text"].as_str().unwrap().trim();
            eprintln!("{name}: {reply:?}");
            if reply != want {
                failures.push(format!(
                    "{name}: reply\n  kjarni: {reply:?}\n  hf:     {want:?}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
