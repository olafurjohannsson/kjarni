//! Models' own Jinja chat templates, rendered as transformers 4.57's
//! `apply_chat_template(tokenize=False)` renders them.
//!
//! The sweep runs when `KJARNI_TEMPLATE_REFERENCE` names the file the
//! reference script writes and `KJARNI_TEMPLATE_MODELS` the folder of model
//! folders it read:
//!
//!     KJARNI_TEMPLATE_REFERENCE=ref.json KJARNI_TEMPLATE_MODELS=models \
//!         cargo test -p kjarni-transformers --test chat_templates -- --nocapture
#![cfg(not(target_arch = "wasm32"))]

use std::path::Path;

use kjarni_transformers::chat::jinja::JinjaChatTemplate;
use kjarni_transformers::{Conversation, Message};
use serde_json::Value;

fn conversation(messages: &Value) -> Conversation {
    let mut c = Conversation::new();
    for m in messages.as_array().unwrap() {
        let content = m["content"].as_str().unwrap();
        c.push(match m["role"].as_str().unwrap() {
            "system" => Message::system(content),
            "user" => Message::user(content),
            "assistant" => Message::assistant(content),
            r => panic!("unknown role {r}"),
        });
    }
    c
}

#[test]
fn templates_render_what_transformers_renders() {
    let (Ok(path), Ok(models)) = (
        std::env::var("KJARNI_TEMPLATE_REFERENCE"),
        std::env::var("KJARNI_TEMPLATE_MODELS"),
    ) else {
        eprintln!("skipping: KJARNI_TEMPLATE_REFERENCE and KJARNI_TEMPLATE_MODELS not set");
        return;
    };
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut failures = Vec::new();
    let mut checked = 0;
    for (name, cases) in reference["models"].as_object().unwrap() {
        let template = match JinjaChatTemplate::from_dir(&Path::new(&models).join(name)) {
            Ok(Some(t)) => t,
            Ok(None) => {
                failures.push(format!("{name}: no template found"));
                continue;
            }
            Err(e) => {
                failures.push(format!("{name}: {e:#}"));
                continue;
            }
        };
        for case in cases.as_array().unwrap() {
            checked += 1;
            let label = format!(
                "{name} {} gen={}",
                case["convo"].as_str().unwrap(),
                case["add_generation_prompt"]
            );
            let got = template.render(
                &conversation(&case["messages"]),
                case["add_generation_prompt"].as_bool().unwrap(),
            );
            match (got, case.get("output").and_then(Value::as_str)) {
                (Ok(g), Some(w)) if g == w => {}
                (Ok(g), Some(w)) => {
                    failures.push(format!("{label}:\n  kjarni: {g:?}\n  hf:     {w:?}"))
                }
                (Ok(g), None) => failures.push(format!(
                    "{label}: transformers fails ({}), Kjarni rendered {g:?}",
                    case["error"]
                )),
                (Err(e), Some(_)) => failures.push(format!("{label}: Kjarni fails: {e:#}")),
                (Err(_), None) => {}
            }
        }
    }
    eprintln!("checked {checked} renderings");
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
