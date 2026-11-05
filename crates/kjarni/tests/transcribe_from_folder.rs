//! Whisper loaded from a folder transcribes as transformers 4.57 does with the
//! same folder in float32 on the CPU: streamed, the ids of
//! `generate(do_sample=False)`; whole, the text of
//! `generate(return_timestamps=True)`, since the transcriber decodes with
//! timestamps to drop text past the end of the audio.
//!
//! Runs when `KJARNI_WHISPER_REFERENCE` names the file the reference script
//! writes:
//!
//!     KJARNI_WHISPER_REFERENCE=ref.json \
//!         cargo test --release -p kjarni --test transcribe_from_folder -- --nocapture

use futures::StreamExt;
use kjarni::transcriber::Transcriber;
use serde_json::Value;

#[tokio::test]
async fn whisper_from_a_folder_matches_transformers() {
    let Ok(path) = std::env::var("KJARNI_WHISPER_REFERENCE") else {
        eprintln!("skipping: KJARNI_WHISPER_REFERENCE not set");
        return;
    };
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let audio = reference["audio"].as_str().unwrap();
    let mut failures = Vec::new();
    for (name, model) in reference["models"].as_object().unwrap() {
        let dir = model["dir"].as_str().unwrap();
        let transcriber = match Transcriber::builder("custom")
            .model_path(dir)
            .cpu()
            .quiet()
            .build()
            .await
        {
            Ok(t) => t,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };

        let mut stream = transcriber.stream_file(audio).await.unwrap();
        let mut ids = Vec::new();
        while let Some(token) = stream.next().await {
            let token = token.unwrap();
            if !token.is_special {
                ids.push(token.id);
            }
        }
        let want: Vec<u32> = serde_json::from_value(model["cases"]["auto"]["ids"].clone()).unwrap();
        // transformers returns the prompt and the closing <|endoftext|> too:
        // every id from <|endoftext|> up is special.
        let end_of_text = if model["multilingual"].as_bool().unwrap() {
            50257
        } else {
            50256
        };
        let want_text: Vec<u32> = want.into_iter().filter(|&id| id < end_of_text).collect();
        if ids != want_text {
            failures.push(format!(
                "{name}: streamed ids\n  kjarni: {ids:?}\n  hf:     {want_text:?}"
            ));
        }

        let whole = transcriber.transcribe_file(audio).unwrap();
        let want = model["cases"]["timestamps"]["text"].as_str().unwrap();
        eprintln!("{name}: {:?}", whole.text);
        if whole.text.trim() != want.trim() {
            failures.push(format!(
                "{name}: text\n  kjarni: {:?}\n  hf:     {want:?}",
                whole.text
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
