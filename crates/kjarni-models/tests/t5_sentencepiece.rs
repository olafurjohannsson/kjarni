//! flan-t5-base loaded from its `spiece.model`, without tokenizer.json, gives
//! what transformers 4.57 gives for the same folder. Skips when flan-t5-base is
//! not in the Hugging Face cache.
//!
//!     cargo test --release -p kjarni-models --test t5_sentencepiece -- --ignored

use std::path::{Path, PathBuf};

use kjarni_models::models::t5::model::T5Model;
use kjarni_transformers::common::{DecodingStrategy, HFGenerationConfig};
use kjarni_transformers::encoder_decoder::{EncoderDecoderGenerator, EncoderDecoderLanguageModel};
use kjarni_transformers::pipeline::Seq2SeqLoader;
use kjarni_transformers::traits::Device;

/// flan-t5-base's cached files in a folder of their own, the weights linked,
/// and tokenizer.json left out unless `with_json`.
fn cached_flan_t5(with_json: bool) -> Option<tempfile::TempDir> {
    let home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".cache/huggingface")))?;
    let snapshot = std::fs::read_dir(home.join("hub/models--google--flan-t5-base/snapshots"))
        .ok()?
        .flatten()
        .next()?
        .path();
    let dir = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(&snapshot).ok()?.flatten() {
        let name = entry.file_name();
        if name == "tokenizer.json" && !with_json {
            continue;
        }
        let src = std::fs::canonicalize(entry.path()).unwrap();
        let dst = dir.path().join(&name);
        #[cfg(unix)]
        std::os::unix::fs::symlink(&src, &dst).unwrap();
        #[cfg(not(unix))]
        std::fs::copy(&src, &dst).unwrap();
    }
    dir.path().join("spiece.model").exists().then_some(dir)
}

async fn greedy(dir: &Path, prompts: &[&str]) -> Vec<String> {
    let model: T5Model = Seq2SeqLoader::load_from_pretrained(
        dir,
        Device::Cpu,
        None,
        None,
        HFGenerationConfig::load_or_default(dir),
    )
    .unwrap();
    let mut config = model.get_generation_config_for_input(prompts[0]);
    config.strategy = DecodingStrategy::Greedy;
    config.max_new_tokens = Some(40);
    let generator = EncoderDecoderGenerator::new(Box::new(model)).unwrap();
    let mut out = Vec::new();
    for p in prompts {
        out.push(generator.generate(p, Some(&config)).await.unwrap());
    }
    out
}

#[tokio::test]
#[ignore = "needs google/flan-t5-base in the Hugging Face cache"]
async fn flan_t5_from_spiece_model_generates_what_transformers_does() {
    let Some(dir) = cached_flan_t5(false) else {
        eprintln!("skipping: google/flan-t5-base not in the Hugging Face cache");
        return;
    };
    // transformers: T5ForConditionalGeneration.generate(num_beams=1,
    // do_sample=False, max_new_tokens=40) with AutoTokenizer on the same folder.
    let cases = [
        (
            "translate English to German: How old are you?",
            "Wie old sind Sie?",
        ),
        (
            "summarize: Þórður Jónsson moved to Reykjavík in 2019.  He   now works   at the hospital.",
            "órur Jónsson has moved to Reykjavk.",
        ),
        (
            "Answer the question: what is the capital of France?",
            "Paris",
        ),
        // Here transformers' ids differ from the published tokenizer.json's.
        (
            "translate English to French: The <extra_id_0> is red.",
            "Le roi est rouge.",
        ),
    ];
    let prompts: Vec<&str> = cases.iter().map(|(p, _)| *p).collect();
    let got = greedy(dir.path(), &prompts).await;
    for ((prompt, want), got) in cases.iter().zip(&got) {
        assert_eq!(got.trim(), *want, "{prompt}");
    }

    // On the prompts where both tokenizers agree, so do the generations.
    let with_json = cached_flan_t5(true).unwrap();
    let published = greedy(with_json.path(), &prompts[..3]).await;
    assert_eq!(published, got[..3]);
}
