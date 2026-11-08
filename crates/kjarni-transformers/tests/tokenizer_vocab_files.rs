//! Tokenizers built from vocab.json and merges.txt, or vocab.txt, give the
//! token ids transformers gives for the same files.
//!
//! The pinned tests use real models from the Hugging Face cache, copied without
//! their tokenizer.json if they have one, and skip when a model is not cached.
//! Their ids are from transformers 4.57's `AutoTokenizer` on the same files.
//!
//! The sweep compares against a reference file written by a script, for as
//! many models and texts as it holds, and runs only when
//! `KJARNI_TOKENIZER_REFERENCE` names that file:
//!
//!     KJARNI_TOKENIZER_REFERENCE=reference.json \
//!         cargo test -p kjarni-transformers --test tokenizer_vocab_files -- --nocapture

use std::path::{Path, PathBuf};

use kjarni_transformers::tokenizer::load_tokenizer;
use serde_json::Value;

const ICELANDIC: &str = "Reykjavík er höfuðborg Íslands. Þórður og Ægir fóru á ströndina.";
const SPECIALS: &str = "a <mask> token, [MASK] token, [SEP] and </s> in text <s>";
const PAIR: (&str, &str) = ("Hvar er Reykjavík?", "Reykjavík er á Íslandi.");

/// A cached Hugging Face model's older tokenizer files, copied to a folder of
/// their own so tokenizer.json cannot be picked up.
fn cached_without_tokenizer_json(repo: &str) -> Option<tempfile::TempDir> {
    let home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".cache/huggingface")))?;
    let snapshots = home
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    let snapshot = std::fs::read_dir(snapshots).ok()?.flatten().next()?.path();
    let dir = tempfile::tempdir().unwrap();
    for f in [
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "config.json",
    ] {
        if snapshot.join(f).exists() {
            std::fs::copy(snapshot.join(f), dir.path().join(f)).unwrap();
        }
    }
    Some(dir)
}

fn encode(dir: &Path, text: &str) -> Vec<u32> {
    let tok = load_tokenizer(dir).unwrap();
    tok.encode(text, true).unwrap().get_ids().to_vec()
}

fn encode_pair(dir: &Path, (a, b): (&str, &str)) -> Vec<u32> {
    let tok = load_tokenizer(dir).unwrap();
    tok.encode((a, b), true).unwrap().get_ids().to_vec()
}

macro_rules! cached {
    ($repo:expr) => {
        match cached_without_tokenizer_json($repo) {
            Some(d) => d,
            None => {
                eprintln!("skipping: {} not in the Hugging Face cache", $repo);
                return;
            }
        }
    };
}

#[test]
fn icebert_from_vocab_json_and_merges() {
    let dir = cached!("vesteinn/IceBERT-ner");
    assert_eq!(
        encode(dir.path(), ICELANDIC),
        [
            0, 958, 304, 2277, 939, 18, 9776, 291, 18058, 2790, 283, 28904, 18, 2
        ]
    );
    assert_eq!(
        encode(dir.path(), SPECIALS),
        [
            0, 1687, 4, 293, 705, 307, 16, 2680, 11890, 4784, 65, 293, 705, 307, 16, 2680, 13121,
            52, 65, 1362, 225, 2, 3868, 15408, 225, 0, 2
        ]
    );
    assert_eq!(
        encode_pair(dir.path(), PAIR),
        [0, 26676, 304, 958, 35, 2, 2, 958, 304, 283, 1359, 18, 2]
    );
}

#[test]
fn toxic_bert_from_vocab_txt_lowercases_and_strips_accents() {
    let dir = cached!("unitary/toxic-bert");
    assert_eq!(
        encode(dir.path(), ICELANDIC),
        [
            101, 28559, 9413, 7570, 11263, 29668, 11755, 3470, 1012, 1101, 2953, 29668, 3126,
            13958, 1097, 5856, 2099, 2005, 2226, 1037, 2358, 4948, 18979, 1012, 102
        ]
    );
    assert_eq!(
        encode(dir.path(), SPECIALS),
        [
            101, 1037, 1026, 7308, 1028, 19204, 1010, 103, 19204, 1010, 102, 1998, 1026, 1013,
            1055, 1028, 1999, 3793, 1026, 1055, 1028, 102
        ]
    );
    assert_eq!(
        encode_pair(dir.path(), PAIR),
        [
            101, 1044, 10755, 9413, 28559, 1029, 102, 28559, 9413, 1037, 2479, 2072, 1012, 102
        ]
    );
}

#[test]
fn mpnet_from_vocab_txt_uses_s_tokens_and_a_double_separator() {
    let dir = cached!("sentence-transformers/all-mpnet-base-v2");
    assert_eq!(
        encode(dir.path(), SPECIALS),
        [
            0, 1041, 30526, 19208, 1014, 1035, 7312, 1037, 19208, 1014, 1035, 19806, 1037, 2002, 2,
            2003, 3797, 0, 2
        ]
    );
    assert_eq!(
        encode_pair(dir.path(), PAIR),
        [
            0, 1048, 10759, 9417, 28563, 1033, 2, 2, 28563, 9417, 1041, 2483, 2076, 1016, 2
        ]
    );
}

#[test]
fn gpt2_from_vocab_json_and_merges_adds_no_special_tokens() {
    let dir = cached!("openai-community/gpt2");
    assert_eq!(
        encode(dir.path(), SPECIALS),
        [
            64, 1279, 27932, 29, 11241, 11, 685, 31180, 42, 60, 11241, 11, 685, 5188, 47, 60, 290,
            7359, 82, 29, 287, 2420, 1279, 82, 29
        ]
    );
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

#[test]
fn matches_transformers_on_every_reference_model() {
    let Ok(path) = std::env::var("KJARNI_TOKENIZER_REFERENCE") else {
        eprintln!("skipping: KJARNI_TOKENIZER_REFERENCE not set");
        return;
    };
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let texts: Vec<&str> = reference["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    let pairs: Vec<(&str, &str)> = reference["pairs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_str().unwrap(), p[1].as_str().unwrap()))
        .collect();

    let mut failures = Vec::new();
    for (name, model) in reference["models"].as_object().unwrap() {
        // Only the files transformers saw: no tokenizer.json.
        let dir = tempfile::tempdir().unwrap();
        for f in model["files"].as_array().unwrap() {
            let f = f.as_str().unwrap();
            std::fs::copy(
                std::path::Path::new(model["dir"].as_str().unwrap()).join(f),
                dir.path().join(f),
            )
            .unwrap();
        }
        let tok = match load_tokenizer(dir.path()) {
            Ok(t) => t,
            Err(e) => {
                failures.push(format!("{name}: {e:#}"));
                continue;
            }
        };
        let mut bad = 0;
        for (i, text) in texts.iter().enumerate() {
            for (key, special) in [("single", true), ("bare", false)] {
                let got = tok.encode(*text, special).unwrap().get_ids().to_vec();
                let want = ids(&model[key][i]);
                if got != want {
                    bad += 1;
                    failures.push(format!(
                        "{name} {key} {text:?}\n   got  {got:?}\n   want {want:?}"
                    ));
                }
            }
        }
        for (i, (a, b)) in pairs.iter().enumerate() {
            let got = tok.encode((*a, *b), true).unwrap().get_ids().to_vec();
            let want = ids(&model["pairs"][i]);
            if got != want {
                bad += 1;
                failures.push(format!(
                    "{name} pair {a:?}\n   got  {got:?}\n   want {want:?}"
                ));
            }
        }
        eprintln!(
            "{name:30} {}",
            if bad == 0 {
                "ok".to_string()
            } else {
                format!("{bad} mismatches")
            }
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// SentencePiece tokenizers against transformers 4.57's conversion of the same
/// `.model` files. `KJARNI_SPM_REFERENCE` names the file the reference script
/// writes; each model's folder is `KJARNI_SPM_MODELS/<name>`, and is used
/// without its tokenizer.json. Besides ids and offsets, the tokenizer itself
/// must equal transformers', section by section.
#[test]
fn sentencepiece_matches_transformers_on_every_reference_model() {
    let (Ok(path), Ok(models)) = (
        std::env::var("KJARNI_SPM_REFERENCE"),
        std::env::var("KJARNI_SPM_MODELS"),
    ) else {
        eprintln!("skipping: KJARNI_SPM_REFERENCE and KJARNI_SPM_MODELS not set");
        return;
    };
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let texts: Vec<&str> = reference["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    let pair = (
        reference["pair"][0].as_str().unwrap(),
        reference["pair"][1].as_str().unwrap(),
    );

    let mut failures = Vec::new();
    let mut checked = 0;
    for (name, want) in reference["models"].as_object().unwrap() {
        let src = Path::new(&models).join(name);
        let dir = tempfile::tempdir().unwrap();
        for entry in std::fs::read_dir(&src).unwrap().flatten() {
            if entry.file_name() != "tokenizer.json" && entry.path().is_file() {
                std::fs::copy(entry.path(), dir.path().join(entry.file_name())).unwrap();
            }
        }
        let loaded = load_tokenizer(dir.path());
        if want.get("error").is_some() {
            if loaded.is_ok() {
                failures.push(format!("{name}: transformers fails, Kjarni loaded it"));
            }
            continue;
        }
        let tok = match loaded {
            Ok(t) => t,
            Err(e) => {
                failures.push(format!("{name}: {e:#}"));
                continue;
            }
        };
        checked += 1;

        let got: Value = serde_json::from_str(&tok.to_string(false).unwrap()).unwrap();
        for section in [
            "model",
            "normalizer",
            "pre_tokenizer",
            "post_processor",
            "decoder",
            "added_tokens",
        ] {
            if got[section] != want["backend"][section] {
                let g = got[section].to_string();
                let w = want["backend"][section].to_string();
                let at = g.chars().zip(w.chars()).take_while(|(a, b)| a == b).count();
                let around = |s: &str| {
                    s.chars()
                        .skip(at.saturating_sub(60))
                        .take(160)
                        .collect::<String>()
                };
                failures.push(format!(
                    "{name}: {section} differs at char {at}\n  kjarni: {}\n  hf:     {}",
                    around(&g),
                    around(&w)
                ));
            }
        }

        for (special, key) in [(true, "with_special"), (false, "without_special")] {
            for (i, text) in texts.iter().enumerate() {
                let enc = tok.encode_char_offsets(*text, special).unwrap();
                let ids = enc.get_ids().to_vec();
                let offsets: Vec<Vec<usize>> =
                    enc.get_offsets().iter().map(|&(a, b)| vec![a, b]).collect();
                let w = &want[key][i];
                let want_ids: Vec<u32> = serde_json::from_value(w["ids"].clone()).unwrap();
                let want_offsets: Vec<Vec<usize>> =
                    serde_json::from_value(w["offsets"].clone()).unwrap();
                if ids != want_ids {
                    failures.push(format!(
                        "{name} {key} {text:?}: ids {ids:?} want {want_ids:?}"
                    ));
                } else if offsets != want_offsets {
                    failures.push(format!(
                        "{name} {key} {text:?}: offsets {offsets:?} want {want_offsets:?}"
                    ));
                }
            }
        }
        let ids = tok.encode(pair, true).unwrap().get_ids().to_vec();
        let want_pair: Vec<u32> = serde_json::from_value(want["pair"].clone()).unwrap();
        if ids != want_pair {
            failures.push(format!("{name} pair: {ids:?} want {want_pair:?}"));
        }
    }
    eprintln!("checked {checked} SentencePiece tokenizers");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Marian folders (`source.spm`, `target.spm`, `vocab.json`) against
/// transformers' `MarianTokenizer`: the ids of each text, with an id the model
/// has no embedding for read as `<unk>` as the seq2seq encoder reads it, and
/// the decoded text of target-language ids. Runs when
/// `KJARNI_MARIAN_REFERENCE` names the file the reference script writes.
#[test]
fn marian_matches_transformers() {
    let Ok(path) = std::env::var("KJARNI_MARIAN_REFERENCE") else {
        eprintln!("skipping: KJARNI_MARIAN_REFERENCE not set");
        return;
    };
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut failures = Vec::new();
    for (name, model) in reference["models"].as_object().unwrap() {
        let dir = std::path::Path::new(model["dir"].as_str().unwrap());
        let tok = match kjarni_transformers::tokenizer::load_tokenizer(dir) {
            Ok(t) => t,
            Err(e) => {
                failures.push(format!("{name}: {e:#}"));
                continue;
            }
        };
        let vocab: std::collections::HashMap<String, u32> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("vocab.json")).unwrap())
                .unwrap();
        let size = vocab.len() as u32;
        let unk = vocab["<unk>"];
        for case in model["cases"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            let want: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            let got: Vec<u32> = tok
                .encode(text, true)
                .unwrap()
                .get_ids()
                .iter()
                .map(|&id| if id >= size { unk } else { id })
                .collect();
            if got != want {
                failures.push(format!(
                    "{name}: {text:?}\n  kjarni: {got:?}\n  hf:     {want:?}"
                ));
            }
        }
        let ids: Vec<u32> = serde_json::from_value(model["decode"]["ids"].clone()).unwrap();
        let want = model["decode"]["text"].as_str().unwrap();
        let got = tok.decode(&ids, true).unwrap();
        if got != want {
            failures.push(format!(
                "{name}: decode\n  kjarni: {got:?}\n  hf:     {want:?}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
