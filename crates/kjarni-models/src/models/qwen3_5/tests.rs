//! Reference values from transformers 5.16.1 running Qwen/Qwen3.5-0.8B in f32
//! on "The capital of France is".

use std::path::PathBuf;

use kjarni_transformers::{Device, cache::Cache, decoder::prelude::*, models::LanguageModel};
use ndarray::Array2;

use super::Qwen3_5Model;

const PROMPT_IDS: [u32; 5] = [760, 6511, 314, 9338, 369];
const LAST_LOGITS_HEAD: [f32; 8] = [
    6.564751625061035,
    6.569220066070557,
    3.525311231613159,
    5.078021049499512,
    2.250756025314331,
    0.7888838648796082,
    6.465867519378662,
    7.816433906555176,
];
/// " Paris.\nThe capital of France is" three times over.
const GREEDY: [u32; 24] = [
    11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369, 11751,
    13, 198, 760, 6511, 314, 9338, 369,
];

fn model_dir() -> Option<PathBuf> {
    let snapshots =
        dirs::home_dir()?.join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .next()
}

fn load() -> Option<Qwen3_5Model> {
    let dir = model_dir()?;
    Some(Qwen3_5Model::from_pretrained(&dir, Device::Cpu, None, None, None).expect("loads"))
}

/// Logits for every position of `ids`, continuing `cache`.
fn logits(model: &Qwen3_5Model, ids: &[u32], cache: &mut dyn Cache) -> Array2<f32> {
    let offset = cache.get_seq_length();
    let tokens = Array2::from_shape_vec((1, ids.len()), ids.to_vec()).unwrap();
    let hidden = model.embed(&tokens, offset).unwrap();
    let mask = CpuDecoderOps::get_attention_mask(model, ids.len(), offset).unwrap();
    let out = CpuDecoderOps::decoder(model)
        .forward(&hidden, &mask, offset, Some(cache))
        .unwrap();
    let logits = CpuDecoderOps::project_to_logits(model, &out).unwrap();
    logits.index_axis_move(ndarray::Axis(0), 0)
}

fn argmax(row: ndarray::ArrayView1<f32>) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

#[test]
#[ignore = "needs Qwen/Qwen3.5-0.8B in the Hugging Face cache"]
fn prefill_logits_match_transformers() {
    let Some(model) = load() else { return };
    let mut cache = model.new_cache(1, 64, 1).unwrap();
    let l = logits(&model, &PROMPT_IDS, cache.as_mut());
    let last = l.row(PROMPT_IDS.len() - 1);
    for (i, (got, want)) in last.iter().zip(LAST_LOGITS_HEAD).enumerate() {
        assert!((got - want).abs() < 1e-4, "logit {i}: {got} vs {want}");
    }
    assert_eq!(argmax(l.row(0)), 2614);
    assert_eq!(argmax(last), 11751);
    let max = last.iter().copied().fold(f32::MIN, f32::max);
    assert!((max - 14.870590209960938).abs() < 1e-4, "{max}");
}

#[test]
#[ignore = "needs Qwen/Qwen3.5-0.8B in the Hugging Face cache"]
fn greedy_decoding_token_by_token_matches_transformers() {
    let Some(model) = load() else { return };
    let mut cache = model.new_cache(1, 64, 1).unwrap();
    let mut next = argmax(logits(&model, &PROMPT_IDS, cache.as_mut()).row(PROMPT_IDS.len() - 1));
    let mut out = vec![next];
    while out.len() < GREEDY.len() {
        next = argmax(logits(&model, &[next], cache.as_mut()).row(0));
        out.push(next);
    }
    assert_eq!(out, GREEDY);
}

/// 272 tokens: long enough that attention runs in blocks and the recurrence
/// carries state across hundreds of steps.
fn long_prompt() -> String {
    "Kjarni is a Rust library for running AI models locally. ".repeat(3)
        + "It loads weights from safetensors files, tokenizes text, and runs transformer layers on the CPU. "
        + &"The quick brown fox jumps over the lazy dog. ".repeat(20)
        + "Question: what does Kjarni do? Answer:"
}

const LONG_LAST8: [f32; 8] = [
    3.4294724464416504,
    1.5867818593978882,
    1.1164700984954834,
    0.03323976695537567,
    -0.07512366771697998,
    -0.15232586860656738,
    1.9753481149673462,
    1.203545093536377,
];
const LONG_POS100_8: [f32; 8] = [
    5.606895446777344,
    2.6140904426574707,
    3.294088125228882,
    -0.38472098112106323,
    0.2976926565170288,
    1.2950232028961182,
    1.283782720565796,
    -0.25118499994277954,
];
/// " Kjarni loads weights from safetensors files, tokenizes text, and"
const LONG_GREEDY: [u32; 16] = [
    218515, 1820, 72, 20269, 13914, 494, 5363, 294, 25681, 3425, 11, 3817, 4603, 1414, 11, 321,
];

fn long_ids(model: &Qwen3_5Model) -> Vec<u32> {
    let ids = model
        .tokenizer()
        .encode(long_prompt(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    assert_eq!(ids.len(), 272);
    // The ends of what transformers tokenized the same text to.
    assert_eq!(&ids[..6], &[42, 73, 1820, 72, 369, 264]);
    assert_eq!(&ids[266..], &[1820, 72, 635, 30, 21134, 25]);
    ids
}

fn assert_close(got: ndarray::ArrayView1<f32>, want: &[f32], tol: f32) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < tol, "logit {i}: {g} vs {w}");
    }
}

#[test]
#[ignore = "needs Qwen/Qwen3.5-0.8B in the Hugging Face cache"]
fn a_long_prompt_matches_transformers() {
    let Some(model) = load() else { return };
    let ids = long_ids(&model);
    let mut cache = model.new_cache(1, 512, 1).unwrap();
    let l = logits(&model, &ids, cache.as_mut());
    assert_close(l.row(100), &LONG_POS100_8, 1e-3);
    assert_eq!(argmax(l.row(100)), 561);
    assert_close(l.row(ids.len() - 1), &LONG_LAST8, 1e-3);

    let mut next = argmax(l.row(ids.len() - 1));
    let mut out = vec![next];
    while out.len() < LONG_GREEDY.len() {
        next = argmax(logits(&model, &[next], cache.as_mut()).row(0));
        out.push(next);
    }
    assert_eq!(out, LONG_GREEDY);
}

/// A prompt prefilled in two pieces ends in the same state as one prefilled
/// whole, so the convolution window and recurrent state carry across calls.
#[test]
#[ignore = "needs Qwen/Qwen3.5-0.8B in the Hugging Face cache"]
fn prefilling_in_pieces_matches_prefilling_whole() {
    let Some(model) = load() else { return };
    let ids = long_ids(&model);
    let mut cache = model.new_cache(1, 512, 1).unwrap();
    logits(&model, &ids[..201], cache.as_mut());
    let l = logits(&model, &ids[201..], cache.as_mut());
    assert_close(l.row(l.nrows() - 1), &LONG_LAST8, 1e-3);
}

/// Cutting the cache back to a snapshot and continuing gives what a fresh
/// cache gives, which is what prefix reuse across chat turns relies on.
#[test]
#[ignore = "needs Qwen/Qwen3.5-0.8B in the Hugging Face cache"]
fn continuing_from_a_snapshot_matches_starting_fresh() {
    let Some(model) = load() else { return };
    let ids = long_ids(&model);
    let mut cache = model.new_cache(1, 512, 1).unwrap();
    logits(&model, &ids[..201], cache.as_mut());
    // A different continuation, then a rewind to the end of the first prefill.
    logits(&model, &[11, 12, 13], cache.as_mut());
    logits(&model, &[14], cache.as_mut());
    assert_eq!(cache.reusable_prefix(203), 201);
    cache.set_seq_length(201);
    let l = logits(&model, &ids[201..], cache.as_mut());
    assert_close(l.row(l.nrows() - 1), &LONG_LAST8, 1e-3);
}

/// A prefill keeps the state from a few tokens before its end, and continuing
/// from there gives what a fresh cache gives. The next chat turn shares the
/// prompt only up to the generation header, so this is the point it reuses.
#[test]
#[ignore = "needs Qwen/Qwen3.5-0.8B in the Hugging Face cache"]
fn continuing_from_just_before_a_prefill_end_matches_starting_fresh() {
    use super::cache::SNAPSHOT_TAIL;
    let Some(model) = load() else { return };
    let ids = long_ids(&model);
    let mut cache = model.new_cache(1, 512, 1).unwrap();
    logits(&model, &ids[..201], cache.as_mut());
    let tail = 201 - SNAPSHOT_TAIL;
    assert_eq!(cache.reusable_prefix(200), tail);
    cache.set_seq_length(tail);
    let l = logits(&model, &ids[tail..], cache.as_mut());
    assert_close(l.row(l.nrows() - 1), &LONG_LAST8, 1e-3);
}

/// The model's own Jinja template, as a folder load renders it, and the
/// built-in one a registry load uses give the same prompts.
#[test]
#[ignore = "needs Qwen/Qwen3.5-0.8B in the Hugging Face cache"]
fn the_folder_template_and_the_built_in_one_agree() {
    use kjarni_transformers::{
        ChatTemplate, Conversation, chat::jinja::JinjaChatTemplate,
        chat::qwen3_5::Qwen3_5ChatTemplate,
    };
    let Some(dir) = model_dir() else { return };
    let jinja = JinjaChatTemplate::from_dir(&dir)
        .unwrap()
        .expect("has a template");
    let built_in = Qwen3_5ChatTemplate::new();

    let mut one = Conversation::with_system("  Be brief. ");
    one.push_user(" What is Rust? \n");
    let mut two = Conversation::new();
    two.push_user("Hi");
    two.push_assistant("<think>\nThe user greets.\n</think>\n\nHello!");
    two.push_user("Bye");
    for c in [one, two] {
        assert_eq!(jinja.try_apply(&c).unwrap(), built_in.apply(&c));
    }
}

/// Qwen/Qwen3.5-4B as `kjarni download` stores it: tied embeddings, 32 value
/// heads over 16 key heads. Reference values from transformers 5.16.1 in f32.
mod four_b {
    use super::*;

    fn load() -> Option<Qwen3_5Model> {
        let dir = dirs::home_dir()?.join(".cache/kjarni/Qwen_Qwen3.5-4B");
        dir.join("config.json").is_file().then(|| {
            Qwen3_5Model::from_pretrained(&dir, Device::Cpu, None, None, None).expect("loads")
        })
    }

    fn greedy(model: &Qwen3_5Model, ids: &[u32], n: usize) -> (Array2<f32>, Vec<u32>) {
        let mut cache = model.new_cache(1, ids.len() + n, 1).unwrap();
        let l = logits(model, ids, cache.as_mut());
        let mut next = argmax(l.row(ids.len() - 1));
        let mut out = vec![next];
        while out.len() < n {
            next = argmax(logits(model, &[next], cache.as_mut()).row(0));
            out.push(next);
        }
        (l, out)
    }

    /// " Paris.\nA. True\nB. False\nAnswer:\nA\n\n"
    const SHORT_GREEDY: [u32; 16] = [
        11751, 13, 198, 32, 13, 2912, 198, 33, 13, 3439, 198, 15666, 25, 198, 32, 271,
    ];
    const SHORT_LAST8: [f32; 8] = [
        8.736597061157227,
        5.8917741775512695,
        3.8924331665039062,
        4.870279312133789,
        4.189123630523682,
        2.8736014366149902,
        5.768157958984375,
        9.242416381835938,
    ];
    /// " Kjarni is a Rust library for running AI models locally. It loads weights"
    const LONG_GREEDY: [u32; 16] = [
        218515, 1820, 72, 369, 264, 32671, 6522, 364, 4162, 14791, 3983, 22756, 13, 1049, 20269,
        13914,
    ];
    const LONG_LAST8: [f32; 8] = [
        5.176456928253174,
        3.4404609203338623,
        1.1258578300476074,
        1.8402273654937744,
        1.1713718175888062,
        1.0172667503356934,
        1.6084625720977783,
        2.238356351852417,
    ];

    #[test]
    #[ignore = "needs Qwen/Qwen3.5-4B in ~/.cache/kjarni and about 17 GB of RAM"]
    fn matches_transformers() {
        let Some(model) = load() else { return };
        let (l, out) = greedy(&model, &PROMPT_IDS, SHORT_GREEDY.len());
        assert_close(l.row(PROMPT_IDS.len() - 1), &SHORT_LAST8, 1e-3);
        assert_eq!(out, SHORT_GREEDY);

        let ids = long_ids(&model);
        let (l, out) = greedy(&model, &ids, LONG_GREEDY.len());
        assert_close(l.row(ids.len() - 1), &LONG_LAST8, 1e-3);
        assert_eq!(out, LONG_GREEDY);
    }
}

/// The tiny model in `testdata`, which `make.py` describes, and what
/// transformers computed with it.
mod tiny {
    use super::*;

    fn dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/qwen3_5/testdata")
    }

    fn expected() -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(dir().join("expected.json")).unwrap())
            .unwrap()
    }

    fn ids(v: &serde_json::Value) -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    }

    fn floats(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    fn model() -> Qwen3_5Model {
        Qwen3_5Model::from_pretrained(&dir(), Device::Cpu, None, None, None).unwrap()
    }

    fn greedy(model: &Qwen3_5Model, prompt: &[u32], n: usize) -> Vec<u32> {
        let mut cache = model.new_cache(1, 256, 1).unwrap();
        let mut next = argmax(logits(model, prompt, cache.as_mut()).row(prompt.len() - 1));
        let mut out = vec![next];
        while out.len() < n {
            next = argmax(logits(model, &[next], cache.as_mut()).row(0));
            out.push(next);
        }
        out
    }

    #[test]
    fn reads_a_text_only_checkpoint_with_shared_key_heads_and_its_own_head() {
        let m = model();
        let c = m.config();
        assert_eq!(c.prefix, "model.");
        assert_eq!((c.linear_num_key_heads, c.linear_num_value_heads), (2, 4));
        assert!(!c.tie_word_embeddings);
        assert_eq!(c.rotary_dim, 8);
    }

    #[test]
    fn every_logit_of_a_prompt_matches_transformers() {
        let e = expected();
        let prompt = ids(&e["short_ids"]);
        let mut cache = model().new_cache(1, 64, 1).unwrap();
        let got = logits(&model(), &prompt, cache.as_mut());
        let want = e["short_logits"].as_array().unwrap();
        for (pos, row) in want.iter().enumerate() {
            for (i, w) in floats(row).iter().enumerate() {
                let g = got[[pos, i]];
                assert!(
                    (g - w).abs() < 1e-4,
                    "position {pos}, logit {i}: {g} vs {w}"
                );
            }
        }
    }

    #[test]
    fn greedy_decoding_matches_transformers() {
        let e = expected();
        let m = model();
        assert_eq!(
            greedy(&m, &ids(&e["short_ids"]), 12),
            ids(&e["short_greedy"])
        );
        assert_eq!(greedy(&m, &ids(&e["long_ids"]), 12), ids(&e["long_greedy"]));
    }

    #[test]
    fn a_prompt_longer_than_an_attention_block_matches_transformers() {
        let e = expected();
        let prompt = ids(&e["long_ids"]);
        let mut cache = model().new_cache(1, 128, 1).unwrap();
        let got = logits(&model(), &prompt, cache.as_mut());
        assert_close(
            got.row(prompt.len() - 1),
            &floats(&e["long_last_logits"]),
            1e-4,
        );
    }
}
