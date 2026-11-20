//! Reference values from transformers 5.16.1 running Qwen/Qwen3.5-0.8B in f32
//! on "The capital of France is".

use std::path::PathBuf;

use kjarni_transformers::{Device, cache::Cache, decoder::prelude::*, models::LanguageModel};
use ndarray::Array2;

use super::Qwen3_5Model;
use super::config::LayerKind;

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
    pub(super) const SHORT_GREEDY: [u32; 16] = [
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
    pub(super) const LONG_GREEDY: [u32; 16] = [
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

    pub(super) fn dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/qwen3_5/testdata")
    }

    pub(super) fn expected() -> serde_json::Value {
        expected_in(&dir())
    }

    pub(super) fn expected_in(dir: &std::path::Path) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(dir.join("expected.json")).unwrap()).unwrap()
    }

    pub(super) fn ids(v: &serde_json::Value) -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    }

    pub(super) fn floats(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    fn model() -> Qwen3_5Model {
        Qwen3_5Model::from_pretrained(&dir(), Device::Cpu, None, None, None).unwrap()
    }

    pub(super) fn greedy(model: &Qwen3_5Model, prompt: &[u32], n: usize) -> Vec<u32> {
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

    /// The same model as llama.cpp's converter lays it out; see `make_gguf.py`.
    /// Layer 0 holds the merged `attn_qkv_gate` and `ssm_beta_alpha` tensors,
    /// the other layers the separate ones, and the value heads are tiled.
    fn gguf_model() -> Qwen3_5Model {
        Qwen3_5Model::from_pretrained(&dir().join("tiny.gguf"), Device::Cpu, None, None, None)
            .unwrap()
    }

    #[test]
    fn reads_the_gguf_layout_from_its_metadata() {
        let m = gguf_model();
        let c = m.config();
        assert!(c.llama_cpp_layout);
        assert_eq!((c.linear_num_key_heads, c.linear_num_value_heads), (2, 4));
        assert_eq!((c.linear_key_head_dim, c.linear_value_head_dim), (16, 16));
        assert!(!c.tie_word_embeddings);
        assert_eq!((c.head_dim, c.rotary_dim, c.vocab_size), (32, 8, 256));
        // From recurrent_layers, which disagrees with full_attention_interval.
        assert_eq!(
            c.layer_kinds,
            [
                LayerKind::LinearAttention,
                LayerKind::LinearAttention,
                LayerKind::FullAttention,
                LayerKind::LinearAttention,
            ]
        );
    }

    #[test]
    fn the_gguf_matches_transformers() {
        let e = expected();
        let m = gguf_model();
        let prompt = ids(&e["short_ids"]);
        let mut cache = m.new_cache(1, 64, 1).unwrap();
        let got = logits(&m, &prompt, cache.as_mut());
        for (pos, row) in e["short_logits"].as_array().unwrap().iter().enumerate() {
            assert_close(got.row(pos), &floats(row), 1e-4);
        }
        let long = ids(&e["long_ids"]);
        let mut cache = m.new_cache(1, 128, 1).unwrap();
        let got = logits(&m, &long, cache.as_mut());
        assert_close(
            got.row(long.len() - 1),
            &floats(&e["long_last_logits"]),
            1e-4,
        );
        assert_eq!(greedy(&m, &prompt, 12), ids(&e["short_greedy"]));
    }
}

/// The tiny mixture-of-experts model in `testdata/moe`, which its `make.py`
/// describes: eight experts, two per token, and a gated shared expert. Layers 0
/// and 1 store a tensor per expert, layers 2 and 3 the experts stacked.
mod tiny_moe {
    use super::tiny::{expected_in, floats, greedy, ids};
    use super::*;
    use crate::models::qwen3_5::config::MoeDims;

    fn dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/qwen3_5/testdata/moe")
    }

    fn check(m: &Qwen3_5Model) {
        let e = expected_in(&dir());
        let prompt = ids(&e["short_ids"]);
        let mut cache = m.new_cache(1, 64, 1).unwrap();
        let got = logits(m, &prompt, cache.as_mut());
        for (pos, row) in e["short_logits"].as_array().unwrap().iter().enumerate() {
            assert_close(got.row(pos), &floats(row), 1e-4);
        }
        let long = ids(&e["long_ids"]);
        let mut cache = m.new_cache(1, 128, 1).unwrap();
        let got = logits(m, &long, cache.as_mut());
        assert_close(
            got.row(long.len() - 1),
            &floats(&e["long_last_logits"]),
            1e-4,
        );
        assert_eq!(greedy(m, &prompt, 12), ids(&e["short_greedy"]));
        assert_eq!(greedy(m, &long, 12), ids(&e["long_greedy"]));
    }

    #[test]
    fn reads_the_experts_from_the_config() {
        let m = Qwen3_5Model::from_pretrained(&dir(), Device::Cpu, None, None, None).unwrap();
        assert_eq!(
            m.config().moe,
            Some(MoeDims {
                num_experts: 8,
                top_k: 2,
                expert_size: 32,
                shared_size: 48
            })
        );
    }

    #[test]
    fn matches_transformers() {
        check(&Qwen3_5Model::from_pretrained(&dir(), Device::Cpu, None, None, None).unwrap());
    }

    /// As llama.cpp's converter lays it out; see `make_gguf.py`. Each expert is
    /// read as its rows of the layer's stacked `ffn_*_exps` tensors.
    #[test]
    fn the_gguf_matches_transformers() {
        let m =
            Qwen3_5Model::from_pretrained(&dir().join("tiny.gguf"), Device::Cpu, None, None, None)
                .unwrap();
        assert!(m.config().llama_cpp_layout);
        assert_eq!(
            m.config().moe.map(|d| (d.num_experts, d.top_k)),
            Some((8, 2))
        );
        check(&m);
    }
}

/// The Q4_K_M files the registry downloads with `--gguf`, against the f32
/// transformers references above. Quantization moves logits by tenths, so these
/// check what survives it: the greedy continuation of the long prompt.
mod q4_k_m {
    use super::*;

    fn load(folder: &str) -> Option<Qwen3_5Model> {
        let path = dirs::home_dir()?
            .join(".cache/kjarni")
            .join(folder)
            .join("model.gguf");
        path.is_file().then(|| {
            Qwen3_5Model::from_pretrained(&path, Device::Cpu, None, None, None).expect("loads")
        })
    }

    fn greedy(model: &Qwen3_5Model, ids: &[u32], n: usize) -> Vec<u32> {
        let mut cache = model.new_cache(1, ids.len() + n, 1).unwrap();
        let mut next = argmax(logits(model, ids, cache.as_mut()).row(ids.len() - 1));
        let mut out = vec![next];
        while out.len() < n {
            next = argmax(logits(model, &[next], cache.as_mut()).row(0));
            out.push(next);
        }
        out
    }

    #[test]
    #[ignore = "needs `kjarni model download qwen3.5-0.8b --gguf`"]
    fn the_0_8b_continues_the_long_prompt_as_transformers_does() {
        let Some(model) = load("Qwen_Qwen3.5-0.8B") else {
            return;
        };
        let ids = long_ids(&model);
        assert_eq!(greedy(&model, &ids, LONG_GREEDY.len()), LONG_GREEDY);
    }

    /// 32 value heads over 16 key heads, so this reads llama.cpp's tiled order.
    #[test]
    #[ignore = "needs `kjarni model download qwen3.5-4b --gguf`"]
    fn the_4b_continues_both_prompts_as_transformers_does() {
        let Some(model) = load("Qwen_Qwen3.5-4B") else {
            return;
        };
        let ids = long_ids(&model);
        assert_eq!(greedy(&model, &ids, 16), four_b::LONG_GREEDY);
        // " Paris.\n", where the f32 model's continuation and this one part ways.
        assert_eq!(greedy(&model, &PROMPT_IDS, 3), four_b::SHORT_GREEDY[..3]);
    }
}

/// The tiny models on the GPU against transformers, and against the GPU
/// itself across cache cut-backs.
#[cfg(not(target_arch = "wasm32"))]
mod gpu {
    use super::tiny::{dir, expected, floats, ids};
    use super::*;
    use crate::models::qwen3_5::cache::SNAPSHOT_TAIL;
    use kjarni_transformers::WgpuContext;

    async fn model(file: Option<&str>) -> (Qwen3_5Model, GpuDecoderBackend) {
        let ctx = WgpuContext::new().await.unwrap();
        let path = file.map_or_else(dir, |f| dir().join(f));
        let model =
            Qwen3_5Model::from_pretrained(&path, Device::Wgpu, Some(ctx.clone()), None, None)
                .unwrap();
        (model, GpuDecoderBackend::new(ctx).unwrap())
    }

    async fn logits(
        model: &Qwen3_5Model,
        backend: &GpuDecoderBackend,
        ids: &[u32],
        cache: &mut dyn Cache,
    ) -> Array2<f32> {
        backend.forward_batch(model, ids, cache).await.unwrap()
    }

    async fn greedy(
        model: &Qwen3_5Model,
        backend: &GpuDecoderBackend,
        prompt: &[u32],
        n: usize,
    ) -> Vec<u32> {
        let mut cache = model.new_cache(1, prompt.len() + n, 1).unwrap();
        let first = logits(model, backend, prompt, cache.as_mut()).await;
        let mut next = argmax(first.row(prompt.len() - 1));
        let mut out = vec![next];
        while out.len() < n {
            next = argmax(logits(model, backend, &[next], cache.as_mut()).await.row(0));
            out.push(next);
        }
        out
    }

    async fn matches_transformers(file: Option<&str>) {
        let e = expected();
        let (m, backend) = model(file).await;
        let prompt = ids(&e["short_ids"]);
        let mut cache = m.new_cache(1, 64, 1).unwrap();
        let got = logits(&m, &backend, &prompt, cache.as_mut()).await;
        for (pos, row) in e["short_logits"].as_array().unwrap().iter().enumerate() {
            assert_close(got.row(pos), &floats(row), 1e-4);
        }
        let long = ids(&e["long_ids"]);
        let mut cache = m.new_cache(1, 128, 1).unwrap();
        let got = logits(&m, &backend, &long, cache.as_mut()).await;
        assert_close(
            got.row(long.len() - 1),
            &floats(&e["long_last_logits"]),
            1e-4,
        );
        assert_eq!(
            greedy(&m, &backend, &prompt, 12).await,
            ids(&e["short_greedy"])
        );
        assert_eq!(
            greedy(&m, &backend, &long, 12).await,
            ids(&e["long_greedy"])
        );
    }

    #[tokio::test]
    #[ignore = "GPU required"]
    async fn the_tiny_model_matches_transformers_on_the_gpu() {
        matches_transformers(None).await;
    }

    /// llama.cpp's layout: merged tensors in layer 0, tiled value heads.
    #[tokio::test]
    #[ignore = "GPU required"]
    async fn the_tiny_gguf_matches_transformers_on_the_gpu() {
        matches_transformers(Some("tiny.gguf")).await;
    }

    /// A prompt that shares a prefix with the last one starts from the
    /// snapshot at the end of that prefix, or the one just before it, and
    /// gives the logits a fresh cache gives.
    #[tokio::test]
    #[ignore = "GPU required"]
    async fn continuing_from_snapshots_matches_starting_fresh() {
        let e = expected();
        let (m, backend) = model(None).await;
        let long = ids(&e["long_ids"]);
        assert!(long.len() > 2 * SNAPSHOT_TAIL);
        let fresh = {
            let mut cache = m.new_cache(1, 256, 1).unwrap();
            logits(&m, &backend, &long, cache.as_mut()).await
        };

        let mut cache = m.new_cache(1, 256, 1).unwrap();
        let half = long.len() / 2;
        logits(&m, &backend, &long[..half], cache.as_mut()).await;
        logits(&m, &backend, &[7, 8, 9], cache.as_mut()).await;
        for point in [half, half - SNAPSHOT_TAIL] {
            assert_eq!(cache.reusable_prefix(half + 1), half);
            assert_eq!(cache.reusable_prefix(point), point);
            cache.set_seq_length(point);
            let rest = logits(&m, &backend, &long[point..], cache.as_mut()).await;
            for (got, want) in rest
                .rows()
                .into_iter()
                .zip(fresh.rows().into_iter().skip(point))
            {
                let want: Vec<f32> = want.to_vec();
                assert_close(got, &want, 1e-4);
            }
            cache.set_seq_length(half);
        }
    }

    async fn load_real(path: Option<PathBuf>) -> Option<(Qwen3_5Model, GpuDecoderBackend)> {
        let path = path.filter(|p| p.exists())?;
        let ctx = WgpuContext::new().await.unwrap();
        let model =
            Qwen3_5Model::from_pretrained(&path, Device::Wgpu, Some(ctx.clone()), None, None)
                .expect("loads");
        Some((model, GpuDecoderBackend::new(ctx).unwrap()))
    }

    fn gguf(folder: &str) -> Option<PathBuf> {
        Some(
            dirs::home_dir()?
                .join(".cache/kjarni")
                .join(folder)
                .join("model.gguf"),
        )
    }

    #[tokio::test]
    #[ignore = "needs a GPU and Qwen/Qwen3.5-0.8B in the Hugging Face cache"]
    async fn the_0_8b_matches_transformers_on_the_gpu() {
        let Some((m, backend)) = load_real(model_dir()).await else {
            return;
        };
        let mut cache = m.new_cache(1, 64, 1).unwrap();
        let l = logits(&m, &backend, &PROMPT_IDS, cache.as_mut()).await;
        let last = l.row(PROMPT_IDS.len() - 1);
        let diff = last
            .iter()
            .zip(LAST_LOGITS_HEAD)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-3, "largest difference {diff}");
        assert_eq!(
            greedy(&m, &backend, &PROMPT_IDS, GREEDY.len()).await,
            GREEDY
        );
        let ids = long_ids(&m);
        assert_eq!(
            greedy(&m, &backend, &ids, LONG_GREEDY.len()).await,
            LONG_GREEDY
        );
    }

    #[tokio::test]
    #[ignore = "needs a GPU and `kjarni model download qwen3.5-4b --gguf`"]
    async fn the_4b_gguf_continues_both_prompts_as_transformers_does_on_the_gpu() {
        let Some((m, backend)) = load_real(gguf("Qwen_Qwen3.5-4B")).await else {
            return;
        };
        let ids = long_ids(&m);
        assert_eq!(greedy(&m, &backend, &ids, 16).await, four_b::LONG_GREEDY);
        assert_eq!(
            greedy(&m, &backend, &PROMPT_IDS, 3).await,
            four_b::SHORT_GREEDY[..3]
        );
    }

    /// Times every GPU kernel of a 512-token prefill and of four decode steps,
    /// totalled per kernel. A tool rather than a check: run it with
    /// `--features kjarni-transformers/profiling --release -- --nocapture`.
    #[tokio::test]
    #[ignore = "a profile: needs a GPU, the 4B GGUF and the profiling feature"]
    async fn profile_the_4b_on_the_gpu() {
        let path = gguf("Qwen_Qwen3.5-4B").unwrap();
        let ctx = WgpuContext::new().await.unwrap();
        let m = Qwen3_5Model::from_pretrained(&path, Device::Wgpu, Some(ctx.clone()), None, None)
            .unwrap();
        let backend = GpuDecoderBackend::new(ctx.clone()).unwrap();
        // PROFILE_CONTEXT sets the prompt length: decoding at long context is
        // where attention costs most.
        let context: usize = std::env::var("PROFILE_CONTEXT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let ids: Vec<u32> = long_ids(&m).into_iter().cycle().take(context).collect();
        let prompt = |t: &[u32]| Array2::from_shape_vec((1, t.len()), t.to_vec()).unwrap();
        let report = |name: &str, totals: Vec<(String, f64, usize)>, wall: f64| {
            let sum: f64 = totals.iter().map(|t| t.1).sum();
            println!("== {name}: wall {wall:.1} ms, timed GPU {sum:.1} ms");
            for (l, ms, n) in totals.iter().take(18) {
                println!("{:>8.1} ms {:>5.1}% {:>5}x  {l}", ms, 100.0 * ms / sum, n);
            }
        };
        let flush = |c: &std::sync::Arc<WgpuContext>| {
            let mut enc = c.device.create_command_encoder(&Default::default());
            c.profiler.process_results(&mut enc);
            c.queue.submit(Some(enc.finish()));
        };
        // A warm-up, whose timings are dropped.
        let mut cache = m.new_cache(1, context + 8, 1).unwrap();
        backend
            .prefill(&m, &prompt(&ids[..512.min(context)]), cache.as_mut())
            .await
            .unwrap();
        flush(&ctx);
        ctx.profiler.totals(&ctx).await;
        let mut cache = m.new_cache(1, context + 8, 1).unwrap();
        let t = std::time::Instant::now();
        let l = backend
            .prefill(&m, &prompt(&ids), cache.as_mut())
            .await
            .unwrap();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        flush(&ctx);
        report(
            &format!("prefill {context}"),
            ctx.profiler.totals(&ctx).await,
            wall,
        );
        let mut next = argmax(l.view());
        let mut tok = backend.new_decode_token().unwrap();
        let t = std::time::Instant::now();
        // Four steps: the profiler holds 2,048 timed dispatches.
        for i in 0..4 {
            backend.update_decode_token(&mut tok, next).unwrap();
            let l = backend
                .decode_one(&m, &tok, 513 + i, cache.as_mut())
                .await
                .unwrap();
            next = argmax(l.view());
        }
        let wall = t.elapsed().as_secs_f64() * 1e3;
        flush(&ctx);
        report("decode 4 tokens", ctx.profiler.totals(&ctx).await, wall);
    }

    /// The daemon's failing turn: about 2,000 tokens in the cache, then 7,000 of
    /// tool output at once. In one submission their attention scores (4.1 GB) are
    /// over the 2 GB a buffer binding may hold, and the work outlasts the 30 s
    /// readback timeout. It runs, and gives the logits of the same tokens
    /// prefilled 256 at a time.
    #[tokio::test]
    #[ignore = "needs a GPU and `kjarni model download qwen3.5-4b --gguf`"]
    async fn the_4b_prefills_7000_tokens_after_2000_on_the_gpu() {
        let Some((m, backend)) = load_real(gguf("Qwen_Qwen3.5-4B")).await else {
            return;
        };
        let ids: Vec<u32> = long_ids(&m).into_iter().cycle().take(9060).collect();
        let prompt =
            |tokens: &[u32]| Array2::from_shape_vec((1, tokens.len()), tokens.to_vec()).unwrap();

        let mut cache = m.new_cache(1, ids.len(), 1).unwrap();
        backend
            .prefill(&m, &prompt(&ids[..1967]), cache.as_mut())
            .await
            .unwrap();
        let at_once = backend
            .prefill(&m, &prompt(&ids[1967..]), cache.as_mut())
            .await
            .unwrap();

        let mut cache = m.new_cache(1, ids.len(), 1).unwrap();
        let mut in_pieces = None;
        for piece in ids.chunks(256) {
            in_pieces = Some(
                backend
                    .prefill(&m, &prompt(piece), cache.as_mut())
                    .await
                    .unwrap(),
            );
        }
        let want: Vec<f32> = in_pieces.unwrap().to_vec();
        assert_close(at_once.view(), &want, 1e-2);
    }
}
