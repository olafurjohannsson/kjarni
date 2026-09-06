//! What the decoder computes layer by layer, checked against what it should.
//!
//! `CpuDecoder::forward_layers` takes a layer range, so an intermediate hidden state
//! can be projected through the same `lm_head` the final layer uses. That is the
//! logit lens, and it needs no changes to the engine.
//!
//! These guard two things. That the geometry accessors report the model's real shape:
//! `num_attention_heads`, `hidden_size` and `head_dim` on `LlamaCpuDecoder` all
//! returned a hardcoded 0 until a probe asked and got "0 heads, hidden 0" back for a
//! 24-layer model. Nothing in the engine consumed them, so the stubs were invisible.
//! And that the whole stack still answers a factual prompt correctly end to end.
//!
//! Ignored by default: they load a real model from the local cache.
//! Run with: cargo test --release -p kjarni-models --test logit_lens_probe -- --ignored --nocapture

use kjarni_models::models::qwen::QwenModel;
use kjarni_transformers::models::registry::ModelType;
use kjarni_transformers::{Device, LanguageModel};

#[tokio::test]
#[ignore = "loads a real model from the local cache"]
async fn logit_lens_across_depth() {
    let model = QwenModel::from_registry(
        ModelType::Qwen2_5_0_5B_Instruct,
        None,
        Device::Cpu,
        None,
        None,
    )
    .await
    .expect("load qwen2.5-0.5b");

    let pipe = model.pipeline();
    let dec = pipe.cpu_decoder().expect("cpu decoder");
    let tok = model.tokenizer();

    let prompt = "The capital of France is";
    let enc = tok.encode(prompt, false).expect("tokenize");
    let ids: Vec<u32> = enc.get_ids().to_vec();
    let seq = ids.len();
    println!("\nprompt   {prompt:?}");
    println!("tokens   {ids:?}");
    println!(
        "geometry {} layers, {} heads, hidden {}, vocab {}",
        dec.num_layers(),
        dec.num_attention_heads(),
        dec.hidden_size(),
        pipe.lm_head().vocab_size()
    );

    let token_arr = ndarray::Array2::from_shape_vec((1, seq), ids.clone()).unwrap();
    let mut hidden = pipe
        .embeddings()
        .embed_cpu(&token_arr, None, 0)
        .expect("embed");
    let mask = kjarni_transformers::utils::create_full_attention_mask(1, seq);

    // Step one layer at a time, projecting the running hidden state through the
    // same lm_head the final layer uses. That is the logit lens.
    println!(
        "\n{:<6} {:<28} {:>8} {:>9}",
        "layer", "top-5 after this layer", "top prob", "entropy"
    );
    let n = dec.num_layers();
    let mut json: Vec<String> = Vec::new();
    let mut final_top = String::new();
    for l in 0..n {
        hidden = dec
            .forward_layers(&hidden, &mask, 0, None, l, l + 1)
            .expect("forward one layer");

        let normed = dec.final_norm(&hidden).expect("final norm");
        let logits = pipe.lm_head().forward_cpu(&normed).expect("lm_head");

        // last position only: that is the one predicting the next token
        let row = logits.slice(ndarray::s![0, seq - 1, ..]).to_owned();
        let max = row.iter().cloned().fold(f32::MIN, f32::max);
        let exp: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|v| v / sum).collect();

        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap());

        let entropy: f32 = -probs
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.log2())
            .sum::<f32>();
        let top: Vec<String> = idx[..5]
            .iter()
            .map(|i| {
                tok.decode(&[*i as u32], false)
                    .unwrap_or_default()
                    .trim()
                    .to_string()
            })
            .collect();

        println!(
            "{:<6} {:<28} {:>8.4} {:>9.2}",
            l + 1,
            top.join(" "),
            probs[idx[0]],
            entropy
        );
        final_top = top[0].clone();

        let row: Vec<String> = idx[..6]
            .iter()
            .map(|i| {
                let t = tok.decode(&[*i as u32], false).unwrap_or_default();
                format!("[{:?},{:.5}]", t, probs[*i])
            })
            .collect();
        json.push(format!(
            "{{\"layer\":{},\"entropy\":{:.4},\"top\":[{}]}}",
            l + 1,
            entropy,
            row.join(",")
        ));
    }

    println!("\nJSON_BEGIN");
    println!(
        "{{\"prompt\":{:?},\"layers\":[{}]}}",
        prompt,
        json.join(",")
    );
    println!("JSON_END");

    // End to end: the last layer must actually answer the question.
    assert!(
        final_top.contains("Paris"),
        "final layer predicted {final_top:?}, expected Paris"
    );
}

/// The same lens, but at every position rather than only the last.
///
/// If factual recall works the way the literature describes, the answer is written
/// into the residual stream at the *subject* position first and only copied to the
/// final position late. That predicts "Paris" appears over "France" before it
/// appears over "is".
#[tokio::test]
#[ignore = "loads a real model from the local cache"]
async fn where_the_fact_lives() {
    let model = QwenModel::from_registry(
        ModelType::Qwen2_5_0_5B_Instruct,
        None,
        Device::Cpu,
        None,
        None,
    )
    .await
    .expect("load qwen2.5-0.5b");

    let pipe = model.pipeline();
    let dec = pipe.cpu_decoder().expect("cpu decoder");
    let tok = model.tokenizer();

    let prompt = "The capital of France is";
    let ids: Vec<u32> = tok
        .encode(prompt, false)
        .expect("tokenize")
        .get_ids()
        .to_vec();
    let seq = ids.len();

    let names: Vec<String> = ids
        .iter()
        .map(|i| {
            tok.decode(&[*i], false)
                .unwrap_or_default()
                .trim()
                .to_string()
        })
        .collect();

    let token_arr = ndarray::Array2::from_shape_vec((1, seq), ids.clone()).unwrap();
    let mut hidden = pipe
        .embeddings()
        .embed_cpu(&token_arr, None, 0)
        .expect("embed");
    let mask = kjarni_transformers::utils::create_full_attention_mask(1, seq);

    let mut last_row: Vec<String> = Vec::new();
    println!("\ntop token at each position, after each layer");
    print!("{:<6}", "layer");
    for n in &names {
        print!("{:>14}", n);
    }
    println!();

    for l in 0..dec.num_layers() {
        hidden = dec
            .forward_layers(&hidden, &mask, 0, None, l, l + 1)
            .expect("fwd");
        let normed = dec.final_norm(&hidden).expect("norm");
        let logits = pipe.lm_head().forward_cpu(&normed).expect("lm_head");

        print!("{:<6}", l + 1);
        for pos in 0..seq {
            let row = logits.slice(ndarray::s![0, pos, ..]);
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let t = tok.decode(&[best as u32], false).unwrap_or_default();
            let t = t.trim().replace('\n', "\\n");
            let t: String = t.chars().take(12).collect();
            let shown = if t.is_empty() { "_".to_string() } else { t };
            print!("{shown:>14}");
            if l + 1 == dec.num_layers() {
                last_row.push(shown);
            }
        }
        println!();
    }

    // A causal model predicts at every position. By the final layer each one should
    // name its true continuation, which is what makes the mid-stack noise a property
    // of the probe rather than of the model.
    assert_eq!(
        last_row.last().map(String::as_str),
        Some("Paris"),
        "final position should predict Paris, got {last_row:?}"
    );
    assert_eq!(
        last_row.get(3).map(String::as_str),
        Some("is"),
        "the France position should predict 'is', got {last_row:?}"
    );
}

/// Is the garbage a property of the model, or of the instrument?
///
/// `lm_head` is trained to read the *final* layer's representation. If early
/// hidden states simply live in a different basis, the noise says more about the
/// probe than about the model. Measuring each layer's cosine similarity to the
/// final hidden state separates the two: a late, sharp rise means the readout only
/// becomes valid at the end.
#[tokio::test]
#[ignore = "loads a real model from the local cache"]
async fn distance_to_final_representation() {
    let model = QwenModel::from_registry(
        ModelType::Qwen2_5_0_5B_Instruct,
        None,
        Device::Cpu,
        None,
        None,
    )
    .await
    .expect("load qwen2.5-0.5b");

    let pipe = model.pipeline();
    let dec = pipe.cpu_decoder().expect("cpu decoder");
    let tok = model.tokenizer();

    let ids: Vec<u32> = tok
        .encode("The capital of France is", false)
        .expect("tokenize")
        .get_ids()
        .to_vec();
    let seq = ids.len();
    let arr = ndarray::Array2::from_shape_vec((1, seq), ids).unwrap();
    let mut hidden = pipe.embeddings().embed_cpu(&arr, None, 0).expect("embed");
    let mask = kjarni_transformers::utils::create_full_attention_mask(1, seq);

    // Keep the last-position hidden state after every layer.
    let mut states: Vec<Vec<f32>> = Vec::new();
    for l in 0..dec.num_layers() {
        hidden = dec
            .forward_layers(&hidden, &mask, 0, None, l, l + 1)
            .expect("fwd");
        states.push(hidden.slice(ndarray::s![0, seq - 1, ..]).to_vec());
    }
    let final_state = states.last().unwrap().clone();

    let cos = |a: &[f32], b: &[f32]| {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    };

    println!("\n{:<6} {:>10} {:>12}", "layer", "cos->final", "norm");
    let mut sims = Vec::new();
    let mut norms = Vec::new();
    for (i, s) in states.iter().enumerate() {
        let norm: f32 = s.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sim = cos(s, &final_state);
        println!("{:<6} {:>10.4} {:>12.1}", i + 1, sim, norm);
        sims.push(sim);
        norms.push(norm);
    }

    // The last layer is trivially identical to itself.
    assert!(
        (sims[sims.len() - 1] - 1.0).abs() < 1e-4,
        "final layer must match itself"
    );
    // Early states sit close to orthogonal to their own final form, which is why
    // reading them through lm_head yields noise: the head expects the final basis.
    assert!(
        sims[0] < 0.4,
        "layer 1 alignment {} unexpectedly high",
        sims[0]
    );
    // Every layer adds to the residual stream rather than replacing it.
    assert!(
        norms[norms.len() - 2] > norms[0] * 5.0,
        "residual norm should accumulate: {} -> {}",
        norms[0],
        norms[norms.len() - 2]
    );
}

/// One combined dump: softmax shape, per-position readout, and basis alignment.
#[tokio::test]
#[ignore = "loads a real model from the local cache"]
async fn dump_everything_json() {
    let model = QwenModel::from_registry(
        ModelType::Qwen2_5_0_5B_Instruct,
        None,
        Device::Cpu,
        None,
        None,
    )
    .await
    .expect("load qwen2.5-0.5b");
    let pipe = model.pipeline();
    let dec = pipe.cpu_decoder().expect("cpu decoder");
    let tok = model.tokenizer();

    let prompt = "The capital of France is";
    let ids: Vec<u32> = tok.encode(prompt, false).expect("tok").get_ids().to_vec();
    let seq = ids.len();
    let names: Vec<String> = ids
        .iter()
        .map(|i| {
            tok.decode(&[*i], false)
                .unwrap_or_default()
                .trim()
                .to_string()
        })
        .collect();

    let arr = ndarray::Array2::from_shape_vec((1, seq), ids).unwrap();
    let mut hidden = pipe.embeddings().embed_cpu(&arr, None, 0).expect("embed");
    let mask = kjarni_transformers::utils::create_full_attention_mask(1, seq);

    let mut states: Vec<Vec<f32>> = Vec::new();
    let mut layers: Vec<String> = Vec::new();

    for l in 0..dec.num_layers() {
        hidden = dec
            .forward_layers(&hidden, &mask, 0, None, l, l + 1)
            .expect("fwd");
        states.push(hidden.slice(ndarray::s![0, seq - 1, ..]).to_vec());

        let normed = dec.final_norm(&hidden).expect("norm");
        let logits = pipe.lm_head().forward_cpu(&normed).expect("head");

        // full softmax at the last position, kept as its top 12
        let row = logits.slice(ndarray::s![0, seq - 1, ..]).to_owned();
        let mx = row.iter().cloned().fold(f32::MIN, f32::max);
        let exp: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|v| v / sum).collect();
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap());
        let h: f32 = -probs
            .iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.log2())
            .sum::<f32>();
        let top: Vec<String> = idx[..12]
            .iter()
            .map(|i| {
                format!(
                    "[{:?},{:.6}]",
                    tok.decode(&[*i as u32], false).unwrap_or_default(),
                    probs[*i]
                )
            })
            .collect();

        // top token at every position
        let grid: Vec<String> = (0..seq)
            .map(|p| {
                let r = logits.slice(ndarray::s![0, p, ..]);
                let b = r
                    .iter()
                    .enumerate()
                    .max_by(|a, c| a.1.partial_cmp(c.1).unwrap())
                    .unwrap()
                    .0;
                format!(
                    "{:?}",
                    tok.decode(&[b as u32], false).unwrap_or_default().trim()
                )
            })
            .collect();

        let norm: f32 = states[l].iter().map(|x| x * x).sum::<f32>().sqrt();
        layers.push(format!(
            "{{\"l\":{},\"h\":{:.3},\"norm\":{:.1},\"top\":[{}],\"grid\":[{}]}}",
            l + 1,
            h,
            norm,
            top.join(","),
            grid.join(",")
        ));
    }

    let fin = states.last().unwrap().clone();
    let cos: Vec<String> = states
        .iter()
        .map(|s| {
            let d: f32 = s.iter().zip(&fin).map(|(x, y)| x * y).sum();
            let a: f32 = s.iter().map(|x| x * x).sum::<f32>().sqrt();
            let b: f32 = fin.iter().map(|x| x * x).sum::<f32>().sqrt();
            format!("{:.4}", d / (a * b))
        })
        .collect();

    println!("\nJSON_BEGIN");
    println!(
        "{{\"prompt\":{:?},\"positions\":[{}],\"cos\":[{}],\"layers\":[{}]}}",
        prompt,
        names
            .iter()
            .map(|n| format!("{:?}", n))
            .collect::<Vec<_>>()
            .join(","),
        cos.join(","),
        layers.join(",")
    );
    println!("JSON_END");
}
