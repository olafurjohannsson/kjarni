//! Checks a decoder against a tiny random checkpoint and what transformers
//! computed with it. Each `testdata` folder holds the checkpoint, the
//! `make.py` that wrote it, and `expected.json`: a short prompt's logits at
//! every position, a long prompt's last logits, and greedy continuations of
//! both.

use std::path::Path;

use kjarni_transformers::{cache::Cache, decoder::prelude::*, models::LanguageModel};
use ndarray::Array2;

pub(crate) struct Expected(serde_json::Value);

impl Expected {
    pub(crate) fn read(dir: &Path) -> Self {
        let text = std::fs::read_to_string(dir.join("expected.json")).unwrap();
        Self(serde_json::from_str(&text).unwrap())
    }

    pub(crate) fn ids(&self, key: &str) -> Vec<u32> {
        self.0[key]
            .as_array()
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
}

/// Logits for `ids`, continuing whatever `cache` holds.
pub(crate) fn logits<M>(model: &M, ids: &[u32], cache: &mut dyn Cache) -> Array2<f32>
where
    M: CpuDecoderOps + LanguageModel,
{
    let offset = cache.get_seq_length();
    let tokens = Array2::from_shape_vec((1, ids.len()), ids.to_vec()).unwrap();
    let hidden = model.embed(&tokens, offset).unwrap();
    let mask = model.get_attention_mask(ids.len(), offset).unwrap();
    let out = model
        .decoder()
        .forward(&hidden, &mask, offset, Some(cache))
        .unwrap();
    model
        .project_to_logits(&out)
        .unwrap()
        .index_axis_move(ndarray::Axis(0), 0)
}

fn argmax(row: ndarray::ArrayView1<f32>) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

/// Greedy decoding, one token at a time through the cache.
pub(crate) fn greedy<M>(model: &M, prompt: &[u32], n: usize) -> Vec<u32>
where
    M: CpuDecoderOps + LanguageModel,
{
    let mut cache = model.new_cache(1, 256, 1).unwrap();
    let mut next = argmax(logits(model, prompt, cache.as_mut()).row(prompt.len() - 1));
    let mut out = vec![next];
    while out.len() < n {
        next = argmax(logits(model, &[next], cache.as_mut()).row(0));
        out.push(next);
    }
    out
}

/// Every logit of the short prompt, the long prompt's last logits, and both
/// greedy continuations, against transformers.
pub(crate) fn assert_matches_transformers<M>(model: &M, dir: &Path, tol: f32)
where
    M: CpuDecoderOps + LanguageModel,
{
    let e = Expected::read(dir);

    let short = e.ids("short_ids");
    let mut cache = model.new_cache(1, 256, 1).unwrap();
    let got = logits(model, &short, cache.as_mut());
    for (pos, row) in e.0["short_logits"].as_array().unwrap().iter().enumerate() {
        for (i, w) in Expected::floats(row).iter().enumerate() {
            let g = got[[pos, i]];
            assert!(
                (g - w).abs() < tol,
                "short prompt, position {pos}, logit {i}: {g} vs {w}"
            );
        }
    }

    let long = e.ids("long_ids");
    let mut cache = model.new_cache(1, 256, 1).unwrap();
    let got = logits(model, &long, cache.as_mut());
    let last = got.row(long.len() - 1);
    for (i, w) in Expected::floats(&e.0["long_last_logits"])
        .iter()
        .enumerate()
    {
        assert!(
            (last[i] - w).abs() < tol,
            "long prompt, logit {i}: {} vs {w}",
            last[i]
        );
    }

    assert_eq!(
        greedy(model, &short, 12),
        e.ids("short_greedy"),
        "short prompt, greedy"
    );
    assert_eq!(
        greedy(model, &long, 12),
        e.ids("long_greedy"),
        "long prompt, greedy"
    );
}
