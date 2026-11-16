//! Phi-3 against transformers, on the tiny checkpoint in `testdata`.

use std::path::PathBuf;

use kjarni_transformers::Device;

use super::PhiModel;
use crate::models::tiny_fixture::assert_matches_transformers;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/phi/testdata")
}

#[test]
fn reads_fused_projections_and_long_rope_from_the_checkpoint() {
    let model = PhiModel::from_pretrained(&dir(), Device::Cpu, None, None, None).unwrap();
    let c = model.config();
    assert_eq!(c.original_max_position_embeddings, Some(64));
    let scaling = c.rope_scaling.as_ref().expect("rope_scaling");
    assert_eq!(scaling.rope_type, "longrope");
    assert_eq!(scaling.short_factor.as_ref().map(Vec::len), Some(8));
}

/// The short prompt stays inside the 64 positions the model was trained on,
/// where LongRoPE uses its short factors; the long one reaches past them.
#[test]
fn logits_and_greedy_decoding_match_transformers() {
    let model = PhiModel::from_pretrained(&dir(), Device::Cpu, None, None, None).unwrap();
    assert_matches_transformers(&model, &dir(), 1e-4);
}
