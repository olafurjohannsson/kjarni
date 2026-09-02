//! The buffered and allocating encoder paths must produce the same numbers.
//!
//! Both exist: `forward` allocates as it goes, `forward_with_buffers` reuses
//! scratch. Which one runs is a performance decision, so they have to be
//! interchangeable, and nothing checked that they were.
//!
//! They were not. RoBERTa's `metadata()` reported `intermediate_size: 0`, so the
//! buffered path allocated a zero-width FFN buffer and sliced past the end of it,
//! while the allocating path was fine. A threshold on token count decided which
//! ran, so the crash only appeared on large batches of a few model families.
//!
//! `#[ignore]`d: needs models from the local cache, and `test-everything.sh`
//! passes `--include-ignored`.

use kjarni_models::SentenceEncoder;
use kjarni_transformers::cpu::encoder::traits::EncoderLanguageModel;
use kjarni_transformers::models::get_default_cache_dir;
use kjarni_transformers::traits::Device;

async fn paths_agree(model: &str, expect_dim: usize) {
    let dir = get_default_cache_dir().join(model);
    assert!(
        dir.join("model.safetensors").exists(),
        "{model} is not in the model cache; this test only runs deliberately"
    );
    let encoder =
        SentenceEncoder::from_pretrained(&dir, Device::Cpu, None, None, None).expect("load");

    let ops = encoder
        .encoder_cpu_ops()
        .expect("cpu encoder ops for a model loaded on Device::Cpu");
    let cpu = ops.encoder();

    // Long enough that the old rule would have chosen the buffered path.
    let docs: Vec<String> = (0..8)
        .map(|i| {
            "The history of Iceland begins with settlement in the ninth century. ".repeat(12)
                + &format!("Document {i}.")
        })
        .collect();
    let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();

    // Tokenize into the padded rectangle the encoder expects.
    let encoded = encoder.encode_batch_texts(&refs).expect("tokenize");
    let seq = encoded
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .expect("non-empty");
    let batch = encoded.len();
    let mut input_ids = ndarray::Array2::<u32>::zeros((batch, seq));
    let mut mask_f32 = ndarray::Array2::<f32>::zeros((batch, seq));
    for (b, e) in encoded.iter().enumerate() {
        let am = e.get_attention_mask();
        for (i, &id) in e.get_ids().iter().enumerate() {
            input_ids[[b, i]] = id;
            mask_f32[[b, i]] = am[i] as f32;
        }
    }

    let embedded = ops.embed_tokens(&input_ids, None, 0).expect("embed");
    let normalized = cpu.embed_norm(&embedded).expect("embed_norm");

    let allocating = cpu
        .forward(&normalized, &mask_f32)
        .expect("allocating path")
        .last_hidden_state;

    let mut buffers = cpu.create_buffers(batch, seq);
    let buffered = cpu
        .forward_with_buffers(&normalized, &mask_f32, &mut buffers)
        .expect("buffered path")
        .last_hidden_state;

    assert_eq!(allocating.dim(), buffered.dim(), "shapes differ");
    assert_eq!(allocating.dim().2, expect_dim, "unexpected hidden size");

    // Relative, not absolute. The two paths reach the same arithmetic through
    // different GEMM kernels (`matmul` dispatches differently from
    // `matmul_noalloc`), so they sum in a different order and the last bits move.
    // Hidden states run to several units in magnitude, so an absolute tolerance
    // tuned for unit-length embeddings rejects a correct result.
    let scale = allocating
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let worst = allocating
        .iter()
        .zip(buffered.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let relative = worst / scale;

    eprintln!(
        "  {model:<40} worst {worst:.3e} absolute, {relative:.3e} relative to a scale of {scale:.2}"
    );
    assert!(
        relative < 1e-4,
        "{model}: the two encoder paths disagree by {relative:.3e} relative"
    );
}

#[tokio::test]
#[ignore = "needs minilm in the local model cache"]
async fn minilm_paths_agree() {
    paths_agree("sentence-transformers_all-MiniLM-L6-v2", 384).await;
}

#[tokio::test]
#[ignore = "needs roberta-base in the local model cache"]
async fn roberta_paths_agree() {
    paths_agree("SamLowe_roberta-base-go_emotions", 768).await;
}

#[tokio::test]
#[ignore = "needs mpnet-base-v2 in the local model cache"]
async fn mpnet_paths_agree() {
    paths_agree("sentence-transformers_all-mpnet-base-v2", 768).await;
}
