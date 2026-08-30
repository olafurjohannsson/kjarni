//! Where a sentence-transformers encoder should truncate, and what it costs.
//!
//! Three files disagree about this model's context length. For `all-MiniLM-L6-v2`
//! the shipped `tokenizer.json` says 128, `sentence_bert_config.json` says 256, and
//! `max_position_embeddings` in `config.json` says 512. sentence-transformers reads
//! the second; Kjarni used to read the third, so on any input past 256 tokens it
//! produced embeddings the reference implementation would never produce.
//!
//! That is not a hypothetical range. The default RAG chunk is 1000 characters,
//! which measures at a median of 286 tokens on ordinary prose, so most chunks
//! anyone indexes land above the reference limit.
//!
//! These tests need `minilm-l6-v2` in the cache and skip without it.

use std::path::{Path, PathBuf};

use kjarni_models::SentenceEncoder;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::traits::Device;

fn model_dir() -> Option<PathBuf> {
    let dir = kjarni_transformers::models::get_default_cache_dir()
        .join("sentence-transformers_all-MiniLM-L6-v2");
    dir.join("model.safetensors").exists().then_some(dir)
}

fn try_at_length(dir: &Path, max_seq_len: Option<usize>) -> anyhow::Result<SentenceEncoder> {
    let config = ModelLoadConfig {
        max_sequence_length: max_seq_len,
        ..Default::default()
    };
    SentenceEncoder::from_pretrained(dir, Device::Cpu, None, Some(config), None)
}

fn at_length(dir: &Path, max_seq_len: Option<usize>) -> SentenceEncoder {
    try_at_length(dir, max_seq_len).expect("load encoder")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-9)
}

/// Varied prose that crosses 256 tokens, with a distinctive fact in the tail.
///
/// Deliberately not repeated filler. A document of one sentence repeated twelve
/// times pools to a degenerate vector that matches nothing, which makes any
/// retrieval comparison over it meaningless. This reads like a real page: several
/// distinct subjects, and the bearing material stated exactly once, at the end,
/// past the 256-token mark.
fn long_document() -> String {
    [
        "The northern facility began operating in the spring of 2019, replacing an \
         older site that had served the region for three decades.",
        "Its throughput is reviewed every quarter by the operations team, who compare \
         the figures against targets agreed at the start of the year.",
        "Weather delays are logged separately so that seasonal storms do not distort \
         the availability numbers reported to the regional board.",
        "Staffing has remained stable, with two shift supervisors and eleven \
         technicians covering a rotating schedule that runs through the weekend.",
        "The training programme was revised after an audit in 2021 recommended more \
         time on electrical isolation procedures before a technician works unsupervised.",
        "Spare parts are held in a bonded store on the eastern side of the site, and \
         stock levels are reconciled against the maintenance log each month.",
        "A shortage of pump seals in 2022 led to a policy of holding six months of \
         consumables rather than the three months previously considered sufficient.",
        "Energy costs rose sharply that year, prompting an assessment of whether the \
         compressors should run overnight when tariffs are lower.",
        "The assessment concluded that overnight running saved less than expected once \
         additional wear on the drive assemblies was accounted for.",
        "Vibration monitoring was introduced on the main drive shafts shortly \
         afterwards, and has since caught two developing faults before failure.",
        "The most recent of these involved a bearing that had begun to run hot under \
         sustained load during the summer months.",
        "The replacement bearing is machined from hafnium alloy.",
    ]
    .join(" ")
}

#[tokio::test]
async fn default_truncation_now_follows_sentence_bert_config() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: minilm-l6-v2 not cached");
        return;
    };
    if !dir.join("sentence_bert_config.json").exists() {
        eprintln!("skipping: sentence_bert_config.json not downloaded for this model");
        return;
    }

    let doc = long_document();
    let default = at_length(&dir, None).encode(&doc).await.unwrap();
    let at_256 = at_length(&dir, Some(256)).encode(&doc).await.unwrap();
    let at_512 = at_length(&dir, Some(512)).encode(&doc).await.unwrap();

    let to_256 = cosine(&default, &at_256);
    let to_512 = cosine(&default, &at_512);
    eprintln!("default vs 256: {to_256:.6}   default vs 512: {to_512:.6}");

    assert!(
        to_256 > 0.9999,
        "default must match the sentence-transformers length of 256, got {to_256:.6}"
    );
    assert!(
        to_512 < 0.9999,
        "256 and 512 must actually differ on this document, or the test proves nothing"
    );
}

#[tokio::test]
async fn truncation_length_changes_embeddings_past_the_limit() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: minilm-l6-v2 not cached");
        return;
    };

    let short = "The replacement bearing is machined from hafnium alloy.";
    let e256 = at_length(&dir, Some(256));
    let e512 = at_length(&dir, Some(512));

    // Below the limit the two are the same computation, so they must agree exactly.
    let a = e256.encode(short).await.unwrap();
    let b = e512.encode(short).await.unwrap();
    let same = cosine(&a, &b);
    assert!(
        same > 0.99999,
        "short input must be identical at both lengths, got {same:.6}"
    );

    // Past the limit they diverge, which is the whole point.
    let doc = long_document();
    let a = e256.encode(&doc).await.unwrap();
    let b = e512.encode(&doc).await.unwrap();
    let differ = cosine(&a, &b);
    eprintln!("long document, 256 vs 512: cosine {differ:.6}");
    assert!(
        differ < 0.999,
        "long input must differ between 256 and 512, got {differ:.6}"
    );
}

/// Which length actually retrieves better, rather than which matches the reference.
///
/// Two queries, deliberately pulling in opposite directions: one targets the fact
/// buried in the tail, which only 512 can see, and one targets the head, where the
/// extra tail can only dilute the pooled vector. Reported, not asserted, because
/// the answer is a judgement about a tradeoff rather than a correctness property.
///
/// Observed on this document with `minilm-l6-v2`:
///
/// ```text
///  tail query: 256 -> 0.1099   512 -> 0.2845   512 wins by +0.1746
///  head query: 256 -> 0.3850   512 -> 0.3286   256 wins by -0.0565
/// ```
///
/// Neither length is simply better. 512 finds what 256 cannot see at all, and pays
/// for it by diluting the pooled vector for everything else. That tradeoff is the
/// argument for chunking rather than for a longer window: chunks that fit under the
/// limit get the tail *and* keep the focus. Matching the reference is the right
/// default because it is predictable, not because it retrieves better.
#[tokio::test]
async fn report_retrieval_tradeoff_between_256_and_512() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: minilm-l6-v2 not cached");
        return;
    };

    let doc = long_document();
    let tail_query = "What material is the replacement bearing made from?";
    let head_query = "How often does the operations team review the schedule?";

    let e256 = at_length(&dir, Some(256));
    let e512 = at_length(&dir, Some(512));

    let d256 = e256.encode(&doc).await.unwrap();
    let d512 = e512.encode(&doc).await.unwrap();

    for (label, query) in [("tail", tail_query), ("head", head_query)] {
        let q = e256.encode(query).await.unwrap();
        let s256 = cosine(&q, &d256);
        let s512 = cosine(&q, &d512);
        eprintln!(
            "{label:>5} query: 256 -> {s256:.4}   512 -> {s512:.4}   {} by {:+.4}",
            if s512 > s256 { "512 wins" } else { "256 wins" },
            s512 - s256
        );
    }
}

/// Asking for more positions than the model has must fail at load.
///
/// It used to succeed. `CpuEmbeddings::forward` clamps the position slice it adds,
/// so a 513-token request on a 512-position model returned a normal-looking 384-dim
/// vector in which the final token carried no positional information whatsoever.
/// Silently wrong beats loudly broken for exactly nobody.
#[tokio::test]
async fn requesting_more_positions_than_the_model_has_is_refused() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: minilm-l6-v2 not cached");
        return;
    };

    // 512 is the height of MiniLM's position table, so this is the last valid value.
    assert!(
        try_at_length(&dir, Some(512)).is_ok(),
        "512 is representable and must still load"
    );

    for over in [513usize, 1024, 8192] {
        let err = try_at_length(&dir, Some(over))
            .err()
            .unwrap_or_else(|| panic!("{over} exceeds the position table but was accepted"));
        let msg = err.to_string();
        assert!(
            msg.contains(&over.to_string()) && msg.contains("512"),
            "error should name both the request and the ceiling, got: {msg}"
        );
    }
}

/// The browser and the native bindings must truncate at the same point.
///
/// They briefly did not. Reading `sentence_bert_config.json` needs a filesystem,
/// which `load_from_bytes` does not have, so native truncated MiniLM at 256 while
/// WASM kept using 512 from `max_position_embeddings`. Same model, same text,
/// different embeddings, and nothing anywhere would have said so.
///
/// `scripts/quantize_model.py` now folds `max_seq_length` into the config it packs
/// into a `.kjq`, and `load_from_bytes` reads it back. This test is the thing that
/// notices if either half of that is dropped.
#[tokio::test]
async fn kjq_and_native_agree_on_a_long_input() {
    use kjarni_transformers::pipeline::EncoderLoader;
    use kjarni_transformers::weights::kjq;

    let Some(dir) = model_dir() else {
        eprintln!("skipping: minilm-l6-v2 not cached");
        return;
    };
    let Some(bytes) = fixture_kjq() else {
        eprintln!("skipping: no .kjq fixture available");
        return;
    };

    let unpacked = kjq::unpack(&bytes).expect("unpack .kjq");
    if !unpacked.config_json.contains("max_seq_length") {
        eprintln!("skipping: fixture predates the packed max_seq_length field");
        return;
    }

    let from_kjq: SentenceEncoder = EncoderLoader::load_from_bytes(
        &unpacked.safetensors,
        &unpacked.config_json,
        unpacked.tokenizer_json.as_bytes(),
        Device::Cpu,
        None,
        None,
    )
    .expect("engine loads the .kjq");

    // Quantisation puts a floor on how close these can be, so an absolute cosine
    // proves nothing on its own: int8 alone costs roughly 0.05. What does prove it
    // is which native length the .kjq lands nearer to. If the packed max_seq_length
    // is being honoured it must sit closer to 256 than to 512, whatever the
    // quantisation offset happens to be.
    let doc = long_document();
    let browser = from_kjq.encode(&doc).await.unwrap();
    let native_256 = at_length(&dir, Some(256)).encode(&doc).await.unwrap();
    let native_512 = at_length(&dir, Some(512)).encode(&doc).await.unwrap();

    let to_256 = cosine(&browser, &native_256);
    let to_512 = cosine(&browser, &native_512);
    eprintln!("kjq vs native: 256 -> {to_256:.6}   512 -> {to_512:.6}");

    assert!(
        to_256 > to_512,
        "the .kjq must truncate where native does (256), but it is nearer 512: \
         {to_256:.6} vs {to_512:.6}"
    );

    // And a short input, below every candidate limit, isolates the quantisation
    // cost on its own so the margin above can be read in context.
    let short = at_length(&dir, None)
        .encode("The replacement bearing is machined from hafnium alloy.")
        .await
        .unwrap();
    let short_kjq = from_kjq
        .encode("The replacement bearing is machined from hafnium alloy.")
        .await
        .unwrap();
    eprintln!(
        "quantisation cost alone (short input): cosine {:.6}",
        cosine(&short, &short_kjq)
    );
}

/// The `.kjq` used by the agreement test, from `KJARNI_KJQ_DIR` or the site checkout.
fn fixture_kjq() -> Option<Vec<u8>> {
    if let Ok(dir) = std::env::var("KJARNI_KJQ_DIR") {
        return std::fs::read(std::path::Path::new(&dir).join("all-MiniLM-L6-v2-q8.kjq")).ok();
    }
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../web.kjarni.ai/src/static/models/all-MiniLM-L6-v2-q8.kjq"),
    )
    .ok()
}
