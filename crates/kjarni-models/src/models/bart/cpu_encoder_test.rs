use crate::models::bart::config::BartConfig;
use crate::models::bart::cpu_encoder::BartCpuEncoder;
use anyhow::Result;
use kjarni_transformers::{
    encoder::{encoder_layer::EncoderLayer, prelude::*},
    encoder_decoder::traits::CpuCrossDecoder,
    feedforward::{FeedForward, StdFeedForward},
    normalization::LayerNorm,
    traits::{Device, TransformerModel},
    utils::linear_algebra::{apply_attention_mask, matmul_4d},
    weights_old::ModelWeights,
};
use ndarray::{Array2, Array3, s};
use std::sync::Arc;
use std::path::Path;


const DISTILBART_PATH: &str =
    "/home/olafurj/.cache/kjarni/olafuraron_distilbart-cnn-12-6/";

// Golden values from Python (HuggingFace transformers)
// Input: "Rust is a multi-paradigm, general-purpose programming language..."
// input_ids: [0, 46541, 16, 10, 3228, 12, 5489, 625, 35045, 6] (10 tokens)
mod golden {
    pub const LAYER0_INPUT: [f32; 10] = [
        -0.012427182,
        -0.1763359,
        0.028129267,
        -0.010629091,
        0.015348487,
        0.00571412,
        0.020377142,
        -0.07212893,
        -0.012256589,
        -0.07150629,
    ];

    pub const ATTN_OUT: [f32; 10] = [
        0.24579018,
        0.14331692,
        -0.022872504,
        -0.1878857,
        -0.064782947,
        0.071185566,
        0.40866822,
        0.07419811,
        0.073571876,
        0.255808,
    ];

    pub const POST_ATTN_LN: [f32; 10] = [
        0.6663479,
        -0.014733588,
        0.11194973,
        -0.43422914,
        -0.11722367,
        0.098294236,
        1.1904591,
        0.029281814,
        0.043691266,
        0.6656334,
    ];

    pub const FC1_OUT: [f32; 10] = [
        -0.84858227,
        -2.0408635,
        -1.4097499,
        -3.1832719,
        0.22090366,
        -0.10113987,
        -1.2940383,
        -1.9374337,
        -2.2026572,
        -0.5642203,
    ];

    pub const FC1_GELU: [f32; 10] = [
        -0.1680675,
        -0.042107336,
        -0.11180281,
        -0.0023179317,
        0.12976241,
        -0.046495996,
        -0.12659079,
        -0.051043857,
        -0.030417403,
        -0.16153744,
    ];

    pub const FFN_OUT: [f32; 10] = [
        -0.062499546,
        -1.6436818,
        -0.4470262,
        -0.64270484,
        2.4917088,
        1.5846581,
        0.26653588,
        0.5026232,
        0.7829591,
        -1.7354881,
    ];

    pub const LAYER0_OUTPUT: [f32; 10] = [
        -0.0076609100,
        -0.053799018,
        -0.014809675,
        -0.009414879,
        0.024039175,
        0.023340283,
        0.007110475,
        -0.00935083,
        0.035594635,
        -0.03405979,
    ];

    pub const SCALED_SCORES: [f32; 5] = [-2.4734669, 0.54202443, 0.2577647, 3.2883463, 0.19397698];

    pub const SOFTMAX: [f32; 5] = [
        0.002194964,
        0.044775307,
        0.033696614,
        0.6978323,
        0.031614296,
    ];

    pub const CONTEXT: [f32; 5] = [
        0.10272437,
        -0.15802623,
        -0.26037022,
        0.06913933,
        -0.010728856,
    ];
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, name: &str) {
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < tol,
            "{} mismatch at {}: expected {}, got {} (diff: {})",
            name,
            i,
            e,
            a,
            (a - e).abs()
        );
    }
}

fn setup_encoder() -> Result<(BartCpuEncoder, Array3<f32>)> {
    let path = Path::new(DISTILBART_PATH);
    if !path.exists() {
        anyhow::bail!("Weights not found at {}", DISTILBART_PATH);
    }

    let weights = ModelWeights::new(path)?;
    let config_json = std::fs::read_to_string(path.join("config.json"))?;
    let config: Arc<BartConfig> = Arc::new(serde_json::from_str(&config_json)?);
    let encoder = BartCpuEncoder::new(&weights, config)?;

    // First 10 tokens of the test input
    let input_ids_vec = vec![0u32, 46541, 16, 10, 3228, 12, 5489, 625, 35045, 6];
    let input_ids = Array2::from_shape_vec((1, 10), input_ids_vec)?;
    let hidden = encoder.embed_and_normalize(&input_ids, None);

    Ok((encoder, hidden))
}

#[tokio::test]
async fn test_layer0_input_embeddings() -> Result<()> {
    let (_, hidden) = setup_encoder()?;
    let actual: Vec<f32> = hidden.slice(s![0, 0, 0..10]).to_vec();
    assert_close(&actual, &golden::LAYER0_INPUT, 1e-5, "Layer0 Input");
    println!("✅ Layer 0 Input (embeddings + layernorm) matches");
    Ok(())
}

#[tokio::test]
async fn test_layer0_self_attention() -> Result<()> {
    let (encoder, hidden) = setup_encoder()?;
    let mask = Array2::<f32>::ones((1, 10));

    let attn_out = encoder.layers[0].self_attn.forward(&hidden, &mask, None)?;
    let actual: Vec<f32> = attn_out.slice(s![0, 0, 0..10]).to_vec();

    assert_close(&actual, &golden::ATTN_OUT, 1e-4, "Self-Attention Output");
    println!("✅ Self-Attention output matches");
    Ok(())
}

#[tokio::test]
async fn test_layer0_post_attn_layernorm() -> Result<()> {
    let (encoder, hidden) = setup_encoder()?;
    let mask = Array2::<f32>::ones((1, 10));

    let attn_out = encoder.layers[0].self_attn.forward(&hidden, &mask, None)?;
    let post_attn = encoder.layers[0]
        .self_attn_layer_norm
        .forward_3d(&(&hidden + &attn_out));
    let actual: Vec<f32> = post_attn.slice(s![0, 0, 0..10]).to_vec();

    assert_close(
        &actual,
        &golden::POST_ATTN_LN,
        1e-4,
        "Post-Attention LayerNorm",
    );
    println!("✅ Post-Attention LayerNorm matches");
    Ok(())
}

#[tokio::test]
async fn test_layer0_fc1_only() -> Result<()> {
    let (encoder, hidden) = setup_encoder()?;
    let mask = Array2::<f32>::ones((1, 10));

    let attn_out = encoder.layers[0].self_attn.forward(&hidden, &mask, None)?;
    let post_attn = encoder.layers[0]
        .self_attn_layer_norm
        .forward_3d(&(&hidden + &attn_out));

    let fc1_out = match &encoder.layers[0].feedforward {
        FeedForward::Standard(ff) => ff.fc1(&post_attn)?,
        FeedForward::Legacy(ff) => ff.fc1(&post_attn)?,
        _ => anyhow::bail!("Unexpected feedforward type"),
    };

    let actual: Vec<f32> = fc1_out.slice(s![0, 0, 0..10]).to_vec();
    assert_close(&actual, &golden::FC1_OUT, 1e-3, "FC1 Output");
    println!("✅ FC1 output matches");
    Ok(())
}

#[tokio::test]
async fn test_layer0_fc1_gelu() -> Result<()> {
    let (encoder, hidden) = setup_encoder()?;
    let mask = Array2::<f32>::ones((1, 10));

    let attn_out = encoder.layers[0].self_attn.forward(&hidden, &mask, None)?;
    let post_attn = encoder.layers[0]
        .self_attn_layer_norm
        .forward_3d(&(&hidden + &attn_out));

    let mut fc1_out = match &encoder.layers[0].feedforward {
        FeedForward::Standard(ff) => ff.fc1(&post_attn)?,
        FeedForward::Legacy(ff) => ff.fc1(&post_attn)?,
        _ => anyhow::bail!("Unexpected feedforward type"),
    };

    match &encoder.layers[0].feedforward {
        FeedForward::Standard(ff) => ff.apply_activation(&mut fc1_out),
        FeedForward::Legacy(ff) => ff.apply_activation(&mut fc1_out),
        _ => anyhow::bail!("Unexpected feedforward type"),
    };

    let actual: Vec<f32> = fc1_out.slice(s![0, 0, 0..10]).to_vec();
    assert_close(&actual, &golden::FC1_GELU, 1e-3, "FC1+GELU Output");
    println!("✅ FC1+GELU output matches");
    Ok(())
}

#[tokio::test]
async fn test_layer0_ffn() -> Result<()> {
    let (encoder, hidden) = setup_encoder()?;
    let mask = Array2::<f32>::ones((1, 10));

    let attn_out = encoder.layers[0].self_attn.forward(&hidden, &mask, None)?;
    let post_attn = encoder.layers[0]
        .self_attn_layer_norm
        .forward_3d(&(&hidden + &attn_out));

    let ffn_out = encoder.layers[0].feedforward.forward(&post_attn)?;
    let actual: Vec<f32> = ffn_out.slice(s![0, 0, 0..10]).to_vec();

    assert_close(&actual, &golden::FFN_OUT, 1e-3, "FFN Output");
    println!("✅ FFN output matches");
    Ok(())
}

#[tokio::test]
async fn test_layer0_full_output() -> Result<()> {
    let (encoder, hidden) = setup_encoder()?;
    let mask = Array2::<f32>::ones((1, 10));

    let layer0_out = encoder.layers[0].forward(hidden, &mask, None, false)?;
    let actual: Vec<f32> = layer0_out.slice(s![0, 0, 0..10]).to_vec();

    assert_close(&actual, &golden::LAYER0_OUTPUT, 1e-4, "Layer 0 Output");
    println!("✅ Layer 0 full output matches");
    Ok(())
}

#[tokio::test]
async fn test_self_attention_scores_and_softmax() -> Result<()> {
    let (encoder, hidden) = setup_encoder()?;
    let self_attn = &encoder.layers[0].self_attn;
    let mask = Array2::<f32>::ones((1, 10));

    let (batch, seq_len, _) = hidden.dim();
    let hidden_dim = self_attn.num_heads * self_attn.head_dim;

    // Project & reshape
    let hidden_2d = hidden
        .view()
        .into_shape_with_order((batch * seq_len, hidden_dim))?;
    let q = self_attn.q_proj.matmul(&hidden_2d.view());
    let k = self_attn.k_proj.matmul(&hidden_2d.view());
    let v = self_attn.v_proj.matmul(&hidden_2d.view());

    let q_heads = q
        .into_shape_with_order((batch, seq_len, self_attn.num_heads, self_attn.head_dim))?
        .permuted_axes([0, 2, 1, 3])
        .to_owned();
    let k_heads_t = k
        .into_shape_with_order((batch, seq_len, self_attn.num_heads, self_attn.head_dim))?
        .permuted_axes([0, 2, 3, 1])
        .to_owned();
    let v_heads = v
        .into_shape_with_order((batch, seq_len, self_attn.num_heads, self_attn.head_dim))?
        .permuted_axes([0, 2, 1, 3])
        .to_owned();

    // Scores
    let mut scores = matmul_4d(&q_heads, &k_heads_t);
    scores.mapv_inplace(|x| x * self_attn.scale_factor);

    // Check scaled scores
    let scaled_actual: Vec<f32> = scores.slice(s![0, 0, 0, 0..5]).to_vec();
    assert_close(
        &scaled_actual,
        &golden::SCALED_SCORES,
        1e-4,
        "Scaled Scores",
    );
    println!("✅ Scaled scores match");

    // Apply mask & softmax
    scores = apply_attention_mask(scores, &mask);
    self_attn.softmax_inplace(&mut scores);

    let softmax_actual: Vec<f32> = scores.slice(s![0, 0, 0, 0..5]).to_vec();
    assert_close(&softmax_actual, &golden::SOFTMAX, 1e-4, "Softmax");
    println!("✅ Softmax matches");

    // Context
    let context = matmul_4d(&scores, &v_heads);
    let context_actual: Vec<f32> = context.slice(s![0, 0, 0, 0..5]).to_vec();
    assert_close(&context_actual, &golden::CONTEXT, 1e-4, "Context");
    println!("✅ Context matches");

    Ok(())
}
