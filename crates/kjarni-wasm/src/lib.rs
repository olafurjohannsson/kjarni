mod wasm_simd;
mod tokenizer;
mod weights;

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Array4, Axis, Zip, s};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Response, Window, WorkerGlobalScope};

use tokenizer::WordPieceTokenizer;
use weights::ModelWeights;

// Model structures

pub struct Model {
    word_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,
    token_type_embeddings: Array2<f32>,
    layers: Vec<BertLayer>,
    layer_norm_final: LayerNorm,
    config: Config,
    tokenizer: WordPieceTokenizer,
}

struct BertLayer {
    attention: MultiHeadAttention,
    intermediate: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

struct MultiHeadAttention {
    query_weight: Array2<f32>,   // (out, in) = (384, 384)
    query_bias: Array1<f32>,
    key_weight: Array2<f32>,
    key_bias: Array1<f32>,
    value_weight: Array2<f32>,
    value_bias: Array1<f32>,
    output_weight: Array2<f32>,
    output_bias: Array1<f32>,
    num_heads: usize,
    head_dim: usize,
    scale_factor: f32,
}

struct FeedForward {
    dense1_weight: Array2<f32>,  // (1536, 384)
    dense1_bias: Array1<f32>,
    dense2_weight: Array2<f32>,  // (384, 1536)
    dense2_bias: Array1<f32>,
}

struct LayerNorm {
    weight: Array1<f32>,
    bias: Array1<f32>,
    eps: f32,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f32,
    pub hidden_act: String,
    pub model_type: String,
}

const SQRT_2_OVER_PI: f32 = 0.7978845608_f32;
const GELU_COEFF: f32 = 0.044715_f32;

// Model loading

impl Model {
    pub fn from_weights(
        weights: ModelWeights,
        tokenizer: WordPieceTokenizer,
        config: Config,
    ) -> Result<Self> {
        let word_embeddings = weights.get_array2("embeddings.word_embeddings.weight")?;
        let position_embeddings = weights.get_array2("embeddings.position_embeddings.weight")?;
        let token_type_embeddings =
            weights.get_array2("embeddings.token_type_embeddings.weight")?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{}", i);

            let attention = MultiHeadAttention {
                query_weight: weights
                    .get_array2(&format!("{}.attention.self.query.weight", prefix))?,
                query_bias: weights.get_array1(&format!("{}.attention.self.query.bias", prefix))?,
                key_weight: weights
                    .get_array2(&format!("{}.attention.self.key.weight", prefix))?,
                key_bias: weights.get_array1(&format!("{}.attention.self.key.bias", prefix))?,
                value_weight: weights
                    .get_array2(&format!("{}.attention.self.value.weight", prefix))?,
                value_bias: weights.get_array1(&format!("{}.attention.self.value.bias", prefix))?,
                output_weight: weights
                    .get_array2(&format!("{}.attention.output.dense.weight", prefix))?,
                output_bias: weights
                    .get_array1(&format!("{}.attention.output.dense.bias", prefix))?,
                num_heads: config.num_attention_heads,
                head_dim: config.hidden_size / config.num_attention_heads,
                scale_factor: 1.0
                    / ((config.hidden_size / config.num_attention_heads) as f32).sqrt(),
            };

            let intermediate = FeedForward {
                dense1_weight: weights
                    .get_array2(&format!("{}.intermediate.dense.weight", prefix))?,
                dense1_bias: weights.get_array1(&format!("{}.intermediate.dense.bias", prefix))?,
                dense2_weight: weights
                    .get_array2(&format!("{}.output.dense.weight", prefix))?,
                dense2_bias: weights.get_array1(&format!("{}.output.dense.bias", prefix))?,
            };

            let layer_norm1 = LayerNorm {
                weight: weights
                    .get_array1(&format!("{}.attention.output.LayerNorm.weight", prefix))?,
                bias: weights.get_array1(&format!("{}.attention.output.LayerNorm.bias", prefix))?,
                eps: config.layer_norm_eps,
            };

            let layer_norm2 = LayerNorm {
                weight: weights.get_array1(&format!("{}.output.LayerNorm.weight", prefix))?,
                bias: weights.get_array1(&format!("{}.output.LayerNorm.bias", prefix))?,
                eps: config.layer_norm_eps,
            };

            layers.push(BertLayer {
                attention,
                intermediate,
                layer_norm1,
                layer_norm2,
            });
        }

        let layer_norm_final = LayerNorm {
            weight: weights.get_array1("embeddings.LayerNorm.weight")?,
            bias: weights.get_array1("embeddings.LayerNorm.bias")?,
            eps: config.layer_norm_eps,
        };

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layers,
            layer_norm_final,
            config,
            tokenizer,
        })
    }

    // Encoding API

    pub fn encode(&self, texts: Vec<&str>, normalize_embeddings: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize WITHOUT padding — each encoding is only as long as needed
        let mut encodings = Vec::with_capacity(texts.len());
        for text in &texts {
            encodings.push(self.tokenizer.encode(text, 512)?);
        }

        let batch_size = encodings.len();

        // Pad to the longest sequence in this batch, NOT to 512
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap();

        let mut input_ids = Array2::<f32>::zeros((batch_size, max_len));
        let mut attention_mask = Array2::<f32>::zeros((batch_size, max_len));

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            // Only fill up to this encoding's actual length
            for j in 0..ids.len() {
                input_ids[[i, j]] = ids[j] as f32;
                attention_mask[[i, j]] = mask[j] as f32;
            }
            // Remaining positions stay as 0 (padding)
        }

        let embeddings = self.forward(&input_ids, &attention_mask)?;

        let final_embeddings = if normalize_embeddings {
            let norms = embeddings.mapv(|x| x.powi(2)).sum_axis(Axis(1));
            let norms = norms.mapv(|x| x.sqrt().max(1e-12));
            let norms_expanded = norms.insert_axis(Axis(1));
            embeddings / &norms_expanded
        } else {
            embeddings
        };

        Ok(final_embeddings.outer_iter().map(|row| row.to_vec()).collect())
    }

    /// Forward pass
    fn forward(
        &self,
        input_ids: &Array2<f32>,
        attention_mask: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        let (batch_size, seq_len) = input_ids.dim();

        // Embedding lookup
        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, self.config.hidden_size));
        for i in 0..batch_size {
            for j in 0..seq_len {
                let token_id = input_ids[[i, j]] as usize;
                let word_emb = self.word_embeddings.row(token_id);
                hidden.slice_mut(s![i, j, ..]).assign(&word_emb);
            }
        }

        let pos_embeddings = self.position_embeddings.slice(s![0..seq_len, ..]);
        hidden += &pos_embeddings;

        let type_embeddings = self.token_type_embeddings.row(0);
        hidden += &type_embeddings;

        let mut hidden = apply_layer_norm_3d(&hidden, &self.layer_norm_final);
        for layer in &self.layers {
            hidden = layer.forward(hidden, attention_mask)?;
        }
        mean_pool(&hidden, attention_mask)
    }
    
    /// Forward pass returning [CLS] token hidden state (for reranking/classification)
    fn forward_cls(
        &self,
        input_ids: &Array2<f32>,
        attention_mask: &Array2<f32>,
        token_type_ids: Option<&Array2<f32>>,
    ) -> Result<Array2<f32>> {
        let (batch_size, seq_len) = input_ids.dim();

        let mut hidden = Array3::<f32>::zeros((batch_size, seq_len, self.config.hidden_size));
        for i in 0..batch_size {
            for j in 0..seq_len {
                let token_id = input_ids[[i, j]] as usize;
                hidden.slice_mut(s![i, j, ..]).assign(&self.word_embeddings.row(token_id));
            }
        }

        let pos_embeddings = self.position_embeddings.slice(s![0..seq_len, ..]);
        hidden += &pos_embeddings;

        // Use token_type_ids if provided (for pair inputs), otherwise default to 0
        if let Some(type_ids) = token_type_ids {
            for i in 0..batch_size {
                for j in 0..seq_len {
                    let type_id = type_ids[[i, j]] as usize;
                    let type_emb = self.token_type_embeddings.row(type_id);
                    let mut slice = hidden.slice_mut(s![i, j, ..]);
                    slice += &type_emb;
                }
            }
        } else {
            let type_embeddings = self.token_type_embeddings.row(0);
            hidden += &type_embeddings;
        }

        let mut hidden = apply_layer_norm_3d(&hidden, &self.layer_norm_final);
        for layer in &self.layers {
            hidden = layer.forward(hidden, attention_mask)?;
        }

        // Extract [CLS] token (position 0) instead of mean pooling
        Ok(hidden.slice(s![.., 0, ..]).to_owned())
    }
}

// Layer forward passes

impl BertLayer {
    fn forward(&self, input: Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array3<f32>> {
        let mut attention_out = self.attention.forward(&input, attention_mask)?;
        attention_out += &input;
        let attention_out = apply_layer_norm_3d(&attention_out, &self.layer_norm1);

        let mut ff_out = self.intermediate.forward(&attention_out)?;
        ff_out += &attention_out;
        Ok(apply_layer_norm_3d(&ff_out, &self.layer_norm2))
    }
}

impl MultiHeadAttention {
    fn forward(&self, hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array3<f32>> {
        let batch_size = hidden.shape()[0];
        let seq_len = hidden.shape()[1];

        let mut q = matmul_3d_2d(hidden, &self.query_weight);
        q += &self.query_bias;

        let mut k = matmul_3d_2d(hidden, &self.key_weight);
        k += &self.key_bias;

        let mut v = matmul_3d_2d(hidden, &self.value_weight);
        v += &self.value_bias;

        let q = q
            .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let k = k
            .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let v = v
            .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]);

        let mut scores = matmul_4d(&q, &k.permuted_axes([0, 1, 3, 2]));
        scores *= self.scale_factor;

        let scores = apply_attention_mask(scores, attention_mask);
        let weights = softmax(&scores);
        let context = matmul_4d(&weights, &v);

        let context = context.permuted_axes([0, 2, 1, 3]);
        let context = context
            .as_standard_layout()
            .into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        let mut output = matmul_3d_2d(&context, &self.output_weight);
        output += &self.output_bias;

        Ok(output)
    }
}

impl FeedForward {
    fn forward(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        let mut intermediate = matmul_3d_2d(hidden, &self.dense1_weight);
        intermediate += &self.dense1_bias;
        gelu(&mut intermediate);

        let mut output = matmul_3d_2d(&intermediate, &self.dense2_weight);
        output += &self.dense2_bias;
        Ok(output)
    }
}

/// C = A @ W^T where A is [batch, m, k] and W is [n, k] (out_features, in_features).
#[inline(always)]
fn matmul_3d_2d(a: &Array3<f32>, w: &Array2<f32>) -> Array3<f32> {
    let (batch, m, k) = a.dim();
    let n = w.shape()[0];
    assert_eq!(k, w.shape()[1], "Dimension mismatch: a[k]={} != w[in]={}", k, w.shape()[1]);

    let a_cont = a.as_standard_layout();
    let w_cont = w.as_standard_layout();
    let mut c = Array3::<f32>::zeros((batch, m, n));

    let a_slice = a_cont.as_slice().expect("A contiguous");
    let w_slice = w_cont.as_slice().expect("W contiguous");
    let c_slice = c.as_slice_mut().expect("C contiguous");

    for batch_idx in 0..batch {
        let a_batch = &a_slice[batch_idx * m * k..(batch_idx + 1) * m * k];
        let c_batch = &mut c_slice[batch_idx * m * n..(batch_idx + 1) * m * n];
        unsafe {
            wasm_simd::wasm_matmul_2d(c_batch, a_batch, w_slice, m, n, k);
        }
    }

    c
}

/// Standard 4D matmul: C = A @ B
/// A: [batch, heads, m, k], B: [batch, heads, k, n], C: [batch, heads, m, n]
#[inline(always)]
fn matmul_4d(a: &Array4<f32>, b: &Array4<f32>) -> Array4<f32> {
    assert_eq!(a.shape()[0], b.shape()[0], "Batch mismatch");
    assert_eq!(a.shape()[1], b.shape()[1], "Heads mismatch");
    assert_eq!(a.shape()[3], b.shape()[2], "Dim mismatch");

    let (batch, heads, seq1, dim) = a.dim();
    let seq2 = b.shape()[3];

    let a_cont = a.as_standard_layout();
    let b_cont = b.as_standard_layout();

    let total = batch * heads;
    let mut output = Array3::<f32>::zeros((total, seq1, seq2));

    let a_slice = a_cont.as_slice().expect("A contiguous");
    let b_slice = b_cont.as_slice().expect("B contiguous");
    let out_slice = output.as_slice_mut().expect("output contiguous");

    for i in 0..total {
        let a_mat = &a_slice[i * seq1 * dim..(i + 1) * seq1 * dim];
        let b_mat = &b_slice[i * dim * seq2..(i + 1) * dim * seq2];
        let c_mat = &mut out_slice[i * seq1 * seq2..(i + 1) * seq1 * seq2];
        unsafe {
            wasm_simd::wasm_matmul_2d_nn(c_mat, a_mat, b_mat, seq1, dim, seq2);
        }
    }

    output
        .into_shape_with_order((batch, heads, seq1, seq2))
        .unwrap()
}

#[inline(always)]
fn softmax(scores: &Array4<f32>) -> Array4<f32> {
    let max_vals = scores.fold_axis(Axis(3), f32::NEG_INFINITY, |&acc, &x| acc.max(x));
    let max_expanded = max_vals.insert_axis(Axis(3));

    let mut result = scores - &max_expanded;
    result.mapv_inplace(f32::exp);

    let sum_exp = result.sum_axis(Axis(3)).insert_axis(Axis(3));
    result /= &sum_exp;
    result
}

#[inline(always)]
fn apply_attention_mask(mut scores: Array4<f32>, mask: &Array2<f32>) -> Array4<f32> {
    let mask_expanded = mask
        .clone()
        .insert_axis(Axis(1))
        .insert_axis(Axis(2));

    if let Some(broadcast_mask) = mask_expanded.broadcast(scores.dim()) {
        Zip::from(&mut scores)
            .and(&broadcast_mask)
            .for_each(|s, &m| {
                if m == 0.0 {
                    *s = f32::NEG_INFINITY;
                }
            });
    }
    scores
}

#[inline(always)]
fn gelu(x: &mut Array3<f32>) {
    x.mapv_inplace(|val| {
        let val_cubed = val * val * val;
        let inner = SQRT_2_OVER_PI * (val + GELU_COEFF * val_cubed);
        val * 0.5 * (1.0 + inner.tanh())
    });
}

#[inline(always)]
fn apply_layer_norm_3d(hidden: &Array3<f32>, ln: &LayerNorm) -> Array3<f32> {
    let mean = hidden.mean_axis(Axis(2)).unwrap();
    let var = hidden.var_axis(Axis(2), 0.0);
    let mean_expanded = mean.insert_axis(Axis(2));
    let var_expanded = var.insert_axis(Axis(2));

    let inv_std = (&var_expanded + ln.eps).mapv(|x| 1.0 / x.sqrt());
    (hidden - &mean_expanded) * &inv_std * &ln.weight + &ln.bias
}

fn mean_pool(hidden: &Array3<f32>, attention_mask: &Array2<f32>) -> Result<Array2<f32>> {
    let mask_expanded = attention_mask.clone().insert_axis(Axis(2));
    let masked_hidden = hidden * &mask_expanded;
    let sum = masked_hidden.sum_axis(Axis(1));
    let count = attention_mask
        .sum_axis(Axis(1))
        .mapv(|x| x.max(1.0))
        .insert_axis(Axis(1));

    Ok(sum / &count)
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let dot: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b + 1e-8)
}

// WASM bindings

use kjarni_rag::{SearchIndex, TextSplitter, SplitterConfig};

#[wasm_bindgen]
pub struct WasmIndexBuilder {
    model: Model,
    index: SearchIndex,
    splitter: TextSplitter,
}

#[wasm_bindgen]
impl WasmIndexBuilder {
    #[wasm_bindgen]
    pub fn new(model_data: &[u8]) -> Result<WasmIndexBuilder, JsValue> {
        let loaded = ModelWeights::from_quantized_bytes(model_data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let config = loaded.weights.config.clone();
        let tokenizer = WordPieceTokenizer::from_json_str(&loaded.tokenizer_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let model = Model::from_weights(loaded.weights, tokenizer, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(WasmIndexBuilder {
            model,
            index: SearchIndex::with_dimension(384),
            splitter: TextSplitter::new(SplitterConfig::default()),
        })
    }

    /// Add a file's content to the index. Splits internally.
    /// Returns number of chunks added.
    #[wasm_bindgen]
    pub fn add_file(&mut self, text: &str, source_path: &str) -> Result<usize, JsValue> {
        let chunks = self.splitter.split(text);
        let n = chunks.len();

        for (i, chunk) in chunks.iter().enumerate() {
            let embedding = self.model
                .encode(vec![chunk.as_str()], true)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            let mut meta = std::collections::HashMap::new();
            meta.insert("source".to_string(), source_path.to_string());
            meta.insert("chunk_index".to_string(), i.to_string());

            self.index
                .add_document(chunk.clone(), embedding[0].clone(), Some(meta))
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }

        Ok(n)
    }

    /// Serialize the built index to bytes (for saving to disk)
    #[wasm_bindgen]
    pub fn finish(&self) -> Result<Vec<u8>, JsValue> {
        let mut buf = Vec::new();
        self.index
            .save_binary(&mut buf)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(buf)
    }

    /// Number of documents indexed so far
    #[wasm_bindgen]
    pub fn doc_count(&self) -> usize {
        self.index.len()
    }
}

#[wasm_bindgen]
pub struct WasmSearch {
    model: Model,
    index: SearchIndex,
}

#[wasm_bindgen]
impl WasmSearch {
    #[wasm_bindgen]
    pub fn load(model_data: &[u8], index_data: &[u8]) -> Result<WasmSearch, JsValue> {
        let loaded = ModelWeights::from_quantized_bytes(model_data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let config = loaded.weights.config.clone();
        let tokenizer = WordPieceTokenizer::from_json_str(&loaded.tokenizer_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let model = Model::from_weights(loaded.weights, tokenizer, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let cursor = std::io::Cursor::new(index_data);
        let index = SearchIndex::load_binary(cursor)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(WasmSearch { model, index })
    }

    #[wasm_bindgen]
    pub fn search(&self, query: &str, limit: usize) -> Result<JsValue, JsValue> {
        let embedding = self.model
            .encode(vec![query], true)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let results = self.index.search_hybrid(query, &embedding[0], limit);

        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn search_semantic(&self, query: &str, limit: usize) -> Result<JsValue, JsValue> {
        let embedding = self.model
            .encode(vec![query], true)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let results = self.index.search_semantic(&embedding[0], limit);

        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn search_keywords(&self, query: &str, limit: usize) -> Result<JsValue, JsValue> {
        let results = self.index.search_keywords(query, limit);

        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn doc_count(&self) -> usize {
        self.index.len()
    }
}

#[wasm_bindgen]
pub struct WasmModel {
    inner: Model,
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub enum WasmModelType {
    MiniLML6V2,
}

enum Global {
    Window(Window),
    Worker(WorkerGlobalScope),
}

fn get_global() -> Result<Global, String> {
    let g = js_sys::global();
    if let Ok(win) = g.clone().dyn_into::<Window>() {
        Ok(Global::Window(win))
    } else if let Ok(worker) = g.clone().dyn_into::<WorkerGlobalScope>() {
        Ok(Global::Worker(worker))
    } else {
        Err("Unknown global scope".to_string())
    }
}

async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    let global = get_global()?;
    let resp_js = match global {
        Global::Window(win) => JsFuture::from(win.fetch_with_str(url)).await,
        Global::Worker(worker) => JsFuture::from(worker.fetch_with_str(url)).await,
    }
    .map_err(|e| format!("Fetch error: {:?}", e))?;

    let resp: Response = resp_js.dyn_into().map_err(|_| "Response cast failed")?;
    let array_buffer = JsFuture::from(resp.array_buffer().map_err(|_| "ArrayBuffer error")?)
        .await
        .map_err(|e| format!("ArrayBuffer await failed: {:?}", e))?;

    Ok(js_sys::Uint8Array::new(&array_buffer).to_vec())
}

async fn fetch_text(url: &str) -> Result<String, String> {
    let global = get_global()?;
    let resp_js = match global {
        Global::Window(win) => JsFuture::from(win.fetch_with_str(url)).await,
        Global::Worker(worker) => JsFuture::from(worker.fetch_with_str(url)).await,
    }
    .map_err(|e| format!("Fetch error: {:?}", e))?;

    let resp: Response = resp_js.dyn_into().map_err(|_| "Response cast failed")?;
    let text_js = JsFuture::from(resp.text().map_err(|_| "Text conversion failed")?)
        .await
        .map_err(|e| format!("Text await failed: {:?}", e))?;

    Ok(text_js.as_string().ok_or("Failed to convert text")?)
}

#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new(
        weights_data: &[u8],
        config_json: &str,
        tokenizer_json: &str,
    ) -> Result<WasmModel, JsValue> {
        let weights = ModelWeights::from_bytes(weights_data, config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let tokenizer = WordPieceTokenizer::from_json_str(tokenizer_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let config =
            serde_json::from_str(config_json).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let model = Model::from_weights(weights, tokenizer, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmModel { inner: model })
    }

    #[wasm_bindgen]
    pub fn from_quantized(data: &[u8]) -> Result<WasmModel, JsValue> {
        let loaded = ModelWeights::from_quantized_bytes(data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let config = loaded.weights.config.clone();
        let tokenizer = WordPieceTokenizer::from_json_str(&loaded.tokenizer_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let model = Model::from_weights(loaded.weights, tokenizer, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmModel { inner: model })
    }

    #[wasm_bindgen]
    pub async fn from_type(model_type: WasmModelType) -> Result<WasmModel, JsValue> {
        let (weights_url, config_url, tokenizer_url) = match model_type {
            WasmModelType::MiniLML6V2 => (
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors",
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
                "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
            ),
        };

        let (weights, config, tokenizer) = futures::future::join3(
            fetch_bytes(weights_url),
            fetch_text(config_url),
            fetch_text(tokenizer_url),
        )
        .await;

        let weights = weights.map_err(|e| JsValue::from_str(&e))?;
        let config = config.map_err(|e| JsValue::from_str(&e))?;
        let tokenizer = tokenizer.map_err(|e| JsValue::from_str(&e))?;

        WasmModel::new(&weights, &config, &tokenizer)
    }

    #[wasm_bindgen]
    pub fn encode(&self, texts: Vec<String>, normalize: bool) -> Result<Vec<f32>, JsValue> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings = self
            .inner
            .encode(text_refs, normalize)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(embeddings.into_iter().flatten().collect())
    }
}


#[wasm_bindgen]
pub struct WasmReranker {
    model: Model,
    // Classification head: score = cls_hidden @ weight^T + bias
    head_weight: Array2<f32>, // (1, hidden_size)
    head_bias: Array1<f32>,   // (1,)
}

#[derive(Serialize)]
struct RerankResult {
    index: usize,
    score: f32,
    text: String,
}

#[wasm_bindgen]
impl WasmReranker {
    #[wasm_bindgen]
    pub fn load(data: &[u8]) -> Result<WasmReranker, JsValue> {
        let loaded = ModelWeights::from_quantized_bytes(data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Extract head weights before building model
        let head_weight = loaded.weights.get_array2("classifier.weight")
            .map_err(|e| JsValue::from_str(&format!("Missing classifier.weight: {}", e)))?;
        let head_bias = loaded.weights.get_array1("classifier.bias")
            .map_err(|e| JsValue::from_str(&format!("Missing classifier.bias: {}", e)))?;

        let config = loaded.weights.config.clone();
        let tokenizer = WordPieceTokenizer::from_json_str(&loaded.tokenizer_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let model = Model::from_weights(loaded.weights, tokenizer, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(WasmReranker { model, head_weight, head_bias })
    }

    #[wasm_bindgen]
    pub fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        limit: usize,
    ) -> Result<JsValue, JsValue> {
        if documents.is_empty() {
            return serde_wasm_bindgen::to_value::<Vec<RerankResult>>(&vec![])
                .map_err(|e| JsValue::from_str(&e.to_string()));
        }

        let n = documents.len();

        // Tokenize all pairs
        let mut all_ids = Vec::with_capacity(n);
        let mut all_masks = Vec::with_capacity(n);
        let mut all_types = Vec::with_capacity(n);
        let mut max_len = 0;

        for doc in &documents {
            let enc = self.model.tokenizer.encode_pair(query, doc, 512)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            max_len = max_len.max(enc.ids.len());
            all_ids.push(enc.ids);
            all_masks.push(enc.attention_mask);
            all_types.push(enc.token_type_ids.unwrap_or_default());
        }

        // Pad and build arrays
        let mut input_ids = Array2::<f32>::zeros((n, max_len));
        let mut attention_mask = Array2::<f32>::zeros((n, max_len));
        let mut token_type_ids = Array2::<f32>::zeros((n, max_len));

        for i in 0..n {
            for j in 0..all_ids[i].len() {
                input_ids[[i, j]] = all_ids[i][j] as f32;
                attention_mask[[i, j]] = all_masks[i][j] as f32;
                token_type_ids[[i, j]] = all_types[i][j] as f32;
            }
        }

        // Forward pass → [CLS] hidden states (n, hidden_size)
        let cls_hidden = self.model.forward_cls(&input_ids, &attention_mask, Some(&token_type_ids))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        // Apply head: score = cls @ weight^T + bias
        let scores_2d = cls_hidden.dot(&self.head_weight.t()) + &self.head_bias;

        // Collect and sort by score descending
        let mut results: Vec<RerankResult> = (0..n).map(|i| {
            RerankResult {
                index: i,
                score: scores_2d[[i, 0]],
                text: documents[i].clone(),
            }
        }).collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn score(&self, query: &str, document: &str) -> Result<f32, JsValue> {
        let enc = self.model.tokenizer.encode_pair(query, document, 512)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let ids = Array2::from_shape_vec((1, enc.ids.len()),
            enc.ids.iter().map(|&x| x as f32).collect()
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let mask = Array2::from_shape_vec((1, enc.attention_mask.len()),
            enc.attention_mask.iter().map(|&x| x as f32).collect()
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let types = enc.token_type_ids.unwrap_or_default();
        let type_ids = Array2::from_shape_vec((1, types.len()),
            types.iter().map(|&x| x as f32).collect()
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let cls = self.model.forward_cls(&ids, &mask, Some(&type_ids))
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let score = cls.dot(&self.head_weight.t()) + &self.head_bias;
        Ok(score[[0, 0]])
    }
}
