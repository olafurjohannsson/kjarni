use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

use crate::wasm_simd;

#[wasm_bindgen]
pub struct WasmFeedForward {
    layer1_weight: Vec<f32>,
    layer1_bias: Vec<f32>,
    layer2_weight: Vec<f32>,
    layer2_bias: Vec<f32>,
    layer3_weight: Vec<f32>,
    layer3_bias: Vec<f32>,
    input_dim: usize,
    hidden1: usize,
    hidden2: usize,
    output_dim: usize,
}

#[derive(Serialize, Deserialize)]
struct WasmFeedForwardResult {
    label: usize,
    confidence: f32,
    scores: Vec<f32>,
}

#[wasm_bindgen]
impl WasmFeedForward {
    /// Load from raw safetensors bytes + config JSON string.
    /// Expected safetensors keys:
    ///   layer1.weight (hidden1, input_dim)
    ///   layer1.bias   (hidden1,)
    ///   layer2.weight (hidden2, hidden1)
    ///   layer2.bias   (hidden2,)
    ///   layer3.weight (output_dim, hidden2)
    ///   layer3.bias   (output_dim,)
    #[wasm_bindgen]
    pub fn load(safetensors_bytes: &[u8], config_json: &str) -> Result<WasmFeedForward, JsValue> {
        let config: WasmFeedForwardConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("Config parse error: {}", e)))?;

        let tensors = safetensors::SafeTensors::deserialize(safetensors_bytes)
            .map_err(|e| JsValue::from_str(&format!("Safetensors error: {}", e)))?;

        let load_f32 = |name: &str| -> Result<Vec<f32>, JsValue> {
            let view = tensors.tensor(name)
                .map_err(|e| JsValue::from_str(&format!("Missing tensor '{}': {}", name, e)))?;
            Ok(view.data().chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        };

        let layer1_weight = load_f32("layer1.weight")?;
        let layer1_bias = load_f32("layer1.bias")?;
        let layer2_weight = load_f32("layer2.weight")?;
        let layer2_bias = load_f32("layer2.bias")?;
        let layer3_weight = load_f32("layer3.weight")?;
        let layer3_bias = load_f32("layer3.bias")?;

        Ok(WasmFeedForward {
            layer1_weight,
            layer1_bias,
            layer2_weight,
            layer2_bias,
            layer3_weight,
            layer3_bias,
            input_dim: config.input_dim,
            hidden1: config.hidden_dim,
            hidden2: config.hidden_dim / 2,
            output_dim: config.output_dim,
        })
    }

    /// Classify a single input. Takes a pre-scaled feature vector (Float32Array).
    /// Returns JSON: { label: 0|1, confidence: 0.0-1.0, scores: [f32, f32] }
    #[wasm_bindgen]
    pub fn classify(&self, features: &[f32]) -> Result<JsValue, JsValue> {
        if features.len() != self.input_dim {
            return Err(JsValue::from_str(&format!(
                "Expected {} features, got {}", self.input_dim, features.len()
            )));
        }

        // Layer 1: Linear(input_dim, hidden1) + ReLU
        let mut h1 = vec![0.0f32; self.hidden1];
        unsafe {
            wasm_simd::wasm_matmul_2d(
                &mut h1, features, &self.layer1_weight,
                1, self.hidden1, self.input_dim,
            );
        }
        for i in 0..self.hidden1 {
            h1[i] = (h1[i] + self.layer1_bias[i]).max(0.0); // ReLU
        }

        // Layer 2: Linear(hidden1, hidden2) + ReLU
        let mut h2 = vec![0.0f32; self.hidden2];
        unsafe {
            wasm_simd::wasm_matmul_2d(
                &mut h2, &h1, &self.layer2_weight,
                1, self.hidden2, self.hidden1,
            );
        }
        for i in 0..self.hidden2 {
            h2[i] = (h2[i] + self.layer2_bias[i]).max(0.0); // ReLU
        }

        // Layer 3: Linear(hidden2, output_dim)
        let mut logits = vec![0.0f32; self.output_dim];
        unsafe {
            wasm_simd::wasm_matmul_2d(
                &mut logits, &h2, &self.layer3_weight,
                1, self.output_dim, self.hidden2,
            );
        }
        for i in 0..self.output_dim {
            logits[i] += self.layer3_bias[i];
        }

        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp() / exp_sum).collect();

        let label = probs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;
        let confidence = probs[label];

        let result = WasmFeedForwardResult {
            label,
            confidence,
            scores: probs,
        };

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Batch classify. Takes flat f32 array (n * input_dim) and returns JSON array of results.
    #[wasm_bindgen]
    pub fn classify_batch(&self, features: &[f32], count: usize) -> Result<JsValue, JsValue> {
        if features.len() != count * self.input_dim {
            return Err(JsValue::from_str(&format!(
                "Expected {} floats ({} x {}), got {}",
                count * self.input_dim, count, self.input_dim, features.len()
            )));
        }

        let mut results = Vec::with_capacity(count);
        for i in 0..count {
            let start = i * self.input_dim;
            let end = start + self.input_dim;
            let result_js = self.classify(&features[start..end])?;
            let result: WasmFeedForwardResult = serde_wasm_bindgen::from_value(result_js)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            results.push(result);
        }

        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

#[derive(Deserialize)]
struct WasmFeedForwardConfig {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
}