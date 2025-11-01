//! WASM bindings for EdgeSearch

use wasm_bindgen::prelude::*;
use crate::*;

#[wasm_bindgen]
pub struct EdgeRAG {
    index: SearchIndex,
}

#[wasm_bindgen]
impl EdgeRAG {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            index: SearchIndex::new(),
        }
    }

    #[wasm_bindgen(js_name = loadIndex)]
    pub fn load_index(&mut self, json: &str) -> Result<(), JsValue> {
        self.index = SearchIndex::load_json(json)
            .map_err(|e| JsValue::from_str(&format!("Load failed: {}", e)))?;
        Ok(())
    }

    #[wasm_bindgen]
    pub fn search(
        &self,
        query_embedding: Vec<f32>,
        query_text: &str,
        k: usize,
    ) -> JsValue {
        let results = self.index.search_hybrid(query_text, &query_embedding, k);
        serde_wasm_bindgen::to_value(&results).unwrap()
    }
}