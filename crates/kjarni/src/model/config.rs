// kjarni/src/config.rs

use kjarni_transformers::common::{DecodingStrategy, GenerationConfig};

pub struct GenerationConfigBuilder {
    max_tokens: Option<usize>,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repetition_penalty: f32,
}

impl GenerationConfigBuilder {
    pub fn new() -> Self {
        Self {
            max_tokens: None,
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
        }
    }
    
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = Some(n);
        self
    }
    
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
    
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }
    
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }
    
    pub fn repetition_penalty(mut self, p: f32) -> Self {
        self.repetition_penalty = p;
        self
    }
    
    pub fn build(self) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: self.max_tokens,
            repetition_penalty: self.repetition_penalty,
            strategy: DecodingStrategy::Sampling {
                temperature: self.temperature,
                top_p: Some(self.top_p),
                top_k: self.top_k,
                min_p: None,
            },
            ..Default::default()
        }
    }
}

impl GenerationConfig {
    pub fn builder() -> GenerationConfigBuilder {
        GenerationConfigBuilder::new()
    }
}