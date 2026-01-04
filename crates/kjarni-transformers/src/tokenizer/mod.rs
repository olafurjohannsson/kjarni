//! Tokenizer implementations for GPT

pub mod bpe;
pub mod wordpiece;

pub trait ModelTokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>>;
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String>;
}


#[derive(Debug, Clone)]
pub struct Encoding {
    ids: Vec<u32>,
    attention_mask: Option<Vec<u32>>,
}

impl Encoding {
    pub fn get_ids(&self) -> &Vec<u32> {
        &self.ids
    }
    
    pub fn len(&self) -> usize {
        self.ids.len()
    }
}