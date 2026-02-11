//! WASM-compatible BPE tokenizer for GPT

use anyhow::{anyhow, Result};
use serde_json::Value;
use std::collections::HashMap;

use crate::tokenizer::{Encoding, ModelTokenizer};


impl ModelTokenizer for BPETokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        Ok(self.encode(text, 1024)?.get_ids().clone())
    }
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.decode(ids, skip_special_tokens)
    }
}


pub struct BPETokenizer {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
    cache: std::sync::Mutex<HashMap<String, Vec<u32>>>, 
}


impl BPETokenizer {
    pub fn from_json_str(content: &str) -> Result<Self> {
        let json: Value = serde_json::from_str(content)?;
        
        let encoder = json["model"]["vocab"]
            .as_object()
            .ok_or_else(|| anyhow!("Could not find vocab"))?
            .iter()
            .map(|(token, id)| (token.clone(), id.as_u64().unwrap() as u32))
            .collect::<HashMap<String, u32>>();
        
        let decoder: HashMap<u32, String> = encoder
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        
        let merges_array = json["model"]["merges"]
            .as_array()
            .ok_or_else(|| anyhow!("Could not find merges"))?;

        let mut bpe_ranks = HashMap::new();
        for (rank, merge) in merges_array.iter().enumerate() {
            if let Some(s) = merge.as_str() {
                let parts: Vec<&str> = s.split_whitespace().collect();
                if parts.len() == 2 {
                    bpe_ranks.insert((parts[0].to_string(), parts[1].to_string()), rank);
                }
            }
        }
        
        let bos_token_id = encoder.get("<|startoftext|>").copied();
        let eos_token_id = encoder.get("<|endoftext|>").copied();
        let pad_token_id = encoder.get("<|pad|>").copied().or(eos_token_id);
        
        Ok(Self {
            encoder,
            decoder,
            bpe_ranks,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            cache: std::sync::Mutex::new(HashMap::new()),
        })
    }
    
    pub fn encode(&self, text: &str, _max_len: usize) -> Result<Encoding> {
        let mut ids = Vec::new();
        
        if let Some(bos_id) = self.bos_token_id {
            ids.push(bos_id);
        }
        for (i, word) in text.split_inclusive(char::is_whitespace).enumerate() {
             let mut token_list: Vec<String> = Vec::new();
             
             for (j, byte) in word.bytes().enumerate() {
                 let s = if j == 0 && i > 0 && !word.starts_with(' ') {
                     format!("Ġ{}", byte as char)
                 } else if byte == b' ' {
                     "Ġ".to_string()
                 } else {
                     (byte as char).to_string()
                 };
                 token_list.push(s);
             }

             let word_ids = self.bpe(&token_list);
             ids.extend(word_ids);
        }

        Ok(Encoding { ids, attention_mask: None })
    }

    fn bpe(&self, word: &[String]) -> Vec<u32> {
        if word.is_empty() { return vec![]; }
        
        let mut word = word.to_vec();

        loop {
            let mut pairs = Vec::new();
            if word.len() < 2 {
                break;
            }
            
            for i in 0..word.len() - 1 {
                pairs.push((word[i].clone(), word[i+1].clone()));
            }

            let mut best_pair: Option<(String, String)> = None;
            let mut best_rank = usize::MAX;

            for pair in pairs {
                if let Some(&rank) = self.bpe_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some(pair);
                    }
                }
            }

            let (first, second) = match best_pair {
                Some(p) => p,
                None => break,
            };

            let mut new_word = Vec::new();
            let mut i = 0;
            while i < word.len() {
                if i < word.len() - 1 && word[i] == first && word[i+1] == second {
                    new_word.push(format!("{}{}", first, second));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            
            word = new_word;
        }

        word.iter()
            .filter_map(|token| self.encoder.get(token).copied())
            .collect()
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| {
                if skip_special_tokens {
                    if Some(id) == self.bos_token_id || 
                       Some(id) == self.eos_token_id || 
                       Some(id) == self.pad_token_id {
                        return None;
                    }
                }
                self.decoder.get(&id).cloned()
            })
            .collect();

        Ok(tokens.join("")
            .replace("Ġ", " ")
            .replace("Ċ", "\n"))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Not;

    use super::*;
    use anyhow::Result;

    const TEST_JSON: &str = r#"
    {
        "model": {
            "vocab": {
                "<|startoftext|>": 0,
                "<|endoftext|>": 1,
                "<|pad|>": 2,
                "Ġh": 3,
                "Ġi": 4,
                "e": 5,
                "l": 6,
                "o": 7,
                " ": 8,
                "h": 9,
                "i": 10
            },
            "merges": [
                "Ġ h",
                "Ġ i"
            ]
        }
    }
    "#;

    #[test]
    fn test_tokenizer_from_json() -> Result<()> {
        let tokenizer = BPETokenizer::from_json_str(TEST_JSON)?;
        assert_eq!(tokenizer.bos_token_id, Some(0));
        assert_eq!(tokenizer.eos_token_id, Some(1));
        assert_eq!(tokenizer.pad_token_id, Some(2));
        assert!(tokenizer.encoder.contains_key("Ġh"));
        Ok(())
    }

    #[test]
    fn test_basic_encode_decode() -> Result<()> {
        let tokenizer = BPETokenizer::from_json_str(TEST_JSON)?;
        let text = "hi";
        let encoding = tokenizer.encode(text, 10)?;
        let ids = encoding.get_ids();
        assert_eq!(ids[0], tokenizer.bos_token_id.unwrap());
        for &id in ids.iter().skip(1) {
            assert!(tokenizer.decoder.contains_key(&id));
        }

        let decoded = tokenizer.decode(ids, true)?;
        assert!(decoded.contains("h"));
        assert!(decoded.contains("i"));
        Ok(())
    }



    #[test]
    fn test_decode_with_unknown_ids() -> Result<()> {
        let tokenizer = BPETokenizer::from_json_str(TEST_JSON)?;
        let ids = vec![0, 999, 1];
        let decoded = tokenizer.decode(&ids, true)?;
        assert!(decoded.contains("<|startoftext|>").not());
        Ok(())
    }
}
