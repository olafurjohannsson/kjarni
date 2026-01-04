//! WASM-compatible BPE tokenizer for GPT

use anyhow::{anyhow, Result};
use serde_json::Value;
use std::collections::HashMap;

use crate::tokenizer::{Encoding, ModelTokenizer};



/// BPE tokenizer for GPT models
pub struct BPETokenizer {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    merges: Vec<(String, String)>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

impl ModelTokenizer for BPETokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        // Map your existing encode to return just Vec<u32>
        Ok(self.encode(text, 1024)?.get_ids().clone())
    }
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.decode(ids, skip_special_tokens)
    }
}

impl BPETokenizer {
    pub fn from_json_str(content: &str) -> Result<Self> {
        let json: Value = serde_json::from_str(content)?;
        
        // Extract vocabulary
        let encoder = json["model"]["vocab"]
            .as_object()
            .ok_or_else(|| anyhow!("Could not find vocab in tokenizer json"))?
            .iter()
            .map(|(token, id)| (token.clone(), id.as_u64().unwrap() as u32))
            .collect::<HashMap<String, u32>>();
        
        let decoder: HashMap<u32, String> = encoder
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        
        // Extract merges
        let merges = json["model"]["merges"]
            .as_array()
            .ok_or_else(|| anyhow!("Could not find merges in tokenizer json"))?
            .iter()
            .filter_map(|merge| {
                merge.as_str().map(|s| {
                    let parts: Vec<&str> = s.split_whitespace().collect();
                    if parts.len() == 2 {
                        Some((parts[0].to_string(), parts[1].to_string()))
                    } else {
                        None
                    }
                })
            })
            .filter_map(|x| x)
            .collect();
        
        // Get special tokens
        let bos_token_id = encoder.get("<|startoftext|>").copied();
        let eos_token_id = encoder.get("<|endoftext|>").copied();
        let pad_token_id = encoder.get("<|pad|>").copied().or(eos_token_id);
        
        Ok(Self {
            encoder,
            decoder,
            merges,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }
    
    /// Simple BPE encoding
    pub fn encode(&self, text: &str, _max_len: usize) -> Result<Encoding> {
        let mut tokens = Vec::new();
        
        // Add BOS token if available
        if let Some(bos_id) = self.bos_token_id {
            tokens.push(bos_id);
        }
        
        // Basic byte-level tokenization
        let bytes = text.as_bytes();
        let mut word = Vec::new();
        
        for &byte in bytes {
            // Convert byte to special token representation
            let token = format!("Ġ{}", byte as char);
            if let Some(&id) = self.encoder.get(&token) {
                word.push(token);
            } else {
                // Fallback to individual byte
                let byte_token = format!("{}", byte as char);
                if let Some(&id) = self.encoder.get(&byte_token) {
                    word.push(byte_token);
                }
            }
            
            // Process at word boundaries (simplified)
            if byte == b' ' || byte == b'\n' {
                tokens.extend(self.bpe_merge(&word));
                word.clear();
            }
        }
        
        // Process remaining word
        if !word.is_empty() {
            tokens.extend(self.bpe_merge(&word));
        }
        
        Ok(Encoding { ids: tokens, attention_mask: None })
    }
    
    /// Apply BPE merges
    fn bpe_merge(&self, word: &[String]) -> Vec<u32> {
        if word.is_empty() {
            return vec![];
        }
        
        // For simplicity, just convert each token to ID
        word.iter()
            .filter_map(|token| self.encoder.get(token).copied())
            .collect()
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| {
                // Get the token string, skip unknown IDs gracefully
                let token = self.decoder.get(&id)?;
                
                // Skip special tokens if requested
                if skip_special_tokens {
                    if let Some(bos) = self.bos_token_id {
                        if id == bos { return None; }
                    }
                    if let Some(eos) = self.eos_token_id {
                        if id == eos { return None; }
                    }
                    if let Some(pad) = self.pad_token_id {
                        if id == pad { return None; }
                    }
                }
                Some(token.clone())
            })
            .collect();

        // Concatenate and clean up common BPE markers
        let text = tokens.join("")
            .replace("Ġ", " ")   // byte-level spaces
            .replace("Ċ", "\n")  // newlines
            .trim()
            .to_string();

        Ok(text)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Not;

    use super::*;
    use anyhow::Result;

    // Minimal JSON for testing
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
                " ": 8
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
        // BOS token should be added
        assert_eq!(ids[0], tokenizer.bos_token_id.unwrap());
        // Other token IDs should exist in vocab
        for &id in ids.iter().skip(1) {
            assert!(tokenizer.decoder.contains_key(&id));
        }

        let decoded = tokenizer.decode(ids, true)?;
        // Decoding should contain original characters
        assert!(decoded.contains("h"));
        assert!(decoded.contains("i"));
        Ok(())
    }

    #[test]
    fn test_bpe_merge() -> Result<()> {
        let tokenizer = BPETokenizer::from_json_str(TEST_JSON)?;
        let word = vec!["Ġh".to_string(), "i".to_string()];
        let merged_ids = tokenizer.bpe_merge(&word);
        // Should convert tokens to correct IDs
        assert_eq!(merged_ids.len(), 1); // "Ġh" exists in vocab
        assert_eq!(merged_ids[0], tokenizer.encoder["Ġh"]);
        Ok(())
    }

    #[test]
    fn test_decode_with_unknown_ids() -> Result<()> {
        let tokenizer = BPETokenizer::from_json_str(TEST_JSON)?;
        let ids = vec![0, 999, 1]; // 999 does not exist
        let decoded = tokenizer.decode(&ids, true)?;
        // Unknown token should be skipped
        assert!(decoded.contains("<|startoftext|>").not()); // actually decoder should skip unknown
        Ok(())
    }
}
