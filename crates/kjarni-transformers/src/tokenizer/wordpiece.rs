//! WASM-compatible BPE tokenizer for GPT

use crate::tokenizer::{Encoding, ModelTokenizer};
use anyhow::{Result, anyhow};
use serde_json::Value;
use std::collections::HashMap;

/// A lightweight, WASM-compatible WordPiece tokenizer
pub struct WordPieceTokenizer {
    vocab: HashMap<String, u32>,
    unk_token_id: u32,
    cls_token_id: u32,
    sep_token_id: u32,
    pad_token_id: u32,
}
impl ModelTokenizer for WordPieceTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> anyhow::Result<Vec<u32>> {
        Ok(self.encode(text, 1024)?.get_ids().clone())
    }
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.decode(ids, skip_special_tokens)
    }
}
impl WordPieceTokenizer {
    pub fn from_json_str(content: &str) -> Result<Self> {
        let json: Value = serde_json::from_str(content)?;

        let vocab = json["model"]["vocab"]
            .as_object()
            .ok_or_else(|| anyhow!("Could not find vocab in tokenizer json"))?
            .iter()
            .map(|(token, id)| (token.clone(), id.as_u64().unwrap() as u32))
            .collect::<HashMap<String, u32>>();

        let get_token_id = |token: &str| -> Result<u32> {
            vocab
                .get(token)
                .copied()
                .ok_or_else(|| anyhow!("Token {} not found in vocabulary", token))
        };

        let cls_token_id = get_token_id("[CLS]")?;
        let sep_token_id = get_token_id("[SEP]")?;
        let pad_token_id = get_token_id("[PAD]")?;
        let unk_token_id = get_token_id("[UNK]")?;

        Ok(Self {
            vocab,
            unk_token_id,
            cls_token_id,
            sep_token_id,
            pad_token_id,
        })
    }

    fn tokenize_word(&self, word: &str) -> Vec<u32> {
        if let Some(id) = self.vocab.get(word) {
            return vec![*id];
        }

        let mut sub_tokens = Vec::new();
        let mut remaining = word;

        while !remaining.is_empty() {
            let mut found = false;

            for i in (1..=remaining.len()).rev() {
                let prefix = &remaining[0..i];
                
                let token_to_check = if remaining.len() != word.len() {
                    format!("##{}", prefix)
                } else {
                    prefix.to_string()
                };

                if let Some(id) = self.vocab.get(&token_to_check) {
                    sub_tokens.push(*id);
                    remaining = &remaining[i..];
                    found = true;
                    break;
                }
            }

            if !found {
                return vec![self.unk_token_id];
            }
        }

        sub_tokens
    }

    pub fn encode(&self, text: &str, max_len: usize) -> Result<Encoding> {
        if max_len == 0 {
            return Err(anyhow!("max_len must be > 0"));
        }
        let mut ids = vec![self.cls_token_id];

        let mut spaced_text = String::new();
        for c in text.to_lowercase().chars() {
            if c.is_ascii_punctuation() {
                spaced_text.push(' ');
                spaced_text.push(c);
                spaced_text.push(' ');
            } else {
                spaced_text.push(c);
            }
        }

        for word in spaced_text.split_whitespace() {
            ids.extend(self.tokenize_word(word));
        }

        if ids.len() >= max_len {
            ids.truncate(max_len - 1);
        }

        ids.push(self.sep_token_id);

        let mut attention_mask = vec![1; ids.len()];

        if ids.len() < max_len {
            let pad_len = max_len - ids.len();
            ids.extend(vec![self.pad_token_id; pad_len]);
            attention_mask.extend(vec![0; pad_len]);
        }

        Ok(Encoding {
            ids,
            attention_mask: Some(attention_mask),
        })
    }

    pub fn encode_batch(&self, texts: Vec<&str>, max_len: usize) -> Result<Vec<Encoding>> {
        texts.iter().map(|t| self.encode(t, max_len)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    const TEST_JSON: &str = "{  
\"model\": {  
    \"vocab\": {  
        \"[CLS]\": 0,  
        \"[SEP]\": 1,  
        \"[PAD]\": 2,  
        \"[UNK]\": 3,  
        \"hello\": 4,  
        \"world\": 5,  
        \"##s\": 6,  
        \"!\": 7  
    }  
}  
}";

    #[test]
    fn test_from_json() -> Result<()> {
        let tokenizer = WordPieceTokenizer::from_json_str(TEST_JSON)?;
        assert_eq!(tokenizer.cls_token_id, 0);
        assert_eq!(tokenizer.sep_token_id, 1);
        assert_eq!(tokenizer.pad_token_id, 2);
        assert_eq!(tokenizer.unk_token_id, 3);
        assert!(tokenizer.vocab.contains_key("hello"));
        Ok(())
    }

    #[test]
    fn test_tokenize_word_known() -> Result<()> {
        let tokenizer = WordPieceTokenizer::from_json_str(TEST_JSON)?;
        let ids = tokenizer.tokenize_word("hello");
        assert_eq!(ids, vec![4]);
        Ok(())
    }

    #[test]
    fn test_tokenize_word_unknown() -> Result<()> {
        let tokenizer = WordPieceTokenizer::from_json_str(TEST_JSON)?;
        let ids = tokenizer.tokenize_word("foobar");
        assert_eq!(ids, vec![tokenizer.unk_token_id]);
        Ok(())
    }

    #[test]
    fn test_tokenize_word_with_subtokens() -> Result<()> {
        let tokenizer = WordPieceTokenizer::from_json_str(TEST_JSON)?;
        let ids = tokenizer.tokenize_word("worlds");
        assert_eq!(ids, vec![5, 6]);
        Ok(())
    }

    #[test]
    fn test_encode_basic() -> Result<()> {
        let tokenizer = WordPieceTokenizer::from_json_str(TEST_JSON)?;
        let text = "hello world!";
        let max_len = 10;
        let encoding = tokenizer.encode(text, max_len)?;

        assert_eq!(encoding.ids.len(), max_len);
        assert_eq!(encoding.ids[0], tokenizer.cls_token_id);
        
        assert_eq!(encoding.ids[1], 4);
        assert_eq!(encoding.ids[2], 5);
        assert_eq!(encoding.ids[3], 7);
        
        assert_eq!(encoding.ids[4], tokenizer.sep_token_id); 
        
        for i in 5..10 {
            assert_eq!(encoding.ids[i], tokenizer.pad_token_id);
        }

        Ok(())
    }
    #[test]
    fn test_encode_truncation_and_padding() -> Result<()> {
        let tokenizer = WordPieceTokenizer::from_json_str(TEST_JSON)?;
        let text = "hello world!";
        let max_len = 5;
        let encoding = tokenizer.encode(text, max_len)?;

        assert_eq!(encoding.ids.len(), max_len);
        assert_eq!(encoding.ids[max_len - 1], tokenizer.sep_token_id);

        if let Some(attn) = &encoding.attention_mask {
            assert_eq!(attn.len(), max_len);
            assert!(attn[max_len - 1] == 1); 
        }

        Ok(())
    }
    #[test]
    fn test_encode_batch() -> Result<()> {
        let tokenizer = WordPieceTokenizer::from_json_str(TEST_JSON)?;
        let texts = vec!["hello", "world"];
        let max_len = 5;
        let encodings = tokenizer.encode_batch(texts.iter().map(|s| *s).collect(), max_len)?;

        assert_eq!(encodings.len(), 2);

        let e0 = &encodings[0];
        assert_eq!(e0.ids[0], 0); // CLS
        assert_eq!(e0.ids[1], 4); // hello
        assert_eq!(e0.ids[2], 1); // SEP
        assert_eq!(e0.ids[3], 2); // PAD
        assert_eq!(e0.ids[4], 2); // PAD

        let e1 = &encodings[1];
        assert_eq!(e1.ids[0], 0);
        assert_eq!(e1.ids[1], 5);
        assert_eq!(e1.ids[2], 1);
        
        Ok(())
    }
}
