//! Tokenizer implementations for GPT

pub mod bpe;
pub mod wordpiece;

use tokenizers::models::ModelWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

/// Split on whitespace before SentencePiece's `Metaspace`, as transformers does
/// when it builds a Unigram tokenizer (XLM-RoBERTa, T5).
///
/// The `tokenizer.json` those models publish has `Metaspace` alone, which turns
/// trailing whitespace into a token of its own, `▁`. SentencePiece drops it, and
/// so does transformers; kept, a text ending in a space or a newline embeds 0.01
/// to 0.03 away from the reference on bge-m3. BPE tokenizers that use `Metaspace`,
/// such as Mistral's, keep their whitespace in transformers too, and are left alone.
pub fn match_sentencepiece_whitespace(tokenizer: &mut tokenizers::Tokenizer) {
    if !matches!(tokenizer.get_model(), ModelWrapper::Unigram(_)) {
        return;
    }
    let Some(PreTokenizerWrapper::Metaspace(metaspace)) = tokenizer.get_pre_tokenizer().cloned()
    else {
        return;
    };
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
        PreTokenizerWrapper::Metaspace(metaspace),
    ])));
}

pub trait ModelTokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> anyhow::Result<Vec<u32>>;
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String>;
}

#[derive(Debug, Clone)]
pub struct Encoding {
    ids: Vec<u32>,
    #[allow(
        dead_code,
        reason = "not referenced yet; kept until the path that needs it lands"
    )]
    attention_mask: Option<Vec<u32>>,
}

impl Encoding {
    pub fn get_ids(&self) -> &Vec<u32> {
        &self.ids
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

#[cfg(test)]
mod sentencepiece_whitespace_tests {
    use super::*;
    use tokenizers::models::unigram::Unigram;
    use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

    /// A Unigram tokenizer with a bare `Metaspace`, as XLM-R and T5 publish it.
    fn unigram() -> tokenizers::Tokenizer {
        let vocab = vec![
            ("<unk>".to_string(), 0.0),
            ("▁".to_string(), -2.0),
            ("▁hi".to_string(), -1.0),
        ];
        let mut tok = tokenizers::Tokenizer::new(Unigram::from(vocab, Some(0), false).unwrap());
        tok.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, true)));
        tok
    }

    fn ids(tok: &tokenizers::Tokenizer, text: &str) -> Vec<u32> {
        tok.encode(text, false).unwrap().get_ids().to_vec()
    }

    #[test]
    fn trailing_whitespace_is_not_a_token() {
        let mut tok = unigram();
        // The published pre-tokenizer alone reads the trailing space as `▁`.
        assert_eq!(ids(&tok, "hi "), vec![2, 1]);
        match_sentencepiece_whitespace(&mut tok);
        // SentencePiece and transformers drop it.
        assert_eq!(ids(&tok, "hi "), vec![2]);
        assert_eq!(ids(&tok, "hi\n"), vec![2]);
        assert_eq!(ids(&tok, "hi hi"), vec![2, 2]);
    }

    #[test]
    fn non_unigram_tokenizers_are_left_alone() {
        let mut tok = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        tok.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, true)));
        match_sentencepiece_whitespace(&mut tok);
        assert!(matches!(
            tok.get_pre_tokenizer(),
            Some(PreTokenizerWrapper::Metaspace(_))
        ));
    }
}
