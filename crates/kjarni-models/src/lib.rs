
pub mod models;


pub use models::cross_encoder::CrossEncoder;
pub use models::sentence_encoder::{BertConfig, DistilBertConfig, MpnetConfig, SentenceEncoder};
pub use models::sequence_classifier::{SequenceClassifier};

/// A callback for streaming generated tokens
pub type TokenCallback<'a> = Box<dyn FnMut(u32, &str) -> bool + 'a>;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(test)]
pub mod tests;

#[cfg(not(target_arch = "wasm32"))]
pub use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
pub use tokenizer::wasm::BPETokenizer;

#[cfg(test)]
mod send_sync_tests {
    use crate::{CrossEncoder, SentenceEncoder, SequenceClassifier, models::{bart::model::BartModel, gpt2::Gpt2Model, llama::LlamaModel, mistral::MistralModel, qwen::QwenModel, t5::T5Model}};
    // Compile time validation of send and sync
     const _: () = {
        const fn assert_send<T: Send>() {}
        const fn assert_sync<T: Sync>() {}
        assert_send::<CrossEncoder>();
        assert_sync::<CrossEncoder>();

        assert_send::<SentenceEncoder>();
        assert_sync::<SentenceEncoder>();

        assert_send::<SequenceClassifier>();
        assert_sync::<SequenceClassifier>();

        assert_send::<LlamaModel>();
        assert_sync::<LlamaModel>();

        assert_send::<QwenModel>();
        assert_sync::<QwenModel>();

        assert_send::<MistralModel>();
        assert_sync::<MistralModel>();

        assert_send::<T5Model>();
        assert_sync::<T5Model>();

        assert_send::<Gpt2Model>();
        assert_sync::<Gpt2Model>();

        assert_send::<BartModel>();
        assert_sync::<BartModel>();
    };

}