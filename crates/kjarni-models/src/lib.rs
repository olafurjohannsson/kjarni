//! GPT implementation using Kjarni Transformers
//! 
//! Provides autoregressive language models for text generation.

pub mod tokenizer;
pub mod cross_encoder;
pub mod sentence_encoder;
pub mod sequence_classifier;
pub mod models;

pub use cross_encoder::CrossEncoder;
pub use sentence_encoder::SentenceEncoder;
pub use sequence_classifier::SequenceClassifier;
// pub use text_generation::TextGenerator;
// pub use generation::Generator;

/// A callback for streaming generated tokens.
///
/// The callback receives the token ID (`u32`) and its decoded text representation (`&str`).
/// It should return `true` to continue generation or `false` to stop early.
pub type TokenCallback<'a> = Box<dyn FnMut(u32, &str) -> bool + 'a>;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(test)]
pub mod tests;

#[cfg(not(target_arch = "wasm32"))]
pub use tokenizers::Tokenizer;

#[cfg(target_arch = "wasm32")]
pub use tokenizer::wasm::BPETokenizer;

// User-facing unified API
// pub enum Model {
//     Decoder(DecoderModel),
//     Encoder(EncoderModel),
//     Seq2Seq(Seq2SeqModel),
//     CrossEncoder(CrossEncoderModel),
// }

// pub enum DecoderModel {
//     Llama(LlamaModel),
//     Phi(PhiModel),
//     Gpt2(Gpt2Model),
// }

// pub enum Seq2SeqModel {
//     Bart(BartModel),
//     T5(T5Model),
// }

// pub enum EncoderModel {
//     MiniLM(SentenceEncoder),
//     // All encoders already share SentenceEncoder struct
// }
// impl Model {
//     pub async fn load(model_type: ModelType, device: Device) -> Result<Self> {
//         match model_type.info().architecture {
//             ModelArchitecture::Decoder => {
//                 let model = match model_type {
//                     ModelType::Llama3_2_1B | ModelType::Llama3_2_1B_Instruct | ... => {
//                         DecoderModel::Llama(LlamaModel::from_registry(model_type, device).await?)
//                     }
//                     ModelType::Phi3Mini => {
//                         DecoderModel::Phi(PhiModel::from_registry(model_type, device).await?)
//                     }
//                     _ => return Err(anyhow!("Unknown decoder"))
//                 };
//                 Ok(Model::Decoder(model))
//             }
//             ModelArchitecture::Seq2Seq => {
//                 let model = match model_type {
//                     ModelType::BartLargeCnn | ModelType::DistilBartCnn => {
//                         Seq2SeqModel::Bart(BartModel::from_registry(model_type, device).await?)
//                     }
//                     ModelType::T5Small | ModelType::T5Base => {
//                         Seq2SeqModel::T5(T5Model::from_registry(model_type, device).await?)
//                     }
//                     _ => return Err(anyhow!("Unknown seq2seq"))
//                 };
//                 Ok(Model::Seq2Seq(model))
//             }
//             // ...
//         }
//     }
// }
// impl DecoderModel {
//     pub async fn generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String> {
//         match self {
//             DecoderModel::Llama(m) => {
//                 let gen = DecoderGenerator::new(Box::new(m.clone()))?;
//                 gen.generate(prompt, config).await
//             }
//             DecoderModel::Phi(m) => {
//                 let gen = DecoderGenerator::new(Box::new(m.clone()))?;
//                 gen.generate(prompt, config).await
//             }
//             DecoderModel::Gpt2(m) => {
//                 let gen = DecoderGenerator::new(Box::new(m.clone()))?;
//                 gen.generate(prompt, config).await
//             }
//         }
//     }
// }
// ro_rules! dispatch_decoder {
//     ($self:expr, $method:ident $(, $arg:expr)*) => {
//         match $self {
//             DecoderModel::Llama(m) => m.$method($($arg),*),
//             DecoderModel::Phi(m) => m.$method($($arg),*),
//             DecoderModel::Gpt2(m) => m.$method($($arg),*),
//         }
//     };
// }

// impl DecoderModel {
//     pub fn vocab_size(&self) -> usize {
//         dispatch_decoder!(self, vocab_size)
//     }
    
//     pub fn tokenizer(&self) -> &Tokenizer {
//         dispatch_decoder!(self, tokenizer)
//     }
// }