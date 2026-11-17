//! The models the server holds: each loaded on its first request and kept.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

use kjarni::chat::Chat;
use kjarni::classifier::{Classifier, ClassifierBuilder};
use kjarni::common::KjarniDevice;
use kjarni::embedder::{Embedder, EmbedderBuilder};
use kjarni::generator::Generator;
use kjarni::ner::EntityRecognizer;
use kjarni::reranker::{Reranker, RerankerBuilder};
use kjarni::summarizer::Summarizer;
use kjarni::transcriber::Transcriber;
use kjarni::translator::Translator;
use kjarni::{ModelTask, ModelType};
use tokio::sync::{OnceCell, Semaphore};

use super::error::ApiError;

/// The model each endpoint uses when a request names none.
#[derive(Debug, Clone)]
pub struct Defaults {
    pub chat: String,
    pub embed: String,
    pub classify: String,
    pub zero_shot: String,
    pub rerank: String,
    pub transcribe: String,
    /// `None` picks a model for the language pair, as `kjarni translate` does.
    pub translate: Option<String>,
    pub summarize: String,
    pub ner: String,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            chat: kjarni_cli::DEFAULT_CHAT_MODEL.to_string(),
            embed: "minilm-l6-v2".to_string(),
            classify: "distilbert-sentiment".to_string(),
            zero_shot: "minilm-zeroshot".to_string(),
            rerank: "minilm-l6-v2-cross-encoder".to_string(),
            transcribe: "whisper-small".to_string(),
            translate: None,
            summarize: "distilbart-cnn".to_string(),
            ner: "distilbert-ner".to_string(),
        }
    }
}

/// A text-generation model. One with a chat template serves both chat and
/// completions; one without, such as GPT-2, serves completions only.
pub enum Decoder {
    Chat(Box<Chat>),
    Raw(Box<Generator>),
}

pub struct LoadedDecoder {
    pub decoder: Decoder,
    /// One generation at a time per model: two at once would share the CPU and
    /// each finish later than if they had queued.
    pub turn: Arc<Semaphore>,
}

impl LoadedDecoder {
    pub fn generator(&self) -> &Generator {
        match &self.decoder {
            Decoder::Chat(chat) => chat.generator(),
            Decoder::Raw(generator) => generator,
        }
    }

    pub fn chat(&self) -> Option<&Chat> {
        match &self.decoder {
            Decoder::Chat(chat) => Some(chat),
            Decoder::Raw(_) => None,
        }
    }
}

/// Models of one kind by name. Concurrent first requests for a model share
/// one load, and a load that fails is tried again on the next request.
struct Pool<T> {
    slots: Mutex<HashMap<String, Arc<OnceCell<Arc<T>>>>>,
}

impl<T> Pool<T> {
    fn new() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
        }
    }

    async fn get<F, Fut>(&self, name: &str, load: F) -> Result<Arc<T>, ApiError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, String>>,
    {
        let slot = self
            .slots
            .lock()
            .expect("model pool lock")
            .entry(name.to_string())
            .or_default()
            .clone();
        slot.get_or_try_init(|| async { load().await.map(Arc::new) })
            .await
            .cloned()
            .map_err(|message| ApiError::bad_request(format!("could not load '{name}': {message}")))
    }

    /// Holds `value` as model `name`, as if a request had loaded it.
    fn insert(&self, name: &str, value: T) -> Arc<T> {
        let value = Arc::new(value);
        self.slots.lock().expect("model pool lock").insert(
            name.to_string(),
            Arc::new(OnceCell::new_with(Some(value.clone()))),
        );
        value
    }

    fn loaded(&self) -> Vec<String> {
        self.slots
            .lock()
            .expect("model pool lock")
            .iter()
            .filter(|(_, slot)| slot.initialized())
            .map(|(name, _)| name.clone())
            .collect()
    }
}

pub struct Models {
    pub defaults: Defaults,
    device: KjarniDevice,
    prefix_cache: bool,
    decoders: Pool<LoadedDecoder>,
    embedders: Pool<Embedder>,
    classifiers: Pool<Classifier>,
    rerankers: Pool<Reranker>,
    transcribers: Pool<Transcriber>,
    translators: Pool<Translator>,
    summarizers: Pool<Summarizer>,
    recognizers: Pool<EntityRecognizer>,
}

/// The registry name a request asked for, or the endpoint's default when it
/// named none. An unknown name is an error, not a quiet fallback to the
/// default, so a client never gets an answer from a model it didn't choose.
pub fn resolve(requested: Option<&str>, default: &str) -> Result<String, ApiError> {
    resolve_in(requested, default, None)
}

/// [`resolve`], with Hugging Face downloads looked for in `cache`. A request
/// may name a Hub repository the server has already downloaded, or the one it
/// was started with; never a new one, so a client cannot fill the server's
/// disk with downloads.
pub fn resolve_in(
    requested: Option<&str>,
    default: &str,
    cache: Option<&std::path::Path>,
) -> Result<String, ApiError> {
    let name = match requested.map(str::trim) {
        None | Some("") | Some("default") => return Ok(default.to_string()),
        Some(name) => name,
    };
    if let Some(model) = ModelType::from_cli_name(name) {
        return Ok(model.cli_name().to_string());
    }
    match kjarni::hub::HubRef::parse(name) {
        Some(_) if name == default => Ok(name.to_string()),
        Some(hub) if kjarni::hub::downloaded(&hub, false, cache).is_some() => Ok(name.to_string()),
        Some(_) => Err(ApiError::hub_model_not_downloaded(name)),
        None => Err(ApiError::model_not_found(name)),
    }
}

/// Runs model work that holds the CPU for a while. Tokio moves its other tasks
/// off this thread first, so the server keeps answering meanwhile.
pub async fn compute<F: Future>(work: F) -> F::Output {
    tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(work))
}

impl Models {
    pub fn new(defaults: Defaults, gpu: bool, prefix_cache: bool) -> Self {
        Self {
            defaults,
            device: if gpu {
                KjarniDevice::Gpu
            } else {
                KjarniDevice::Cpu
            },
            prefix_cache,
            decoders: Pool::new(),
            embedders: Pool::new(),
            classifiers: Pool::new(),
            rerankers: Pool::new(),
            transcribers: Pool::new(),
            translators: Pool::new(),
            summarizers: Pool::new(),
            recognizers: Pool::new(),
        }
    }

    fn gpu(&self) -> bool {
        self.device == KjarniDevice::Gpu
    }

    pub fn loaded(&self) -> Vec<String> {
        let mut names = self.decoders.loaded();
        names.extend(self.embedders.loaded());
        names.extend(self.classifiers.loaded());
        names.extend(self.rerankers.loaded());
        names.extend(self.transcribers.loaded());
        names.extend(self.translators.loaded());
        names.extend(self.summarizers.loaded());
        names.extend(self.recognizers.loaded());
        names
    }

    /// Serves `chat`, loaded by the caller with settings of its own, as
    /// model `name`.
    pub fn insert_chat(&self, name: &str, chat: Chat) -> Arc<LoadedDecoder> {
        self.decoders.insert(
            name,
            LoadedDecoder {
                decoder: Decoder::Chat(Box::new(chat)),
                turn: Arc::new(Semaphore::new(1)),
            },
        )
    }

    pub async fn decoder(&self, name: &str) -> Result<Arc<LoadedDecoder>, ApiError> {
        let device = self.device;
        let prefix_cache = self.prefix_cache;
        self.decoders
            .get(name, || async move {
                let mut chat = Chat::builder(name).device(device).quiet();
                if prefix_cache {
                    chat = chat.prefix_cache(true);
                }
                let decoder = match compute(chat.build()).await {
                    Ok(chat) => Decoder::Chat(Box::new(chat)),
                    // Not a chat model; it can still complete text.
                    Err(chat_error) => {
                        let generator = Generator::builder(name).device(device).quiet();
                        match compute(generator.build()).await {
                            Ok(generator) => Decoder::Raw(Box::new(generator)),
                            Err(_) => return Err(chat_error.to_string()),
                        }
                    }
                };
                Ok(LoadedDecoder {
                    decoder,
                    turn: Arc::new(Semaphore::new(1)),
                })
            })
            .await
    }

    pub async fn embedder(&self, name: &str) -> Result<Arc<Embedder>, ApiError> {
        let gpu = self.gpu();
        self.embedders
            .get(name, || async move {
                let builder = EmbedderBuilder::new(name).quiet(true);
                let builder = if gpu { builder.gpu() } else { builder.cpu() };
                compute(builder.build()).await.map_err(|e| e.to_string())
            })
            .await
    }

    pub async fn classifier(&self, name: &str) -> Result<Arc<Classifier>, ApiError> {
        let gpu = self.gpu();
        self.classifiers
            .get(name, || async move {
                let builder = ClassifierBuilder::new(name).quiet(true);
                let builder = if gpu { builder.gpu() } else { builder.cpu() };
                compute(builder.build()).await.map_err(|e| e.to_string())
            })
            .await
    }

    pub async fn reranker(&self, name: &str) -> Result<Arc<Reranker>, ApiError> {
        let gpu = self.gpu();
        self.rerankers
            .get(name, || async move {
                let builder = RerankerBuilder::new(name).quiet(true);
                let builder = if gpu { builder.gpu() } else { builder.cpu() };
                compute(builder.build()).await.map_err(|e| e.to_string())
            })
            .await
    }

    pub async fn transcriber(&self, name: &str) -> Result<Arc<Transcriber>, ApiError> {
        let gpu = self.gpu();
        self.transcribers
            .get(name, || async move {
                let builder = Transcriber::builder(name).quiet();
                let builder = if gpu { builder.gpu() } else { builder.cpu() };
                compute(builder.build()).await.map_err(|e| e.to_string())
            })
            .await
    }

    pub async fn translator(&self, name: &str) -> Result<Arc<Translator>, ApiError> {
        let device = self.device;
        self.translators
            .get(name, || async move {
                let builder = Translator::builder(name).device(device).quiet();
                compute(builder.build()).await.map_err(|e| e.to_string())
            })
            .await
    }

    pub async fn summarizer(&self, name: &str) -> Result<Arc<Summarizer>, ApiError> {
        let device = self.device;
        self.summarizers
            .get(name, || async move {
                let builder = Summarizer::builder(name).device(device).quiet();
                compute(builder.build()).await.map_err(|e| e.to_string())
            })
            .await
    }

    pub async fn recognizer(&self, name: &str) -> Result<Arc<EntityRecognizer>, ApiError> {
        let device = self.device;
        self.recognizers
            .get(name, || async move {
                let builder = EntityRecognizer::builder(name).device(device).quiet(true);
                compute(builder.build()).await.map_err(|e| e.to_string())
            })
            .await
    }
}

/// The endpoint a registry model is for, as `/v1/models` reports it.
pub fn endpoint_for(task: ModelTask) -> &'static str {
    match task {
        ModelTask::Embedding => "embeddings",
        ModelTask::ReRanking => "rerank",
        ModelTask::Classification
        | ModelTask::SentimentAnalysis
        | ModelTask::ZeroShotClassification => "classify",
        ModelTask::TokenClassification => "ner",
        ModelTask::Chat | ModelTask::Reasoning => "chat",
        ModelTask::Generation => "completions",
        ModelTask::Summarization => "summarize",
        ModelTask::Translation => "translate",
        ModelTask::SpeechToText => "audio",
        // Image search runs through `kjarni image`, not this server.
        ModelTask::ImageEmbedding => "none",
        ModelTask::Seq2Seq | ModelTask::TextToText => "summarize",
    }
}
