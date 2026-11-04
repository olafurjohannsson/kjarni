//! Token classification: a label for every token, as in named-entity recognition.
//!
//! Loads `BertForTokenClassification`, `DistilBertForTokenClassification`,
//! `RobertaForTokenClassification` and similar checkpoints: the same encoder as
//! [`SequenceClassifier`], with the final linear layer applied to each token's
//! hidden state instead of a pooled one. [`TokenClassifier::entities`] groups the
//! tokens into entities the way transformers'
//! `pipeline("ner", aggregation_strategy="simple")` does.

use anyhow::{Result, anyhow};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::gpu::{GpuFrameContext, GpuTensor, GpuTensorPool};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
use kjarni_transformers::models::base::ModelInput;
#[cfg(not(target_arch = "wasm32"))]
use kjarni_transformers::pipeline::EncoderLoader;
use kjarni_transformers::{
    WgpuContext,
    activations::softmax_inplace,
    cpu::encoder::{
        batching::{PaddedBatch, pad},
        classifier::CpuSequenceClassificationHead,
        config::PoolingStrategy,
        traits::{CpuEncoder, GpuEncoder},
    },
    models::ModelType,
    models::base::ModelLoadConfig,
    pipeline::{EncoderModelFactory, EncoderPipeline},
    traits::{Device, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};
use ndarray::Array3;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;
use tokenizers::{Encoding, Tokenizer};

use crate::SequenceClassifier;

#[cfg(test)]
mod tests;

/// A token with its predicted label.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenLabel {
    /// The token as the tokenizer writes it, such as `ĠReykjavík` or `##vík`.
    pub token: String,
    /// The most likely label, such as `B-Person` or `O`.
    pub label: String,
    /// The probability of that label.
    pub score: f32,
    /// Where the token is in the text, in characters (not bytes), end exclusive.
    pub start: usize,
    pub end: usize,
}

/// A run of tokens that make up one entity, such as a person's full name.
#[derive(Debug, Clone, PartialEq)]
pub struct Entity {
    /// The entity type, with any `B-` or `I-` prefix removed: `Person`, `LOC`.
    pub label: String,
    /// The mean probability of the tokens' labels.
    pub score: f32,
    /// The entity's text, exactly as it appears in the input.
    pub text: String,
    /// Where the entity is in the text, in characters (not bytes), end exclusive.
    pub start: usize,
    pub end: usize,
}

/// How tokens become entities, as transformers' `aggregation_strategy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Aggregation {
    /// Each token keeps its own label, and neighbours with the same tag join.
    /// A word split into pieces can come out as two entities.
    Simple,
    /// A word takes the label of its first token, so it is never split.
    #[default]
    First,
    /// A word takes the label of its most confident token.
    Max,
    /// A word takes the label with the highest probability averaged over its
    /// tokens.
    Average,
}

/// A token with its label probabilities, before aggregation.
struct ScoredToken {
    token: String,
    scores: Vec<f32>,
    start: usize,
    end: usize,
    is_subword: bool,
}

/// How transformers decides a token continues the word before it.
///
/// A tokenizer that marks word pieces, as WordPiece does with `##`, says so
/// itself: the token is longer than the text it covers. For the others,
/// byte-level BPE and SentencePiece among them, transformers falls back to a
/// heuristic: a token continues a word unless it starts the text or a space
/// sits right before or at its start. So `Reykjavík.` is one word there.
struct SubwordRule {
    marked: bool,
    unk_id: Option<u32>,
}

impl SubwordRule {
    fn of(tokenizer: &Tokenizer) -> Self {
        use tokenizers::models::ModelWrapper;
        let (marked, unk) = match tokenizer.get_model() {
            ModelWrapper::WordPiece(m) => (
                !m.continuing_subword_prefix.is_empty(),
                Some(m.unk_token.clone()),
            ),
            ModelWrapper::BPE(m) => (
                m.continuing_subword_prefix
                    .as_deref()
                    .is_some_and(|p| !p.is_empty()),
                m.unk_token.clone(),
            ),
            ModelWrapper::WordLevel(m) => (false, Some(m.unk_token.clone())),
            ModelWrapper::Unigram(_) => (false, Some("<unk>".to_string())),
        };
        Self {
            marked,
            unk_id: unk.and_then(|u| tokenizer.token_to_id(&u)),
        }
    }

    fn is_subword(&self, token: &str, text: &[char], start: usize, end: usize) -> bool {
        if self.marked {
            token.chars().count() != end.saturating_sub(start)
        } else {
            start > 0 && !text[start - 1..(start + 1).min(text.len())].contains(&' ')
        }
    }
}

/// An encoder with a label for each token.
pub struct TokenClassifier {
    pipeline: EncoderPipeline,
    tokenizer: Tokenizer,
    labels: Vec<String>,
}

impl EncoderModelFactory for TokenClassifier {
    fn load_config(weights: &ModelWeights) -> Result<Arc<dyn ModelConfig>> {
        <SequenceClassifier as EncoderModelFactory>::load_config(weights)
    }

    fn build_backends(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        load_config: ModelLoadConfig,
        context: Option<&Arc<WgpuContext>>,
        device: Device,
    ) -> Result<(Option<Box<dyn CpuEncoder>>, Option<Box<dyn GpuEncoder>>)> {
        <SequenceClassifier as EncoderModelFactory>::build_backends(
            weights,
            meta,
            layout,
            load_config,
            context,
            device,
        )
    }

    fn build_head(
        weights: &ModelWeights,
        load_config: &ModelLoadConfig,
    ) -> Result<Option<CpuSequenceClassificationHead>> {
        <SequenceClassifier as EncoderModelFactory>::build_head(weights, load_config)
    }

    fn pooling_strategy() -> PoolingStrategy {
        // Unused: no pooling happens, every token keeps its own state.
        PoolingStrategy::Cls
    }

    fn new_from_pipeline(
        pipeline: EncoderPipeline,
        tokenizer: Tokenizer,
        _config: Arc<dyn ModelConfig>,
        _model_type: Option<ModelType>,
    ) -> Self {
        let labels = pipeline.labels().map(|l| l.to_vec()).unwrap_or_default();
        Self {
            pipeline,
            tokenizer,
            labels,
        }
    }
}

impl TokenClassifier {
    /// Load a `*ForTokenClassification` model from a folder.
    ///
    /// Fails for a folder whose `config.json` names a different architecture,
    /// since a sequence classifier's weights would load and give nonsense.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_pretrained(
        model_path: &Path,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        check_architecture(model_path)?;
        let model: Self =
            EncoderLoader::load_from_pretrained(model_path, device, context, load_config, None)?;
        if model.labels.is_empty() {
            return Err(anyhow!(
                "{} has no id2label in config.json, so its token labels have no names",
                model_path.display()
            ));
        }
        Ok(model)
    }

    /// Load a registry model, downloading it first if needed.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<std::path::PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        load_config: Option<ModelLoadConfig>,
    ) -> Result<Self> {
        EncoderLoader::load_from_registry::<Self>(
            model_type,
            cache_dir,
            device,
            context,
            load_config,
        )
        .await
    }

    /// The label names, in the order of the model's outputs.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn max_seq_length(&self) -> usize {
        self.pipeline.max_seq_length()
    }

    pub fn device(&self) -> Device {
        self.pipeline.plan().layers
    }

    /// The label of every token of every text, special tokens left out.
    ///
    /// Text past the model's input limit is cut off, as everywhere else.
    pub async fn tokens(&self, texts: &[&str]) -> Result<Vec<Vec<TokenLabel>>> {
        let scored = self.scored_tokens(texts).await?;
        Ok(scored
            .into_iter()
            .map(|tokens| {
                tokens
                    .into_iter()
                    .map(|t| {
                        let (best, score) = argmax(&t.scores);
                        TokenLabel {
                            token: t.token,
                            label: self.label(best),
                            score,
                            start: t.start,
                            end: t.end,
                        }
                    })
                    .collect()
            })
            .collect())
    }

    /// The entities in each text, `O` (outside any entity) left out.
    ///
    /// `aggregation` decides how tokens become entities, with the meaning
    /// transformers gives it; see [`Aggregation`].
    pub async fn entities(
        &self,
        texts: &[&str],
        aggregation: Aggregation,
    ) -> Result<Vec<Vec<Entity>>> {
        let scored = self.scored_tokens(texts).await?;
        Ok(texts
            .iter()
            .zip(scored)
            .map(|(text, tokens)| {
                let labelled = match aggregation {
                    Aggregation::Simple => tokens
                        .iter()
                        .map(|t| {
                            let (best, score) = argmax(&t.scores);
                            (self.label(best), score, t.start, t.end)
                        })
                        .collect(),
                    _ => words(&tokens)
                        .into_iter()
                        .map(|word| {
                            let (best, score) = aggregate_word(word, aggregation);
                            (
                                self.label(best),
                                score,
                                word[0].start,
                                word[word.len() - 1].end,
                            )
                        })
                        .collect::<Vec<_>>(),
                };
                let labelled: Vec<TokenLabel> = labelled
                    .into_iter()
                    .map(|(label, score, start, end)| TokenLabel {
                        token: String::new(),
                        label,
                        score,
                        start,
                        end,
                    })
                    .collect();
                group_entities(text, &labelled)
            })
            .collect())
    }

    fn label(&self, index: usize) -> String {
        self.labels
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("LABEL_{index}"))
    }

    /// Every ordinary token with its label probabilities and whether it
    /// continues the word before it.
    async fn scored_tokens(&self, texts: &[&str]) -> Result<Vec<Vec<ScoredToken>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let encodings = self
            .tokenizer
            .encode_batch_char_offsets(texts.to_vec(), true)
            .map_err(|e| anyhow!("Tokenization failed: {e}"))?;
        let logits = self.logits(&encodings).await?;
        let subword_rule = SubwordRule::of(&self.tokenizer);

        let mut out = Vec::with_capacity(texts.len());
        for (row, (encoding, text)) in encodings.iter().zip(texts).enumerate() {
            let chars: Vec<char> = text.chars().collect();
            let mut tokens = Vec::new();
            for (i, &id) in encoding.get_ids().iter().enumerate() {
                if encoding.get_special_tokens_mask()[i] == 1
                    || encoding.get_attention_mask()[i] == 0
                {
                    continue;
                }
                let mut scores = logits.slice(ndarray::s![row, i, ..]).to_vec();
                softmax_inplace(&mut scores);
                let (start, end) = encoding.get_offsets()[i];
                let token = self.tokenizer.id_to_token(id).unwrap_or_default();
                let is_subword = if Some(id) == subword_rule.unk_id {
                    false
                } else {
                    subword_rule.is_subword(&token, &chars, start, end)
                };
                tokens.push(ScoredToken {
                    token,
                    scores,
                    start,
                    end,
                    is_subword,
                });
            }
            out.push(tokens);
        }
        Ok(out)
    }

    /// Raw logits, `[batch, tokens, labels]`.
    async fn logits(&self, encodings: &[Encoding]) -> Result<Array3<f32>> {
        let head = self
            .pipeline
            .cpu_head()
            .ok_or_else(|| anyhow!("No classification head available"))?;
        let PaddedBatch {
            input_ids,
            attention_mask,
            token_type_ids,
        } = pad(&encodings.iter().collect::<Vec<_>>());

        if let Some(hidden) = self
            .forward_gpu(&input_ids, &attention_mask, &token_type_ids)
            .await?
        {
            return head.forward_tokens(&hidden);
        }

        let encoder = self
            .pipeline
            .cpu_encoder()
            .ok_or_else(|| anyhow!("No CPU encoder available"))?;
        let embedded =
            self.pipeline
                .embeddings()
                .embed_cpu(&input_ids, Some(&token_type_ids), 0)?;
        let normalized = encoder.embed_norm(&embedded)?;
        let (rows, width) = input_ids.dim();
        let mut buffers = encoder.create_buffers(rows, width);
        let hidden = encoder
            .forward_with_buffers(&normalized, &attention_mask, &mut buffers)?
            .last_hidden_state;
        encoder.recycle_buffers(buffers);
        head.forward_tokens(&hidden)
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-gpu"))]
    async fn forward_gpu(
        &self,
        input_ids: &ndarray::Array2<u32>,
        attention_mask: &ndarray::Array2<f32>,
        token_type_ids: &ndarray::Array2<u32>,
    ) -> Result<Option<Array3<f32>>> {
        let Some(gpu_encoder) = self.pipeline.gpu_encoder() else {
            return Ok(None);
        };
        let context = self
            .pipeline
            .context()
            .ok_or_else(|| anyhow!("GPU context required for GPU encoder"))?;
        let pool: Arc<tokio::sync::Mutex<GpuTensorPool>> = context.get_inference_pool();
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(context, pool_guard);
        let (encoder_cmd, pool_ref) = frame.resources();

        let input_ids_gpu = GpuTensor::from_ndarray(context, input_ids)?;
        let attention_mask_gpu = GpuTensor::from_ndarray(context, attention_mask)?;
        let token_types_gpu = GpuTensor::from_ndarray(context, token_type_ids)?;

        #[allow(
            deprecated,
            reason = "internal caller of an API deprecated without a named replacement"
        )]
        let output = gpu_encoder.forward(
            encoder_cmd,
            pool_ref,
            ModelInput::TokensGpu(&input_ids_gpu),
            &attention_mask_gpu,
            Some(ModelInput::TokensGpu(&token_types_gpu)),
        )?;
        frame.finish();
        Ok(Some(output.last_hidden_state.to_ndarray_3d().await?))
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-gpu")))]
    async fn forward_gpu(
        &self,
        _input_ids: &ndarray::Array2<u32>,
        _attention_mask: &ndarray::Array2<f32>,
        _token_type_ids: &ndarray::Array2<u32>,
    ) -> Result<Option<Array3<f32>>> {
        Ok(None)
    }
}

/// Refuse a folder whose config names an architecture other than token
/// classification. A folder with no `architectures` entry is allowed.
#[cfg(not(target_arch = "wasm32"))]
fn check_architecture(model_path: &Path) -> Result<()> {
    let config_path = model_path.join("config.json");
    let text = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow!("reading {}: {e}", config_path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&text)?;
    let Some(archs) = config.get("architectures").and_then(|a| a.as_array()) else {
        return Ok(());
    };
    let names: Vec<&str> = archs.iter().filter_map(|a| a.as_str()).collect();
    if names.is_empty() || names.iter().any(|a| a.ends_with("ForTokenClassification")) {
        Ok(())
    } else {
        Err(anyhow!(
            "{} is a {}, not a token classification model",
            model_path.display(),
            names.join(", ")
        ))
    }
}

fn argmax(scores: &[f32]) -> (usize, f32) {
    scores
        .iter()
        .copied()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, s)| {
            if s > best.1 { (i, s) } else { best }
        })
}

/// Tokens grouped into words: a token joins the word before it when it is a
/// subword, as in transformers' `aggregate_words`.
fn words(tokens: &[ScoredToken]) -> Vec<&[ScoredToken]> {
    let mut out = Vec::new();
    let mut first = 0;
    for i in 1..tokens.len() {
        if !tokens[i].is_subword {
            out.push(&tokens[first..i]);
            first = i;
        }
    }
    if !tokens.is_empty() {
        out.push(&tokens[first..]);
    }
    out
}

/// One label and score for a word, as transformers' `aggregate_word`.
fn aggregate_word(word: &[ScoredToken], aggregation: Aggregation) -> (usize, f32) {
    match aggregation {
        Aggregation::Simple | Aggregation::First => argmax(&word[0].scores),
        Aggregation::Max => {
            // The first token with the highest top score, as Python's max().
            let best = word.iter().fold(&word[0], |best, t| {
                if argmax(&t.scores).1 > argmax(&best.scores).1 {
                    t
                } else {
                    best
                }
            });
            argmax(&best.scores)
        }
        Aggregation::Average => {
            let n = word[0].scores.len();
            let mean: Vec<f32> = (0..n)
                .map(|j| word.iter().map(|t| t.scores[j]).sum::<f32>() / word.len() as f32)
                .collect();
            argmax(&mean)
        }
    }
}

/// `B-PER` → (`B`, `PER`); `I-PER` → (`I`, `PER`); anything else, `O` included,
/// counts as inside an entity of its own name, as in transformers' `get_tag`.
fn split_tag(label: &str) -> (&str, &str) {
    if let Some(tag) = label.strip_prefix("B-") {
        ("B", tag)
    } else if let Some(tag) = label.strip_prefix("I-") {
        ("I", tag)
    } else {
        ("I", label)
    }
}

/// Merge consecutive tokens into entities, as transformers'
/// `aggregation_strategy="simple"`: a token joins the previous entity when its
/// tag matches and it is not a `B-` tag. The entity's score is the mean of its
/// tokens', and its span runs from the first token's start to the last's end.
fn group_entities(text: &str, tokens: &[TokenLabel]) -> Vec<Entity> {
    let mut groups: Vec<&[TokenLabel]> = Vec::new();
    let mut first = 0;
    for i in 1..tokens.len() {
        let (bi, tag) = split_tag(&tokens[i].label);
        let (_, last_tag) = split_tag(&tokens[i - 1].label);
        if !(tag == last_tag && bi != "B") {
            groups.push(&tokens[first..i]);
            first = i;
        }
    }
    if !tokens.is_empty() {
        groups.push(&tokens[first..]);
    }

    groups
        .into_iter()
        .filter_map(|group| {
            let head = &group[0];
            let label = head
                .label
                .split_once('-')
                .map_or(head.label.as_str(), |(_, t)| t);
            if head.label == "O" || label == "O" {
                return None;
            }
            let start = head.start;
            let end = group[group.len() - 1].end;
            let score = group.iter().map(|t| t.score).sum::<f32>() / group.len() as f32;
            Some(Entity {
                label: label.to_string(),
                score,
                text: text.chars().skip(start).take(end - start).collect(),
                start,
                end,
            })
        })
        .collect()
}
