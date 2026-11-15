//! The encoder and sequence-to-sequence endpoints: `/v1/embeddings` in
//! OpenAI's format, `/v1/rerank` in Cohere's and Jina's, and Kjarni's own
//! `/v1/classify`, `/v1/translate`, `/v1/summarize` and `/v1/ner`.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use base64::Engine;
use kjarni::ModelType;
use kjarni::classifier::{ClassificationResult, ZeroShotOptions};
use kjarni::seq2seq::Seq2SeqOverrides;
use serde::Deserialize;
use serde_json::{Value, json};

use super::AppState;
use super::error::{ApiError, ApiJson};
use super::models::{compute, endpoint_for, resolve};

/// One text or several, as every task endpoint takes them.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Input {
    One(String),
    Many(Vec<String>),
    Other(serde::de::IgnoredAny),
}

impl Input {
    pub fn texts(self) -> Result<Vec<String>, ApiError> {
        match self {
            Input::One(text) => Ok(vec![text]),
            Input::Many(texts) if texts.is_empty() => {
                Err(ApiError::invalid("input", "input must not be empty"))
            }
            Input::Many(texts) => Ok(texts),
            Input::Other(_) => Err(ApiError::invalid(
                "input",
                "input must be a string or an array of strings",
            )),
        }
    }
}

fn refs(texts: &[String]) -> Vec<&str> {
    texts.iter().map(String::as_str).collect()
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: Input,
    pub encoding_format: Option<String>,
    pub dimensions: Option<usize>,
}

pub async fn embeddings(
    State(app): State<Arc<AppState>>,
    ApiJson(request): ApiJson<EmbeddingRequest>,
) -> Result<Json<Value>, ApiError> {
    let base64 = match request.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return Err(ApiError::invalid(
                "encoding_format",
                format!("encoding_format must be 'float' or 'base64', not '{other}'"),
            ));
        }
    };
    let texts = request.input.texts()?;
    let model = resolve(request.model.as_deref(), &app.models.defaults.embed)?;
    let embedder = app.models.embedder(&model).await?;
    if let Some(dimensions) = request.dimensions
        && dimensions != embedder.dimension()
    {
        return Err(ApiError::unsupported(
            "dimensions",
            format!(
                "'{model}' makes {}-dimensional embeddings and can't shorten them",
                embedder.dimension()
            ),
        ));
    }
    let tokens: usize = embedder
        .count_tokens(&refs(&texts))
        .map_err(ApiError::internal)?
        .iter()
        .sum();
    let vectors = compute(embedder.embed_batch(&refs(&texts)))
        .await
        .map_err(ApiError::internal)?;

    let data: Vec<Value> = vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            let embedding = if base64 {
                let bytes: Vec<u8> = vector.iter().flat_map(|x| x.to_le_bytes()).collect();
                json!(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                json!(vector)
            };
            json!({ "object": "embedding", "index": index, "embedding": embedding })
        })
        .collect();
    Ok(Json(json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": { "prompt_tokens": tokens, "total_tokens": tokens },
    })))
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Document {
    Text(String),
    Object { text: String },
}

#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<Document>,
    pub top_n: Option<usize>,
    pub return_documents: Option<bool>,
}

pub async fn rerank(
    State(app): State<Arc<AppState>>,
    ApiJson(request): ApiJson<RerankRequest>,
) -> Result<Json<Value>, ApiError> {
    if request.documents.is_empty() {
        return Err(ApiError::invalid(
            "documents",
            "documents must not be empty",
        ));
    }
    let documents: Vec<String> = request
        .documents
        .into_iter()
        .map(|d| match d {
            Document::Text(text) | Document::Object { text } => text,
        })
        .collect();
    let model = resolve(request.model.as_deref(), &app.models.defaults.rerank)?;
    let reranker = app.models.reranker(&model).await?;
    let mut ranked = compute(reranker.rerank(&request.query, &refs(&documents)))
        .await
        .map_err(ApiError::internal)?;
    if let Some(n) = request.top_n {
        ranked.truncate(n);
    }
    let return_documents = request.return_documents.unwrap_or(true);
    let results: Vec<Value> = ranked
        .iter()
        .map(|r| {
            // Cohere's and Jina's scores run 0 to 1, so clients threshold on
            // that scale; the logit is the cross-encoder's raw output.
            let mut result = json!({
                "index": r.index,
                "relevance_score": r.probability(),
                "logit": r.score,
            });
            if return_documents {
                result["document"] = json!({ "text": r.document });
            }
            result
        })
        .collect();
    Ok(Json(json!({
        "id": super::generation::new_id("rerank"),
        "model": model,
        "results": results,
    })))
}

#[derive(Debug, Deserialize)]
pub struct ClassifyRequest {
    pub model: Option<String>,
    pub input: Input,
    /// Candidate labels for zero-shot classification. Without them, the
    /// model's own labels are used.
    pub labels: Option<Vec<String>>,
    pub multi_label: Option<bool>,
    pub hypothesis_template: Option<String>,
    pub top_k: Option<usize>,
}

fn classification(index: usize, result: &ClassificationResult, top_k: Option<usize>) -> Value {
    let k = top_k.unwrap_or(result.all_scores.len());
    let scores: Vec<Value> = result
        .all_scores
        .iter()
        .take(k)
        .map(|(label, score)| json!({ "label": label, "score": score }))
        .collect();
    json!({ "index": index, "label": result.label, "score": result.score, "scores": scores })
}

pub async fn classify(
    State(app): State<Arc<AppState>>,
    ApiJson(request): ApiJson<ClassifyRequest>,
) -> Result<Json<Value>, ApiError> {
    let texts = request.input.texts()?;
    let results = match &request.labels {
        Some(labels) => {
            if labels.is_empty() {
                return Err(ApiError::invalid("labels", "labels must not be empty"));
            }
            let model = resolve(request.model.as_deref(), &app.models.defaults.zero_shot)?;
            let classifier = app.models.classifier(&model).await?;
            let mut options =
                ZeroShotOptions::default().multi_label(request.multi_label.unwrap_or(false));
            if let Some(template) = &request.hypothesis_template {
                options = options.hypothesis_template(template.clone());
            }
            let results =
                compute(classifier.zero_shot_batch(&refs(&texts), &refs(labels), &options))
                    .await
                    .map_err(|e| ApiError::invalid("labels", e.to_string()))?;
            (model, results)
        }
        None => {
            if request.multi_label.is_some() || request.hypothesis_template.is_some() {
                return Err(ApiError::invalid(
                    "labels",
                    "multi_label and hypothesis_template are for zero-shot classification, which needs labels",
                ));
            }
            let model = resolve(request.model.as_deref(), &app.models.defaults.classify)?;
            let classifier = app.models.classifier(&model).await?;
            let results = compute(classifier.classify_batch(&refs(&texts)))
                .await
                .map_err(ApiError::internal)?;
            (model, results)
        }
    };
    let (model, results) = results;
    let data: Vec<Value> = results
        .iter()
        .enumerate()
        .map(|(i, r)| classification(i, r, request.top_k))
        .collect();
    Ok(Json(
        json!({ "object": "list", "model": model, "data": data }),
    ))
}

#[derive(Debug, Deserialize)]
pub struct TranslateRequest {
    pub model: Option<String>,
    pub input: Input,
    #[serde(alias = "source", alias = "source_language")]
    pub from: Option<String>,
    #[serde(alias = "target", alias = "target_language")]
    pub to: Option<String>,
}

pub async fn translate(
    State(app): State<Arc<AppState>>,
    ApiJson(request): ApiJson<TranslateRequest>,
) -> Result<Json<Value>, ApiError> {
    let texts = request.input.texts()?;
    let (Some(from), Some(to)) = (request.from, request.to) else {
        return Err(ApiError::invalid(
            "from",
            "give both languages as 'from' and 'to', such as \"en\" and \"de\"",
        ));
    };
    let default = match &app.models.defaults.translate {
        Some(model) => model.clone(),
        None => kjarni::translator::default_model(&from, &to).to_string(),
    };
    let model = resolve(request.model.as_deref(), &default)?;
    let translator = app.models.translator(&model).await?;
    let mut data = Vec::with_capacity(texts.len());
    for (index, text) in texts.iter().enumerate() {
        let translated = compute(translator.translate(text, &from, &to))
            .await
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        data.push(json!({ "index": index, "text": translated }));
    }
    Ok(Json(
        json!({ "object": "list", "model": model, "data": data }),
    ))
}

#[derive(Debug, Deserialize)]
pub struct SummarizeRequest {
    pub model: Option<String>,
    pub input: Input,
    pub max_length: Option<usize>,
    pub min_length: Option<usize>,
}

pub async fn summarize(
    State(app): State<Arc<AppState>>,
    ApiJson(request): ApiJson<SummarizeRequest>,
) -> Result<Json<Value>, ApiError> {
    let texts = request.input.texts()?;
    let model = resolve(request.model.as_deref(), &app.models.defaults.summarize)?;
    let summarizer = app.models.summarizer(&model).await?;
    let overrides = Seq2SeqOverrides {
        max_length: request.max_length,
        min_length: request.min_length,
        ..Default::default()
    };
    let mut data = Vec::with_capacity(texts.len());
    for (index, text) in texts.iter().enumerate() {
        let summary = compute(summarizer.summarize_with_config(text, &overrides))
            .await
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        data.push(json!({ "index": index, "text": summary }));
    }
    Ok(Json(
        json!({ "object": "list", "model": model, "data": data }),
    ))
}

#[derive(Debug, Deserialize)]
pub struct NerRequest {
    pub model: Option<String>,
    pub input: Input,
}

pub async fn ner(
    State(app): State<Arc<AppState>>,
    ApiJson(request): ApiJson<NerRequest>,
) -> Result<Json<Value>, ApiError> {
    let texts = request.input.texts()?;
    let model = resolve(request.model.as_deref(), &app.models.defaults.ner)?;
    let recognizer = app.models.recognizer(&model).await?;
    let found = compute(recognizer.recognize_batch(&refs(&texts)))
        .await
        .map_err(ApiError::internal)?;
    let data: Vec<Value> = found
        .iter()
        .enumerate()
        .map(|(index, entities)| {
            let entities: Vec<Value> = entities
                .iter()
                .map(|e| {
                    json!({
                        "label": e.label,
                        "text": e.text,
                        "score": e.score,
                        "start": e.start,
                        "end": e.end,
                    })
                })
                .collect();
            json!({ "index": index, "entities": entities })
        })
        .collect();
    Ok(Json(
        json!({ "object": "list", "model": model, "data": data }),
    ))
}

fn model_entry(model: ModelType, loaded: &[String]) -> Value {
    let name = model.cli_name();
    json!({
        "id": name,
        "object": "model",
        "created": 0,
        "owned_by": "kjarni",
        "endpoint": endpoint_for(model.info().task),
        "loaded": loaded.iter().any(|l| l == name),
    })
}

pub async fn list_models(State(app): State<Arc<AppState>>) -> Json<Value> {
    let loaded = app.models.loaded();
    let data: Vec<Value> = ModelType::all().map(|m| model_entry(m, &loaded)).collect();
    Json(json!({ "object": "list", "data": data }))
}

pub async fn get_model(
    State(app): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let model = ModelType::from_cli_name(&id).ok_or_else(|| ApiError::model_not_found(&id))?;
    Ok(Json(model_entry(model, &app.models.loaded())))
}
