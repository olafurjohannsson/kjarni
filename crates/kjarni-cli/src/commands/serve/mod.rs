//! `kjarni serve`: Kjarni over HTTP.
//!
//! Chat, completions, embeddings and speech-to-text follow OpenAI's API, so
//! its client libraries and the tools built on them work by changing the base
//! URL. Reranking follows Cohere's and Jina's format. Classification,
//! translation, summarization and entity recognition have no common standard
//! and use Kjarni's own, in the same style.
//!
//! Each model loads on the first request that names it and then stays loaded.

mod audio;
mod error;
mod generation;
mod models;
mod stop;
mod tasks;
#[cfg(test)]
mod tests;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{HeaderMap, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use tower_http::cors::CorsLayer;

use error::ApiError;
pub use models::{Defaults, LoadedDecoder, Models, compute};

/// Large enough for long recordings: OpenAI's own limit is 25 MB.
const MAX_BODY_BYTES: usize = 256 * 1024 * 1024;

pub struct AppState {
    pub models: Models,
}

pub struct ServeOptions {
    pub host: String,
    pub port: u16,
    pub defaults: Defaults,
    /// Models to load before taking requests, so the first ones aren't slow.
    pub preload: Vec<(Endpoint, String)>,
    pub api_key: Option<String>,
    pub cors: bool,
    pub gpu: bool,
    pub prefix_cache: bool,
    pub quiet: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Endpoint {
    Chat,
    Embed,
    Classify,
    Rerank,
    Transcribe,
    Translate,
    Summarize,
    Ner,
}

pub struct RouterOptions {
    pub api_key: Option<String>,
    pub cors: bool,
    pub log: bool,
}

pub fn router(state: Arc<AppState>, options: RouterOptions) -> Router {
    let v1 = Router::new()
        .route("/models", get(tasks::list_models))
        .route("/models/{id}", get(tasks::get_model))
        .route("/chat/completions", post(generation::chat_completions))
        .route("/completions", post(generation::completions))
        .route("/embeddings", post(tasks::embeddings))
        .route("/audio/transcriptions", post(audio::transcriptions))
        .route("/audio/translations", post(audio::translations))
        .route("/rerank", post(tasks::rerank))
        .route("/classify", post(tasks::classify))
        .route("/translate", post(tasks::translate))
        .route("/summarize", post(tasks::summarize))
        .route("/ner", post(tasks::ner))
        .fallback(|| async {
            ApiError {
                status: axum::http::StatusCode::NOT_FOUND,
                ..ApiError::bad_request("no such endpoint")
            }
        })
        .route_layer(middleware::from_fn_with_state(
            Arc::new(options.api_key),
            require_key,
        ));

    let mut app = Router::new()
        .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
        .nest("/v1", v1)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state);
    if options.cors {
        app = app.layer(CorsLayer::permissive());
    }
    if options.log {
        app = app.layer(middleware::from_fn(log_request));
    }
    app
}

async fn require_key(
    State(key): State<Arc<Option<String>>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    if let Some(key) = key.as_deref() {
        let sent = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or_default();
        if !same(sent.as_bytes(), key.as_bytes()) {
            return ApiError::unauthorized().into_response();
        }
    }
    next.run(request).await
}

/// Compares in time that depends only on the lengths, so a wrong key's
/// response time says nothing about how much of it was right.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn log_request(request: Request, next: Next) -> Response {
    let (method, path) = (request.method().clone(), request.uri().path().to_string());
    let started = Instant::now();
    let response = next.run(request).await;
    eprintln!(
        "{method} {path} {} {:.2?}",
        response.status().as_u16(),
        started.elapsed()
    );
    response
}

pub async fn run(options: ServeOptions) -> Result<()> {
    let models = Models::new(options.defaults, options.gpu, options.prefix_cache);
    for (endpoint, name) in &options.preload {
        if !options.quiet {
            eprintln!("Loading {name}...");
        }
        let name = models::resolve(Some(name), name).map_err(|e| anyhow::anyhow!(e.message))?;
        let loaded = match endpoint {
            Endpoint::Chat => models.decoder(&name).await.map(drop),
            Endpoint::Embed => models.embedder(&name).await.map(drop),
            Endpoint::Classify => models.classifier(&name).await.map(drop),
            Endpoint::Rerank => models.reranker(&name).await.map(drop),
            Endpoint::Transcribe => models.transcriber(&name).await.map(drop),
            Endpoint::Translate => models.translator(&name).await.map(drop),
            Endpoint::Summarize => models.summarizer(&name).await.map(drop),
            Endpoint::Ner => models.recognizer(&name).await.map(drop),
        };
        loaded.map_err(|e| anyhow::anyhow!(e.message))?;
    }

    let state = Arc::new(AppState { models });
    let app = router(
        state,
        RouterOptions {
            api_key: options.api_key,
            cors: options.cors,
            log: !options.quiet,
        },
    );
    let address: SocketAddr = format!("{}:{}", options.host, options.port)
        .parse()
        .with_context(|| format!("'{}' is not an IP address to listen on", options.host))?;
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .with_context(|| format!("could not listen on {address}"))?;
    if !options.quiet {
        eprintln!("Kjarni is serving an OpenAI-compatible API at http://{address}/v1");
        eprintln!("Models load on their first request. Press Ctrl+C to stop.");
    }
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("the server stopped with an error")
}
