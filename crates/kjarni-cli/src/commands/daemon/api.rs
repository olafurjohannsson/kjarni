//! The daemon's own routes, under `/api`, and the sign-in page.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use futures::Stream;
use kjarni::agent::{Approval, Memory};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::broadcast::error::RecvError;

use super::auth::{COOKIE, Guard};
use super::conversations::{Conversations, Event, Live, SendError};
use super::schedules::{self, Notify, Schedule, Spec, StartError, Store};

pub struct DaemonState {
    pub conversations: Conversations,
    pub guard: Arc<Guard>,
    pub memory: Option<Arc<Memory>>,
    pub model: String,
    pub gpu: bool,
    pub schedules: Arc<Store>,
}

type AppState = Arc<DaemonState>;

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message.into() } })),
    )
        .into_response()
}

fn internal(e: anyhow::Error) -> Response {
    error(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))
}

fn not_found(what: &str) -> Response {
    error(StatusCode::NOT_FOUND, format!("there is no {what}"))
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/login", get(login).post(login_with_token))
        .route("/api/status", get(status))
        .route("/api/login-code", post(login_code))
        .route(
            "/api/conversations",
            get(list_conversations).post(create_conversation),
        )
        .route(
            "/api/conversations/{id}",
            get(get_conversation).delete(delete_conversation),
        )
        .route("/api/conversations/{id}/messages", post(send_message))
        .route("/api/conversations/{id}/events", get(events))
        .route(
            "/api/conversations/{id}/approvals/{approval}",
            post(answer_approval),
        )
        .route("/api/conversations/{id}/stop", post(stop))
        .route("/api/memory", get(list_memory).post(remember))
        .route("/api/memory/{id}", delete(forget))
        .route("/api/schedules", get(list_schedules).post(create_schedule))
        .route(
            "/api/schedules/{id}",
            put(update_schedule).delete(delete_schedule),
        )
        .route("/api/schedules/{id}/run", post(run_schedule))
        .route("/api/notifications", put(set_notifications))
        .route("/api/notifications/test", post(test_notification))
}

async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let setup = state.conversations.setup();
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "model": state.model,
        "gpu": state.gpu,
        "workdir": setup.workdir,
        "memory": state.memory.is_some(),
    }))
}

/// A one-time code that signs a browser in at `/login?code=...`.
async fn login_code(State(state): State<AppState>, headers: HeaderMap) -> Json<serde_json::Value> {
    let code = state.guard.new_login_code();
    let host = headers
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("127.0.0.1");
    Json(json!({
        "code": code,
        "url": format!("http://{host}/login?code={code}"),
    }))
}

#[derive(Deserialize)]
struct LoginQuery {
    code: String,
}

async fn login(State(state): State<AppState>, Query(q): Query<LoginQuery>) -> Response {
    if !state.guard.redeem(&q.code) {
        return error(
            StatusCode::UNAUTHORIZED,
            "this sign-in link has expired or was used already; run `kjarni daemon open` for a new one",
        );
    }
    (
        [(header::SET_COOKIE, sign_in_cookie(&state.guard))],
        Redirect::to("/"),
    )
        .into_response()
}

/// The cookie that signs a browser in. It lasts a year: the browser stays
/// signed in until the token changes.
fn sign_in_cookie(guard: &Guard) -> String {
    format!(
        "{COOKIE}={}; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000",
        guard.token()
    )
}

#[derive(Deserialize)]
struct TokenBody {
    token: String,
}

/// Signs in a browser on another device, where `kjarni daemon open` cannot
/// run, with the token the user copied from the daemon's token file.
async fn login_with_token(State(state): State<AppState>, Json(body): Json<TokenBody>) -> Response {
    if !state.guard.is_token(body.token.trim()) {
        return error(StatusCode::UNAUTHORIZED, "that is not the daemon's token");
    }
    (
        StatusCode::NO_CONTENT,
        [(header::SET_COOKIE, sign_in_cookie(&state.guard))],
    )
        .into_response()
}

async fn list_conversations(State(state): State<AppState>) -> Response {
    match state.conversations.list() {
        Ok(list) => {
            let items: Vec<_> = list
                .iter()
                .map(|s| {
                    json!({
                        "id": s.id,
                        "title": s.title,
                        "modified": s.modified,
                        "busy": state.conversations.busy(&s.id),
                    })
                })
                .collect();
            Json(json!({ "conversations": items })).into_response()
        }
        Err(e) => internal(e),
    }
}

async fn create_conversation(State(state): State<AppState>) -> Response {
    match state.conversations.create() {
        Ok(live) => (StatusCode::CREATED, Json(json!({ "id": live.id }))).into_response(),
        Err(e) => internal(e),
    }
}

/// The error is boxed: a `Response` is large to pass back by value.
fn find(state: &DaemonState, id: &str) -> Result<Arc<Live>, Box<Response>> {
    match state.conversations.get(id) {
        Ok(Some(live)) => Ok(live),
        Ok(None) => Err(Box::new(not_found("such conversation"))),
        Err(e) => Err(Box::new(internal(e))),
    }
}

async fn get_conversation(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let live = match find(&state, &id) {
        Ok(live) => live,
        Err(r) => return *r,
    };
    let snap = live.snapshot();
    Json(json!({
        "id": live.id,
        "busy": snap.busy,
        "summary": snap.transcript["summary"],
        "messages": snap.transcript["messages"],
        "events": snap.events,
        "pending_approval": snap.pending_approval,
    }))
    .into_response()
}

async fn delete_conversation(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.conversations.delete(&id) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => not_found("such conversation"),
        Err(e) => error(StatusCode::CONFLICT, format!("{e:#}")),
    }
}

#[derive(Deserialize)]
struct SendBody {
    text: String,
}

async fn send_message(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<SendBody>,
) -> Response {
    let live = match find(&state, &id) {
        Ok(live) => live,
        Err(r) => return *r,
    };
    match live.send(&body.text) {
        Ok(()) => StatusCode::ACCEPTED.into_response(),
        Err(SendError::Busy) => error(
            StatusCode::CONFLICT,
            "the conversation is still answering the last message",
        ),
        Err(SendError::Empty) => error(StatusCode::BAD_REQUEST, "the message is empty"),
    }
}

#[derive(Deserialize)]
struct EventsQuery {
    after: Option<u64>,
}

/// The conversation's events as server-sent events: those of the current
/// turn after `after` (or the `Last-Event-ID` a reconnecting browser sends),
/// then new ones as they happen.
async fn events(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<EventsQuery>,
    headers: HeaderMap,
) -> Response {
    let live = match find(&state, &id) {
        Ok(live) => live,
        Err(r) => return *r,
    };
    let after = q.after.or_else(|| {
        headers
            .get("last-event-id")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
    });
    let (past, rx) = live.follow(after.unwrap_or(0));
    Sse::new(event_stream(past, rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn sse(event: &Event) -> SseEvent {
    SseEvent::default()
        .id(event.seq.to_string())
        .json_data(event)
        .expect("events serialize")
}

fn event_stream(
    past: Vec<Event>,
    rx: tokio::sync::broadcast::Receiver<Event>,
) -> impl Stream<Item = Result<SseEvent, Infallible>> {
    let last = past.last().map_or(0, |e| e.seq);
    let past = futures::stream::iter(past.into_iter().map(|e| Ok(sse(&e))));
    let live = futures::stream::unfold((rx, last), |(mut rx, last)| async move {
        loop {
            match rx.recv().await {
                // Already sent from the log.
                Ok(e) if e.seq <= last => continue,
                Ok(e) => return Some((Ok(sse(&e)), (rx, e.seq))),
                // A client too slow to keep up reconnects and reads the log.
                Err(RecvError::Lagged(_)) | Err(RecvError::Closed) => return None,
            }
        }
    });
    futures::StreamExt::chain(past, live)
}

#[derive(Deserialize)]
struct ApprovalBody {
    decision: String,
}

async fn answer_approval(
    State(state): State<AppState>,
    Path((id, approval)): Path<(String, u64)>,
    Json(body): Json<ApprovalBody>,
) -> Response {
    let live = match find(&state, &id) {
        Ok(live) => live,
        Err(r) => return *r,
    };
    let decision = match body.decision.as_str() {
        "once" => Approval::Once,
        "always" => Approval::Always,
        "deny" => Approval::Deny,
        other => {
            return error(
                StatusCode::BAD_REQUEST,
                format!("decision must be once, always or deny, not '{other}'"),
            );
        }
    };
    if live.answer(approval, decision) {
        StatusCode::NO_CONTENT.into_response()
    } else {
        not_found("such approval waiting")
    }
}

async fn stop(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    let live = match find(&state, &id) {
        Ok(live) => live,
        Err(r) => return *r,
    };
    if live.stop() {
        StatusCode::NO_CONTENT.into_response()
    } else {
        error(StatusCode::CONFLICT, "the conversation is not answering")
    }
}

fn memory(state: &DaemonState) -> Result<&Memory, Box<Response>> {
    state
        .memory
        .as_deref()
        .ok_or_else(|| Box::new(error(StatusCode::NOT_FOUND, "memory is off (--no-memory)")))
}

async fn list_memory(State(state): State<AppState>) -> Response {
    let memory = match memory(&state) {
        Ok(m) => m,
        Err(r) => return *r,
    };
    let facts: Vec<_> = memory
        .entries()
        .into_iter()
        .map(|e| json!({ "id": e.id, "text": e.text, "created": e.created }))
        .collect();
    Json(json!({ "facts": facts })).into_response()
}

#[derive(Deserialize)]
struct RememberBody {
    text: String,
}

async fn remember(State(state): State<AppState>, Json(body): Json<RememberBody>) -> Response {
    let memory = match memory(&state) {
        Ok(m) => m,
        Err(r) => return *r,
    };
    match memory.remember(&body.text).await {
        Ok(e) => (
            StatusCode::CREATED,
            Json(json!({ "id": e.id, "text": e.text, "created": e.created })),
        )
            .into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

async fn forget(State(state): State<AppState>, Path(id): Path<u64>) -> Response {
    let memory = match memory(&state) {
        Ok(m) => m,
        Err(r) => return *r,
    };
    match memory.forget(id) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => not_found("such fact"),
        Err(e) => internal(e),
    }
}

/// A schedule as clients see it: its settings, when it runs next and
/// whether it is running now.
fn schedule_json(store: &Store, s: &Schedule) -> serde_json::Value {
    let mut value = serde_json::to_value(s).expect("schedules serialize");
    value["next_run"] = json!(s.next_run(&chrono::Local).timestamp());
    value["running"] = json!(store.is_running(&s.id));
    value
}

async fn list_schedules(State(state): State<AppState>) -> Json<serde_json::Value> {
    let store = &state.schedules;
    let list: Vec<_> = store
        .list()
        .iter()
        .map(|s| schedule_json(store, s))
        .collect();
    // The token stays in the daemon; clients only learn there is one.
    let notify = store
        .notify()
        .map(|n| json!({ "url": n.url, "has_token": n.token.is_some(), "open_url": n.open_url }));
    Json(json!({
        "schedules": list,
        "notify": notify,
        "tools": schedules::ASKING_TOOLS,
    }))
}

async fn create_schedule(State(state): State<AppState>, Json(spec): Json<Spec>) -> Response {
    let store = &state.schedules;
    match store.create(spec, chrono::Utc::now().timestamp()) {
        Ok(s) => (StatusCode::CREATED, Json(schedule_json(store, &s))).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

async fn update_schedule(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(spec): Json<Spec>,
) -> Response {
    let store = &state.schedules;
    match store.update(&id, spec) {
        Ok(Some(s)) => Json(schedule_json(store, &s)).into_response(),
        Ok(None) => not_found("such schedule"),
        Err(e) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

async fn delete_schedule(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.schedules.delete(&id) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => not_found("such schedule"),
        Err(e) => internal(e),
    }
}

/// Runs a schedule now. The answer names the run's conversation, which
/// the client can follow like any other.
async fn run_schedule(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match schedules::start(&state, &id) {
        Ok(conversation) => (
            StatusCode::ACCEPTED,
            Json(json!({ "conversation": conversation })),
        )
            .into_response(),
        Err(StartError::NotFound) => not_found("such schedule"),
        Err(StartError::Running) => error(StatusCode::CONFLICT, "that schedule is running now"),
        Err(StartError::Failed(e)) => internal(e),
    }
}

#[derive(Deserialize)]
struct NotifyBody {
    /// Empty turns notifications off.
    url: String,
    /// Absent keeps the saved token while the address stays the same;
    /// empty removes it.
    token: Option<String>,
    open_url: Option<String>,
}

async fn set_notifications(
    State(state): State<AppState>,
    Json(body): Json<NotifyBody>,
) -> Response {
    let store = &state.schedules;
    let url = body.url.trim().to_string();
    if url.is_empty() {
        return match store.set_notify(None) {
            Ok(()) => StatusCode::NO_CONTENT.into_response(),
            Err(e) => internal(e),
        };
    }
    if let Err(e) = schedules::split_topic(&url) {
        return error(StatusCode::BAD_REQUEST, format!("{e:#}"));
    }
    let open_url = body
        .open_url
        .map(|u| u.trim().trim_end_matches('/').to_string())
        .filter(|u| !u.is_empty());
    if let Some(u) = &open_url
        && let Err(e) = schedules::check_open_url(u)
    {
        return error(StatusCode::BAD_REQUEST, format!("{e:#}"));
    }
    let token = match body.token {
        Some(t) if t.trim().is_empty() => None,
        Some(t) => Some(t.trim().to_string()),
        None => store
            .notify()
            .filter(|old| old.url == url)
            .and_then(|old| old.token),
    };
    match store.set_notify(Some(Notify {
        url,
        token,
        open_url,
    })) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => internal(e),
    }
}

async fn test_notification(State(state): State<AppState>) -> Response {
    let Some(notify) = state.schedules.notify() else {
        return error(StatusCode::BAD_REQUEST, "notifications are not set up");
    };
    match schedules::send_notification(
        &notify,
        "Kjarni",
        "Notifications from Kjarni reach this device.",
        None,
    )
    .await
    {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => error(StatusCode::BAD_GATEWAY, format!("{e:#}")),
    }
}
