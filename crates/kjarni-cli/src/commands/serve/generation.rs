//! `/v1/chat/completions` and `/v1/completions`: text generation in OpenAI's
//! format, streamed as server-sent events or returned whole.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt};
use kjarni::agent::{ToolCall, ToolSpec, displayable_prefix_len, parse_tool_calls};
use kjarni::chat::Message;
use kjarni::generation::GenerationOverrides;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::AppState;
use super::error::{ApiError, ApiJson};
use super::models::{LoadedDecoder, resolve};
use super::stop::{StopFilter, Trim};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    fn into_vec(self) -> Vec<String> {
        match self {
            Stop::One(s) => vec![s],
            Stop::Many(v) => v,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// The sampling fields chat and completion requests share. `top_k`, `min_p`
/// and `repetition_penalty` are not OpenAI's, but vLLM and llama.cpp accept
/// them too.
#[derive(Debug, Default, Deserialize)]
pub struct Sampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub n: Option<usize>,
    pub logprobs: Option<Value>,
    pub top_logprobs: Option<Value>,
}

impl Sampling {
    pub fn overrides(&self) -> Result<GenerationOverrides, ApiError> {
        if self.n.is_some_and(|n| n != 1) {
            return Err(ApiError::unsupported("n", "only n = 1 is supported"));
        }
        if self
            .logprobs
            .as_ref()
            .is_some_and(|v| !v.is_null() && v != &json!(false))
            || self.top_logprobs.as_ref().is_some_and(|v| !v.is_null())
        {
            return Err(ApiError::unsupported(
                "logprobs",
                "log probabilities are not supported yet",
            ));
        }
        let mut overrides = GenerationOverrides::default();
        match self.temperature {
            Some(t) if !(0.0..=2.0).contains(&t) => {
                return Err(ApiError::invalid(
                    "temperature",
                    "temperature must be between 0 and 2",
                ));
            }
            // OpenAI's 0 is the most deterministic setting there is: greedy.
            Some(0.0) => overrides.do_sample = Some(false),
            Some(t) => {
                overrides.do_sample = Some(true);
                overrides.temperature = Some(t);
            }
            None => {}
        }
        if let Some(p) = self.top_p {
            if !(p > 0.0 && p <= 1.0) {
                return Err(ApiError::invalid(
                    "top_p",
                    "top_p must be above 0 and at most 1",
                ));
            }
            overrides.top_p = Some(p);
        }
        overrides.top_k = self.top_k;
        overrides.min_p = self.min_p;
        overrides.repetition_penalty = self.repetition_penalty;
        match self.max_completion_tokens.or(self.max_tokens) {
            Some(0) => {
                return Err(ApiError::invalid(
                    "max_tokens",
                    "max_tokens must be at least 1",
                ));
            }
            max => overrides.max_new_tokens = max,
        }
        Ok(overrides)
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<InMessage>,
    #[serde(flatten)]
    pub sampling: Sampling,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub stop: Option<Stop>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub response_format: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct InMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Content>,
    /// The calls an earlier assistant message made.
    #[serde(default)]
    pub tool_calls: Vec<InToolCall>,
}

#[derive(Debug, Deserialize)]
pub struct InToolCall {
    pub function: InFunction,
}

#[derive(Debug, Deserialize)]
pub struct InFunction {
    pub name: String,
    /// A JSON object, or a string holding one, as OpenAI sends it.
    #[serde(default)]
    pub arguments: Value,
}

impl InToolCall {
    fn into_call(self) -> Result<ToolCall, ApiError> {
        let arguments = match self.function.arguments {
            Value::Null => serde_json::Map::new(),
            Value::Object(map) => map,
            Value::String(s) if s.trim().is_empty() => serde_json::Map::new(),
            Value::String(s) => match serde_json::from_str(&s) {
                Ok(Value::Object(map)) => map,
                _ => {
                    return Err(ApiError::invalid(
                        "messages",
                        format!(
                            "the arguments of the call to '{}' are not a JSON object",
                            self.function.name
                        ),
                    ));
                }
            },
            _ => {
                return Err(ApiError::invalid(
                    "messages",
                    format!(
                        "the arguments of the call to '{}' are not a JSON object",
                        self.function.name
                    ),
                ));
            }
        };
        Ok(ToolCall {
            name: self.function.name,
            arguments,
        })
    }
}

/// OpenAI's `tools` as the definitions a template renders. Only functions
/// exist; each needs a name.
pub fn tool_specs(tools: &[Value]) -> Result<Vec<ToolSpec>, ApiError> {
    tools
        .iter()
        .map(|tool| {
            if tool.get("type").and_then(Value::as_str) != Some("function") {
                return Err(ApiError::unsupported(
                    "tools",
                    "only tools of type 'function' are supported",
                ));
            }
            let function = tool.get("function").unwrap_or(&Value::Null);
            let Some(name) = function.get("name").and_then(Value::as_str) else {
                return Err(ApiError::invalid("tools", "every function needs a name"));
            };
            let parameters = match function.get("parameters") {
                None | Some(Value::Null) => json!({"type": "object", "properties": {}}),
                Some(p) => p.clone(),
            };
            Ok(ToolSpec::new(
                name,
                function
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                parameters,
            ))
        })
        .collect()
}

/// The tools to offer under `tool_choice`: all of them for `auto` or none
/// given, none for `none`. Forcing a call can't be guaranteed yet.
pub fn offered_tools(
    tools: Option<&[Value]>,
    tool_choice: Option<&Value>,
) -> Result<Vec<ToolSpec>, ApiError> {
    match tool_choice {
        None => {}
        Some(c) if c == "auto" => {}
        Some(c) if c == "none" => return Ok(Vec::new()),
        Some(_) => {
            return Err(ApiError::unsupported(
                "tool_choice",
                "only tool_choice \"auto\" and \"none\" are supported; a call can't be forced yet",
            ));
        }
    }
    tool_specs(tools.unwrap_or_default())
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Debug, Deserialize)]
pub struct Part {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
}

/// OpenAI's messages as Kjarni's. Roles and content Kjarni can't use yet are
/// refused, never dropped: a model answering without the image it was sent
/// would answer a different question. An assistant message's calls are written
/// into its text by `format_call`, as the model writes them.
pub fn messages(
    input: Vec<InMessage>,
    format_call: &dyn Fn(&ToolCall) -> String,
) -> Result<Vec<Message>, ApiError> {
    if input.is_empty() {
        return Err(ApiError::invalid("messages", "messages must not be empty"));
    }
    input
        .into_iter()
        .map(|m| {
            let content = match m.content {
                None => String::new(),
                Some(Content::Text(text)) => text,
                Some(Content::Parts(parts)) => {
                    let mut text = String::new();
                    for part in parts {
                        match (part.kind.as_str(), part.text) {
                            ("text", Some(t)) => text.push_str(&t),
                            (kind, _) => {
                                return Err(ApiError::unsupported(
                                    "messages",
                                    format!("message content of type '{kind}' is not supported; only text is"),
                                ));
                            }
                        }
                    }
                    text
                }
            };
            match m.role.as_str() {
                "system" | "developer" => Ok(Message::system(content)),
                "user" => Ok(Message::user(content)),
                "assistant" => {
                    let mut text = content;
                    for (i, call) in m.tool_calls.into_iter().enumerate() {
                        let call = call.into_call()?;
                        if i > 0 {
                            text.push('\n');
                        } else if !text.trim().is_empty() {
                            text.push_str("\n\n");
                        }
                        text.push_str(&format_call(&call));
                    }
                    Ok(Message::assistant(text))
                }
                "tool" => Ok(Message::tool(content)),
                "function" => Err(ApiError::unsupported(
                    "messages",
                    "the legacy 'function' role is not supported; send tool results with role 'tool'",
                )),
                other => Err(ApiError::invalid(
                    "messages",
                    format!("unknown message role '{other}'"),
                )),
            }
        })
        .collect()
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: Prompt,
    #[serde(flatten)]
    pub sampling: Sampling,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub stop: Option<Stop>,
    #[serde(default)]
    pub echo: bool,
    pub suffix: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    One(String),
    Many(Vec<String>),
    Tokens(serde::de::IgnoredAny),
}

/// What a generation run sends back to the request handler.
#[derive(Debug, PartialEq)]
pub enum Piece {
    Text(String),
    Done {
        finish_reason: &'static str,
        completion_tokens: usize,
    },
    Failed(String),
}

/// A generation run that has started: the prompt was accepted and tokens are
/// on their way.
pub struct Run {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub prompt_tokens: usize,
    pub pieces: mpsc::Receiver<Piece>,
}

pub fn new_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{nanos:x}{n:04x}")
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Starts generating `prompt`, after checking it fits the model's context.
///
/// The run holds the model's turn until it ends, and ends early when the
/// receiver is dropped, which is what a client disconnecting does.
pub async fn start(
    decoder: Arc<LoadedDecoder>,
    model: String,
    prompt: String,
    overrides: GenerationOverrides,
    stops: Vec<String>,
    trim: bool,
) -> Result<Run, ApiError> {
    let generator = decoder.generator();
    let prompt_tokens = generator
        .count_tokens(&prompt)
        .map_err(ApiError::internal)?;
    let context = generator.context_size();
    if prompt_tokens >= context {
        return Err(ApiError {
            code: Some("context_length_exceeded"),
            param: Some("messages"),
            ..ApiError::bad_request(format!(
                "the prompt is {prompt_tokens} tokens and '{model}' takes at most {context}"
            ))
        });
    }
    let max_new_tokens = generator.max_new_tokens(&overrides);
    // Chat templates end the reply with their own markers, such as
    // `<|im_end|>`; cut there if the model writes one as plain text.
    let mut stops = stops;
    if decoder.chat().is_some() {
        stops.extend(generator.decoder().stop_sequences());
    }
    let opening = decoder
        .chat()
        .map(|chat| chat.reply_prefix(&prompt).to_string())
        .unwrap_or_default();

    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        let Ok(_turn) = decoder.turn.clone().acquire_owned().await else {
            return;
        };
        let generator = decoder.generator();
        let mut tokens = match generator.stream_with_config(&prompt, overrides).await {
            Ok(tokens) => tokens,
            Err(e) => {
                let _ = tx.send(Piece::Failed(e.to_string())).await;
                return;
            }
        };
        let mut filter = StopFilter::new(stops);
        let mut trimmer = trim.then(Trim::default);
        if !opening.is_empty() && !send_text(&tx, opening, &mut trimmer).await {
            return;
        }
        let mut count = 0;
        while let Some(token) = tokens.next().await {
            let token = match token {
                Ok(token) => token,
                Err(e) => {
                    let _ = tx.send(Piece::Failed(e.to_string())).await;
                    return;
                }
            };
            count += 1;
            if token.is_special {
                continue;
            }
            let (text, stopped) = filter.push(&token.text);
            if !send_text(&tx, text, &mut trimmer).await {
                return;
            }
            if stopped {
                let _ = tx
                    .send(Piece::Done {
                        finish_reason: "stop",
                        completion_tokens: count,
                    })
                    .await;
                return;
            }
        }
        if !send_text(&tx, filter.finish(), &mut trimmer).await {
            return;
        }
        let finish_reason = if max_new_tokens.is_some_and(|max| count >= max) {
            "length"
        } else {
            "stop"
        };
        let _ = tx
            .send(Piece::Done {
                finish_reason,
                completion_tokens: count,
            })
            .await;
    });

    Ok(Run {
        id: String::new(),
        created: unix_now(),
        model,
        prompt_tokens,
        pieces: rx,
    })
}

/// Sends text on, trimmed when the run trims. False once the receiver is gone.
async fn send_text(tx: &mpsc::Sender<Piece>, text: String, trimmer: &mut Option<Trim>) -> bool {
    let text = match trimmer {
        Some(trimmer) => trimmer.push(&text),
        None => text,
    };
    text.is_empty() || tx.send(Piece::Text(text)).await.is_ok()
}

/// Waits for a whole run: its text, finish reason and token count.
pub async fn collect(run: &mut Run) -> Result<(String, &'static str, usize), ApiError> {
    let mut text = String::new();
    while let Some(piece) = run.pieces.recv().await {
        match piece {
            Piece::Text(t) => text.push_str(&t),
            Piece::Done {
                finish_reason,
                completion_tokens,
            } => {
                return Ok((text, finish_reason, completion_tokens));
            }
            Piece::Failed(message) => return Err(ApiError::internal(message)),
        }
    }
    Err(ApiError::internal("generation ended without finishing"))
}

fn usage(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

/// The server-sent events for a run: one per chunk `to_chunk` builds from a
/// piece of text or the finish, an optional usage chunk, then `[DONE]`.
fn events<F>(
    mut run: Run,
    include_usage: bool,
    to_chunk: F,
) -> impl Stream<Item = Result<Event, Infallible>>
where
    F: Fn(&Run, Option<&str>, Option<&str>) -> Value + Send + 'static,
{
    async_stream_events(move |tx| async move {
        while let Some(piece) = run.pieces.recv().await {
            match piece {
                Piece::Text(text) => {
                    if tx.send(to_chunk(&run, Some(&text), None)).await.is_err() {
                        return;
                    }
                }
                Piece::Done {
                    finish_reason,
                    completion_tokens,
                } => {
                    let _ = tx.send(to_chunk(&run, None, Some(finish_reason))).await;
                    if include_usage {
                        let mut chunk = to_chunk(&run, None, None);
                        chunk["choices"] = json!([]);
                        chunk["usage"] = usage(run.prompt_tokens, completion_tokens);
                        let _ = tx.send(chunk).await;
                    }
                    break;
                }
                Piece::Failed(message) => {
                    let _ = tx.send(ApiError::internal(message).body()).await;
                    break;
                }
            }
        }
    })
}

/// A stream of JSON events fed by `produce`, ending with OpenAI's `[DONE]`.
fn async_stream_events<P, Fut>(produce: P) -> impl Stream<Item = Result<Event, Infallible>>
where
    P: FnOnce(mpsc::Sender<Value>) -> Fut,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<Value>(64);
    tokio::spawn(produce(tx));
    let json_events = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|value| (Ok(Event::default().data(value.to_string())), rx))
    });
    json_events.chain(futures::stream::once(async {
        Ok(Event::default().data("[DONE]"))
    }))
}

pub async fn chat_completions(
    State(app): State<Arc<AppState>>,
    ApiJson(request): ApiJson<ChatRequest>,
) -> Result<Response, ApiError> {
    let tools = offered_tools(request.tools.as_deref(), request.tool_choice.as_ref())?;
    if let Some(format) = &request.response_format
        && format.get("type").and_then(Value::as_str) != Some("text")
    {
        return Err(ApiError::unsupported(
            "response_format",
            "only response_format {\"type\": \"text\"} is supported; JSON output can't be guaranteed yet",
        ));
    }
    let overrides = request.sampling.overrides()?;
    let model = resolve(request.model.as_deref(), &app.models.defaults.chat)?;
    let decoder = app.models.decoder(&model).await?;
    let Some(chat) = decoder.chat() else {
        return Err(ApiError::invalid(
            "model",
            format!("'{model}' has no chat template; use /v1/completions with it instead"),
        ));
    };
    if !tools.is_empty() && !chat.supports_tools() {
        return Err(ApiError::invalid(
            "tools",
            format!(
                "'{model}' has no tool-calling format; use a Qwen 3.5 model, such as qwen3.5-4b"
            ),
        ));
    }
    let messages = messages(request.messages, &|call| chat.format_tool_call(call))?;
    let prompt = chat
        .render_messages_with_tools(&messages, &tools)
        .map_err(|e| ApiError::invalid("messages", e.to_string()))?;
    let stops = request.stop.map(Stop::into_vec).unwrap_or_default();
    let mut run = start(decoder.clone(), model, prompt, overrides, stops, true).await?;
    run.id = new_id("chatcmpl");

    if request.stream {
        let include_usage = request.stream_options.unwrap_or_default().include_usage;
        let stream = chat_events(run, include_usage, tools);
        return Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let (text, finish_reason, completion_tokens) = collect(&mut run).await?;
    let reply = parse_tool_calls(&text, &tools);
    let (message, finish_reason) = if tools.is_empty() || reply.tool_calls.is_empty() {
        (
            json!({ "role": "assistant", "content": text, "refusal": null }),
            finish_reason,
        )
    } else {
        let content = (!reply.content.is_empty()).then_some(reply.content);
        (
            json!({
                "role": "assistant",
                "content": content,
                "refusal": null,
                "tool_calls": reply.tool_calls.iter().map(|c| tool_call_json(c, None)).collect::<Vec<_>>(),
            }),
            "tool_calls",
        )
    };
    Ok(Json(json!({
        "id": run.id,
        "object": "chat.completion",
        "created": run.created,
        "model": run.model,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": null,
            "finish_reason": finish_reason,
        }],
        "usage": usage(run.prompt_tokens, completion_tokens),
    }))
    .into_response())
}

/// A call in OpenAI's shape, with `index` when it is a streamed delta.
fn tool_call_json(call: &ToolCall, index: Option<usize>) -> Value {
    let mut v = json!({
        "id": new_id("call"),
        "type": "function",
        "function": { "name": call.name, "arguments": call.arguments_json() },
    });
    if let Some(i) = index {
        v["index"] = json!(i);
    }
    v
}

/// The server-sent events of a chat run. Text streams as it comes; when tools
/// were offered, text from a `<tool_call>` on is held back, read as calls at
/// the end, and sent as `tool_calls` deltas.
fn chat_events(
    mut run: Run,
    include_usage: bool,
    tools: Vec<ToolSpec>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream_events(move |tx| async move {
        let chunk = |run: &Run, delta: Value, finish: Option<&str>| {
            json!({
                "id": run.id,
                "object": "chat.completion.chunk",
                "created": run.created,
                "model": run.model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "logprobs": null,
                    "finish_reason": finish,
                }],
            })
        };
        if tx
            .send(chunk(
                &run,
                json!({ "role": "assistant", "content": "" }),
                None,
            ))
            .await
            .is_err()
        {
            return;
        }
        let mut text = String::new();
        let mut shown = 0;
        while let Some(piece) = run.pieces.recv().await {
            match piece {
                Piece::Text(t) => {
                    text.push_str(&t);
                    let showable = if tools.is_empty() {
                        text.len()
                    } else {
                        displayable_prefix_len(&text)
                    };
                    if showable > shown {
                        let delta = json!({ "content": &text[shown..showable] });
                        shown = showable;
                        if tx.send(chunk(&run, delta, None)).await.is_err() {
                            return;
                        }
                    }
                }
                Piece::Done {
                    finish_reason,
                    completion_tokens,
                } => {
                    let calls = if tools.is_empty() {
                        Vec::new()
                    } else {
                        parse_tool_calls(&text, &tools).tool_calls
                    };
                    let finish = if calls.is_empty() {
                        if shown < text.len() {
                            let delta = json!({ "content": &text[shown..] });
                            let _ = tx.send(chunk(&run, delta, None)).await;
                        }
                        finish_reason
                    } else {
                        for (i, call) in calls.iter().enumerate() {
                            let delta = json!({ "tool_calls": [tool_call_json(call, Some(i))] });
                            let _ = tx.send(chunk(&run, delta, None)).await;
                        }
                        "tool_calls"
                    };
                    let _ = tx.send(chunk(&run, json!({}), Some(finish))).await;
                    if include_usage {
                        let mut last = chunk(&run, json!({}), None);
                        last["choices"] = json!([]);
                        last["usage"] = usage(run.prompt_tokens, completion_tokens);
                        let _ = tx.send(last).await;
                    }
                    break;
                }
                Piece::Failed(message) => {
                    let _ = tx.send(ApiError::internal(message).body()).await;
                    break;
                }
            }
        }
    })
}

pub async fn completions(
    State(app): State<Arc<AppState>>,
    ApiJson(request): ApiJson<CompletionRequest>,
) -> Result<Response, ApiError> {
    if request.echo {
        return Err(ApiError::unsupported("echo", "echo is not supported"));
    }
    if request.suffix.is_some() {
        return Err(ApiError::unsupported("suffix", "suffix is not supported"));
    }
    let prompt = match request.prompt {
        Prompt::One(p) => p,
        Prompt::Many(mut many) if many.len() == 1 => many.remove(0),
        Prompt::Many(_) => {
            return Err(ApiError::unsupported(
                "prompt",
                "send one prompt per request",
            ));
        }
        Prompt::Tokens(_) => {
            return Err(ApiError::unsupported(
                "prompt",
                "token-id prompts are not supported; send text",
            ));
        }
    };
    let overrides = request.sampling.overrides()?;
    let model = resolve(request.model.as_deref(), &app.models.defaults.chat)?;
    let decoder = app.models.decoder(&model).await?;
    let stops = request.stop.map(Stop::into_vec).unwrap_or_default();
    let mut run = start(decoder, model, prompt, overrides, stops, false).await?;
    run.id = new_id("cmpl");

    let chunk = |run: &Run, text: Option<&str>, finish: Option<&str>| {
        json!({
            "id": run.id,
            "object": "text_completion",
            "created": run.created,
            "model": run.model,
            "choices": [{
                "index": 0,
                "text": text.unwrap_or(""),
                "logprobs": null,
                "finish_reason": finish,
            }],
        })
    };
    if request.stream {
        let include_usage = request.stream_options.unwrap_or_default().include_usage;
        let stream = events(run, include_usage, chunk);
        return Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let (text, finish_reason, completion_tokens) = collect(&mut run).await?;
    let mut body = chunk(&run, Some(&text), Some(finish_reason));
    body["usage"] = usage(run.prompt_tokens, completion_tokens);
    Ok(Json(body).into_response())
}
