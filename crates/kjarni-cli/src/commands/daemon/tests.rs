//! The daemon's API end to end, with a scripted model in place of a real one.

use std::collections::VecDeque;
use std::sync::Mutex;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use futures::StreamExt;
use serde_json::{Value as Json, json};
use tower::ServiceExt;

use super::*;

const TOKEN: &str = "0123456789abcdef0123456789abcdef";

/// Replies from a list, in order, each streamed in a few pieces. A reply of
/// `None` says "Working" and never finishes.
struct Scripted(Mutex<VecDeque<Option<String>>>);

impl Scripted {
    fn new(replies: &[Option<&str>]) -> Arc<Self> {
        Arc::new(Self(Mutex::new(
            replies.iter().map(|r| r.map(str::to_string)).collect(),
        )))
    }
}

#[async_trait]
impl AgentModel for Scripted {
    fn render(&self, _: &Conversation) -> Result<String> {
        Ok(String::new())
    }
    async fn stream(&self, _: String) -> Result<TextStream> {
        let next = self
            .0
            .lock()
            .unwrap()
            .pop_front()
            .expect("a scripted reply");
        Ok(match next {
            Some(text) => {
                let pieces: Vec<Result<String>> = text
                    .split_inclusive(' ')
                    .map(|p| Ok(p.to_string()))
                    .collect();
                Box::pin(futures::stream::iter(pieces))
            }
            None => Box::pin(
                futures::stream::iter([Ok("Working".to_string())])
                    .chain(futures::stream::pending()),
            ),
        })
    }
    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(text.len() / 4)
    }
    fn context_size(&self) -> usize {
        32_768
    }
    fn name(&self) -> String {
        "scripted".into()
    }
}

struct Daemon {
    app: Router,
    workdir: tempfile::TempDir,
    _data: tempfile::TempDir,
}

fn daemon(model: Arc<dyn AgentModel>, any_host: bool) -> Daemon {
    let workdir = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    let memory = Arc::new(Memory::open(data.path(), None).unwrap());
    let state = Arc::new(DaemonState {
        conversations: Conversations::new(
            AgentSetup {
                model,
                memory: Some(memory.clone()),
                workdir: workdir.path().to_path_buf(),
                max_steps: 4,
                approve_all: false,
            },
            data.path().join("sessions"),
        ),
        guard: Arc::new(Guard::new(TOKEN.into(), any_host)),
        memory: Some(memory),
        model: "scripted".into(),
        gpu: false,
    });
    let serve_state = Arc::new(serve::AppState {
        models: serve::Models::new(serve::Defaults::default(), false, false),
    });
    Daemon {
        app: router(state, serve_state, false),
        workdir,
        _data: data,
    }
}

fn request(method: &str, path: &str) -> axum::http::request::Builder {
    Request::builder()
        .method(method)
        .uri(path)
        .header(header::HOST, "127.0.0.1:7710")
}

impl Daemon {
    async fn call(&self, req: Request<Body>) -> (StatusCode, Json) {
        let response = self.app.clone().oneshot(req).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json = serde_json::from_slice(&bytes).unwrap_or(Json::Null);
        (status, json)
    }

    /// An authorized request with an optional JSON body.
    async fn api(&self, method: &str, path: &str, body: Option<Json>) -> (StatusCode, Json) {
        let builder =
            request(method, path).header(header::AUTHORIZATION, format!("Bearer {TOKEN}"));
        let req = match body {
            Some(b) => builder
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(b.to_string())),
            None => builder.body(Body::empty()),
        };
        self.call(req.unwrap()).await
    }

    async fn new_conversation(&self) -> String {
        let (status, body) = self.api("POST", "/api/conversations", None).await;
        assert_eq!(status, StatusCode::CREATED);
        body["id"].as_str().unwrap().to_string()
    }

    /// Reads the conversation's events after `after` until `stop` says to.
    async fn follow(&self, id: &str, after: u64, mut stop: impl FnMut(&Json) -> bool) -> Vec<Json> {
        let req = request(
            "GET",
            &format!("/api/conversations/{id}/events?after={after}"),
        )
        .header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
        .body(Body::empty())
        .unwrap();
        let response = self.app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body().into_data_stream();
        let mut buffer = String::new();
        let mut events = Vec::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let chunk = tokio::time::timeout_at(deadline, body.next())
                .await
                .expect("events before the timeout")
                .expect("the stream stays open")
                .unwrap();
            buffer.push_str(std::str::from_utf8(&chunk).unwrap());
            while let Some((frame, rest)) = buffer.split_once("\n\n") {
                let frame = frame.to_string();
                buffer = rest.to_string();
                let Some(data) = frame.lines().find_map(|l| l.strip_prefix("data: ")) else {
                    continue;
                };
                let event: Json = serde_json::from_str(data).unwrap();
                let done = stop(&event);
                events.push(event);
                if done {
                    return events;
                }
            }
        }
    }
}

fn ends_turn(e: &Json) -> bool {
    matches!(e["type"].as_str(), Some("done" | "error" | "stopped"))
}

fn types(events: &[Json]) -> Vec<&str> {
    events
        .iter()
        .map(|e| e["type"].as_str().unwrap())
        .filter(|t| *t != "text")
        .collect()
}

#[tokio::test]
async fn the_token_is_required_and_other_hosts_and_origins_are_refused() {
    let d = daemon(Scripted::new(&[]), false);
    let no_token = request("GET", "/api/status").body(Body::empty()).unwrap();
    assert_eq!(d.call(no_token).await.0, StatusCode::UNAUTHORIZED);
    let wrong = request("GET", "/api/status")
        .header(header::AUTHORIZATION, "Bearer nope")
        .body(Body::empty())
        .unwrap();
    assert_eq!(d.call(wrong).await.0, StatusCode::UNAUTHORIZED);
    let (status, body) = d.api("GET", "/api/status", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["model"], "scripted");

    // The OpenAI-compatible routes sit behind the same token.
    let v1 = request("GET", "/v1/models").body(Body::empty()).unwrap();
    assert_eq!(d.call(v1).await.0, StatusCode::UNAUTHORIZED);
    assert_eq!(d.api("GET", "/v1/models", None).await.0, StatusCode::OK);

    // A page on another domain pointed at 127.0.0.1 (DNS rebinding).
    let rebound = Request::builder()
        .uri("/api/status")
        .header(header::HOST, "evil.example:7710")
        .header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
        .body(Body::empty())
        .unwrap();
    assert_eq!(d.call(rebound).await.0, StatusCode::FORBIDDEN);

    // A page on another origin, even with the cookie the browser attaches.
    let cross = request("POST", "/api/conversations")
        .header(header::ORIGIN, "https://evil.example")
        .header(header::COOKIE, format!("{}={TOKEN}", auth::COOKIE))
        .body(Body::empty())
        .unwrap();
    assert_eq!(d.call(cross).await.0, StatusCode::FORBIDDEN);
    let same = request("GET", "/api/status")
        .header(header::ORIGIN, "http://127.0.0.1:7710")
        .header(header::COOKIE, format!("{}={TOKEN}", auth::COOKIE))
        .body(Body::empty())
        .unwrap();
    assert_eq!(d.call(same).await.0, StatusCode::OK);

    // Listening on a network address accepts any host, still with the token.
    let open = daemon(Scripted::new(&[]), true);
    let remote = Request::builder()
        .uri("/api/status")
        .header(header::HOST, "192.168.1.5:7710")
        .header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
        .body(Body::empty())
        .unwrap();
    assert_eq!(open.call(remote).await.0, StatusCode::OK);
}

#[tokio::test]
async fn a_login_code_signs_a_browser_in_once() {
    let d = daemon(Scripted::new(&[]), false);
    let (status, body) = d.api("POST", "/api/login-code", None).await;
    assert_eq!(status, StatusCode::OK);
    let code = body["code"].as_str().unwrap();
    assert_eq!(
        body["url"],
        format!("http://127.0.0.1:7710/login?code={code}")
    );

    let login = || {
        request("GET", &format!("/login?code={code}"))
            .body(Body::empty())
            .unwrap()
    };
    let response = d.app.clone().oneshot(login()).await.unwrap();
    assert_eq!(response.status(), StatusCode::SEE_OTHER);
    let cookie = response.headers()[header::SET_COOKIE].to_str().unwrap();
    assert_eq!(
        cookie,
        format!("kjarni_token={TOKEN}; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000")
    );
    assert_eq!(
        d.call(login()).await.0,
        StatusCode::UNAUTHORIZED,
        "used twice"
    );

    let bad = request("GET", "/login?code=guess")
        .body(Body::empty())
        .unwrap();
    assert_eq!(d.call(bad).await.0, StatusCode::UNAUTHORIZED);
}

#[test]
fn loopback_hosts() {
    for host in [
        "127.0.0.1",
        "127.0.0.1:7710",
        "localhost:80",
        "LOCALHOST",
        "[::1]",
        "[::1]:7710",
    ] {
        assert!(auth::is_loopback_host(host), "{host}");
    }
    for host in [
        "",
        "evil.example",
        "127.0.0.1.evil.example",
        "localhost.evil.example:7710",
        "[::1]x",
        "127.0.0.2",
        "localhost:abc",
    ] {
        assert!(!auth::is_loopback_host(host), "{host}");
    }
}

#[test]
fn the_token_file_is_made_once_and_private() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("daemon/token");
    let token = auth::load_or_create_token(&path).unwrap();
    assert_eq!(token.len(), 64);
    assert!(token.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(auth::load_or_create_token(&path).unwrap(), token);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}

#[tokio::test]
async fn a_message_streams_its_reply_and_is_saved() {
    let d = daemon(
        Scripted::new(&[Some("<think>easy</think>Hello there.")]),
        false,
    );
    let id = d.new_conversation().await;
    let (status, _) = d
        .api(
            "POST",
            &format!("/api/conversations/{id}/messages"),
            Some(json!({"text": "Hi"})),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED);

    let events = d.follow(&id, 0, ends_turn).await;
    assert_eq!(types(&events), ["user", "done"]);
    assert_eq!(events[0]["text"], "Hi");
    let streamed: String = events.iter().filter_map(|e| e["delta"].as_str()).collect();
    assert_eq!(streamed, "<think>easy</think>Hello there.");
    let done = events.last().unwrap();
    assert_eq!(done["text"], "Hello there.");
    let seqs: Vec<u64> = events.iter().map(|e| e["seq"].as_u64().unwrap()).collect();
    assert!(seqs.windows(2).all(|w| w[0] < w[1]));

    // Reconnecting after an event replays only what came after it.
    let replay = d.follow(&id, seqs[1], ends_turn).await;
    assert_eq!(replay.first().unwrap()["seq"].as_u64(), Some(seqs[2]));

    let (_, conversation) = d
        .api("GET", &format!("/api/conversations/{id}"), None)
        .await;
    assert_eq!(conversation["busy"], false);
    assert_eq!(conversation["events"], json!([]));
    assert_eq!(
        conversation["messages"],
        json!([
            {"role": "user", "text": "Hi"},
            {"role": "assistant", "text": "Hello there.", "reasoning": "easy", "tool_calls": []},
        ])
    );
    let (_, list) = d.api("GET", "/api/conversations", None).await;
    assert_eq!(list["conversations"][0]["id"], id.as_str());
    assert_eq!(list["conversations"][0]["title"], "Hi");
}

const WRITE_CALL: &str = "I'll save it.\n<tool_call>\n<function=write_file>\n<parameter=path>\nnote.txt\n</parameter>\n<parameter=content>\nhello\n</parameter>\n</function>\n</tool_call>";

#[tokio::test]
async fn a_write_waits_for_approval_from_the_client() {
    let d = daemon(
        Scripted::new(&[
            Some(WRITE_CALL),
            Some("Saved."),
            Some(WRITE_CALL),
            Some("Left it."),
        ]),
        false,
    );
    let id = d.new_conversation().await;
    d.api(
        "POST",
        &format!("/api/conversations/{id}/messages"),
        Some(json!({"text": "Save hello"})),
    )
    .await;
    let events = d.follow(&id, 0, |e| e["type"] == "approval").await;
    assert_eq!(types(&events), ["user", "tool_call", "approval"]);
    let call = events.iter().find(|e| e["type"] == "tool_call").unwrap();
    assert_eq!(call["name"], "write_file");
    assert_eq!(call["access"], "write");
    assert_eq!(
        call["arguments"],
        json!({"path": "note.txt", "content": "hello"})
    );
    let approval = events.last().unwrap()["approval"].as_u64().unwrap();
    assert!(!d.workdir.path().join("note.txt").exists());

    // Opening the conversation now shows the turn so far and what it waits for.
    let (_, open) = d
        .api("GET", &format!("/api/conversations/{id}"), None)
        .await;
    assert_eq!(open["busy"], true);
    assert_eq!(open["pending_approval"], approval);
    assert_eq!(open["messages"], json!([]));
    assert_eq!(open["events"].as_array().unwrap().len(), events.len());

    // A second message while it waits is refused; so is a wrong approval.
    let (busy, _) = d
        .api(
            "POST",
            &format!("/api/conversations/{id}/messages"),
            Some(json!({"text": "and?"})),
        )
        .await;
    assert_eq!(busy, StatusCode::CONFLICT);
    let path = format!("/api/conversations/{id}/approvals/{}", approval + 100);
    assert_eq!(
        d.api("POST", &path, Some(json!({"decision": "once"})))
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    let path = format!("/api/conversations/{id}/approvals/{approval}");
    assert_eq!(
        d.api("POST", &path, Some(json!({"decision": "maybe"})))
            .await
            .0,
        StatusCode::BAD_REQUEST
    );

    assert_eq!(
        d.api("POST", &path, Some(json!({"decision": "once"})))
            .await
            .0,
        StatusCode::NO_CONTENT
    );
    let last = events.last().unwrap()["seq"].as_u64().unwrap();
    let rest = d.follow(&id, last, ends_turn).await;
    assert_eq!(types(&rest), ["approved", "tool_result", "done"]);
    assert_eq!(rest[0]["decision"], "once");
    assert_eq!(rest[1]["ok"], true);
    assert_eq!(rest.last().unwrap()["text"], "Saved.");
    assert_eq!(
        std::fs::read_to_string(d.workdir.path().join("note.txt")).unwrap(),
        "hello"
    );

    // Declined, the call does not run and the model is told.
    std::fs::remove_file(d.workdir.path().join("note.txt")).unwrap();
    d.api(
        "POST",
        &format!("/api/conversations/{id}/messages"),
        Some(json!({"text": "Again"})),
    )
    .await;
    let events = d.follow(&id, 0, |e| e["type"] == "approval").await;
    let approval = events.last().unwrap()["approval"].as_u64().unwrap();
    let path = format!("/api/conversations/{id}/approvals/{approval}");
    d.api("POST", &path, Some(json!({"decision": "deny"})))
        .await;
    let rest = d.follow(&id, approval, ends_turn).await;
    let result = rest.iter().find(|e| e["type"] == "tool_result").unwrap();
    assert_eq!(
        (result["ok"].as_bool(), result["denied"].as_bool()),
        (Some(false), Some(true))
    );
    assert!(!d.workdir.path().join("note.txt").exists());

    let (_, conversation) = d
        .api("GET", &format!("/api/conversations/{id}"), None)
        .await;
    let messages = conversation["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 8);
    assert_eq!(messages[1]["tool_calls"][0]["name"], "write_file");
    assert_eq!(messages[1]["text"], "I'll save it.");
    assert_eq!(messages[2]["role"], "tool");
}

#[tokio::test]
async fn stopping_a_turn_frees_the_conversation() {
    let d = daemon(Scripted::new(&[None, Some("Back.")]), false);
    let id = d.new_conversation().await;
    let stop = format!("/api/conversations/{id}/stop");
    assert_eq!(
        d.api("POST", &stop, None).await.0,
        StatusCode::CONFLICT,
        "nothing to stop"
    );

    d.api(
        "POST",
        &format!("/api/conversations/{id}/messages"),
        Some(json!({"text": "Go"})),
    )
    .await;
    // Once the model is generating.
    d.follow(&id, 0, |e| e["type"] == "text").await;
    assert_eq!(d.api("POST", &stop, None).await.0, StatusCode::NO_CONTENT);
    let events = d.follow(&id, 0, ends_turn).await;
    assert_eq!(types(&events), ["user", "stopped"]);

    // The stopped message stays in the conversation, and the next one runs.
    let (status, _) = d
        .api(
            "POST",
            &format!("/api/conversations/{id}/messages"),
            Some(json!({"text": "Again"})),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let events = d.follow(&id, 0, ends_turn).await;
    assert_eq!(events.last().unwrap()["text"], "Back.");
    let (_, conversation) = d
        .api("GET", &format!("/api/conversations/{id}"), None)
        .await;
    let users: Vec<&str> = conversation["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["role"] == "user")
        .map(|m| m["text"].as_str().unwrap())
        .collect();
    assert_eq!(users, ["Go", "Again"]);
}

#[tokio::test]
async fn conversations_are_found_by_id_and_deleted() {
    let d = daemon(Scripted::new(&[Some("Hello.")]), false);
    let id = d.new_conversation().await;
    for bad in ["nope", "..%2F..%2Fetc%2Fpasswd", "a.b"] {
        let (status, _) = d
            .api("GET", &format!("/api/conversations/{bad}"), None)
            .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{bad}");
    }
    let (empty, _) = d
        .api(
            "POST",
            &format!("/api/conversations/{id}/messages"),
            Some(json!({"text": "  "})),
        )
        .await;
    assert_eq!(empty, StatusCode::BAD_REQUEST);

    let path = format!("/api/conversations/{id}");
    assert_eq!(d.api("DELETE", &path, None).await.0, StatusCode::NO_CONTENT);
    assert_eq!(d.api("GET", &path, None).await.0, StatusCode::NOT_FOUND);
    assert_eq!(d.api("DELETE", &path, None).await.0, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn memory_can_be_listed_added_to_and_forgotten() {
    let d = daemon(Scripted::new(&[]), false);
    let (status, fact) = d
        .api(
            "POST",
            "/api/memory",
            Some(json!({"text": "Prefers tabs."})),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED);
    let id = fact["id"].as_u64().unwrap();
    let (_, list) = d.api("GET", "/api/memory", None).await;
    assert_eq!(list["facts"][0]["text"], "Prefers tabs.");
    assert_eq!(
        d.api("POST", "/api/memory", Some(json!({"text": " "})))
            .await
            .0,
        StatusCode::BAD_REQUEST
    );

    assert_eq!(
        d.api("DELETE", &format!("/api/memory/{id}"), None).await.0,
        StatusCode::NO_CONTENT
    );
    assert_eq!(
        d.api("DELETE", &format!("/api/memory/{id}"), None).await.0,
        StatusCode::NOT_FOUND
    );
    let (_, list) = d.api("GET", "/api/memory", None).await;
    assert_eq!(list["facts"], json!([]));
}

#[test]
fn install_downloads_only_what_the_cache_lacks() {
    let cache = tempfile::tempdir().unwrap();
    let model = kjarni::ModelType::from_cli_name("qwen3.5-4b").unwrap();
    let dir = model.cache_dir(cache.path());
    std::fs::create_dir_all(&dir).unwrap();
    assert!(!super::is_cached_in(cache.path(), "qwen3.5-4b", true));
    for file in ["config.json", "tokenizer.json", "model.gguf"] {
        std::fs::write(dir.join(file), "").unwrap();
    }
    assert!(super::is_cached_in(cache.path(), "qwen3.5-4b", true));
    // Full precision loads safetensors, which are not there.
    assert!(!super::is_cached_in(cache.path(), "qwen3.5-4b", false));
    std::fs::write(dir.join("model.safetensors"), "").unwrap();
    assert!(super::is_cached_in(cache.path(), "qwen3.5-4b", false));
    assert!(!super::is_cached_in(cache.path(), "no-such-model", false));
}
