//! The daemon's API end to end, with a scripted model in place of a real one.

use std::collections::{HashMap, HashSet, VecDeque};
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

/// Loads models from a list for a change of model, and logs what it does.
/// A model not on the list fails to load. While `blocked`, loads wait for
/// it to be released.
#[derive(Default)]
struct FakeLoader {
    models: HashMap<String, Arc<dyn AgentModel>>,
    downloaded: Mutex<HashSet<String>>,
    log: Mutex<Vec<String>>,
    blocked: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

impl FakeLoader {
    fn log(&self, line: String) {
        self.log.lock().unwrap().push(line);
    }
}

#[async_trait]
impl model::Loader for FakeLoader {
    fn downloaded(&self, name: &str) -> bool {
        self.downloaded.lock().unwrap().contains(name)
    }
    fn bytes_on_disk(&self, _: &str) -> u64 {
        0
    }
    async fn download(&self, name: &str) -> Result<()> {
        self.log(format!("download {name}"));
        self.downloaded.lock().unwrap().insert(name.to_string());
        Ok(())
    }
    async fn unload(&self, name: &str) {
        self.log(format!("unload {name}"));
    }
    async fn load(&self, name: &str) -> Result<Arc<dyn AgentModel>> {
        self.log(format!("load {name}"));
        let blocked = self.blocked.lock().unwrap().take();
        if let Some(rx) = blocked {
            let _ = rx.await;
        }
        self.models
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("{name} would not load"))
    }
}

struct Daemon {
    app: Router,
    workdir: tempfile::TempDir,
    data: tempfile::TempDir,
}

fn daemon(model: Arc<dyn AgentModel>, any_host: bool) -> Daemon {
    daemon_with_guard(model, Guard::new(TOKEN.into(), any_host))
}

fn daemon_with_guard(model: Arc<dyn AgentModel>, guard: Guard) -> Daemon {
    daemon_with(model, guard, Arc::new(FakeLoader::default()))
}

/// A daemon running `model` as "scripted", that changes model with `loader`.
fn daemon_with(model: Arc<dyn AgentModel>, guard: Guard, loader: Arc<FakeLoader>) -> Daemon {
    let workdir = tempfile::tempdir().unwrap();
    let data = tempfile::tempdir().unwrap();
    let memory = Arc::new(Memory::open(data.path(), None).unwrap());
    let slot = Arc::new(model::Slot::new(model));
    let state = Arc::new(DaemonState {
        conversations: Arc::new(Conversations::new(
            AgentSetup {
                model: slot.clone(),
                memory: Some(memory.clone()),
                workdir: workdir.path().to_path_buf(),
                max_steps: 4,
                approve_all: false,
            },
            data.path().join("sessions"),
        )),
        guard: Arc::new(guard),
        memory: Some(memory),
        models: Arc::new(model::Models::new(
            slot,
            loader,
            "scripted".into(),
            model::settings_path(data.path()),
            None,
        )),
        gpu: false,
        schedules: Arc::new(schedules::Store::open(data.path().join("schedules.json")).unwrap()),
    });
    let serve_state = Arc::new(serve::AppState {
        models: serve::Models::new(serve::Defaults::default(), false, false),
    });
    Daemon {
        app: router(state, serve_state, false),
        workdir,
        data,
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
    assert_eq!(types(&events), ["user", "context", "done"]);
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

    // A wrong approval is refused.
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
    assert_eq!(
        types(&rest),
        ["approved", "file_changed", "tool_result", "context", "done"]
    );
    assert_eq!(rest[0]["decision"], "once");
    assert_eq!(rest[2]["ok"], true);
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

/// A decline can say what to do instead: the model reads it, and followers
/// see it with the decision.
#[tokio::test]
async fn a_decline_with_a_reason_tells_the_model_and_followers() {
    let d = daemon(Scripted::new(&[Some(WRITE_CALL), Some("Okay.")]), false);
    let id = d.new_conversation().await;
    d.api(
        "POST",
        &format!("/api/conversations/{id}/messages"),
        Some(json!({"text": "Save hello"})),
    )
    .await;
    let events = d.follow(&id, 0, |e| e["type"] == "approval").await;
    let approval = events.last().unwrap()["approval"].as_u64().unwrap();
    let path = format!("/api/conversations/{id}/approvals/{approval}");

    let too_long = "x".repeat(2001);
    let (status, _) = d
        .api(
            "POST",
            &path,
            Some(json!({"decision": "deny", "reason": too_long})),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = d
        .api(
            "POST",
            &path,
            Some(json!({"decision": "deny", "reason": " Put it in notes/ "})),
        )
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    let last = events.last().unwrap()["seq"].as_u64().unwrap();
    let rest = d.follow(&id, last, ends_turn).await;
    assert_eq!(rest[0]["type"], "approved");
    assert_eq!(rest[0]["decision"], "deny");
    assert_eq!(rest[0]["reason"], "Put it in notes/");
    let result = rest.iter().find(|e| e["type"] == "tool_result").unwrap();
    assert_eq!(result["denied"], true);
    assert!(
        result["output"]
            .as_str()
            .unwrap()
            .contains("They said: \"Put it in notes/\"")
    );
}

/// What a conversation wrote is listed with diffs and saved beside it, and
/// deleting the conversation deletes the list.
#[tokio::test]
async fn the_files_a_conversation_changed_are_listed_with_their_diffs() {
    let d = daemon(Scripted::new(&[Some(WRITE_CALL), Some("Saved.")]), false);
    std::fs::write(d.workdir.path().join("note.txt"), "old\n").unwrap();
    let id = d.new_conversation().await;
    let changes = format!("/api/conversations/{id}/changes");
    assert_eq!(d.api("GET", &changes, None).await.1, json!({"changes": []}));

    d.api(
        "POST",
        &format!("/api/conversations/{id}/messages"),
        Some(json!({"text": "Save hello"})),
    )
    .await;
    let events = d.follow(&id, 0, |e| e["type"] == "approval").await;
    let approval = events.last().unwrap()["approval"].as_u64().unwrap();
    d.api(
        "POST",
        &format!("/api/conversations/{id}/approvals/{approval}"),
        Some(json!({"decision": "once"})),
    )
    .await;
    let last = events.last().unwrap()["seq"].as_u64().unwrap();
    let rest = d.follow(&id, last, ends_turn).await;
    assert_eq!(
        types(&rest),
        ["approved", "file_changed", "tool_result", "context", "done"]
    );
    assert_eq!(rest[1]["path"], "note.txt");

    let (status, list) = d.api("GET", &changes, None).await;
    assert_eq!(status, StatusCode::OK);
    let file = &list["changes"][0];
    assert_eq!(file["path"], "note.txt");
    assert_eq!(file["status"], "modified");
    assert_eq!(
        file["hunks"][0]["lines"],
        json!([{"sign": "-", "text": "old"}, {"sign": "+", "text": "hello"}])
    );
    let (_, open) = d
        .api("GET", &format!("/api/conversations/{id}"), None)
        .await;
    assert_eq!(open["changed_files"], 1);

    let saved = d
        .data
        .path()
        .join("sessions")
        .join(format!("{id}.changes.json"));
    assert!(saved.exists(), "{}", saved.display());
    d.api("DELETE", &format!("/api/conversations/{id}"), None)
        .await;
    assert!(!saved.exists());
    assert_eq!(d.api("GET", &changes, None).await.0, StatusCode::NOT_FOUND);
}

/// A message sent while the agent works is queued, then read before its
/// next step, in the same turn.
#[tokio::test]
async fn a_message_sent_mid_turn_joins_the_turn() {
    let d = daemon(
        Scripted::new(&[Some(WRITE_CALL), Some("Saved, and noted.")]),
        false,
    );
    let id = d.new_conversation().await;
    let messages = format!("/api/conversations/{id}/messages");
    let (_, sent) = d
        .api("POST", &messages, Some(json!({"text": "Save hello"})))
        .await;
    assert_eq!(sent, json!({"queued": false}));
    let events = d.follow(&id, 0, |e| e["type"] == "approval").await;
    let approval = events.last().unwrap()["approval"].as_u64().unwrap();

    let (status, sent) = d
        .api("POST", &messages, Some(json!({"text": " also add bread "})))
        .await;
    assert_eq!(
        (status, sent),
        (StatusCode::ACCEPTED, json!({"queued": true}))
    );
    d.api(
        "POST",
        &format!("/api/conversations/{id}/approvals/{approval}"),
        Some(json!({"decision": "once"})),
    )
    .await;

    let all = d.follow(&id, 0, ends_turn).await;
    let steps: Vec<_> = types(&all).into_iter().filter(|t| *t != "text").collect();
    assert_eq!(
        steps,
        [
            "user",
            "tool_call",
            "approval",
            "queued",
            "approved",
            "file_changed",
            "tool_result",
            "user",
            "context",
            "done"
        ]
    );
    let users: Vec<_> = all
        .iter()
        .filter(|e| e["type"] == "user")
        .map(|e| e["text"].as_str().unwrap())
        .collect();
    assert_eq!(users, ["Save hello", "also add bread"]);
    let (_, open) = d
        .api("GET", &format!("/api/conversations/{id}"), None)
        .await;
    let roles: Vec<_> = open["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["user", "assistant", "tool", "user", "assistant"]);
    assert!(open["context"]["used"].as_u64().is_some());
}

/// Stopping a turn discards the messages queued for it, and says which.
#[tokio::test]
async fn stopping_drops_the_messages_queued_for_the_turn() {
    let d = daemon(Scripted::new(&[None]), false);
    let id = d.new_conversation().await;
    let messages = format!("/api/conversations/{id}/messages");
    d.api("POST", &messages, Some(json!({"text": "Think hard"})))
        .await;
    d.follow(&id, 0, |e| e["type"] == "text").await;
    d.api("POST", &messages, Some(json!({"text": "never mind"})))
        .await;

    d.api("POST", &format!("/api/conversations/{id}/stop"), None)
        .await;

    let all = d.follow(&id, 0, ends_turn).await;
    let stopped = all.last().unwrap();
    assert_eq!(stopped["type"], "stopped");
    assert_eq!(stopped["dropped"], json!(["never mind"]));
}

/// Compacting by hand summarizes the earlier messages now and reports how
/// full the prompt is; while a turn runs it is refused.
#[tokio::test]
async fn a_conversation_can_be_compacted_by_hand() {
    let d = daemon(
        Scripted::new(&[
            Some("One."),
            Some("Two."),
            Some("Three."),
            Some("Four."),
            Some("The user counted to four."),
            None,
        ]),
        false,
    );
    let id = d.new_conversation().await;
    let compact = format!("/api/conversations/{id}/compact");
    let (status, body) = d.api("POST", &compact, None).await;
    assert_eq!(
        (status, body["compacted"].clone()),
        (StatusCode::OK, json!(false))
    );

    for text in ["1", "2", "3", "4"] {
        d.ask_in(&id, text).await;
    }
    let (status, body) = d.api("POST", &compact, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["compacted"], true);
    assert!(body["context"]["budget"].as_u64().unwrap() > 0);
    let (_, open) = d
        .api("GET", &format!("/api/conversations/{id}"), None)
        .await;
    assert_eq!(open["summary"], "The user counted to four.");
    assert_eq!(open["messages"].as_array().unwrap().len(), 6);

    d.api(
        "POST",
        &format!("/api/conversations/{id}/messages"),
        Some(json!({"text": "5"})),
    )
    .await;
    assert_eq!(d.api("POST", &compact, None).await.0, StatusCode::CONFLICT);
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

#[test]
fn the_daemon_downloads_its_models_with_progress_before_loading() {
    let cache = tempfile::tempdir().unwrap();
    // An empty cache: the chat model with progress, then the memory model quietly.
    assert_eq!(
        super::missing(cache.path(), "qwen3.5-4b", true, true),
        vec![
            ("qwen3.5-4b", true, false),
            (super::MEMORY_MODEL, false, true)
        ]
    );
    // Without memory, only the chat model.
    assert_eq!(
        super::missing(cache.path(), "qwen3.5-4b", true, false),
        vec![("qwen3.5-4b", true, false)]
    );
    // Nothing once both are cached.
    for (name, files) in [
        (
            "qwen3.5-4b",
            &["config.json", "tokenizer.json", "model.gguf"][..],
        ),
        (
            super::MEMORY_MODEL,
            &["config.json", "tokenizer.json", "model.safetensors"][..],
        ),
    ] {
        let dir = kjarni::ModelType::from_cli_name(name)
            .unwrap()
            .cache_dir(cache.path());
        std::fs::create_dir_all(&dir).unwrap();
        for file in files {
            std::fs::write(dir.join(file), "").unwrap();
        }
    }
    assert!(super::missing(cache.path(), "qwen3.5-4b", true, true).is_empty());
}

#[tokio::test]
async fn the_token_signs_in_a_browser_on_another_device() {
    let d = daemon(Scripted::new(&[]), true);
    let sign_in = |token: &str| {
        request("POST", "/login")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json!({ "token": token }).to_string()))
            .unwrap()
    };
    let response = d.app.clone().oneshot(sign_in(TOKEN)).await.unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert_eq!(
        response.headers()[header::SET_COOKIE].to_str().unwrap(),
        format!("kjarni_token={TOKEN}; Path=/; HttpOnly; SameSite=Strict; Max-Age=31536000")
    );
    let wrong = d.app.clone().oneshot(sign_in("guess")).await.unwrap();
    assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED);
    assert!(wrong.headers().get(header::SET_COOKIE).is_none());

    // Another web page cannot sign the user's browser in to its own choice.
    let forged = request("POST", "/login")
        .header(header::ORIGIN, "http://evil.example")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(json!({ "token": TOKEN }).to_string()))
        .unwrap();
    assert_eq!(d.call(forged).await.0, StatusCode::FORBIDDEN);
}

/// A signed-in device shows an eight-digit code; typed on another device's
/// sign-in page, it signs that browser in once.
#[tokio::test]
async fn a_device_code_signs_another_browser_in_once() {
    let d = daemon(Scripted::new(&[]), true);
    let (status, made) = d.api("POST", "/api/device-code", None).await;
    assert_eq!(status, StatusCode::OK);
    let code = made["code"].as_str().unwrap().to_string();
    assert_eq!(code.len(), 8);
    assert!(code.chars().all(|c| c.is_ascii_digit()), "{code}");
    assert_eq!(made["expires_in"], 120);

    let sign_in = |code: &str| {
        request("POST", "/login")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json!({ "code": code }).to_string()))
            .unwrap()
    };
    let typed = format!("{} {}", &code[..4], &code[4..]);
    let response = d.app.clone().oneshot(sign_in(&typed)).await.unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert!(
        response.headers()[header::SET_COOKIE]
            .to_str()
            .unwrap()
            .starts_with(&format!("kjarni_token={TOKEN};"))
    );
    let again = d.app.clone().oneshot(sign_in(&code)).await.unwrap();
    assert_eq!(again.status(), StatusCode::UNAUTHORIZED);
}

/// Ten wrong codes in a row cancel every pending one, so a short code cannot
/// be found by guessing.
#[test]
fn wrong_codes_cancel_the_pending_ones() {
    let guard = Guard::new(TOKEN.into(), false);
    let code = guard.new_device_code();
    for _ in 0..9 {
        assert!(!guard.redeem("00000000x"));
    }
    let kept = guard.new_device_code();
    assert!(guard.redeem(&code));
    for _ in 0..10 {
        assert!(!guard.redeem("nope"));
    }
    assert!(!guard.redeem(&kept));

    let fresh = guard.new_device_code();
    assert!(guard.redeem(&fresh));
}

#[tokio::test]
async fn the_web_ui_is_served_without_signing_in() {
    let d = daemon(Scripted::new(&[]), false);
    let get = |path: &str| request("GET", path).body(Body::empty()).unwrap();

    for path in ["/", "/c/abc123"] {
        let response = d.app.clone().oneshot(get(path)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        let headers = response.headers();
        assert_eq!(headers[header::CONTENT_TYPE], "text/html; charset=utf-8");
        let csp = headers[header::CONTENT_SECURITY_POLICY].to_str().unwrap();
        assert!(csp.contains("script-src 'self'"), "{csp}");
        assert!(csp.contains("frame-ancestors 'none'"), "{csp}");
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("/assets/app.js"));
    }

    let js = d.app.clone().oneshot(get("/assets/app.js")).await.unwrap();
    assert_eq!(js.status(), StatusCode::OK);
    assert_eq!(
        js.headers()[header::CONTENT_TYPE],
        "text/javascript; charset=utf-8"
    );
    let font = d
        .app
        .clone()
        .oneshot(get("/fonts/martian-mono-latin.woff2"))
        .await
        .unwrap();
    assert_eq!(font.headers()[header::CONTENT_TYPE], "font/woff2");
    assert!(
        font.headers()[header::CACHE_CONTROL]
            .to_str()
            .unwrap()
            .contains("immutable")
    );

    assert_eq!(
        d.call(get("/assets/missing.js")).await.0,
        StatusCode::NOT_FOUND
    );
    // The API's own paths never fall through to the page.
    assert_eq!(
        d.api("GET", "/api/no-such-thing", None).await.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(d.call(get("/api/status")).await.0, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn an_allowed_host_name_works_over_https_as_tailscale_serve_sends_it() {
    let guard = Guard::new(TOKEN.into(), false).allow_hosts(vec!["box.tail1234.ts.net".into()]);
    let d = daemon_with_guard(Scripted::new(&[]), guard);
    // The web UI offers these names when it signs in another device.
    assert_eq!(
        d.api("GET", "/api/status", None).await.1["allowed_hosts"],
        json!(["box.tail1234.ts.net"])
    );
    let status = |host: &str, origin: Option<&str>| {
        let mut r = Request::builder()
            .method("POST")
            .uri("/api/conversations")
            .header(header::HOST, host)
            .header(header::AUTHORIZATION, format!("Bearer {TOKEN}"));
        if let Some(o) = origin {
            r = r.header(header::ORIGIN, o);
        }
        r.body(Body::empty()).unwrap()
    };

    for host in [
        "box.tail1234.ts.net",
        "BOX.tail1234.ts.net:443",
        "127.0.0.1:7710",
    ] {
        assert_eq!(
            d.call(status(host, None)).await.0,
            StatusCode::CREATED,
            "{host}"
        );
    }
    assert_eq!(
        d.call(status(
            "box.tail1234.ts.net",
            Some("https://box.tail1234.ts.net")
        ))
        .await
        .0,
        StatusCode::CREATED
    );

    // Another name is refused, with what to do if it is this computer's.
    let (code, body) = d.call(status("other.tail1234.ts.net", None)).await;
    assert_eq!(code, StatusCode::FORBIDDEN);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("--allow-host other.tail1234.ts.net")
    );
    // A page on another site still cannot use the allowed name.
    assert_eq!(
        d.call(status("box.tail1234.ts.net", Some("https://evil.example")))
            .await
            .0,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        d.call(status(
            "box.tail1234.ts.net",
            Some("https://box.tail1234.ts.net.evil.example")
        ))
        .await
        .0,
        StatusCode::FORBIDDEN
    );
}

#[test]
fn allow_host_takes_a_name_or_a_pasted_address() {
    assert_eq!(
        auth::allowed_host("box.tail1234.ts.net").unwrap(),
        "box.tail1234.ts.net"
    );
    assert_eq!(
        auth::allowed_host(" https://Box.Tail1234.ts.net/ ").unwrap(),
        "box.tail1234.ts.net"
    );
    assert_eq!(auth::allowed_host("box.lan:7710").unwrap(), "box.lan");
    for bad in ["", "https://box.ts.net/path", "box ts net", "user@box"] {
        assert!(auth::allowed_host(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn install_carries_allowed_hosts_into_the_startup_entry() {
    let options = super::InstallOptions {
        model: None,
        gpu: false,
        cpu: false,
        full_precision: false,
        port: 7710,
        allow_hosts: vec!["box.tail1234.ts.net".into()],
        dir: None,
        yes: false,
        no_memory: false,
        max_steps: 16,
        no_linger: false,
    };
    let args = super::run_arguments(
        &options,
        true,
        std::path::Path::new("/home/ada/kjarni"),
        std::path::Path::new("/tmp/daemon.log"),
    );
    let at = args.iter().position(|a| a == "--allow-host").unwrap();
    assert_eq!(args[at + 1], "box.tail1234.ts.net");
    assert!(args.contains(&"--gpu".to_string()));
    // The daemon loads the saved model, so a change in the web UI lasts.
    assert!(!args.contains(&"--model".to_string()));
}

const RUN_CALL: &str = "Let me look.\n<tool_call>\n<function=run_command>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>";

/// A stand-in ntfy server: each publish arrives on the channel as its
/// `Authorization` header and JSON body.
async fn fake_ntfy() -> (
    String,
    tokio::sync::mpsc::UnboundedReceiver<(Option<String>, Json)>,
) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let app = axum::Router::new().route(
        "/",
        axum::routing::post(move |headers: axum::http::HeaderMap, body: String| {
            let tx = tx.clone();
            async move {
                let auth = headers
                    .get(header::AUTHORIZATION)
                    .map(|v| v.to_str().unwrap().to_string());
                tx.send((auth, serde_json::from_str(&body).unwrap()))
                    .unwrap();
                "{}"
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), rx)
}

#[tokio::test]
async fn a_scheduled_run_uses_only_the_tools_it_allows_and_notifies() {
    let d = daemon(
        Scripted::new(&[Some(WRITE_CALL), Some(RUN_CALL), Some("Saved the note.")]),
        false,
    );
    let (ntfy, mut published) = fake_ntfy().await;
    let (status, _) = d
        .api(
            "PUT",
            "/api/notifications",
            Some(json!({
                "url": format!("{ntfy}/kjarni-test"),
                "token": "tk_secret",
                "open_url": "https://box.tail1234.ts.net/",
            })),
        )
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    let (status, made) = d
        .api(
            "POST",
            "/api/schedules",
            Some(json!({
                "name": "Morning note",
                "prompt": "Save hello",
                "when": {"kind": "daily", "time": "07:00", "days": ["mon", "fri"]},
                "allow": ["write_file"],
            })),
        )
        .await;
    assert_eq!(status, StatusCode::CREATED, "{made}");
    assert_eq!(made["running"], false);
    assert_eq!(made["notify"], true);
    assert!(made["next_run"].as_i64().unwrap() > made["created"].as_i64().unwrap());
    let id = made["id"].as_str().unwrap();

    let (status, run) = d
        .api("POST", &format!("/api/schedules/{id}/run"), None)
        .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let conversation = run["conversation"].as_str().unwrap();
    let events = d.follow(conversation, 0, ends_turn).await;
    assert_eq!(
        types(&events),
        [
            "user",
            "tool_call",
            "approval",
            "approved",
            "file_changed",
            "tool_result",
            "tool_call",
            "approval",
            "notice",
            "approved",
            "tool_result",
            "context",
            "done"
        ]
    );
    let decisions: Vec<_> = events
        .iter()
        .filter(|e| e["type"] == "approved")
        .map(|e| e["decision"].as_str().unwrap())
        .collect();
    assert_eq!(decisions, ["once", "deny"]);
    let notice = events.iter().find(|e| e["type"] == "notice").unwrap();
    assert!(
        notice["message"]
            .as_str()
            .unwrap()
            .contains("run_command is not allowed")
    );
    assert_eq!(
        std::fs::read_to_string(d.workdir.path().join("note.txt")).unwrap(),
        "hello"
    );

    let (auth, body) = tokio::time::timeout(Duration::from_secs(10), published.recv())
        .await
        .expect("a notification")
        .unwrap();
    assert_eq!(auth.as_deref(), Some("Bearer tk_secret"));
    assert_eq!(body["topic"], "kjarni-test");
    assert_eq!(body["title"], "Morning note");
    assert_eq!(body["message"], "Saved the note.");
    assert_eq!(
        body["click"],
        format!("https://box.tail1234.ts.net/c/{conversation}")
    );

    let (_, listed) = d.api("GET", "/api/schedules", None).await;
    let s = &listed["schedules"][0];
    assert_eq!(s["last_result"], "done");
    assert_eq!(s["last_conversation"], conversation);
    assert_eq!(s["running"], false);
    // Clients learn a token is set, never the token.
    assert_eq!(listed["notify"]["has_token"], true);
    assert!(!listed.to_string().contains("tk_secret"));
    assert_eq!(
        listed["tools"],
        json!([
            "fetch_url",
            "write_file",
            "edit_file",
            "run_command",
            "remember",
            "forget"
        ])
    );
}

#[tokio::test]
async fn a_running_schedule_cannot_start_twice_and_a_stop_is_recorded() {
    let d = daemon(Scripted::new(&[None]), false);
    let (_, made) = d
        .api(
            "POST",
            "/api/schedules",
            Some(json!({
                "name": "Slow",
                "prompt": "Take your time",
                "when": {"kind": "every", "minutes": 30},
                "notify": false,
            })),
        )
        .await;
    let id = made["id"].as_str().unwrap();
    let (_, run) = d
        .api("POST", &format!("/api/schedules/{id}/run"), None)
        .await;
    let conversation = run["conversation"].as_str().unwrap();
    d.follow(conversation, 0, |e| e["type"] == "text").await;

    let (again, _) = d
        .api("POST", &format!("/api/schedules/{id}/run"), None)
        .await;
    assert_eq!(again, StatusCode::CONFLICT);
    let (_, listed) = d.api("GET", "/api/schedules", None).await;
    assert_eq!(listed["schedules"][0]["running"], true);
    assert_eq!(listed["schedules"][0]["last_result"], "running");

    let (stopped, _) = d
        .api(
            "POST",
            &format!("/api/conversations/{conversation}/stop"),
            None,
        )
        .await;
    assert_eq!(stopped, StatusCode::NO_CONTENT);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let (_, listed) = d.api("GET", "/api/schedules", None).await;
        if listed["schedules"][0]["last_result"] == "stopped" {
            assert_eq!(listed["schedules"][0]["running"], false);
            break;
        }
        assert!(tokio::time::Instant::now() < deadline, "{listed}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test]
async fn schedules_and_notification_settings_are_checked() {
    let d = daemon(Scripted::new(&[]), false);
    let base = json!({"name": "X", "prompt": "Y", "when": {"kind": "every", "minutes": 60}});

    let mut reads = base.clone();
    reads["allow"] = json!(["read_file"]);
    let (status, body) = d.api("POST", "/api/schedules", Some(reads)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("fetch_url")
    );

    let mut often = base.clone();
    often["when"] = json!({"kind": "every", "minutes": 1});
    let (status, _) = d.api("POST", "/api/schedules", Some(often)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (status, _) = d
        .api("PUT", "/api/schedules/nope", Some(base.clone()))
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = d.api("POST", "/api/schedules/nope/run", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = d.api("DELETE", "/api/schedules/nope", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) = d.api("POST", "/api/notifications/test", None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    for bad in [
        json!({"url": "ntfy.sh/topic"}),
        json!({"url": "https://ntfy.sh/"}),
        json!({"url": "https://ntfy.sh/t", "open_url": "javascript:alert(1)"}),
    ] {
        let (status, _) = d.api("PUT", "/api/notifications", Some(bad.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}");
    }

    // Leaving the token out keeps it while the address stays the same.
    d.api(
        "PUT",
        "/api/notifications",
        Some(json!({"url": "https://ntfy.sh/t", "token": "tk"})),
    )
    .await;
    d.api(
        "PUT",
        "/api/notifications",
        Some(json!({"url": "https://ntfy.sh/t"})),
    )
    .await;
    let (_, listed) = d.api("GET", "/api/schedules", None).await;
    assert_eq!(listed["notify"]["has_token"], true);
    d.api(
        "PUT",
        "/api/notifications",
        Some(json!({"url": "https://ntfy.sh/u"})),
    )
    .await;
    let (_, listed) = d.api("GET", "/api/schedules", None).await;
    assert_eq!(listed["notify"]["has_token"], false);
    let (status, _) = d
        .api("PUT", "/api/notifications", Some(json!({"url": ""})))
        .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (_, listed) = d.api("GET", "/api/schedules", None).await;
    assert_eq!(listed["notify"], Json::Null);
}

#[test]
fn schedules_can_allow_exactly_the_tools_that_ask() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Arc::new(Memory::open(dir.path(), None).unwrap());
    let workspace = Arc::new(kjarni::agent::tools::Workspace::new(dir.path()));
    let mut asking: Vec<String> = kjarni::agent::tools::builtin(workspace, Some(memory))
        .iter()
        .filter(|t| t.access() != kjarni::agent::Access::Read)
        .map(|t| t.spec().name.clone())
        .collect();
    asking.sort();
    let mut listed: Vec<String> = schedules::ASKING_TOOLS
        .iter()
        .map(|s| s.to_string())
        .collect();
    listed.sort();
    assert_eq!(asking, listed);
}

impl Daemon {
    /// Sends `text` in a new conversation and returns the reply.
    async fn ask(&self, text: &str) -> String {
        let id = self.new_conversation().await;
        self.ask_in(&id, text).await
    }

    /// The answer to `text` in conversation `id`, once its turn ends.
    async fn ask_in(&self, id: &str, text: &str) -> String {
        let (status, _) = self
            .api(
                "POST",
                &format!("/api/conversations/{id}/messages"),
                Some(json!({ "text": text })),
            )
            .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let events = self.follow(id, 0, ends_turn).await;
        events.last().unwrap()["text"].as_str().unwrap().to_string()
    }

    /// `/api/models` once `phase` is no longer downloading or loading.
    async fn settled_models(&self) -> Json {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let (_, body) = self.api("GET", "/api/models", None).await;
            let state = body["phase"]["state"].as_str().unwrap();
            if state == "idle" || state == "failed" {
                return body;
            }
            assert!(tokio::time::Instant::now() < deadline, "the switch ended");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }
}

#[tokio::test]
async fn switching_models_frees_the_old_one_before_loading_and_saves_the_choice() {
    let (release, blocked) = tokio::sync::oneshot::channel();
    let loader = Arc::new(FakeLoader {
        models: HashMap::from([(
            "qwen3.5-9b".to_string(),
            Scripted::new(&[Some("From the new model.")]) as Arc<dyn AgentModel>,
        )]),
        blocked: Mutex::new(Some(blocked)),
        ..Default::default()
    });
    let d = daemon_with(
        Scripted::new(&[]),
        Guard::new(TOKEN.into(), false),
        loader.clone(),
    );

    let (_, body) = d.api("GET", "/api/models", None).await;
    assert_eq!(body["current"], "scripted");
    let names: Vec<_> = body["choices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, model::CHOICES);
    assert_eq!(body["choices"][3]["params_millions"], 9653);

    let (status, _) = d
        .api(
            "PUT",
            "/api/models/current",
            Some(json!({ "model": "qwen3.5-9b" })),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED);

    // While it loads, messages and a second switch wait.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while d.api("GET", "/api/models", None).await.1["phase"]["state"] != "loading" {
        assert!(tokio::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let id = d.new_conversation().await;
    let (status, body) = d
        .api(
            "POST",
            &format!("/api/conversations/{id}/messages"),
            Some(json!({ "text": "Hello?" })),
        )
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("switching models")
    );
    let (status, _) = d
        .api(
            "PUT",
            "/api/models/current",
            Some(json!({ "model": "qwen3.5-2b" })),
        )
        .await;
    assert_eq!(status, StatusCode::CONFLICT);

    release.send(()).unwrap();
    let body = d.settled_models().await;
    assert_eq!(body["phase"]["state"], "idle");
    assert_eq!(body["current"], "qwen3.5-9b");
    assert_eq!(
        *loader.log.lock().unwrap(),
        ["download qwen3.5-9b", "unload scripted", "load qwen3.5-9b"]
    );
    assert_eq!(
        d.api("GET", "/api/status", None).await.1["model"],
        "qwen3.5-9b"
    );
    assert_eq!(
        model::Settings::load(&model::settings_path(d.data.path()))
            .model
            .as_deref(),
        Some("qwen3.5-9b")
    );
    assert_eq!(d.ask("Hello?").await, "From the new model.");
}

#[tokio::test]
async fn a_model_that_will_not_load_leaves_the_old_one_running() {
    let loader = Arc::new(FakeLoader {
        // Loaded again after the new one fails.
        models: HashMap::from([(
            "scripted".to_string(),
            Scripted::new(&[Some("Still here.")]) as Arc<dyn AgentModel>,
        )]),
        downloaded: Mutex::new(HashSet::from(["qwen3.5-9b".to_string()])),
        ..Default::default()
    });
    let d = daemon_with(
        Scripted::new(&[]),
        Guard::new(TOKEN.into(), false),
        loader.clone(),
    );
    let (status, _) = d
        .api(
            "PUT",
            "/api/models/current",
            Some(json!({ "model": "qwen3.5-9b" })),
        )
        .await;
    assert_eq!(status, StatusCode::ACCEPTED);
    let body = d.settled_models().await;
    assert_eq!(body["phase"]["state"], "failed");
    assert_eq!(body["phase"]["model"], "qwen3.5-9b");
    assert!(
        body["phase"]["message"]
            .as_str()
            .unwrap()
            .contains("would not load")
    );
    assert_eq!(body["current"], "scripted");
    assert_eq!(
        *loader.log.lock().unwrap(),
        ["unload scripted", "load qwen3.5-9b", "load scripted"]
    );
    assert!(!model::settings_path(d.data.path()).exists());
    assert_eq!(d.ask("Hello?").await, "Still here.");
}

#[tokio::test]
async fn only_tool_models_are_offered_and_not_while_a_conversation_answers() {
    let d = daemon(Scripted::new(&[None]), false);
    let (status, body) = d
        .api(
            "PUT",
            "/api/models/current",
            Some(json!({ "model": "llama3.2-1b-instruct" })),
        )
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("tool calls")
    );

    // A reply that never finishes keeps the conversation answering.
    let id = d.new_conversation().await;
    d.api(
        "POST",
        &format!("/api/conversations/{id}/messages"),
        Some(json!({ "text": "Go on" })),
    )
    .await;
    let (status, body) = d
        .api(
            "PUT",
            "/api/models/current",
            Some(json!({ "model": "qwen3.5-2b" })),
        )
        .await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("answering")
    );
    assert_eq!(
        d.api("GET", "/api/models", None).await.1["phase"]["state"],
        "idle"
    );
}

#[test]
fn the_model_given_wins_over_the_saved_one_and_the_saved_one_over_the_default() {
    let dir = tempfile::tempdir().unwrap();
    let path = model::settings_path(dir.path());
    let none = model::Settings::load(&path);
    assert_eq!(
        model::starting_model(None, &none),
        kjarni_cli::DEFAULT_AGENT_MODEL
    );
    model::Settings {
        model: Some("qwen3.5-9b".into()),
    }
    .save(&path)
    .unwrap();
    let saved = model::Settings::load(&path);
    assert_eq!(model::starting_model(None, &saved), "qwen3.5-9b");
    assert_eq!(
        model::starting_model(Some("qwen3.5-2b"), &saved),
        "qwen3.5-2b"
    );
}
