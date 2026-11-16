//! The agent loop, driven by a model that replies from a script and renders
//! prompts with Qwen 3.5's template.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use kjarni_transformers::chat::qwen3_5::Qwen3_5ChatTemplate;
use kjarni_transformers::{ChatTemplate, Conversation, Role, ToolCall};

use super::*;

struct ScriptedModel {
    replies: Mutex<VecDeque<String>>,
    prompts: Mutex<Vec<String>>,
}

impl ScriptedModel {
    fn new(replies: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
            prompts: Mutex::new(Vec::new()),
        })
    }

    fn prompts(&self) -> Vec<String> {
        self.prompts.lock().unwrap().clone()
    }
}

#[async_trait]
impl AgentModel for ScriptedModel {
    fn render(&self, conversation: &Conversation) -> anyhow::Result<String> {
        Ok(Qwen3_5ChatTemplate::new().apply(conversation))
    }

    async fn stream(&self, prompt: String) -> anyhow::Result<TextStream> {
        self.prompts.lock().unwrap().push(prompt);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("the script ran out of replies");
        // Delivered a few characters at a time, as a model streams.
        let pieces: Vec<anyhow::Result<String>> = reply
            .chars()
            .collect::<Vec<_>>()
            .chunks(3)
            .map(|c| Ok(c.iter().collect()))
            .collect();
        Ok(Box::pin(futures::stream::iter(pieces)))
    }

    fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        Ok(text.len() / 4)
    }

    fn context_size(&self) -> usize {
        1_000_000
    }

    fn name(&self) -> String {
        "scripted".into()
    }
}

/// Records what a front end would show, and answers approvals from a list.
#[derive(Default)]
struct Recorder {
    text: String,
    calls: Vec<String>,
    approvals_asked: Vec<String>,
    answers: VecDeque<Approval>,
    notices: Vec<String>,
}

#[async_trait]
impl AgentEvents for Recorder {
    async fn text(&mut self, delta: &str) {
        self.text.push_str(delta);
    }
    async fn tool_call(&mut self, _call: &ToolCall, summary: &str, _access: Access) {
        self.calls.push(summary.to_string());
    }
    async fn approve(&mut self, _call: &ToolCall, summary: &str, _access: Access) -> Approval {
        self.approvals_asked.push(summary.to_string());
        self.answers.pop_front().unwrap_or(Approval::Deny)
    }
    async fn notice(&mut self, message: &str) {
        self.notices.push(message.to_string());
    }
}

fn call(name: &str, args: &[(&str, &str)]) -> String {
    let mut out = format!("<tool_call>\n<function={name}>\n");
    for (k, v) in args {
        out.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
    }
    out.push_str("</function>\n</tool_call>");
    out
}

fn agent_in(dir: &std::path::Path, model: Arc<ScriptedModel>) -> Agent {
    let workspace = Arc::new(tools::Workspace::new(dir));
    Agent::new(
        model,
        tools::builtin(workspace, None),
        AgentConfig::default(),
    )
}

#[tokio::test]
async fn reads_a_file_and_answers_from_it() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.txt"), "buy milk\n").unwrap();
    let read = call("read_file", &[("path", "notes.txt")]);
    let model = ScriptedModel::new(&[&format!("Let me check.\n\n{read}"), "It says: buy milk."]);
    let mut agent = agent_in(dir.path(), model.clone());
    let mut events = Recorder::default();

    let reply = agent
        .send("What's in notes.txt?", &mut events)
        .await
        .unwrap();

    assert_eq!(reply.text, "It says: buy milk.");
    assert_eq!((reply.steps, reply.tool_calls), (2, 1));
    // The markup never reaches the screen; the prose either side does.
    assert_eq!(events.text, "Let me check.\n\nIt says: buy milk.");
    assert_eq!(events.calls, ["read_file notes.txt"]);
    assert!(events.approvals_asked.is_empty(), "reads need no approval");

    let second = &model.prompts()[1];
    assert!(
        second.ends_with(&format!(
            "<|im_start|>assistant\n<think>\n\n</think>\n\nLet me check.\n\n{read}<|im_end|>\n\
             <|im_start|>user\n<tool_response>\n1\tbuy milk\n</tool_response><|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        )),
        "{second}"
    );
    let roles: Vec<Role> = agent.messages().iter().map(|m| m.role.clone()).collect();
    assert_eq!(
        roles,
        [Role::User, Role::Assistant, Role::Tool, Role::Assistant]
    );
}

#[tokio::test]
async fn a_declined_write_does_not_happen_and_the_model_is_told() {
    let dir = tempfile::tempdir().unwrap();
    let write = call("write_file", &[("path", "out.txt"), ("content", "hi")]);
    let model = ScriptedModel::new(&[&write, "Okay, I won't."]);
    let mut agent = agent_in(dir.path(), model.clone());
    let mut events = Recorder::default();

    agent
        .send("Write hi to out.txt", &mut events)
        .await
        .unwrap();

    assert_eq!(events.approvals_asked, ["write_file out.txt (1 lines)"]);
    assert!(!dir.path().join("out.txt").exists());
    assert!(
        agent.messages()[2]
            .content
            .starts_with("The user declined this call")
    );
}

#[tokio::test]
async fn always_approves_later_calls_of_the_same_tool() {
    let dir = tempfile::tempdir().unwrap();
    let a = call("write_file", &[("path", "a.txt"), ("content", "1")]);
    let b = call("write_file", &[("path", "b.txt"), ("content", "2")]);
    let model = ScriptedModel::new(&[&a, &b, "Done."]);
    let mut agent = agent_in(dir.path(), model);
    let mut events = Recorder {
        answers: VecDeque::from([Approval::Always]),
        ..Default::default()
    };

    agent.send("Write two files", &mut events).await.unwrap();

    assert_eq!(events.approvals_asked.len(), 1);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
        "1"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("b.txt")).unwrap(),
        "2"
    );
}

#[tokio::test]
async fn unknown_tools_and_unreadable_calls_are_reported_to_the_model() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(&[
        &call("launch_rocket", &[]),
        "<tool_call>{oops}</tool_call>",
        "Sorry.",
    ]);
    let mut agent = agent_in(dir.path(), model);

    let reply = agent.send("Go", &mut Recorder::default()).await.unwrap();

    assert_eq!(reply.text, "Sorry.");
    let results: Vec<&str> = agent
        .messages()
        .iter()
        .filter(|m| m.role == Role::Tool)
        .map(|m| m.content.as_str())
        .collect();
    assert!(results[0].starts_with(
        "Error: there is no tool named 'launch_rocket'. The tools are: list_directory, read_file"
    ));
    assert!(results[1].starts_with("Error: this tool call could not be read"));
}

#[tokio::test]
async fn stops_after_max_steps() {
    let dir = tempfile::tempdir().unwrap();
    let list = call("list_directory", &[]);
    let model = ScriptedModel::new(&[&list, &list, &list]);
    let workspace = Arc::new(tools::Workspace::new(dir.path()));
    let config = AgentConfig {
        max_steps: 3,
        ..Default::default()
    };
    let mut agent = Agent::new(model, tools::builtin(workspace, None), config);
    let mut events = Recorder::default();

    let reply = agent.send("Loop", &mut events).await.unwrap();

    assert!(reply.stopped_early);
    assert_eq!(reply.steps, 3);
    assert_eq!(
        events.notices,
        ["Stopped after 3 steps without a final answer."]
    );
}

#[tokio::test]
async fn folds_old_messages_into_a_summary_and_saves_it() {
    let dir = tempfile::tempdir().unwrap();
    let sessions = dir.path().join("sessions");
    let model = ScriptedModel::new(&[
        "Hello!",
        "Rust is a language.",
        // The summary, asked for when the long third message pushes the
        // prompt over budget.
        "<think>\n\n</think>\n\nThe user greeted and asked about Rust.",
        "Yes.",
    ]);
    let workspace = Arc::new(tools::Workspace::new(dir.path()));
    let config = AgentConfig {
        context_budget: 8_000,
        keep_recent: 1,
        ..Default::default()
    };
    let session = Session::create(&sessions).unwrap();
    let mut agent = Agent::new(model.clone(), tools::builtin(workspace, None), config)
        .with_session(session.clone());
    let mut events = Recorder::default();

    agent.send("Hi", &mut events).await.unwrap();
    agent.send("What is Rust?", &mut events).await.unwrap();
    let long = format!("Is it fast? {}", "Context. ".repeat(4000));
    agent.send(&long, &mut events).await.unwrap();

    assert_eq!(
        agent.summary(),
        Some("The user greeted and asked about Rust.")
    );
    assert_eq!(agent.messages().len(), 2);
    assert_eq!(
        events.notices,
        ["Summarizing 4 earlier messages to make room."]
    );
    assert!(
        model.prompts()[3].contains(
            "Summary of this conversation so far:\nThe user greeted and asked about Rust."
        )
    );

    let restored = session.load().unwrap();
    assert_eq!(restored.summary.as_deref(), agent.summary());
    assert_eq!(restored.messages.len(), 2);
    assert!(restored.messages[0].content.starts_with("Is it fast?"));
}

/// The system prompt lists saved facts and stays the same through a
/// conversation, so the prompt cache is reused after `remember`. A fact saved
/// now joins it in the next conversation; a forgotten fact leaves at once.
#[tokio::test]
async fn saved_facts_appear_in_the_system_prompt() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Arc::new(Memory::open(dir.path(), None).unwrap());
    let short = memory
        .remember("The user prefers short answers.")
        .await
        .unwrap();
    let model = ScriptedModel::new(&[
        &call("remember", &[("fact", "The user's dog is Loki.")]),
        "Noted.",
    ]);
    let workspace = Arc::new(tools::Workspace::new(dir.path()));
    let mut agent = Agent::new(
        model.clone(),
        tools::builtin(workspace, Some(memory.clone())),
        AgentConfig::default(),
    )
    .with_memory(memory.clone());
    let mut events = Recorder {
        answers: VecDeque::from([Approval::Once]),
        ..Default::default()
    };

    agent.send("My dog is Loki", &mut events).await.unwrap();

    let prompts = model.prompts();
    let system = |p: &str| p[..p.find("<|im_end|>").unwrap()].to_string();
    assert!(system(&prompts[0]).ends_with(
        "What you remember from earlier conversations:\n- The user prefers short answers."
    ));
    assert_eq!(memory.len(), 2);
    assert_eq!(system(&prompts[1]), system(&prompts[0]));
    assert!(prompts[1].starts_with(&prompts[0]));

    agent.start_over(None);
    assert!(
        agent
            .system_prompt()
            .ends_with("- The user prefers short answers.\n- The user's dog is Loki.")
    );

    memory.forget(short.id).unwrap();
    assert!(
        agent
            .system_prompt()
            .ends_with("What you remember from earlier conversations:\n- The user's dog is Loki.")
    );
}

/// The agent on a real model: Qwen 3.5 0.8B reads a file it is pointed at and
/// answers from it, through the `Chat` implementation of [`AgentModel`].
#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads qwen3.5-0.8b"]
async fn a_real_model_reads_a_file_and_answers_from_it() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("notes.txt"),
        "The project code word is heron.\n",
    )
    .unwrap();
    let chat = crate::chat::Chat::builder("qwen3.5-0.8b")
        .temperature(0.0)
        .max_tokens(200)
        .cpu()
        .quiet()
        .with_load_config(|c| c.prefer_gguf(true))
        .build()
        .await
        .unwrap();
    assert!(chat.supports_tools());
    assert!(AgentModel::context_size(&chat) >= 32_768);

    let workspace = Arc::new(tools::Workspace::new(dir.path()));
    let agent_config = AgentConfig {
        workdir: Some(dir.path().display().to_string()),
        ..Default::default()
    };
    let mut agent = Agent::new(
        Arc::new(chat),
        tools::builtin(workspace, None),
        agent_config,
    );
    let mut events = Recorder::default();
    let reply = agent
        .send("What is the code word in notes.txt?", &mut events)
        .await
        .unwrap();
    assert!(
        events.calls.iter().any(|c| c.starts_with("read_file")),
        "{:?}",
        events.calls
    );
    assert!(
        reply.text.to_lowercase().contains("heron"),
        "{}",
        reply.text
    );
    assert!(!reply.text.contains("<tool_call>"), "{}", reply.text);
    assert!(!reply.stopped_early);
}

/// `search_documents` indexes the working folder on its first call, names
/// files relative to it, and searches a subfolder through the same index.
#[tokio::test(flavor = "multi_thread")]
async fn search_documents_finds_passages_by_meaning() {
    use crate::folder_search::FolderSearch;
    use tool::Tool;

    let cache = tempfile::tempdir().unwrap();
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("notes")).unwrap();
    std::fs::write(
        dir.path().join("notes/boat.md"),
        "The ferry leaves the harbour at nine.",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("notes/bread.md"),
        "Knead the dough and let it rise.",
    )
    .unwrap();
    std::fs::write(dir.path().join("todo.txt"), "Pump the bicycle tyres.").unwrap();
    let search = FolderSearch::new()
        .cache_dir(cache.path())
        .reranker(tools::documents::RERANK_MODEL);
    let tool = tools::SearchDocuments::new(
        Arc::new(tools::Workspace::new(dir.path())),
        Arc::new(search),
    );
    let args = |pairs: serde_json::Value| pairs.as_object().unwrap().clone();
    let boat = std::path::Path::new("notes").join("boat.md");

    let first = tool
        .call(&args(
            serde_json::json!({"query": "when does the boat sail", "limit": 1}),
        ))
        .await
        .unwrap();
    assert_eq!(
        first,
        format!(
            "(Indexed the working folder (3 files).)\n\n1. {}\nThe ferry leaves the harbour at nine.",
            boat.display()
        )
    );

    let inside = tool
        .call(&args(
            serde_json::json!({"query": "tyres", "path": "notes", "limit": 5}),
        ))
        .await
        .unwrap();
    assert!(!inside.contains("Indexed"), "{inside}");
    assert!(!inside.contains("todo.txt"), "{inside}");
    assert!(inside.starts_with("1. notes"), "{inside}");
    assert_eq!(inside.matches("\n\n").count(), 1, "{inside}");
}
