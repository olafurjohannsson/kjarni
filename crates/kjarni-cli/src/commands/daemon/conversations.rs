//! Conversations the daemon holds: an agent for each, the events of its
//! current turn for clients to follow, and approvals answered from a client.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use kjarni::agent::{
    Access, Agent, AgentConfig, AgentEvents, AgentModel, Approval, FileChange, Memory, Session,
    SessionInfo, ToolCall, ToolOutcome, tools,
};

use super::changes::Changes;
use serde::Serialize;
use serde_json::{Value as Json, json};
use tokio::sync::{broadcast, oneshot};

/// One thing that happened in a turn. `seq` grows across the daemon's life,
/// so a client that reconnects asks for what came after the last it saw.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Event {
    pub seq: u64,
    #[serde(flatten)]
    pub kind: EventKind,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventKind {
    /// A turn started with this message from the user.
    User {
        text: String,
    },
    /// Reply text as it is generated, reasoning included.
    Text {
        delta: String,
    },
    /// The model asked for a tool call. `call` numbers the calls of a turn.
    ToolCall {
        call: u32,
        name: String,
        summary: String,
        access: &'static str,
        arguments: Json,
    },
    /// Call `call` waits for the user to answer approval `approval`.
    Approval {
        approval: u64,
        call: u32,
        summary: String,
        access: &'static str,
    },
    /// The user answered, with what they said when they declined.
    Approved {
        approval: u64,
        decision: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// Call `call` changed the file at `path`, relative to the working folder
    /// when inside it.
    FileChanged {
        call: u32,
        path: String,
    },
    ToolResult {
        call: u32,
        ok: bool,
        denied: bool,
        output: String,
    },
    Notice {
        message: String,
    },
    /// The turn ended with this answer.
    Done {
        text: String,
        steps: usize,
        tool_calls: usize,
        stopped_early: bool,
    },
    /// The turn ended with an error.
    Error {
        message: String,
    },
    /// The user stopped the turn. `dropped` are the messages they sent
    /// meanwhile that the agent had not read, which are discarded.
    Stopped {
        #[serde(skip_serializing_if = "Vec::is_empty")]
        dropped: Vec<String>,
    },
    /// The user sent `text` while the turn ran; the agent reads it before its
    /// next step, when a `user` event with the same text follows.
    Queued {
        text: String,
    },
    /// How full the prompt is: `used` tokens of the `budget` past which
    /// earlier messages are summarized.
    Context {
        used: usize,
        budget: usize,
    },
}

impl EventKind {
    pub fn ends_turn(&self) -> bool {
        matches!(
            self,
            EventKind::Done { .. } | EventKind::Error { .. } | EventKind::Stopped { .. }
        )
    }
}

fn decision_name(a: &Approval) -> &'static str {
    match a {
        Approval::Once => "once",
        Approval::Always => "always",
        Approval::Deny | Approval::DenyWith(_) => "deny",
    }
}

fn decision_reason(a: &Approval) -> Option<String> {
    match a {
        Approval::DenyWith(words) => Some(words.clone()),
        _ => None,
    }
}

/// The part of a conversation that changes during a turn.
struct Turn {
    busy: bool,
    /// The events of the current or last turn.
    events: Vec<Event>,
    pending: Option<(u64, oneshot::Sender<Approval>)>,
    task: Option<tokio::task::AbortHandle>,
    /// Messages sent during the turn that the agent has not read yet.
    queued: Vec<String>,
    /// The saved messages as a client shows them, up to the start of the
    /// current turn. The agent is locked for a whole turn, so this copy is
    /// what a client opening the conversation mid-turn reads.
    transcript: Json,
}

/// What a client opening a conversation is shown.
pub struct Snapshot {
    pub transcript: Json,
    pub busy: bool,
    /// The current turn's events; empty when no turn is running, since the
    /// transcript then holds the last one.
    pub events: Vec<Event>,
    pub pending_approval: Option<u64>,
}

fn transcript_of(agent: &Agent) -> Json {
    let specs = agent.tools();
    let messages: Vec<Json> = agent
        .messages()
        .iter()
        .map(|m| super::display::message(m, specs))
        .collect();
    json!({ "summary": agent.summary(), "messages": messages })
}

/// A conversation with its agent.
pub struct Live {
    pub id: String,
    agent: tokio::sync::Mutex<Agent>,
    turn: Mutex<Turn>,
    tx: broadcast::Sender<Event>,
    hub: Arc<Hub>,
    /// The files this conversation changed.
    pub changes: Changes,
    /// The agent's working folder, which changed paths are shown relative to.
    pub workdir: PathBuf,
}

/// What the conversations share: the sequence numbers of events and
/// approvals, and whether new turns are held back.
#[derive(Default)]
struct Hub {
    seq: Mutex<u64>,
    /// True while new turns wait, such as during a change of model. A turn
    /// starts under a read lock, so taking the write lock waits out turns
    /// that are starting.
    held: RwLock<bool>,
}

impl Hub {
    fn next(&self) -> u64 {
        let mut seq = self.seq.lock().expect("event counter lock");
        *seq += 1;
        *seq
    }
}

/// Who answers a turn's approvals.
#[derive(Clone)]
pub enum Answers {
    /// The user, from a client.
    User,
    /// A list fixed in advance, for a turn nobody is watching: calls to these
    /// tools run, any other call that needs approval is declined.
    Allowed(Arc<HashSet<String>>),
}

/// What became of a user's message.
#[derive(Debug, PartialEq)]
pub enum Sent {
    /// It started a turn.
    Started,
    /// A turn is running; the agent reads it before its next step.
    Queued,
}

/// A conversation could not be summarized.
#[derive(Debug)]
pub enum CompactError {
    /// A turn is running.
    Busy,
    /// Turns are held back for now.
    Held,
    /// The summary could not be written.
    Failed(anyhow::Error),
}

/// A user's message could not start a turn.
#[derive(Debug, PartialEq)]
pub enum SendError {
    Busy,
    Empty,
    /// New turns are held back for now.
    Held,
}

/// Holds new turns back until dropped.
pub struct Hold(Arc<Hub>);

impl Drop for Hold {
    fn drop(&mut self) {
        *self.0.held.write().expect("hold lock") = false;
    }
}

impl Live {
    fn emit(&self, kind: EventKind) -> Event {
        self.emit_with(kind, None)
    }

    /// Emits `kind`, first setting the transcript to `transcript` when given.
    fn emit_with(&self, kind: EventKind, transcript: Option<Json>) -> Event {
        let event = Event {
            seq: self.hub.next(),
            kind,
        };
        let mut turn = self.turn.lock().expect("turn lock");
        if let Some(t) = transcript {
            turn.transcript = t;
        }
        if event.kind.ends_turn() {
            turn.busy = false;
            turn.task = None;
            turn.pending = None;
        }
        turn.events.push(event.clone());
        // No subscribers is fine: the event is in the log.
        let _ = self.tx.send(event.clone());
        event
    }

    /// The events of the current or last turn after `after`, and a receiver
    /// for those still to come, with none missed or repeated between them.
    pub fn follow(&self, after: u64) -> (Vec<Event>, broadcast::Receiver<Event>) {
        let turn = self.turn.lock().expect("turn lock");
        let rx = self.tx.subscribe();
        let past = turn
            .events
            .iter()
            .filter(|e| e.seq > after)
            .cloned()
            .collect();
        (past, rx)
    }

    pub fn busy(&self) -> bool {
        self.turn.lock().expect("turn lock").busy
    }

    pub fn snapshot(&self) -> Snapshot {
        let turn = self.turn.lock().expect("turn lock");
        Snapshot {
            transcript: turn.transcript.clone(),
            busy: turn.busy,
            events: if turn.busy {
                turn.events.clone()
            } else {
                Vec::new()
            },
            pending_approval: turn.pending.as_ref().map(|(id, _)| *id),
        }
    }

    /// Starts a turn answering `text`, or, while one runs, gives `text` to
    /// it to read before its next step. Events follow on
    /// [`follow`](Self::follow).
    pub fn send(self: &Arc<Self>, text: &str) -> Result<Sent, SendError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(SendError::Empty);
        }
        {
            let mut turn = self.turn.lock().expect("turn lock");
            if turn.busy {
                turn.queued.push(text.to_string());
                drop(turn);
                self.emit(EventKind::Queued {
                    text: text.to_string(),
                });
                return Ok(Sent::Queued);
            }
        }
        match self.send_with(text, Answers::User) {
            // Another message started a turn since the check: join it.
            Err(SendError::Busy) => self.send(text),
            other => other.map(|()| Sent::Started),
        }
    }

    /// Summarizes the conversation's earlier messages now. False when there
    /// were too few. Counts as a turn while it runs, so no message or change
    /// of model starts meanwhile.
    pub async fn compact(self: &Arc<Self>) -> Result<bool, CompactError> {
        {
            let held = self.hub.held.read().expect("hold lock");
            if *held {
                return Err(CompactError::Held);
            }
            let mut turn = self.turn.lock().expect("turn lock");
            if turn.busy {
                return Err(CompactError::Busy);
            }
            turn.busy = true;
        }
        let mut agent = self.agent.lock().await;
        let mut events = Relay {
            live: self.clone(),
            call: 0,
            answers: Answers::User,
            workdir: self.workdir.clone(),
        };
        let done = agent.compact(&mut events).await;
        let transcript = transcript_of(&agent);
        {
            let mut turn = self.turn.lock().expect("turn lock");
            turn.transcript = transcript;
            turn.busy = false;
        }
        self.emit_context(&agent);
        done.map_err(CompactError::Failed)
    }

    /// How full the prompt is, when the agent is not busy with a turn.
    pub fn context(&self) -> Option<(usize, usize)> {
        let agent = self.agent.try_lock().ok()?;
        agent.context_usage().ok()
    }

    fn emit_context(&self, agent: &Agent) {
        if let Ok((used, budget)) = agent.context_usage() {
            self.emit(EventKind::Context { used, budget });
        }
    }

    /// Starts a turn answering `text`, with its approvals answered by `answers`.
    pub fn send_with(self: &Arc<Self>, text: &str, answers: Answers) -> Result<(), SendError> {
        let text = text.trim().to_string();
        if text.is_empty() {
            return Err(SendError::Empty);
        }
        {
            let held = self.hub.held.read().expect("hold lock");
            if *held {
                return Err(SendError::Held);
            }
            let mut turn = self.turn.lock().expect("turn lock");
            if turn.busy {
                return Err(SendError::Busy);
            }
            turn.busy = true;
            turn.events.clear();
        }
        self.emit(EventKind::User { text: text.clone() });
        let live = self.clone();
        let task = tokio::spawn(async move {
            let mut agent = live.agent.lock().await;
            let mut events = Relay {
                live: live.clone(),
                call: 0,
                answers,
                workdir: live.workdir.clone(),
            };
            let result = agent.send(&text, &mut events).await;
            live.emit_context(&agent);
            let kind = match result {
                Ok(reply) => EventKind::Done {
                    text: super::display::split_reasoning(&reply.text).1.to_string(),
                    steps: reply.steps,
                    tool_calls: reply.tool_calls,
                    stopped_early: reply.stopped_early,
                },
                Err(e) => EventKind::Error {
                    message: format!("{e:#}"),
                },
            };
            let transcript = transcript_of(&agent);
            drop(agent);
            live.emit_with(kind, Some(transcript));
            // Messages sent after the agent last asked for them start the
            // next turn.
            let left = std::mem::take(&mut live.turn.lock().expect("turn lock").queued);
            if !left.is_empty() {
                let _ = live.send(&left.join("\n\n"));
            }
        });
        let mut turn = self.turn.lock().expect("turn lock");
        // A turn that already finished has cleared `busy`; its handle is moot.
        if turn.busy {
            turn.task = Some(task.abort_handle());
        }
        Ok(())
    }

    /// Answers approval `id`. False when it is not the one waiting.
    pub fn answer(&self, id: u64, decision: Approval) -> bool {
        let sender = {
            let mut turn = self.turn.lock().expect("turn lock");
            match turn.pending.take() {
                Some((pending, sender)) if pending == id => sender,
                other => {
                    turn.pending = other;
                    return false;
                }
            }
        };
        self.emit(EventKind::Approved {
            approval: id,
            decision: decision_name(&decision),
            reason: decision_reason(&decision),
        });
        sender.send(decision).is_ok()
    }

    /// Stops the turn in progress. False when there is none.
    pub fn stop(self: &Arc<Self>) -> bool {
        let task = {
            let mut turn = self.turn.lock().expect("turn lock");
            if !turn.busy {
                return false;
            }
            turn.pending = None;
            (turn.task.take(), std::mem::take(&mut turn.queued))
        };
        let (task, dropped) = task;
        if let Some(task) = task {
            task.abort();
        }
        self.emit(EventKind::Stopped { dropped });
        // The stopped turn kept what it had saved; show that once its task
        // has let go of the agent.
        let live = self.clone();
        tokio::spawn(async move {
            let agent = live.agent.lock().await;
            let transcript = transcript_of(&agent);
            drop(agent);
            live.turn.lock().expect("turn lock").transcript = transcript;
        });
        true
    }
}

/// Reports a turn's events to a conversation's followers, and waits for
/// approvals from them.
struct Relay {
    live: Arc<Live>,
    call: u32,
    answers: Answers,
    workdir: PathBuf,
}

#[async_trait]
impl AgentEvents for Relay {
    async fn text(&mut self, delta: &str) {
        self.live.emit(EventKind::Text {
            delta: delta.to_string(),
        });
    }

    async fn tool_call(&mut self, call: &ToolCall, summary: &str, access: Access) {
        self.call += 1;
        self.live.emit(EventKind::ToolCall {
            call: self.call,
            name: call.name.clone(),
            summary: summary.to_string(),
            access: access.as_str(),
            arguments: Json::Object(call.arguments.clone()),
        });
    }

    async fn approve(&mut self, call: &ToolCall, summary: &str, access: Access) -> Approval {
        let id = self.live.hub.next();
        // Announced like any approval, so clients show a call answered in
        // advance the same way as one the user answered.
        let asked = EventKind::Approval {
            approval: id,
            call: self.call,
            summary: summary.to_string(),
            access: access.as_str(),
        };
        if let Answers::Allowed(allowed) = &self.answers {
            self.live.emit(asked);
            let decision = if allowed.contains(&call.name) {
                Approval::Once
            } else {
                self.live.emit(EventKind::Notice {
                    message: format!(
                        "{} is not allowed in this schedule, so the call was declined.",
                        call.name
                    ),
                });
                Approval::Deny
            };
            self.live.emit(EventKind::Approved {
                approval: id,
                decision: decision_name(&decision),
                reason: None,
            });
            return decision;
        }
        let (tx, rx) = oneshot::channel();
        self.live.turn.lock().expect("turn lock").pending = Some((id, tx));
        self.live.emit(asked);
        rx.await.unwrap_or(Approval::Deny)
    }

    async fn file_changed(&mut self, _call: &ToolCall, change: &FileChange) {
        if let Err(e) = self.live.changes.record(change) {
            eprintln!("Could not save the list of changed files: {e:#}");
        }
        let workdir = &self.workdir;
        self.live.emit(EventKind::FileChanged {
            call: self.call,
            path: change
                .path
                .strip_prefix(workdir)
                .unwrap_or(&change.path)
                .display()
                .to_string(),
        });
    }

    async fn tool_result(&mut self, _call: &ToolCall, outcome: &ToolOutcome) {
        self.live.emit(EventKind::ToolResult {
            call: self.call,
            ok: outcome.ok,
            denied: outcome.denied,
            output: outcome.output.clone(),
        });
    }

    async fn notice(&mut self, message: &str) {
        self.live.emit(EventKind::Notice {
            message: message.to_string(),
        });
    }

    async fn interjections(&mut self) -> Vec<String> {
        let texts = std::mem::take(&mut self.live.turn.lock().expect("turn lock").queued);
        for text in &texts {
            self.live.emit(EventKind::User { text: text.clone() });
        }
        texts
    }
}

/// Everything a conversation's agent is made from.
pub struct AgentSetup {
    pub model: Arc<dyn AgentModel>,
    pub memory: Option<Arc<Memory>>,
    pub workdir: PathBuf,
    pub max_steps: usize,
    /// Run every call without asking, as `kjarni agent --yes` does.
    pub approve_all: bool,
}

/// The daemon's conversations, loaded from disk as they are used.
pub struct Conversations {
    setup: AgentSetup,
    sessions_dir: PathBuf,
    live: Mutex<HashMap<String, Arc<Live>>>,
    hub: Arc<Hub>,
}

/// Where conversation `id`'s list of changed files is saved.
fn changes_file(sessions_dir: &std::path::Path, id: &str) -> PathBuf {
    sessions_dir.join(format!("{id}.changes.json"))
}

/// Whether `id` can name a session file: no separators, no `..`.
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

impl Conversations {
    pub fn new(setup: AgentSetup, sessions_dir: PathBuf) -> Self {
        Self {
            setup,
            sessions_dir,
            live: Mutex::new(HashMap::new()),
            hub: Arc::new(Hub::default()),
        }
    }

    pub fn setup(&self) -> &AgentSetup {
        &self.setup
    }

    /// Saved conversations, most recent first.
    pub fn list(&self) -> Result<Vec<SessionInfo>> {
        Session::list(&self.sessions_dir)
    }

    /// Whether conversation `id` is in the middle of a turn.
    pub fn busy(&self, id: &str) -> bool {
        self.live
            .lock()
            .expect("conversations lock")
            .get(id)
            .is_some_and(|l| l.busy())
    }

    /// Whether any conversation is in the middle of a turn.
    pub fn any_busy(&self) -> bool {
        self.live
            .lock()
            .expect("conversations lock")
            .values()
            .any(|l| l.busy())
    }

    /// Whether new turns are held back now.
    pub fn held(&self) -> bool {
        *self.hub.held.read().expect("hold lock")
    }

    /// Holds new turns back until the returned [`Hold`] is dropped. `None`
    /// while a turn is running, or turns are held already.
    pub fn hold(&self) -> Option<Hold> {
        let mut held = self.hub.held.write().expect("hold lock");
        if *held || self.any_busy() {
            return None;
        }
        *held = true;
        Some(Hold(self.hub.clone()))
    }

    /// A new, empty conversation.
    pub fn create(&self) -> Result<Arc<Live>> {
        let session = Session::create(&self.sessions_dir)?;
        let id = session.id().to_string();
        let live = self.wrap(id.clone(), self.agent().with_session(session));
        self.live
            .lock()
            .expect("conversations lock")
            .insert(id, live.clone());
        Ok(live)
    }

    /// Conversation `id`, read from disk if it is not loaded. `None` when
    /// there is no such conversation.
    pub fn get(&self, id: &str) -> Result<Option<Arc<Live>>> {
        if !valid_id(id) {
            return Ok(None);
        }
        if let Some(live) = self.live.lock().expect("conversations lock").get(id) {
            return Ok(Some(live.clone()));
        }
        if !self.sessions_dir.join(format!("{id}.jsonl")).exists() {
            return Ok(None);
        }
        let session = Session::open(&self.sessions_dir, id)?;
        let restored = session.load()?;
        let agent = self.agent().with_session(session).with_history(restored);
        let mut live = self.live.lock().expect("conversations lock");
        // Another request may have loaded it meanwhile; keep that one.
        let entry = live
            .entry(id.to_string())
            .or_insert_with(|| self.wrap(id.to_string(), agent));
        Ok(Some(entry.clone()))
    }

    /// Deletes conversation `id`'s file. False when there is no such
    /// conversation; an error while it is mid-turn.
    pub fn delete(&self, id: &str) -> Result<bool> {
        if !valid_id(id) {
            return Ok(false);
        }
        let mut live = self.live.lock().expect("conversations lock");
        if live.get(id).is_some_and(|l| l.busy()) {
            return Err(anyhow!("the conversation is answering; stop it first"));
        }
        let path = self.sessions_dir.join(format!("{id}.jsonl"));
        if !path.exists() {
            return Ok(false);
        }
        std::fs::remove_file(&path)?;
        let _ = std::fs::remove_file(changes_file(&self.sessions_dir, id));
        live.remove(id);
        Ok(true)
    }

    fn agent(&self) -> Agent {
        let workdir = &self.setup.workdir;
        let notes = kjarni::agent::project_notes(workdir);
        let config = AgentConfig {
            max_steps: self.setup.max_steps,
            workdir: Some(workdir.display().to_string()),
            project_notes: notes.map(|n| n.text),
            ..Default::default()
        };
        let workspace = Arc::new(tools::Workspace::new(workdir));
        let mut agent = Agent::new(
            self.setup.model.clone(),
            tools::builtin(workspace, self.setup.memory.clone()),
            config,
        );
        if let Some(memory) = &self.setup.memory {
            agent = agent.with_memory(memory.clone());
        }
        if self.setup.approve_all {
            for access in [Access::Write, Access::Execute, Access::Network] {
                agent.approve_access(access);
            }
        }
        agent
    }

    fn wrap(&self, id: String, agent: Agent) -> Arc<Live> {
        let (tx, _) = broadcast::channel(1024);
        let transcript = transcript_of(&agent);
        let changes = Changes::open(changes_file(&self.sessions_dir, &id));
        Arc::new(Live {
            id,
            agent: tokio::sync::Mutex::new(agent),
            turn: Mutex::new(Turn {
                busy: false,
                events: Vec::new(),
                pending: None,
                task: None,
                queued: Vec::new(),
                transcript,
            }),
            tx,
            hub: self.hub.clone(),
            changes,
            workdir: self.setup.workdir.clone(),
        })
    }
}
