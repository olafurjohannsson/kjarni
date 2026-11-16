//! The agent loop: the model replies, its tool calls run, their results go
//! back to it, until it answers without calling anything.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use kjarni_transformers::chat::tools::displayable_prefix_len;
use kjarni_transformers::{Conversation, Message, Role, ToolCall, ToolSpec, parse_tool_calls};

use super::memory::Memory;
use super::model::AgentModel;
use super::session::{Restored, Session, today};
use super::tool::{Access, Tool};
use super::tools::shell::clip;

/// What the agent is told it is, unless the caller gives instructions of its own.
pub const DEFAULT_INSTRUCTIONS: &str = "You are Kjarni, an assistant running entirely on the user's own computer. \
You can use tools to look at and change files, run commands, read web pages, and remember things between conversations. \
Use a tool when it gets you facts you do not have or does what the user asked; otherwise answer directly. \
When a question may be answered by the user's files, look in the working folder before saying you do not know: search_documents finds passages by meaning, search_files finds exact text, and read_file shows a whole file. \
Look before you change: read a file before editing it. \
When the user tells you something worth keeping, such as a preference or a fact about their work, save it with remember. \
Keep answers short and concrete.";

/// How the agent runs.
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// Instructions that replace [`DEFAULT_INSTRUCTIONS`].
    pub instructions: Option<String>,
    /// Model replies per user message at most, so a model that keeps calling
    /// tools stops.
    pub max_steps: usize,
    /// Bytes of a tool's output the model sees at most.
    pub max_tool_output: usize,
    /// Prompt tokens at which earlier messages are folded into a summary.
    pub context_budget: usize,
    /// Messages kept word for word when folding.
    pub keep_recent: usize,
    /// Saved facts listed in the prompt, newest last.
    pub facts_in_prompt: usize,
    /// The folder the tools work in, as the prompt names it.
    pub workdir: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            instructions: None,
            max_steps: 16,
            max_tool_output: 12_000,
            context_budget: 16_384,
            keep_recent: 6,
            facts_in_prompt: 20,
            workdir: None,
        }
    }
}

/// The user's answer to a call that needs approval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Approval {
    /// Run this call.
    Once,
    /// Run this call and later calls of the same tool without asking.
    Always,
    /// Do not run it; the model is told so.
    Deny,
}

/// The result of one call, as a front end shows it.
#[derive(Clone, Debug)]
pub struct ToolOutcome {
    /// Whether the tool ran and succeeded.
    pub ok: bool,
    /// Whether the user declined the call.
    pub denied: bool,
    /// What the model is shown.
    pub output: String,
}

/// What happens during a turn, for a front end to show. Every method has a
/// default; [`approve`](Self::approve)'s declines.
#[async_trait]
pub trait AgentEvents: Send {
    /// Reply text as it is generated. Tool-call markup is not included.
    async fn text(&mut self, _delta: &str) {}

    /// The model asked for `call`.
    async fn tool_call(&mut self, _call: &ToolCall, _summary: &str, _access: Access) {}

    /// Whether `call` may run. Asked only for calls whose access is not
    /// approved already.
    async fn approve(&mut self, _call: &ToolCall, _summary: &str, _access: Access) -> Approval {
        Approval::Deny
    }

    /// `call` finished.
    async fn tool_result(&mut self, _call: &ToolCall, _outcome: &ToolOutcome) {}

    /// Something the user should know about, such as earlier messages being
    /// summarized.
    async fn notice(&mut self, _message: &str) {}
}

/// Events that go nowhere, and no approvals.
pub struct NoEvents;

#[async_trait]
impl AgentEvents for NoEvents {}

/// The end of a turn.
#[derive(Clone, Debug, Default)]
pub struct AgentReply {
    /// The final answer.
    pub text: String,
    /// Model replies in the turn.
    pub steps: usize,
    /// Tool calls made.
    pub tool_calls: usize,
    /// Whether the turn ended because it reached `max_steps`.
    pub stopped_early: bool,
}

/// An assistant that uses tools, remembers, and keeps its conversation.
pub struct Agent {
    model: Arc<dyn AgentModel>,
    tools: Vec<Arc<dyn Tool>>,
    specs: Vec<ToolSpec>,
    config: AgentConfig,
    messages: Vec<Message>,
    summary: Option<String>,
    approved_access: HashSet<Access>,
    approved_tools: HashSet<String>,
    memory: Option<Arc<Memory>>,
    session: Option<Session>,
}

impl Agent {
    pub fn new(model: Arc<dyn AgentModel>, tools: Vec<Arc<dyn Tool>>, config: AgentConfig) -> Self {
        let specs = tools.iter().map(|t| t.spec()).collect();
        Self {
            model,
            tools,
            specs,
            config,
            messages: Vec::new(),
            summary: None,
            approved_access: HashSet::from([Access::Read]),
            approved_tools: HashSet::new(),
            memory: None,
            session: None,
        }
    }

    /// Lists the memory's facts in the prompt.
    pub fn with_memory(mut self, memory: Arc<Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Saves the conversation to `session` as it goes.
    pub fn with_session(mut self, session: Session) -> Self {
        self.session = Some(session);
        self
    }

    /// Continues a saved conversation.
    pub fn with_history(mut self, restored: Restored) -> Self {
        self.messages = restored.messages;
        self.summary = restored.summary;
        self
    }

    /// Runs calls needing `access` without asking. Reads are approved from
    /// the start.
    pub fn approve_access(&mut self, access: Access) {
        self.approved_access.insert(access);
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    pub fn tools(&self) -> &[ToolSpec] {
        &self.specs
    }

    pub fn session(&self) -> Option<&Session> {
        self.session.as_ref()
    }

    pub fn model_name(&self) -> String {
        self.model.name()
    }

    /// Starts a new conversation, saved to `session` when given. The previous
    /// session's file is left as it is.
    pub fn start_over(&mut self, session: Option<Session>) {
        self.messages.clear();
        self.summary = None;
        self.session = session;
    }

    /// The system prompt: instructions, where and when, saved facts, and the
    /// summary of earlier messages.
    pub fn system_prompt(&self) -> String {
        let mut prompt = self
            .config
            .instructions
            .clone()
            .unwrap_or_else(|| DEFAULT_INSTRUCTIONS.to_string());
        prompt.push_str(&format!("\n\nToday's date: {}.", today()));
        if let Some(dir) = &self.config.workdir {
            prompt.push_str(&format!(" Working directory: {dir}."));
        }
        if let Some(memory) = &self.memory {
            let facts = memory.recent(self.config.facts_in_prompt);
            if !facts.is_empty() {
                prompt.push_str("\n\nWhat you remember from earlier conversations:");
                for fact in facts {
                    prompt.push_str(&format!("\n- {}", fact.text));
                }
            }
        }
        if let Some(summary) = &self.summary {
            prompt.push_str("\n\nSummary of this conversation so far:\n");
            prompt.push_str(summary);
        }
        prompt
    }

    /// The conversation as the model sees it next.
    pub fn conversation(&self) -> Conversation {
        let mut c = Conversation::with_system(self.system_prompt());
        c.set_tools(self.specs.clone());
        for m in &self.messages {
            c.push(m.clone());
        }
        c
    }

    fn push(&mut self, message: Message) -> anyhow::Result<()> {
        if let Some(session) = &self.session {
            session.append_message(&message)?;
        }
        self.messages.push(message);
        Ok(())
    }

    /// Answers `input`, calling tools as the model asks, and reports each step
    /// to `events`.
    pub async fn send(
        &mut self,
        input: &str,
        events: &mut dyn AgentEvents,
    ) -> anyhow::Result<AgentReply> {
        self.push(Message::user(input))?;
        let mut reply = AgentReply::default();
        for _ in 0..self.config.max_steps {
            self.fold_if_needed(events).await?;
            let prompt = self.model.render(&self.conversation())?;
            let raw = self.generate(prompt, events).await?;
            reply.steps += 1;

            let parsed = parse_tool_calls(&raw, &self.specs);
            self.push(Message::assistant(raw.trim()))?;
            if parsed.tool_calls.is_empty() && parsed.malformed.is_empty() {
                reply.text = parsed.content;
                return Ok(reply);
            }
            for call in &parsed.tool_calls {
                reply.tool_calls += 1;
                let outcome = self.run(call, events).await;
                events.tool_result(call, &outcome).await;
                self.push(Message::tool(outcome.output))?;
            }
            for block in &parsed.malformed {
                let output = format!(
                    "Error: this tool call could not be read, so nothing ran:\n{}\nWrite calls exactly in the format the system prompt shows.",
                    clip(block)
                );
                self.push(Message::tool(output))?;
            }
        }
        reply.stopped_early = true;
        events
            .notice(&format!(
                "Stopped after {} steps without a final answer.",
                self.config.max_steps
            ))
            .await;
        Ok(reply)
    }

    /// The model's reply to `prompt`, with its prose streamed to `events` as
    /// it comes and tool-call markup held back.
    async fn generate(
        &self,
        prompt: String,
        events: &mut dyn AgentEvents,
    ) -> anyhow::Result<String> {
        let mut stream = self.model.stream(prompt).await?;
        let mut text = String::new();
        let mut shown = 0;
        while let Some(piece) = stream.next().await {
            text.push_str(&piece?);
            let showable = displayable_prefix_len(&text);
            if showable > shown {
                events.text(&text[shown..showable]).await;
                shown = showable;
            }
        }
        // A trailing fragment held back in case it began a tag, which it did not.
        if !text.contains("<tool_call>") && shown < text.len() {
            events.text(&text[shown..]).await;
        }
        Ok(strip_stop_sequences(&text).to_string())
    }

    async fn run(&mut self, call: &ToolCall, events: &mut dyn AgentEvents) -> ToolOutcome {
        let Some(tool) = self
            .tools
            .iter()
            .find(|t| t.spec().name == call.name)
            .cloned()
        else {
            let names: Vec<_> = self.specs.iter().map(|s| s.name.as_str()).collect();
            return ToolOutcome {
                ok: false,
                denied: false,
                output: format!(
                    "Error: there is no tool named '{}'. The tools are: {}.",
                    call.name,
                    names.join(", ")
                ),
            };
        };
        let access = tool.access();
        let summary = tool.summary(&call.arguments);
        events.tool_call(call, &summary, access).await;

        let approved =
            self.approved_access.contains(&access) || self.approved_tools.contains(&call.name);
        if !approved {
            match events.approve(call, &summary, access).await {
                Approval::Once => {}
                Approval::Always => {
                    self.approved_tools.insert(call.name.clone());
                }
                Approval::Deny => {
                    return ToolOutcome {
                        ok: false,
                        denied: true,
                        output: "The user declined this call, so it did not run. Ask them what to do instead, or continue without it.".into(),
                    };
                }
            }
        }
        match tool.call(&call.arguments).await {
            Ok(output) => ToolOutcome {
                ok: true,
                denied: false,
                output: clip_to(&output, self.config.max_tool_output),
            },
            Err(e) => ToolOutcome {
                ok: false,
                denied: false,
                output: format!("Error: {e:#}"),
            },
        }
    }

    /// Folds the older messages into the summary when the prompt has grown
    /// past the budget.
    async fn fold_if_needed(&mut self, events: &mut dyn AgentEvents) -> anyhow::Result<()> {
        let prompt = self.model.render(&self.conversation())?;
        let budget = self.config.context_budget.min(self.model.context_size());
        if self.model.count_tokens(&prompt)? <= budget {
            return Ok(());
        }
        let Some(cut) = fold_point(&self.messages, self.config.keep_recent) else {
            return Ok(());
        };
        events
            .notice(&format!("Summarizing {cut} earlier messages to make room."))
            .await;
        let summary = self.summarize(&self.messages[..cut]).await?;
        self.messages.drain(..cut);
        if let Some(session) = &self.session {
            session.append_summary(&summary, cut)?;
        }
        self.summary = Some(summary);
        Ok(())
    }

    async fn summarize(&self, messages: &[Message]) -> anyhow::Result<String> {
        let mut transcript = String::new();
        if let Some(previous) = &self.summary {
            transcript.push_str(&format!("Earlier summary:\n{previous}\n\n"));
        }
        for m in messages {
            let content = match m.role {
                Role::Tool => clip_to(&m.content, 1500),
                _ => m.content.clone(),
            };
            transcript.push_str(&format!("{}: {}\n\n", m.role, content));
        }
        let mut c = Conversation::with_system(
            "You write short, factual summaries of conversations between a user and an assistant that uses tools.",
        );
        c.push_user(format!(
            "Summarize this conversation so the assistant can continue it. Keep what the user wants, decisions made, files and commands involved, results, and anything still to do. Use at most 200 words.\n\n{transcript}"
        ));
        let prompt = self.model.render(&c)?;
        let mut stream = self.model.stream(prompt).await?;
        let mut text = String::new();
        while let Some(piece) = stream.next().await {
            text.push_str(&piece?);
        }
        Ok(strip_reasoning(strip_stop_sequences(&text))
            .trim()
            .to_string())
    }
}

/// Where to cut `messages` so that at least `keep` recent ones stay and the
/// kept part starts at a user message, never inside a tool exchange. `None`
/// when there is nothing to fold.
fn fold_point(messages: &[Message], keep: usize) -> Option<usize> {
    let latest = messages.len().saturating_sub(keep);
    (1..=latest).rev().find(|&i| messages[i].role == Role::User)
}

/// `text` without a trailing stop sequence the generator passed through.
fn strip_stop_sequences(text: &str) -> &str {
    let mut text = text.trim_end();
    for stop in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"] {
        if let Some(t) = text.strip_suffix(stop) {
            text = t.trim_end();
        }
    }
    text
}

/// `text` without a leading `<think>...</think>` block.
fn strip_reasoning(text: &str) -> &str {
    match text.split_once("</think>") {
        Some((_, after)) => after,
        None => text,
    }
}

fn clip_to(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let half = max / 2;
    let mut head = half;
    while !text.is_char_boundary(head) {
        head -= 1;
    }
    let mut tail = text.len() - half;
    while !text.is_char_boundary(tail) {
        tail += 1;
    }
    format!(
        "{}\n[... {} bytes left out ...]\n{}",
        &text[..head],
        tail - head,
        &text[tail..]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folding_keeps_whole_tool_exchanges() {
        let messages = vec![
            Message::user("a"),
            Message::assistant("call"),
            Message::tool("result"),
            Message::assistant("done"),
            Message::user("b"),
            Message::assistant("call"),
            Message::tool("result"),
            Message::tool("result"),
            Message::assistant("done"),
        ];
        // Keeping 3 would start inside the second exchange; the cut moves back
        // to the user message that opened it.
        assert_eq!(fold_point(&messages, 3), Some(4));
        // Nothing before the first user message to fold.
        assert_eq!(fold_point(&messages[..4], 2), None);
    }

    #[test]
    fn long_output_keeps_its_start_and_end() {
        let text = "a".repeat(50) + &"b".repeat(50);
        let clipped = clip_to(&text, 20);
        assert!(clipped.starts_with("aaaaaaaaaa\n[... 80 bytes left out ...]\nbbbbbbbbbb"));
        assert_eq!(clip_to("short", 20), "short");
    }
}
