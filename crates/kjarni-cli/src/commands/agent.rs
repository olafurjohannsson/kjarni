//! `kjarni agent`: an assistant in the terminal that uses tools, remembers
//! between sessions, and keeps its conversations.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use colored::Colorize;

use kjarni::agent::{
    Access, Agent, AgentConfig, AgentEvents, Approval, Embed, Memory, Session, ToolCall,
    ToolOutcome, data_dir, tools,
};
use kjarni::chat::Chat;
use kjarni::embedder::Embedder;

/// Embedding model for recalling memories by meaning.
const MEMORY_MODEL: &str = "minilm-l6-v2";

pub struct AgentArgs {
    pub model: String,
    pub model_path: Option<String>,
    pub full_precision: bool,
    pub dir: Option<String>,
    pub yes: bool,
    pub continue_last: bool,
    pub resume: Option<String>,
    pub no_memory: bool,
    pub prompt: Option<String>,
    pub temperature: f32,
    pub max_tokens: usize,
    pub max_steps: usize,
    pub quiet: bool,
    pub list_sessions: bool,
}

pub async fn run(args: AgentArgs) -> Result<()> {
    let home = data_dir();
    let sessions_dir = home.join("sessions");

    if args.list_sessions {
        return list_sessions(&sessions_dir);
    }

    let workdir = match &args.dir {
        Some(d) => PathBuf::from(d),
        None => std::env::current_dir()?,
    };
    let workdir = workdir
        .canonicalize()
        .with_context(|| format!("cannot open {}", workdir.display()))?;

    if args.model_path.is_none() {
        let _ = crate::commands::util::resolve_model(&args.model, Some("decoder"))?;
    }
    let config = AgentConfig {
        max_steps: args.max_steps,
        workdir: Some(workdir.display().to_string()),
        ..Default::default()
    };

    if !args.quiet {
        eprintln!(
            "{}",
            format!(
                "Loading {}...",
                args.model_path.as_deref().unwrap_or(&args.model)
            )
            .dimmed()
        );
    }
    let mut builder = Chat::builder(&args.model)
        .temperature(args.temperature)
        .max_tokens(args.max_tokens)
        .prefix_cache_tokens(config.context_budget + args.max_tokens)
        .cpu()
        .quiet();
    if let Some(path) = &args.model_path {
        builder = builder.model_path(path);
    }
    if !args.full_precision {
        builder = builder.with_load_config(|c| c.prefer_gguf(true));
    }
    let chat = builder
        .build()
        .await
        .map_err(|e| anyhow!("failed to load the model: {e}"))?;
    if !chat.supports_tools() {
        return Err(anyhow!(
            "{} has no tool-calling format. Use a Qwen 3.5 model, such as qwen3.5-4b.",
            chat.model_name()
        ));
    }

    let memory = if args.no_memory {
        None
    } else {
        let embedder: Option<Arc<dyn Embed>> =
            match Embedder::builder(MEMORY_MODEL).quiet(true).build().await {
                Ok(e) => Some(Arc::new(e)),
                Err(e) => {
                    eprintln!(
                        "{}",
                        format!("Memory will match words, not meaning: {e}").yellow()
                    );
                    None
                }
            };
        Some(Arc::new(Memory::open(&home, embedder)?))
    };

    let (session, restored) = if let Some(id) = &args.resume {
        let s = Session::open(&sessions_dir, id)?;
        let r = s.load()?;
        (s, Some(r))
    } else if args.continue_last {
        match Session::latest(&sessions_dir)? {
            Some(s) => {
                let r = s.load()?;
                (s, Some(r))
            }
            None => (Session::create(&sessions_dir)?, None),
        }
    } else {
        (Session::create(&sessions_dir)?, None)
    };

    let workspace = Arc::new(tools::Workspace::new(&workdir));
    let model_name = chat.model_name().to_string();
    let mut agent = Agent::new(
        Arc::new(chat),
        tools::builtin(workspace, memory.clone()),
        config,
    )
    .with_session(session);
    if let Some(memory) = &memory {
        agent = agent.with_memory(memory.clone());
    }
    if let Some(restored) = restored {
        agent = agent.with_history(restored);
    }
    if args.yes {
        for access in [Access::Write, Access::Execute, Access::Network] {
            agent.approve_access(access);
        }
    }

    let mut events = Terminal::default();

    if let Some(prompt) = &args.prompt {
        let reply = agent.send(prompt, &mut events).await?;
        events.finish_text();
        if reply.stopped_early {
            std::process::exit(2);
        }
        return Ok(());
    }

    if !args.quiet {
        print_welcome(&model_name, &workdir, &agent, memory.as_deref(), args.yes);
    }

    let stdin = io::stdin();
    loop {
        print!("{} ", "›".bold().cyan());
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input.starts_with('/') {
            match command(input, &mut agent, memory.as_deref(), &sessions_dir) {
                Flow::Quit => break,
                Flow::Continue => continue,
            }
        }
        println!();
        match agent.send(input, &mut events).await {
            Ok(_) => events.finish_text(),
            Err(e) => {
                events.finish_text();
                eprintln!("{} {e:#}", "error:".red().bold());
            }
        }
        println!();
    }
    Ok(())
}

enum Flow {
    Quit,
    Continue,
}

fn command(
    input: &str,
    agent: &mut Agent,
    memory: Option<&Memory>,
    sessions: &std::path::Path,
) -> Flow {
    let mut parts = input.splitn(2, ' ');
    let name = parts.next().unwrap_or("");
    let arg = parts.next().unwrap_or("").trim();
    match name {
        "/quit" | "/exit" | "/q" => return Flow::Quit,
        "/help" | "/?" => print_help(),
        "/clear" | "/new" => match Session::create(sessions) {
            Ok(s) => {
                let id = s.id().to_string();
                agent.start_over(Some(s));
                println!("{}", format!("Started a new conversation ({id}).").dimmed());
            }
            Err(e) => eprintln!("{} {e:#}", "error:".red().bold()),
        },
        "/tools" => {
            for spec in agent.tools() {
                println!("  {}  {}", spec.name.bold(), spec.description.dimmed());
            }
        }
        "/memory" => match memory {
            None => println!("{}", "Memory is off (--no-memory).".dimmed()),
            Some(m) if m.is_empty() => println!("{}", "Nothing remembered yet.".dimmed()),
            Some(m) => {
                for e in m.entries() {
                    println!("  {} {}", format!("#{}", e.id).dimmed(), e.text);
                }
                println!("{}", format!("  Kept in {}", m.path().display()).dimmed());
            }
        },
        "/forget" => match (memory, arg.trim_start_matches('#').parse::<u64>()) {
            (Some(m), Ok(id)) => match m.forget(id) {
                Ok(true) => println!("{}", format!("Forgot #{id}.").dimmed()),
                Ok(false) => println!("{}", format!("There is no #{id}.").dimmed()),
                Err(e) => eprintln!("{} {e:#}", "error:".red().bold()),
            },
            (None, _) => println!("{}", "Memory is off (--no-memory).".dimmed()),
            (_, Err(_)) => println!(
                "{}",
                "Usage: /forget <number>, as /memory lists them.".dimmed()
            ),
        },
        "/sessions" => {
            if let Err(e) = list_sessions(sessions) {
                eprintln!("{} {e:#}", "error:".red().bold());
            }
        }
        _ => println!(
            "{}",
            format!("Unknown command {name}. Type /help.").dimmed()
        ),
    }
    Flow::Continue
}

fn list_sessions(dir: &std::path::Path) -> Result<()> {
    let sessions = Session::list(dir)?;
    if sessions.is_empty() {
        println!("{}", "No saved conversations yet.".dimmed());
        return Ok(());
    }
    for s in sessions.iter().take(20) {
        println!("  {}  {}", s.id.dimmed(), s.title);
    }
    println!(
        "{}",
        "  Continue one with: kjarni agent --resume <id>   (or -c for the latest)".dimmed()
    );
    Ok(())
}

fn print_welcome(
    model: &str,
    workdir: &std::path::Path,
    agent: &Agent,
    memory: Option<&Memory>,
    yes: bool,
) {
    println!();
    println!("  {}  {}", "kjarni agent".bold(), model.dimmed());
    println!("  {} {}", "folder".dimmed(), workdir.display());
    if let Some(m) = memory {
        println!("  {} {} facts", "memory".dimmed(), m.len());
    }
    let resumed = agent.messages().len();
    if resumed > 0 || agent.summary().is_some() {
        println!("  {} continuing, {resumed} messages", "session".dimmed());
    }
    if yes {
        println!(
            "  {}",
            "every tool call runs without asking (--yes)".yellow()
        );
    } else {
        println!(
            "  {}",
            "asks before writing files, running commands or going online".dimmed()
        );
    }
    println!("  {}", "/help for commands, Ctrl-D to quit".dimmed());
    println!();
}

fn print_help() {
    let rows = [
        ("/tools", "list the tools the agent can use"),
        ("/memory", "show what it remembers"),
        ("/forget N", "delete remembered fact N"),
        ("/sessions", "list saved conversations"),
        ("/clear", "start a new conversation"),
        ("/quit", "leave (or Ctrl-D)"),
    ];
    for (cmd, what) in rows {
        println!("  {:<12} {}", cmd.bold(), what.dimmed());
    }
}

/// Shows a turn in the terminal and asks for approvals on stdin.
#[derive(Default)]
struct Terminal {
    /// Whether text is mid-line.
    in_text: bool,
}

impl Terminal {
    fn finish_text(&mut self) {
        if self.in_text {
            println!();
            self.in_text = false;
        }
    }
}

#[async_trait]
impl AgentEvents for Terminal {
    async fn text(&mut self, delta: &str) {
        let delta = if self.in_text {
            delta
        } else {
            delta.trim_start()
        };
        if delta.is_empty() {
            return;
        }
        print!("{delta}");
        let _ = io::stdout().flush();
        self.in_text = true;
    }

    async fn tool_call(&mut self, _call: &ToolCall, summary: &str, _access: Access) {
        self.finish_text();
        println!("{} {}", "●".cyan(), summary.bold());
    }

    async fn approve(&mut self, _call: &ToolCall, _summary: &str, access: Access) -> Approval {
        let verb = match access {
            Access::Write => "write",
            Access::Execute => "run this",
            Access::Network => "go online",
            Access::Read => "read",
        };
        loop {
            print!(
                "  {} {} ",
                format!("Allow it to {verb}?").yellow(),
                "[y]es  [a]lways  [n]o ›".dimmed()
            );
            let _ = io::stdout().flush();
            let mut line = String::new();
            if io::stdin().lock().read_line(&mut line).unwrap_or(0) == 0 {
                return Approval::Deny;
            }
            match line.trim().to_lowercase().as_str() {
                "y" | "yes" => return Approval::Once,
                "a" | "always" => return Approval::Always,
                "n" | "no" | "" => return Approval::Deny,
                _ => continue,
            }
        }
    }

    async fn tool_result(&mut self, _call: &ToolCall, outcome: &ToolOutcome) {
        let first = outcome.output.lines().next().unwrap_or("").trim();
        let first: String = first.chars().take(100).collect();
        let more = outcome.output.lines().count().saturating_sub(1);
        let tail = if more > 0 {
            format!("  (+{more} lines)")
        } else {
            String::new()
        };
        if outcome.denied {
            println!("  {} {}", "⎿".dimmed(), "declined".red());
        } else if outcome.ok {
            println!("  {} {}{}", "⎿".dimmed(), first.dimmed(), tail.dimmed());
        } else {
            println!("  {} {}", "⎿".dimmed(), first.red());
        }
    }

    async fn notice(&mut self, message: &str) {
        self.finish_text();
        println!("{}", message.dimmed());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kjarni::agent::{AgentModel, TextStream};

    /// A model the slash commands never call.
    struct Idle;

    #[async_trait]
    impl AgentModel for Idle {
        fn render(&self, _: &kjarni::Conversation) -> anyhow::Result<String> {
            Ok(String::new())
        }
        async fn stream(&self, _: String) -> anyhow::Result<TextStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
            Ok(text.len())
        }
        fn context_size(&self) -> usize {
            32_768
        }
        fn name(&self) -> String {
            "idle".into()
        }
    }

    fn agent(dir: &std::path::Path) -> Agent {
        let workspace = Arc::new(tools::Workspace::new(dir));
        Agent::new(
            Arc::new(Idle),
            tools::builtin(workspace, None),
            AgentConfig::default(),
        )
    }

    #[test]
    fn quit_commands_end_the_loop_and_others_do_not() {
        let dir = tempfile::tempdir().unwrap();
        let mut a = agent(dir.path());
        for q in ["/quit", "/exit", "/q"] {
            assert!(
                matches!(command(q, &mut a, None, dir.path()), Flow::Quit),
                "{q}"
            );
        }
        for other in [
            "/help",
            "/tools",
            "/memory",
            "/sessions",
            "/forget 1",
            "/nonsense",
        ] {
            assert!(
                matches!(command(other, &mut a, None, dir.path()), Flow::Continue),
                "{other}"
            );
        }
    }

    #[test]
    fn clear_starts_a_new_saved_conversation() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path().join("sessions");
        let mut a = agent(dir.path());
        assert!(a.session().is_none());
        command("/clear", &mut a, None, &sessions);
        let session = a.session().expect("a session");
        assert!(session.path().starts_with(&sessions) && session.path().exists());
        assert!(a.messages().is_empty());
    }

    #[tokio::test]
    async fn forget_takes_the_number_memory_lists() {
        let dir = tempfile::tempdir().unwrap();
        let memory = Memory::open(dir.path(), None).unwrap();
        let first = memory.remember("Prefers tabs.").await.unwrap();
        let second = memory.remember("Lives in Reykjavik.").await.unwrap();
        let mut a = agent(dir.path());
        command(
            &format!("/forget #{}", first.id),
            &mut a,
            Some(&memory),
            dir.path(),
        );
        command("/forget not-a-number", &mut a, Some(&memory), dir.path());
        command("/forget 9999", &mut a, Some(&memory), dir.path());
        let left: Vec<u64> = memory.entries().iter().map(|e| e.id).collect();
        assert_eq!(left, [second.id]);
    }

    #[test]
    fn listing_sessions_works_with_none_saved() {
        let dir = tempfile::tempdir().unwrap();
        assert!(list_sessions(&dir.path().join("sessions")).is_ok());
    }
}
