//! `kjarni daemon`: the agent, its memory and the OpenAI-compatible API
//! behind one local web address.
//!
//! The daemon listens on loopback only unless told otherwise, and every
//! request needs the token kept in the daemon's folder (see [`auth`]). The
//! chat model is loaded once and shared: the agent's conversations and
//! `/v1/chat/completions` take turns on it.

mod api;
mod auth;
mod conversations;
mod display;
#[cfg(test)]
mod tests;

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use axum::Router;
use futures::StreamExt;
use kjarni::Conversation;
use kjarni::agent::{AgentModel, Embed, Memory, TextStream};
use kjarni::chat::Chat;
use kjarni::embedder::Embedder;
use kjarni_cli::DaemonCommands;
use serde::{Deserialize, Serialize};

use super::serve::{self, LoadedDecoder};
pub use api::DaemonState;
pub use auth::Guard;
pub use conversations::{AgentSetup, Conversations};

/// Embedding model for recalling memories by meaning.
const MEMORY_MODEL: &str = "minilm-l6-v2";

/// Tokens of conversation the agent keeps before summarizing, plus room for
/// the reply; the prefix cache holds this much.
const AGENT_MAX_TOKENS: usize = 4096;

/// The daemon's own files: its token and, while it runs, where it listens.
/// `$KJARNI_DAEMON_DIR`, or `kjarni/daemon` in the user's data folder.
pub fn daemon_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("KJARNI_DAEMON_DIR") {
        return dir.into();
    }
    dirs::data_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("kjarni")
        .join("daemon")
}

/// Where a running daemon says it listens, so `open` and `status` find it.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Running {
    pid: u32,
    port: u16,
}

fn running_path(dir: &Path) -> PathBuf {
    dir.join("daemon.json")
}

pub async fn run(action: DaemonCommands) -> Result<()> {
    match action {
        DaemonCommands::Run {
            model,
            gpu,
            full_precision,
            port,
            listen,
            dir,
            yes,
            no_memory,
            max_steps,
            quiet,
        } => {
            serve_daemon(RunOptions {
                model,
                gpu,
                full_precision,
                port,
                listen,
                dir,
                yes,
                no_memory,
                max_steps,
                quiet,
            })
            .await
        }
        DaemonCommands::Open => open(),
        DaemonCommands::Status => status(),
    }
}

struct RunOptions {
    model: String,
    gpu: bool,
    full_precision: bool,
    port: u16,
    listen: Option<String>,
    dir: Option<String>,
    yes: bool,
    no_memory: bool,
    max_steps: usize,
    quiet: bool,
}

/// The chat model the server loaded, as the agent uses it. A reply holds the
/// model's turn until it ends, so the agent and API requests take turns.
struct SharedChat(Arc<LoadedDecoder>);

impl SharedChat {
    fn chat(&self) -> &Chat {
        self.0.chat().expect("the daemon loads a chat model")
    }
}

#[async_trait]
impl AgentModel for SharedChat {
    fn render(&self, conversation: &Conversation) -> Result<String> {
        Ok(self.chat().render_conversation(conversation)?)
    }

    async fn stream(&self, prompt: String) -> Result<TextStream> {
        let turn = self.0.turn.clone().acquire_owned().await?;
        let stream = self
            .chat()
            .generate_stream(prompt, Default::default())
            .await?;
        Ok(Box::pin(stream.map(move |piece| {
            let _held = &turn;
            piece.map_err(anyhow::Error::from)
        })))
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(self.chat().count_tokens(text)?)
    }

    fn context_size(&self) -> usize {
        self.chat().context_size()
    }

    fn name(&self) -> String {
        self.chat().model_name().to_string()
    }
}

/// The daemon's routes and the API's, behind the guard.
pub fn router(state: Arc<DaemonState>, serve_state: Arc<serve::AppState>, log: bool) -> Router {
    let guard = state.guard.clone();
    let api = serve::router(
        serve_state,
        serve::RouterOptions {
            api_key: None,
            cors: false,
            log,
        },
    );
    api::routes()
        .with_state(state)
        .merge(api)
        .layer(axum::middleware::from_fn_with_state(guard, auth::guard))
}

async fn load_chat(options: &RunOptions) -> Result<Chat> {
    let _ = crate::commands::util::resolve_model(&options.model, Some("decoder"))?;
    let mut builder = Chat::builder(&options.model)
        .temperature(0.7)
        .max_tokens(AGENT_MAX_TOKENS)
        .prefix_cache_tokens(
            kjarni::agent::AgentConfig::default().context_budget + AGENT_MAX_TOKENS,
        )
        .quiet();
    if options.gpu {
        if let Some(mt) = kjarni::ModelType::from_cli_name(&options.model) {
            crate::commands::util::check_gpu_capacity(mt, !options.full_precision)?;
        }
        builder = builder.gpu();
    } else {
        builder = builder.cpu();
    }
    if !options.full_precision {
        builder = builder.with_load_config(|c| c.prefer_gguf(true));
    }
    let chat = builder
        .build()
        .await
        .map_err(|e| anyhow!("failed to load {}: {e}", options.model))?;
    if !chat.supports_tools() {
        bail!(
            "{} has no tool-calling format. Use a Qwen 3.5 model, such as qwen3.5-4b.",
            chat.model_name()
        );
    }
    Ok(chat)
}

async fn serve_daemon(options: RunOptions) -> Result<()> {
    let dir = daemon_dir();
    let token = auth::load_or_create_token(&dir.join("token"))?;

    let workdir = match &options.dir {
        Some(d) => PathBuf::from(d),
        None => dirs::home_dir()
            .ok_or_else(|| anyhow!("no home folder; pass --dir"))?
            .join("kjarni"),
    };
    std::fs::create_dir_all(&workdir).with_context(|| format!("creating {}", workdir.display()))?;
    let workdir = workdir.canonicalize()?;

    let host = options.listen.clone().unwrap_or_else(|| "127.0.0.1".into());
    let address: SocketAddr = format!("{host}:{}", options.port)
        .parse()
        .with_context(|| format!("'{host}' is not an IP address to listen on"))?;
    // Bind before loading, so a second daemon fails at once.
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .with_context(|| {
            format!("could not listen on {address}; is the daemon running already?")
        })?;

    eprintln!("Loading {}...", options.model);
    let chat = load_chat(&options).await?;
    let model = chat.model_name().to_string();
    let models = serve::Models::new(
        serve::Defaults {
            chat: model.clone(),
            ..Default::default()
        },
        options.gpu,
        true,
    );
    let decoder = models.insert_chat(&model, chat);

    let agent_home = kjarni::agent::data_dir();
    let memory = if options.no_memory {
        None
    } else {
        let embedder: Option<Arc<dyn Embed>> =
            match Embedder::builder(MEMORY_MODEL).quiet(true).build().await {
                Ok(e) => Some(Arc::new(e)),
                Err(e) => {
                    eprintln!("Memory will match words, not meaning: {e}");
                    None
                }
            };
        Some(Arc::new(Memory::open(&agent_home, embedder)?))
    };

    let state = Arc::new(DaemonState {
        conversations: Conversations::new(
            AgentSetup {
                model: Arc::new(SharedChat(decoder)),
                memory: memory.clone(),
                workdir: workdir.clone(),
                max_steps: options.max_steps,
                approve_all: options.yes,
            },
            agent_home.join("sessions"),
        ),
        guard: Arc::new(Guard::new(token, options.listen.is_some())),
        memory,
        model: model.clone(),
        gpu: options.gpu,
    });
    let app = router(state, Arc::new(serve::AppState { models }), !options.quiet);

    let running = running_path(&dir);
    let port = listener.local_addr()?.port();
    std::fs::write(
        &running,
        serde_json::to_string(&Running {
            pid: std::process::id(),
            port,
        })?,
    )?;

    if options.listen.is_some() {
        eprintln!(
            "Warning: listening on {address}, reachable from other machines. Every request needs the token in {}.",
            dir.join("token").display()
        );
    }
    eprintln!(
        "Kjarni daemon at http://127.0.0.1:{port}, with {model}, working in {}",
        workdir.display()
    );
    eprintln!("Sign in from this computer with `kjarni daemon open`. Ctrl+C stops it.");

    let served = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("the daemon stopped with an error");
    let _ = std::fs::remove_file(&running);
    served
}

/// Ctrl+C, or the service manager asking it to stop.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = signal(SignalKind::terminate()).expect("listening for SIGTERM");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// The running daemon's port and token, or why there is none.
fn find_running(dir: &Path) -> Result<(u16, String)> {
    let not_running = || anyhow!("the daemon is not running; start it with `kjarni daemon run`");
    let text = std::fs::read_to_string(running_path(dir)).map_err(|_| not_running())?;
    let running: Running = serde_json::from_str(&text).map_err(|_| not_running())?;
    let token = std::fs::read_to_string(dir.join("token"))
        .with_context(|| format!("reading {}", dir.join("token").display()))?;
    Ok((running.port, token.trim().to_string()))
}

/// A request to the daemon on this machine: `(status, body)`.
fn local_request(port: u16, method: &str, path: &str, token: &str) -> Result<(u16, String)> {
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(3))
        .map_err(|_| anyhow!("the daemon is not answering on port {port}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    write!(
        stream,
        "{method} {path} HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {token}\r\nContent-Length: 0\r\n\r\n"
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    let (head, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| anyhow!("the daemon sent a malformed response"))?;
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| anyhow!("the daemon sent a malformed response"))?;
    // HTTP/1.0 gets the body whole, never chunked.
    Ok((status, body.to_string()))
}

fn open() -> Result<()> {
    let (port, token) = find_running(&daemon_dir())?;
    let (status, body) = local_request(port, "POST", "/api/login-code", &token)?;
    if status != 200 {
        bail!("the daemon refused to sign in ({status}): {body}");
    }
    let json: serde_json::Value = serde_json::from_str(&body)?;
    let url = json["url"]
        .as_str()
        .ok_or_else(|| anyhow!("the daemon sent no sign-in link"))?;
    if open_browser(url).is_err() {
        println!("Open this link in your browser within two minutes:\n{url}");
    } else {
        println!("Opened Kjarni in your browser.");
    }
    Ok(())
}

fn open_browser(url: &str) -> std::io::Result<()> {
    let mut command = if cfg!(target_os = "macos") {
        std::process::Command::new("open")
    } else if cfg!(windows) {
        let mut c = std::process::Command::new("cmd");
        c.args(["/C", "start", ""]);
        c
    } else {
        std::process::Command::new("xdg-open")
    };
    let status = command
        .arg(url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other("the browser did not open"))
    }
}

fn status() -> Result<()> {
    let (port, token) = find_running(&daemon_dir())?;
    let (code, body) = local_request(port, "GET", "/api/status", &token)?;
    if code != 200 {
        bail!("the daemon answered {code}: {body}");
    }
    let s: serde_json::Value = serde_json::from_str(&body)?;
    println!("Running at http://127.0.0.1:{port}");
    println!(
        "  model    {}{}",
        s["model"].as_str().unwrap_or("?"),
        if s["gpu"].as_bool() == Some(true) {
            " (GPU)"
        } else {
            ""
        }
    );
    println!("  folder   {}", s["workdir"].as_str().unwrap_or("?"));
    println!(
        "  memory   {}",
        if s["memory"].as_bool() == Some(true) {
            "on"
        } else {
            "off"
        }
    );
    println!("  version  {}", s["version"].as_str().unwrap_or("?"));
    Ok(())
}
