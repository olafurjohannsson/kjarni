//! `kjarni daemon`: the agent, its memory and the OpenAI-compatible API
//! behind one local web address, started at login once installed.
//!
//! The daemon listens on loopback only unless told otherwise, and every
//! request needs the token kept in the daemon's folder (see [`auth`]). The
//! chat model is loaded once and shared: the agent's conversations and
//! `/v1/chat/completions` take turns on it. Schedules (see [`schedules`])
//! start conversations by themselves at set times.

mod api;
mod auth;
mod conversations;
mod display;
mod logs;
mod machine;
mod schedules;
mod service;
#[cfg(test)]
mod tests;
mod web;

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
            allow_host,
            dir,
            yes,
            no_memory,
            max_steps,
            quiet,
            log_file,
        } => {
            let allow_hosts = allowed_hosts(&allow_host)?;
            if let Some(path) = &log_file {
                logs::redirect_output(Path::new(path))?;
            }
            serve_daemon(RunOptions {
                model,
                gpu,
                full_precision,
                port,
                listen,
                allow_hosts,
                dir,
                yes,
                no_memory,
                max_steps,
                quiet,
                background: log_file.is_some(),
            })
            .await
        }
        DaemonCommands::Install {
            model,
            gpu,
            cpu,
            full_precision,
            port,
            allow_host,
            dir,
            yes,
            no_memory,
            max_steps,
            no_linger,
        } => {
            install(InstallOptions {
                allow_hosts: allowed_hosts(&allow_host)?,
                model,
                gpu,
                cpu,
                full_precision,
                port,
                dir,
                yes,
                no_memory,
                max_steps,
                no_linger,
            })
            .await
        }
        DaemonCommands::Uninstall { purge } => uninstall(purge),
        DaemonCommands::Start => start(),
        DaemonCommands::Stop => stop(),
        DaemonCommands::Logs { lines, follow } => {
            logs::show(&logs::log_path(&daemon_dir()), lines, follow)
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
    /// Names besides loopback the daemon answers to.
    allow_hosts: Vec<String>,
    dir: Option<String>,
    yes: bool,
    no_memory: bool,
    max_steps: usize,
    quiet: bool,
    /// Started by the startup entry, with no terminal to press Ctrl+C in.
    background: bool,
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
        .fallback(web::serve)
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

/// `--allow-host` names as the guard compares them.
fn allowed_hosts(given: &[String]) -> Result<Vec<String>> {
    given.iter().map(|h| auth::allowed_host(h)).collect()
}

/// The folder the agent works in, `~/kjarni` unless given, created if
/// missing.
fn working_folder(dir: Option<&str>) -> Result<PathBuf> {
    let workdir = match dir {
        Some(d) => PathBuf::from(d),
        None => dirs::home_dir()
            .ok_or_else(|| anyhow!("no home folder; pass --dir"))?
            .join("kjarni"),
    };
    std::fs::create_dir_all(&workdir).with_context(|| format!("creating {}", workdir.display()))?;
    Ok(workdir.canonicalize()?)
}

async fn serve_daemon(options: RunOptions) -> Result<()> {
    let dir = daemon_dir();
    let token = auth::load_or_create_token(&dir.join("token"))?;

    let workdir = working_folder(options.dir.as_deref())?;

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
        guard: Arc::new(
            Guard::new(token, options.listen.is_some()).allow_hosts(options.allow_hosts.clone()),
        ),
        memory,
        model: model.clone(),
        gpu: options.gpu,
        schedules: Arc::new(schedules::Store::open(dir.join("schedules.json"))?),
    });
    tokio::spawn(schedules::run(state.clone()));
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

    if !options.allow_hosts.is_empty() {
        eprintln!(
            "Also answering to {}, for a proxy such as `tailscale serve` that forwards to this port.",
            options.allow_hosts.join(", ")
        );
    }
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
    eprintln!(
        "Sign in from this computer with `kjarni daemon open`. {} stops it.",
        if options.background {
            "`kjarni daemon stop`"
        } else {
            "Ctrl+C"
        }
    );

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

fn not_running() -> anyhow::Error {
    if service::is_installed() {
        anyhow!(
            "the daemon is not running; start it with `kjarni daemon start`, and see `kjarni daemon logs` if it stopped by itself"
        )
    } else {
        anyhow!(
            "the daemon is not running; install it with `kjarni daemon install`, or run it here with `kjarni daemon run`"
        )
    }
}

/// What the running daemon wrote about itself, if anything.
fn read_running(dir: &Path) -> Option<Running> {
    let text = std::fs::read_to_string(running_path(dir)).ok()?;
    serde_json::from_str(&text).ok()
}

/// The running daemon's port and token, or why there is none.
fn find_running(dir: &Path) -> Result<(u16, String)> {
    let running = read_running(dir).ok_or_else(not_running)?;
    let token = std::fs::read_to_string(dir.join("token"))
        .with_context(|| format!("reading {}", dir.join("token").display()))?;
    Ok((running.port, token.trim().to_string()))
}

/// The daemon's `/api/status`, when one is running and answering.
fn running_status(dir: &Path) -> Option<(Running, serde_json::Value)> {
    let running = read_running(dir)?;
    let token = std::fs::read_to_string(dir.join("token")).ok()?;
    match local_request(running.port, "GET", "/api/status", token.trim()) {
        Ok((200, body)) => Some((running, serde_json::from_str(&body).ok()?)),
        _ => None,
    }
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
    let dir = daemon_dir();
    let installed = service::is_installed();
    match running_status(&dir) {
        Some((running, s)) => {
            println!("Running at http://127.0.0.1:{}", running.port);
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
        }
        None if installed => {
            println!("Not running, or still loading its model.");
            println!("  `kjarni daemon logs` shows which; `kjarni daemon start` starts it.");
        }
        None => println!("Not running."),
    }
    if installed {
        println!("Starts at login: yes ({})", service::describe());
    } else {
        println!("Starts at login: no (`kjarni daemon install` sets that up)");
    }
    Ok(())
}

struct InstallOptions {
    model: Option<String>,
    gpu: bool,
    cpu: bool,
    full_precision: bool,
    port: u16,
    allow_hosts: Vec<String>,
    dir: Option<String>,
    yes: bool,
    no_memory: bool,
    max_steps: usize,
    no_linger: bool,
}

/// Variables that move Kjarni's folders, carried into the startup entry so
/// the daemon uses the same folders as the command that installed it.
const CARRIED_VARIABLES: [&str; 3] = ["KJARNI_DAEMON_DIR", "KJARNI_AGENT_DIR", "KJARNI_CACHE_DIR"];

/// Records that install turned linger on, so uninstall turns it off again
/// and leaves it alone when the user had it on already.
const LINGER_MARKER: &str = "linger-enabled-by-install";

/// The command line the startup entry runs.
fn run_arguments(
    options: &InstallOptions,
    model: &str,
    gpu: bool,
    workdir: &Path,
    log: &Path,
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "daemon".into(),
        "run".into(),
        "--model".into(),
        model.into(),
        "--port".into(),
        options.port.to_string(),
        "--dir".into(),
        workdir.to_string_lossy().into_owned(),
        "--max-steps".into(),
        options.max_steps.to_string(),
        "--log-file".into(),
        log.to_string_lossy().into_owned(),
    ];
    for host in &options.allow_hosts {
        args.push("--allow-host".into());
        args.push(host.clone());
    }
    for (on, flag) in [
        (gpu, "--gpu"),
        (options.full_precision, "--full-precision"),
        (options.yes, "--yes"),
        (options.no_memory, "--no-memory"),
    ] {
        if on {
            args.push(flag.into());
        }
    }
    args
}

async fn install(options: InstallOptions) -> Result<()> {
    let dir = daemon_dir();
    let installed = service::is_installed();
    let before = running_status(&dir);
    if let Some((running, _)) = &before
        && !installed
    {
        bail!(
            "a daemon started by hand is running (process {}, port {}); stop it with Ctrl+C first",
            running.pid,
            running.port
        );
    }

    let (model, gpu) = match &options.model {
        Some(m) => (m.clone(), options.gpu),
        None => {
            let pick = machine::pick(machine::Machine::this(), options.gpu, options.cpu);
            println!(
                "Picked {} on the {} for this machine ({}). `--model` picks another.",
                pick.model,
                if pick.gpu { "GPU" } else { "CPU" },
                pick.because
            );
            (pick.model.to_string(), pick.gpu)
        }
    };
    crate::commands::util::resolve_model(&model, Some("decoder"))?;

    // Downloaded now, where the user sees the progress, rather than by the
    // daemon in the background.
    if !is_cached(&model, !options.full_precision) {
        crate::commands::model::download(&model, !options.full_precision, false).await?;
    }
    if !options.no_memory && !is_cached(MEMORY_MODEL, false) {
        crate::commands::model::download(MEMORY_MODEL, false, true).await?;
    }

    auth::load_or_create_token(&dir.join("token"))?;
    let workdir = working_folder(options.dir.as_deref())?;
    let log = logs::log_path(&dir);
    if let Some(logs_dir) = log.parent() {
        std::fs::create_dir_all(logs_dir)
            .with_context(|| format!("creating {}", logs_dir.display()))?;
    }
    let program = std::env::current_exe()
        .and_then(|p| p.canonicalize())
        .context("finding this program's path")?;
    let mut env = Vec::new();
    for name in CARRIED_VARIABLES {
        if let Ok(value) = std::env::var(name) {
            let path = std::path::absolute(&value).unwrap_or_else(|_| value.clone().into());
            env.push((name.to_string(), path.to_string_lossy().into_owned()));
        }
    }
    if cfg!(windows) && !env.is_empty() {
        eprintln!(
            "Warning: a logon task cannot carry {}; the daemon will use the default folders.",
            env.iter()
                .map(|(k, _)| k.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let entry = service::Entry {
        program,
        args: run_arguments(&options, &model, gpu, &workdir, &log),
        env,
        log,
    };
    // A stale record from a daemon that was killed would look like the new one.
    if before.is_none() {
        let _ = std::fs::remove_file(running_path(&dir));
    }
    service::install(&entry)?;
    println!("Kjarni will start at login ({}).", service::describe());

    if let Some(on) = service::linger() {
        let marker = dir.join(LINGER_MARKER);
        if options.no_linger {
            if marker.exists() {
                service::set_linger(false)?;
                let _ = std::fs::remove_file(&marker);
            }
            println!("It stops when you log out (--no-linger).");
        } else if on {
            println!("It keeps running after you log out.");
        } else {
            match service::set_linger(true) {
                Ok(()) => {
                    std::fs::write(&marker, "")?;
                    println!(
                        "It keeps running after you log out (loginctl enable-linger; --no-linger turns that off)."
                    );
                }
                Err(e) => eprintln!("It stops when you log out: could not turn on linger: {e:#}"),
            }
        }
    }

    println!("Loading {model}...");
    let old_pid = before.map(|(r, _)| r.pid);
    let ready = wait_until_ready(&dir, old_pid, Duration::from_secs(300));
    match ready {
        Some(port) => {
            println!(
                "Ready at http://127.0.0.1:{port}, working in {}.",
                workdir.display()
            );
            println!("Sign in with `kjarni daemon open`.");
            Ok(())
        }
        None => bail!(
            "the daemon has not answered after five minutes; `kjarni daemon logs` shows what it is doing"
        ),
    }
}

/// Whether registry model `name` is downloaded in the format a load picks:
/// the GGUF when `gguf` and the model has one, otherwise safetensors.
fn is_cached(name: &str, gguf: bool) -> bool {
    is_cached_in(&kjarni::registry::cache_dir(), name, gguf)
}

fn is_cached_in(cache: &Path, name: &str, gguf: bool) -> bool {
    let Some(model) = kjarni::ModelType::from_cli_name(name) else {
        return false;
    };
    if gguf && model.info().paths.gguf_url.is_some() {
        let dir = model.cache_dir(cache);
        dir.join("config.json").exists() && dir.join("model.gguf").exists()
    } else {
        model.is_downloaded(cache)
    }
}

/// Waits for a daemon other than process `old_pid` to answer, returning its
/// port, or `None` after `limit`.
fn wait_until_ready(dir: &Path, old_pid: Option<u32>, limit: Duration) -> Option<u16> {
    let start = std::time::Instant::now();
    while start.elapsed() < limit {
        if let Some((running, _)) = running_status(dir)
            && Some(running.pid) != old_pid
        {
            return Some(running.port);
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    None
}

/// Waits up to `limit` for the daemon to stop answering, then clears its
/// record if it left one: a task ended on Windows gets no chance to.
fn wait_until_stopped(dir: &Path, limit: Duration) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < limit {
        if running_status(dir).is_none() {
            let _ = std::fs::remove_file(running_path(dir));
            return true;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    false
}

fn uninstall(purge: bool) -> Result<()> {
    let dir = daemon_dir();
    if service::uninstall()? {
        wait_until_stopped(&dir, Duration::from_secs(15));
        println!("Kjarni no longer starts at login, and is stopped.");
    } else {
        println!("The daemon was not installed.");
    }
    let marker = dir.join(LINGER_MARKER);
    if marker.exists() {
        match service::set_linger(false) {
            Ok(()) => {
                let _ = std::fs::remove_file(&marker);
                println!("Turned linger back off (loginctl disable-linger).");
            }
            Err(e) => eprintln!("Could not turn linger back off: {e:#}"),
        }
    }
    if !purge {
        println!(
            "Memory and conversations stay in {}; `--purge` deletes them.",
            kjarni::agent::data_dir().display()
        );
        return Ok(());
    }
    let agent = kjarni::agent::data_dir();
    let doomed: Vec<(&Path, &str)> = [
        (dir.as_path(), "token and logs"),
        (
            agent.as_path(),
            "memory and conversations, shared with `kjarni agent`",
        ),
    ]
    .into_iter()
    .filter(|(p, _)| p.exists())
    .collect();
    if doomed.is_empty() {
        println!("Nothing to delete.");
        return Ok(());
    }
    println!("This deletes, for good:");
    for (path, what) in &doomed {
        println!("  {}  ({what})", path.display());
    }
    print!("Delete them? [y/N] ");
    std::io::stdout().flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    if answer.trim().eq_ignore_ascii_case("y") {
        for (path, _) in doomed {
            std::fs::remove_dir_all(path)
                .with_context(|| format!("deleting {}", path.display()))?;
            println!("Deleted {}", path.display());
        }
        println!(
            "Downloaded models stay in {}, shared with the other kjarni commands; `kjarni model remove` deletes one.",
            kjarni::registry::cache_dir().display()
        );
    } else {
        println!("Kept them.");
    }
    Ok(())
}

fn start() -> Result<()> {
    let dir = daemon_dir();
    if let Some((running, _)) = running_status(&dir) {
        println!("Already running at http://127.0.0.1:{}.", running.port);
        return Ok(());
    }
    service::start()?;
    println!(
        "Starting. `kjarni daemon status` shows when it is ready; `kjarni daemon logs -f` shows it loading."
    );
    Ok(())
}

fn stop() -> Result<()> {
    let dir = daemon_dir();
    service::stop()?;
    if wait_until_stopped(&dir, Duration::from_secs(15)) {
        println!("Stopped. It starts again at the next login, or with `kjarni daemon start`.");
        Ok(())
    } else {
        bail!("the daemon is still answering after 15 seconds")
    }
}
