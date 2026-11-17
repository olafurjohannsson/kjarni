//! Who may talk to the daemon.
//!
//! Every request carries the daemon's token, as a bearer token from programs
//! or as a cookie from the web UI. The browser gets the cookie through a
//! one-time login code, so the token itself never appears in a URL or in the
//! browser's history.
//!
//! Requests must also name the daemon by a loopback host, or by a name the
//! user allowed with `--allow-host` (such as the one `tailscale serve` gives
//! the computer), and a browser request must come from the daemon's own
//! origin. Together these stop a web page the user visits from driving the
//! daemon through their browser, directly or by pointing its own domain at
//! 127.0.0.1 (DNS rebinding).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// The cookie the web UI is signed in with.
pub const COOKIE: &str = "kjarni_token";

/// How long a login code works for.
const CODE_LIFETIME: Duration = Duration::from_secs(120);

/// Wrong codes tried before every pending code stops working, so a short
/// code cannot be guessed by trying many.
const MAX_WRONG_CODES: u32 = 10;

/// `n` random bytes as lowercase hex.
pub fn random_hex(n: usize) -> String {
    let mut bytes = vec![0u8; n];
    getrandom::fill(&mut bytes).expect("the system's random number generator failed");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The token in `path`, created if the file is missing. Only the user can
/// read the file.
pub fn load_or_create_token(path: &Path) -> Result<String> {
    if let Ok(text) = std::fs::read_to_string(path) {
        let token = text.trim();
        if token.len() >= 32 {
            return Ok(token.to_string());
        }
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let token = random_hex(32);
    write_private(path, &token).with_context(|| format!("writing {}", path.display()))?;
    Ok(token)
}

/// Writes `text` to `path`, readable and writable by the owner alone.
pub fn write_private(path: &Path, text: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    // A file that already existed keeps its old mode through `open`.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(text.as_bytes())
}

/// What the guard checks requests against.
pub struct Guard {
    token: String,
    /// Accept any `Host`, for a daemon listening on a network address.
    any_host: bool,
    /// Names besides loopback the daemon answers to, lowercase, no port.
    allowed_hosts: Vec<String>,
    codes: Mutex<Codes>,
}

/// Login codes not yet used, and wrong codes tried since the last reset.
#[derive(Default)]
struct Codes {
    live: HashMap<String, Instant>,
    wrong: u32,
}

impl Codes {
    fn add(&mut self, code: String) {
        self.live.retain(|_, made| made.elapsed() < CODE_LIFETIME);
        if self.live.is_empty() {
            self.wrong = 0;
        }
        self.live.insert(code, Instant::now());
    }
}

impl Guard {
    pub fn new(token: String, any_host: bool) -> Self {
        Self {
            token,
            any_host,
            allowed_hosts: Vec::new(),
            codes: Mutex::new(Codes::default()),
        }
    }

    /// Also answers to `hosts`, as [`allowed_host`] wrote them.
    /// The names besides loopback the daemon answers to.
    pub fn allowed_hosts(&self) -> &[String] {
        &self.allowed_hosts
    }

    pub fn allow_hosts(mut self, hosts: Vec<String>) -> Self {
        self.allowed_hosts = hosts;
        self
    }

    /// Whether a request naming `host` is for this daemon.
    fn answers_to(&self, host: &str) -> bool {
        if self.any_host || is_loopback_host(host) {
            return true;
        }
        host_name(host).is_some_and(|name| self.allowed_hosts.contains(&name))
    }

    pub fn token(&self) -> &str {
        &self.token
    }

    /// A code that signs one browser in, once, within two minutes.
    pub fn new_login_code(&self) -> String {
        let code = random_hex(16);
        self.codes
            .lock()
            .expect("login codes lock")
            .add(code.clone());
        code
    }

    /// An eight-digit code to type on another device, which signs it in
    /// once, within two minutes.
    pub fn new_device_code(&self) -> String {
        let mut bytes = [0u8; 8];
        getrandom::fill(&mut bytes).expect("the system's random number generator failed");
        let code = format!("{:08}", u64::from_le_bytes(bytes) % 100_000_000);
        self.codes
            .lock()
            .expect("login codes lock")
            .add(code.clone());
        code
    }

    /// Whether `code` is a live login code, which it no longer is after this.
    /// Spaces and dashes are ignored. After [`MAX_WRONG_CODES`] wrong ones,
    /// every pending code stops working.
    pub fn redeem(&self, code: &str) -> bool {
        let code: String = code
            .chars()
            .filter(|c| !c.is_whitespace() && *c != '-')
            .collect();
        let mut codes = self.codes.lock().expect("login codes lock");
        match codes.live.remove(&code) {
            Some(made) if made.elapsed() < CODE_LIFETIME => true,
            _ => {
                codes.wrong += 1;
                if codes.wrong >= MAX_WRONG_CODES {
                    codes.live.clear();
                    codes.wrong = 0;
                }
                false
            }
        }
    }

    /// Whether `candidate` is the token.
    pub fn is_token(&self, candidate: &str) -> bool {
        same(candidate.as_bytes(), self.token.as_bytes())
    }

    fn signed_in(&self, headers: &HeaderMap) -> bool {
        let bearer = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        if bearer.is_some_and(|b| same(b.trim().as_bytes(), self.token.as_bytes())) {
            return true;
        }
        cookie(headers, COOKIE).is_some_and(|c| same(c.as_bytes(), self.token.as_bytes()))
    }
}

/// The value of cookie `name`.
pub fn cookie<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .filter_map(|pair| pair.trim().split_once('='))
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v)
}

/// Compares in time that depends only on the lengths.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The name in `host`, a `Host` header or an origin's host part, without its
/// port and in lowercase. `None` when it is not a host.
pub fn host_name(host: &str) -> Option<String> {
    let name = if let Some(rest) = host.strip_prefix('[') {
        // [::1] or [::1]:port
        match rest.split_once(']') {
            Some((name, tail)) if tail.is_empty() || port(tail) => name,
            _ => return None,
        }
    } else {
        match host.split_once(':') {
            Some((name, tail)) if port(&format!(":{tail}")) => name,
            Some(_) => return None,
            None => host,
        }
    };
    let valid = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | ':'));
    valid.then(|| name.to_ascii_lowercase())
}

/// Whether `tail` is `:` and a port number.
fn port(tail: &str) -> bool {
    tail.strip_prefix(':')
        .is_some_and(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
}

/// Whether `host` names this machine's loopback interface.
pub fn is_loopback_host(host: &str) -> bool {
    host_name(host).is_some_and(|name| name == "localhost" || name == "127.0.0.1" || name == "::1")
}

/// A name for `--allow-host`, as the guard compares it. Takes a pasted
/// address too: `https://box.tail1234.ts.net/` gives `box.tail1234.ts.net`.
pub fn allowed_host(text: &str) -> Result<String> {
    let trimmed = text.trim();
    let bare = trimmed
        .strip_prefix("https://")
        .or_else(|| trimmed.strip_prefix("http://"))
        .unwrap_or(trimmed);
    let bare = bare.strip_suffix('/').unwrap_or(bare);
    match host_name(bare) {
        Some(name) if !bare.contains('/') => Ok(name),
        _ => bail!("'{text}' is not a host name, such as mybox.tail1234.ts.net"),
    }
}

fn refuse(status: StatusCode, message: &str) -> Response {
    (
        status,
        axum::Json(json!({ "error": { "message": message } })),
    )
        .into_response()
}

/// Paths that work without the token: the sign-in page, the health check,
/// and the web UI's own files, which hold nothing private.
fn open_path(path: &str) -> bool {
    path == "/login" || path == "/health" || !(path.starts_with("/api") || path.starts_with("/v1"))
}

/// Middleware: refuses requests from the wrong host or origin, and requests
/// without the token on paths that need it.
pub async fn guard(State(guard): State<Arc<Guard>>, request: Request, next: Next) -> Response {
    let headers = request.headers();
    let host = headers
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    if !guard.answers_to(host) {
        let message = match host_name(host) {
            Some(name) => format!(
                "the daemon does not answer to '{name}'. If that is this computer, start the daemon with --allow-host {name}"
            ),
            None => "the daemon answers only to localhost, 127.0.0.1, [::1] and names given with --allow-host".into(),
        };
        return refuse(StatusCode::FORBIDDEN, &message);
    }
    // Over HTTPS when a proxy such as `tailscale serve` adds it.
    if let Some(origin) = headers.get(header::ORIGIN) {
        let origin = origin.to_str().unwrap_or_default();
        let ours = origin
            .strip_prefix("http://")
            .or_else(|| origin.strip_prefix("https://"))
            .is_some_and(|rest| rest == host);
        if !ours {
            return refuse(
                StatusCode::FORBIDDEN,
                "requests from other web pages are refused",
            );
        }
    }
    if !open_path(request.uri().path()) && !guard.signed_in(headers) {
        return refuse(
            StatusCode::UNAUTHORIZED,
            "sign in with `kjarni daemon open`, or send the token from the daemon's token file as a bearer token",
        );
    }
    next.run(request).await
}
