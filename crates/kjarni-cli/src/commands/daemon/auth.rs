//! Who may talk to the daemon.
//!
//! Every request carries the daemon's token, as a bearer token from programs
//! or as a cookie from the web UI. The browser gets the cookie through a
//! one-time login code, so the token itself never appears in a URL or in the
//! browser's history.
//!
//! Requests must also name the daemon by a loopback host, and a browser
//! request must come from the daemon's own origin. Together these stop a web
//! page the user visits from driving the daemon through their browser,
//! directly or by pointing its own domain at 127.0.0.1 (DNS rebinding).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// The cookie the web UI is signed in with.
pub const COOKIE: &str = "kjarni_token";

/// How long a login code works for.
const CODE_LIFETIME: Duration = Duration::from_secs(120);

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
    codes: Mutex<HashMap<String, Instant>>,
}

impl Guard {
    pub fn new(token: String, any_host: bool) -> Self {
        Self {
            token,
            any_host,
            codes: Mutex::new(HashMap::new()),
        }
    }

    pub fn token(&self) -> &str {
        &self.token
    }

    /// A code that signs one browser in, once, within two minutes.
    pub fn new_login_code(&self) -> String {
        let code = random_hex(16);
        let mut codes = self.codes.lock().expect("login codes lock");
        codes.retain(|_, made| made.elapsed() < CODE_LIFETIME);
        codes.insert(code.clone(), Instant::now());
        code
    }

    /// Whether `code` is a live login code, which it no longer is after this.
    pub fn redeem(&self, code: &str) -> bool {
        let mut codes = self.codes.lock().expect("login codes lock");
        matches!(codes.remove(code), Some(made) if made.elapsed() < CODE_LIFETIME)
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

/// Whether `host`, a `Host` header or an origin's host part, names this
/// machine's loopback interface.
pub fn is_loopback_host(host: &str) -> bool {
    let name = if let Some(rest) = host.strip_prefix('[') {
        // [::1] or [::1]:port
        match rest.split_once(']') {
            Some((name, tail)) if tail.is_empty() || tail.starts_with(':') => name,
            _ => return false,
        }
    } else {
        match host.rsplit_once(':') {
            Some((name, port)) if port.chars().all(|c| c.is_ascii_digit()) => name,
            Some(_) => return false,
            None => host,
        }
    };
    let name = name.to_ascii_lowercase();
    name == "localhost" || name == "127.0.0.1" || name == "::1"
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
    if !guard.any_host && !is_loopback_host(host) {
        return refuse(
            StatusCode::FORBIDDEN,
            "the daemon answers only to localhost, 127.0.0.1 or [::1]",
        );
    }
    if let Some(origin) = headers.get(header::ORIGIN) {
        let expected = format!("http://{host}");
        if origin.to_str().ok() != Some(expected.as_str()) {
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
