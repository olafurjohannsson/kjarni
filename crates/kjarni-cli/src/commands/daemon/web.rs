//! The web UI. Its source is in `web/`; `npm run build` there writes
//! `web/dist/`, which is committed and embedded here, so building Kjarni
//! needs no Node.

use axum::http::{HeaderValue, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use serde_json::json;

struct Asset {
    path: &'static str,
    mime: &'static str,
    body: &'static [u8],
    /// Never changes under the same name, so browsers may keep it.
    immutable: bool,
}

macro_rules! asset {
    ($path:literal, $mime:literal, $immutable:literal) => {
        Asset {
            path: $path,
            mime: $mime,
            body: include_bytes!(concat!("../../../web/dist", $path)),
            immutable: $immutable,
        }
    };
}

const ASSETS: &[Asset] = &[
    asset!("/assets/app.js", "text/javascript; charset=utf-8", false),
    asset!("/assets/app.css", "text/css; charset=utf-8", false),
    asset!("/icon.svg", "image/svg+xml", false),
    asset!("/fonts/schibsted-grotesk-latin.woff2", "font/woff2", true),
    asset!(
        "/fonts/schibsted-grotesk-latin-ext.woff2",
        "font/woff2",
        true
    ),
    asset!("/fonts/martian-mono-latin.woff2", "font/woff2", true),
    asset!("/fonts/martian-mono-latin-ext.woff2", "font/woff2", true),
];

const INDEX: &[u8] = include_bytes!("../../../web/dist/index.html");

/// The page may load only its own files and talk only to this daemon. Model
/// replies can carry text from web pages the agent read, so even if some
/// slipped past the sanitizer it could neither run nor send anything out.
/// `frame-ancestors` stops another site from framing the page to trick a
/// click on an approval button.
const CONTENT_SECURITY_POLICY: &str = "default-src 'self'; script-src 'self'; \
     style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; \
     connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'";

fn with_headers(mut response: Response, mime: &'static str, cache: &'static str) -> Response {
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(mime));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static(cache));
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    response
}

/// Serves the UI's files, and its page for any other path outside the API,
/// so `/c/<conversation>` links open the page.
pub async fn serve(uri: Uri) -> Response {
    let path = uri.path();
    if let Some(asset) = ASSETS.iter().find(|a| a.path == path) {
        let cache = if asset.immutable {
            "public, max-age=31536000, immutable"
        } else {
            // A new build reuses the names, so the browser asks each time.
            "no-cache"
        };
        return with_headers(asset.body.into_response(), asset.mime, cache);
    }
    let is_api = path == "/api" || path.starts_with("/api/") || path.starts_with("/v1/");
    if is_api
        || path
            .rsplit('/')
            .next()
            .is_some_and(|last| last.contains('.'))
    {
        return (
            StatusCode::NOT_FOUND,
            axum::Json(json!({ "error": { "message": "there is no such path" } })),
        )
            .into_response();
    }
    let mut response = with_headers(
        INDEX.into_response(),
        "text/html; charset=utf-8",
        "no-cache",
    );
    response.headers_mut().insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(CONTENT_SECURITY_POLICY),
    );
    response
}
