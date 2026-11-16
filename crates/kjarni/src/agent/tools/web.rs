//! Fetching a web page as readable text.

use std::time::Duration;

use anyhow::{Context, bail};
use async_trait::async_trait;
use serde_json::{Map, Value as Json, json};

use crate::agent::tool::{Access, Tool, ToolSpec, string_arg};

/// Bytes of a response read at most.
const MAX_BYTES: usize = 3 * 1024 * 1024;

/// Characters of page text returned at most.
const MAX_CHARS: usize = 20_000;

pub struct FetchUrl {
    client: reqwest::Client,
}

impl Default for FetchUrl {
    fn default() -> Self {
        Self::new()
    }
}

impl FetchUrl {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent(concat!("kjarni-agent/", env!("CARGO_PKG_VERSION")))
            .build()
            .unwrap_or_default();
        Self { client }
    }
}

#[async_trait]
impl Tool for FetchUrl {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "fetch_url",
            "Download a web page and return its text, without HTML markup.",
            json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The http or https address."}
                },
                "required": ["url"]
            }),
        )
    }

    fn access(&self) -> Access {
        Access::Network
    }

    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String> {
        let url = string_arg(args, "url")?.trim();
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            bail!("the url must start with http:// or https://");
        }
        let mut response = self
            .client
            .get(url)
            .send()
            .await
            .with_context(|| format!("cannot fetch {url}"))?;
        let status = response.status();
        let is_html = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_none_or(|t| t.contains("html"));
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            body.extend_from_slice(&chunk);
            if body.len() >= MAX_BYTES {
                break;
            }
        }
        let raw = String::from_utf8_lossy(&body);
        let text = if is_html {
            html_to_text(&raw)
        } else {
            raw.into_owned()
        };
        let cut = text.chars().count() > MAX_CHARS || body.len() >= MAX_BYTES;
        let mut text: String = text.chars().take(MAX_CHARS).collect();
        if cut {
            text.push_str("\n[page cut short]");
        }
        if !status.is_success() {
            return Ok(format!("HTTP {status}\n{text}"));
        }
        Ok(text)
    }
}

/// The readable text of an HTML page: no tags, scripts, styles or comments;
/// block elements on their own lines; common entities decoded.
pub(crate) fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 3);
    let lower = html.to_ascii_lowercase();
    let mut i = 0;
    let bytes = html.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip comments, and the contents of script, style and similar.
            if lower[i..].starts_with("<!--") {
                i = lower[i..].find("-->").map_or(bytes.len(), |e| i + e + 3);
                continue;
            }
            let mut skipped = false;
            for tag in ["script", "style", "noscript", "svg", "head"] {
                if lower[i + 1..].starts_with(tag) {
                    let close = format!("</{tag}");
                    i = lower[i..].find(&close).map_or(bytes.len(), |e| {
                        let end = i + e;
                        lower[end..].find('>').map_or(bytes.len(), |g| end + g + 1)
                    });
                    skipped = true;
                    break;
                }
            }
            if skipped {
                continue;
            }
            let end = lower[i..].find('>').map_or(bytes.len(), |e| i + e + 1);
            let tag = lower[i + 1..end.saturating_sub(1).max(i + 1)]
                .trim_start_matches('/')
                .split(|c: char| c.is_whitespace() || c == '>' || c == '/')
                .next()
                .unwrap_or("");
            if matches!(
                tag,
                "p" | "div"
                    | "br"
                    | "li"
                    | "tr"
                    | "h1"
                    | "h2"
                    | "h3"
                    | "h4"
                    | "h5"
                    | "h6"
                    | "section"
                    | "article"
                    | "header"
                    | "footer"
                    | "pre"
                    | "blockquote"
                    | "table"
                    | "ul"
                    | "ol"
                    | "nav"
                    | "main"
            ) {
                let closing = lower[i + 1..].starts_with('/');
                // A list item starts a line; its end adds no blank line.
                if !(closing && tag == "li") {
                    out.push('\n');
                }
                if tag == "li" && !closing {
                    out.push_str("- ");
                }
            }
            i = end;
        } else {
            let next = html[i..].find('<').map_or(html.len(), |e| i + e);
            out.push_str(&decode_entities(&html[i..next]));
            i = next;
        }
    }
    // Collapse runs of spaces, and of blank lines.
    let mut cleaned = String::with_capacity(out.len());
    let mut blank_run = 0;
    for line in out.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        cleaned.push_str(&line);
        cleaned.push('\n');
    }
    cleaned.trim().to_string()
}

fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp..];
        let Some(semi) = after[..after.len().min(12)].find(';') else {
            out.push('&');
            rest = &after[1..];
            continue;
        };
        let entity = &after[1..semi];
        let decoded = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" | "#39" => Some('\''),
            "nbsp" => Some(' '),
            _ if entity.starts_with("#x") || entity.starts_with("#X") => {
                u32::from_str_radix(&entity[2..], 16)
                    .ok()
                    .and_then(char::from_u32)
            }
            _ if entity.starts_with('#') => entity[1..].parse().ok().and_then(char::from_u32),
            _ => None,
        };
        match decoded {
            Some(c) => {
                out.push(c);
                rest = &after[semi + 1..];
            }
            None => {
                out.push('&');
                rest = &after[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_becomes_readable_text() {
        let html = "<html><head><title>T</title><style>p{}</style></head><body>\
            <h1>Kjarni</h1><!-- hidden --><p>Runs  models&nbsp;locally &amp; offline.</p>\
            <script>alert(1)</script><ul><li>Fast</li><li>Small</li></ul><p>&#169; 2026 &#x2014; done</p></body></html>";
        assert_eq!(
            html_to_text(html),
            "Kjarni\n\nRuns models locally & offline.\n\n- Fast\n- Small\n\n© 2026 — done"
        );
    }

    #[test]
    fn unknown_entities_and_bare_ampersands_stay() {
        assert_eq!(decode_entities("a & b &bogus; &lt;"), "a & b &bogus; <");
    }
}
