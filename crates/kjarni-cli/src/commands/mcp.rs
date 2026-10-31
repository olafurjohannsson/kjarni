//! `kjarni mcp`: Kjarni's capabilities as tools for AI agents, over the Model
//! Context Protocol on stdio.
//!
//! An agent (Claude Code, Cursor, Claude Desktop, any MCP client) starts this
//! as a subprocess and calls tools: semantic search over a folder, speech to
//! text, classification, reranking and similarity, all on the user's machine.
//!
//! The protocol is newline-delimited JSON-RPC 2.0. Only the part of MCP a tool
//! server needs is implemented: initialize, ping, tools/list and tools/call.
//! Models load on first use and stay loaded for the session, so only the first
//! call of each kind pays for it.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, anyhow, bail};
use kjarni::classifier::Classifier;
use kjarni::embedder::Embedder;
use kjarni::reranker::Reranker;
use kjarni::searcher::{SearchOptions, Searcher};
use kjarni::transcriber::Transcriber;
use kjarni::{Indexer, SearchMode};
use serde_json::{Value, json};
use tokio::io::AsyncBufReadExt;

/// Protocol versions this server speaks, newest first. A client asking for
/// one of these gets it back; anything else gets the newest.
const PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// The embedding model behind `search_folder`. Small and fast, which matters
/// more than the last few points of quality when an agent is waiting.
const SEARCH_MODEL: &str = "minilm-l6-v2";
const RERANK_MODEL: &str = "minilm-l6-v2-cross-encoder";
const TRANSCRIBE_MODEL: &str = "whisper-small";
const CLASSIFY_MODEL: &str = "distilbert-sentiment";

/// Never indexed by `search_folder`: build output, dependencies and virtual
/// environments, which in a code repository outnumber the sources many times
/// over. Hidden folders such as .git are skipped as well.
pub(crate) const EXCLUDED: &[&str] = &[
    "target",
    "node_modules",
    "dist",
    "build",
    "out",
    "bin",
    "obj",
    "vendor",
    "venv",
    "__pycache__",
    "*.min.js",
];

const INSTRUCTIONS: &str = "Kjarni runs AI models locally on the user's machine; nothing leaves it. \
Use search_folder to find passages in a folder of documents or code by meaning, not just by \
exact words; the first search of a folder builds an index, later ones reuse it until files \
change. Use transcribe for audio or video files, classify for sentiment, emotion or toxicity, \
rerank to order candidate passages by relevance to a query, and similarity to compare two \
texts. Models download once on first use.";

pub async fn run() -> Result<()> {
    let mut out = protocol_stdout();
    let mut server = Server::default();
    let mut lines = tokio::io::BufReader::new(tokio::io::stdin()).lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(&line) {
            Ok(message) => server.handle(&message).await,
            Err(e) => Some(error_response(
                Value::Null,
                -32700,
                &format!("parse error: {e}"),
            )),
        };
        if let Some(response) = response {
            writeln!(out, "{response}")?;
            out.flush()?;
        }
    }
    Ok(())
}

/// Where protocol messages go. Anything else the process prints, a library's
/// progress line or a download notice, would corrupt the stream, so stdout
/// itself is pointed at stderr and the protocol keeps a private copy of the
/// original.
#[cfg(unix)]
fn protocol_stdout() -> Box<dyn Write + Send> {
    use std::os::fd::FromRawFd;
    // SAFETY: plain descriptor duplication on 0..2, which this process owns.
    unsafe {
        let saved = libc::dup(1);
        if saved >= 0 && libc::dup2(2, 1) >= 0 {
            return Box::new(std::fs::File::from_raw_fd(saved));
        }
    }
    Box::new(std::io::stdout())
}

/// On Windows the library's output is kept quiet instead: every model is
/// built with `quiet`, and downloads are the one remaining source of output.
#[cfg(not(unix))]
fn protocol_stdout() -> Box<dyn Write + Send> {
    Box::new(std::io::stdout())
}

#[derive(Default)]
struct Server {
    searcher: Option<Searcher>,
    rerank_searcher: Option<Searcher>,
    indexer: Option<Indexer>,
    embedder: Option<Embedder>,
    reranker: Option<Reranker>,
    transcriber: Option<Transcriber>,
    classifiers: HashMap<String, Classifier>,
}

impl Server {
    /// Answers one message. Notifications get no response.
    async fn handle(&mut self, message: &Value) -> Option<Value> {
        let id = message.get("id").cloned()?;
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let params = message.get("params").cloned().unwrap_or(Value::Null);

        let result = match method {
            "initialize" => Ok(initialize(&params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tools() })),
            "tools/call" => Ok(self.call(&params).await),
            _ => Err((-32601, format!("method not found: {method}"))),
        };
        Some(match result {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err((code, message)) => error_response(id, code, &message),
        })
    }

    /// Runs a tool. Failures are results with `isError`, as MCP asks, so the
    /// agent sees what went wrong and can correct its call.
    async fn call(&mut self, params: &Value) -> Value {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let args = params
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let outcome = match name {
            "search_folder" => self.search_folder(&args).await,
            "transcribe" => self.transcribe(&args).await,
            "classify" => self.classify(&args).await,
            "rerank" => self.rerank(&args).await,
            "similarity" => self.similarity(&args).await,
            _ => Err(anyhow!("unknown tool: {name}")),
        };
        match outcome {
            Ok(text) => json!({ "content": [{ "type": "text", "text": text }] }),
            Err(e) => {
                json!({ "content": [{ "type": "text", "text": format!("{e:#}") }], "isError": true })
            }
        }
    }

    async fn search_folder(&mut self, args: &Value) -> Result<String> {
        let folder = PathBuf::from(string(args, "path")?);
        let folder = folder
            .canonicalize()
            .with_context(|| format!("cannot open {}", folder.display()))?;
        if !folder.is_dir() {
            bail!("{} is not a folder", folder.display());
        }
        let query = string(args, "query")?;
        let mode: SearchMode = optional_string(args, "mode")
            .unwrap_or("hybrid")
            .parse()
            .map_err(|e: String| anyhow!(e))?;
        let top_k = args
            .get("top_k")
            .and_then(Value::as_u64)
            .unwrap_or(5)
            .clamp(1, 50) as usize;
        let rerank = args.get("rerank").and_then(Value::as_bool).unwrap_or(false);

        let (index, rebuilt) = self.index_for(&folder).await?;
        let index = index.to_string_lossy().into_owned();
        let options = SearchOptions::new().mode(mode).top_k(top_k).rerank(rerank);
        let searcher = if rerank {
            if self.rerank_searcher.is_none() {
                let s = Searcher::builder(SEARCH_MODEL)
                    .reranker(RERANK_MODEL)
                    .quiet(true)
                    .build()
                    .await?;
                self.rerank_searcher = Some(s);
            }
            self.rerank_searcher.as_ref().expect("built above")
        } else {
            if self.searcher.is_none() {
                self.searcher = Some(Searcher::builder(SEARCH_MODEL).quiet(true).build().await?);
            }
            self.searcher.as_ref().expect("built above")
        };
        let results = searcher
            .search_with_options(&index, query, &options)
            .await?;

        let mut text = String::new();
        if rebuilt {
            text.push_str(&format!(
                "(Indexed {} for this search.)\n\n",
                folder.display()
            ));
        }
        if results.is_empty() {
            text.push_str("No matching passages.");
        }
        for (i, r) in results.iter().enumerate() {
            let source = r.metadata.get("source").map(String::as_str).unwrap_or("?");
            text.push_str(&format!(
                "{}. {source}  (score {:.3})\n{}\n\n",
                i + 1,
                r.score,
                clip(r.text.trim(), 800)
            ));
        }
        Ok(text.trim_end().to_string())
    }

    /// The index for a folder, built when there is none or the folder's files
    /// have changed since. Indexes live in Kjarni's cache, never in the folder.
    async fn index_for(&mut self, folder: &Path) -> Result<(PathBuf, bool)> {
        let root = kjarni::registry::cache_dir().join("mcp-indexes");
        let index = root.join(format!(
            "{:016x}",
            fnv1a(folder.as_os_str().as_encoded_bytes())
        ));
        let stamp_file = index.with_extension("stamp");
        let stamp = fingerprint(folder);

        let current = std::fs::read_to_string(&stamp_file).ok();
        if index.exists() && current.as_deref() == Some(stamp.as_str()) {
            return Ok((index, false));
        }
        if self.indexer.is_none() {
            let mut builder = Indexer::builder(SEARCH_MODEL).quiet(true);
            for pattern in EXCLUDED {
                builder = builder.exclude(pattern);
            }
            self.indexer = Some(builder.build().await?);
        }
        std::fs::create_dir_all(&root)?;
        if index.exists() {
            std::fs::remove_dir_all(&index).context("removing the outdated index")?;
        }
        let index_str = index.to_string_lossy().into_owned();
        let folder_str = folder.to_string_lossy().into_owned();
        self.indexer
            .as_ref()
            .expect("built above")
            .create(&index_str, &[folder_str.as_str()])
            .await?;
        std::fs::write(&stamp_file, stamp)?;
        Ok((index, true))
    }

    async fn transcribe(&mut self, args: &Value) -> Result<String> {
        let path = PathBuf::from(string(args, "path")?);
        if !path.is_file() {
            bail!("{} is not a file", path.display());
        }
        if self.transcriber.is_none() {
            self.transcriber = Some(
                Transcriber::builder(TRANSCRIBE_MODEL)
                    .quiet()
                    .build()
                    .await?,
            );
        }
        let transcriber = self.transcriber.as_ref().expect("built above");
        // Minutes of work on the CPU: run it where it will not stall the
        // runtime's other threads.
        let result = tokio::task::block_in_place(|| transcriber.transcribe_file(&path))?;

        let mut text = format!(
            "Language: {}. Duration: {:.0} s.\n\n",
            result.language, result.duration_secs
        );
        for s in &result.segments {
            text.push_str(&format!("[{}] {}\n", timestamp(s.start), s.text.trim()));
        }
        if result.segments.is_empty() {
            text.push_str(result.text.trim());
        }
        Ok(text.trim_end().to_string())
    }

    async fn classify(&mut self, args: &Value) -> Result<String> {
        let text = string(args, "text")?;
        let model = optional_string(args, "model")
            .unwrap_or(CLASSIFY_MODEL)
            .to_string();
        if !self.classifiers.contains_key(&model) {
            let classifier = Classifier::builder(model.clone())
                .quiet(true)
                .build()
                .await?;
            self.classifiers.insert(model.clone(), classifier);
        }
        let result = self.classifiers[&model].classify(text).await?;
        let scores: Vec<String> = result
            .all_scores
            .iter()
            .map(|(label, score)| format!("{label}: {:.3}", score))
            .collect();
        Ok(format!(
            "{} ({:.3})\nAll: {}",
            result.label,
            result.score,
            scores.join(", ")
        ))
    }

    async fn rerank(&mut self, args: &Value) -> Result<String> {
        let query = string(args, "query")?;
        let documents: Vec<String> = args
            .get("documents")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("documents must be an array of strings"))?
            .iter()
            .map(|d| {
                d.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| anyhow!("documents must be strings"))
            })
            .collect::<Result<_>>()?;
        if documents.is_empty() {
            bail!("documents is empty");
        }
        if self.reranker.is_none() {
            self.reranker = Some(Reranker::builder(RERANK_MODEL).quiet(true).build().await?);
        }
        let refs: Vec<&str> = documents.iter().map(String::as_str).collect();
        let ranked = self
            .reranker
            .as_ref()
            .expect("built above")
            .rerank(query, &refs)
            .await?;
        let lines: Vec<String> = ranked
            .iter()
            .enumerate()
            .map(|(rank, r)| {
                format!(
                    "{}. [{}] relevance {:.3}: {}",
                    rank + 1,
                    r.index,
                    r.probability(),
                    clip(&documents[r.index], 200)
                )
            })
            .collect();
        Ok(lines.join("\n"))
    }

    async fn similarity(&mut self, args: &Value) -> Result<String> {
        let a = string(args, "a")?;
        let b = string(args, "b")?;
        if self.embedder.is_none() {
            self.embedder = Some(Embedder::builder(SEARCH_MODEL).quiet(true).build().await?);
        }
        let score = self
            .embedder
            .as_ref()
            .expect("built above")
            .similarity(a, b)
            .await?;
        Ok(format!("{score:.4}"))
    }
}

fn initialize(params: &Value) -> Value {
    let requested = params
        .get("protocolVersion")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let version = PROTOCOL_VERSIONS
        .iter()
        .find(|v| **v == requested)
        .unwrap_or(&PROTOCOL_VERSIONS[0]);
    json!({
        "protocolVersion": version,
        "capabilities": { "tools": { "listChanged": false } },
        "serverInfo": { "name": "kjarni", "title": "Kjarni", "version": env!("CARGO_PKG_VERSION") },
        "instructions": INSTRUCTIONS,
    })
}

fn tools() -> Value {
    json!([
        {
            "name": "search_folder",
            "title": "Search a folder by meaning",
            "description": "Find the passages in a folder of documents or code (txt, md, source code, json, yaml, html, ...) that best match a query, by meaning as well as by keyword. Runs locally. The first search of a folder indexes it, which takes seconds to minutes depending on its size; later searches reuse the index until files change. Build output, dependencies and hidden folders are skipped.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path of the folder to search, subfolders included." },
                    "query": { "type": "string", "description": "What to find, in natural language or keywords." },
                    "mode": { "type": "string", "enum": ["hybrid", "semantic", "keyword"], "description": "hybrid (default) combines meaning and keywords; semantic is meaning only; keyword is BM25 only." },
                    "top_k": { "type": "integer", "minimum": 1, "maximum": 50, "description": "How many passages to return. Default 5." },
                    "rerank": { "type": "boolean", "description": "Reorder results with a cross-encoder for better precision. Slower. Default false." }
                },
                "required": ["path", "query"]
            },
            "annotations": { "readOnlyHint": true, "openWorldHint": false }
        },
        {
            "name": "transcribe",
            "title": "Transcribe audio or video",
            "description": "Speech to text with Whisper, locally. Detects the language. Returns timestamped segments. Takes roughly a tenth of the recording's length on a desktop CPU.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path of a wav, mp3, flac or ogg file." }
                },
                "required": ["path"]
            },
            "annotations": { "readOnlyHint": true, "openWorldHint": false }
        },
        {
            "name": "classify",
            "title": "Classify text",
            "description": "Sentiment, emotion or toxicity of a text, locally.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string" },
                    "model": { "type": "string", "enum": ["distilbert-sentiment", "roberta-sentiment", "roberta-emotions", "distilroberta-emotion", "toxic-bert", "bert-sentiment-multilingual"], "description": "Default distilbert-sentiment (positive/negative)." }
                },
                "required": ["text"]
            },
            "annotations": { "readOnlyHint": true, "openWorldHint": false }
        },
        {
            "name": "rerank",
            "title": "Rank passages by relevance",
            "description": "Order candidate passages by how well each answers a query, with a cross-encoder. Returns each passage's original index and a relevance from 0 to 1.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "documents": { "type": "array", "items": { "type": "string" }, "minItems": 1 }
                },
                "required": ["query", "documents"]
            },
            "annotations": { "readOnlyHint": true, "openWorldHint": false }
        },
        {
            "name": "similarity",
            "title": "Compare two texts",
            "description": "Semantic similarity of two texts as a cosine between -1 and 1; above about 0.6 usually means the same topic.",
            "inputSchema": {
                "type": "object",
                "properties": { "a": { "type": "string" }, "b": { "type": "string" } },
                "required": ["a", "b"]
            },
            "annotations": { "readOnlyHint": true, "openWorldHint": false }
        }
    ])
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn string<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    optional_string(args, key).ok_or_else(|| anyhow!("missing argument: {key}"))
}

fn optional_string<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// What says a folder has changed since it was indexed: how many files it
/// has outside the excluded and hidden folders, their total size, and the
/// newest modification time.
fn fingerprint(folder: &Path) -> String {
    let (mut files, mut bytes, mut newest) = (0u64, 0u64, 0u128);
    let walker = walkdir::WalkDir::new(folder).into_iter().filter_entry(|e| {
        e.depth() == 0 || {
            let name = e.file_name().to_string_lossy();
            !name.starts_with('.') && !EXCLUDED.iter().any(|p| wildcard(p, &name))
        }
    });
    for entry in walker.filter_map(|e| e.ok()) {
        if let Ok(meta) = entry.metadata()
            && meta.is_file()
        {
            files += 1;
            bytes += meta.len();
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok());
            newest = newest.max(mtime.map_or(0, |d| d.as_nanos()));
        }
    }
    format!("{SEARCH_MODEL} {files} {bytes} {newest}")
}

/// `*` matches any run of characters, as in the indexer's own exclusions.
fn wildcard(pattern: &str, name: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == name,
        Some((head, tail)) => {
            name.len() >= head.len() + tail.len() && name.starts_with(head) && name.ends_with(tail)
        }
    }
}

/// A stable hash, so the same folder maps to the same index across runs.
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

fn timestamp(secs: f32) -> String {
    let s = secs.max(0.0) as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

fn clip(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let cut: String = text.chars().take(max_chars).collect();
    format!("{}…", cut.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn ask(server: &mut Server, message: Value) -> Value {
        server
            .handle(&message)
            .await
            .expect("a request gets a response")
    }

    #[tokio::test]
    async fn speaks_the_protocol_handshake() {
        let mut server = Server::default();
        let init = ask(
            &mut server,
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"t","version":"1"}}}),
        )
        .await;
        assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(init["result"]["serverInfo"]["name"], "kjarni");
        assert!(init["result"]["capabilities"]["tools"].is_object());

        // An unknown version gets the newest one this server speaks.
        let init = ask(&mut server, json!({"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"1999-01-01"}})).await;
        assert_eq!(init["result"]["protocolVersion"], PROTOCOL_VERSIONS[0]);

        // Notifications get no answer; ping gets an empty one.
        assert!(
            server
                .handle(&json!({"jsonrpc":"2.0","method":"notifications/initialized"}))
                .await
                .is_none()
        );
        assert_eq!(
            ask(
                &mut server,
                json!({"jsonrpc":"2.0","id":"p","method":"ping"})
            )
            .await["result"],
            json!({})
        );

        let unknown = ask(
            &mut server,
            json!({"jsonrpc":"2.0","id":3,"method":"resources/list"}),
        )
        .await;
        assert_eq!(unknown["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn lists_tools_with_schemas() {
        let mut server = Server::default();
        let list = ask(
            &mut server,
            json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
        )
        .await;
        let tools = list["result"]["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            [
                "search_folder",
                "transcribe",
                "classify",
                "rerank",
                "similarity"
            ]
        );
        for tool in tools {
            assert_eq!(tool["inputSchema"]["type"], "object", "{}", tool["name"]);
            assert!(tool["description"].as_str().unwrap().len() > 20);
        }
    }

    /// Bad arguments are the agent's to fix, so they come back as a tool
    /// result it can read, not as a protocol error.
    #[tokio::test]
    async fn tool_errors_are_results_the_agent_can_read() {
        let mut server = Server::default();
        let r = ask(&mut server, json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_folder","arguments":{"query":"x"}}})).await;
        assert_eq!(r["result"]["isError"], true);
        assert!(
            r["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("missing argument: path")
        );

        let r = ask(
            &mut server,
            json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"nope"}}),
        )
        .await;
        assert_eq!(r["result"]["isError"], true);
    }

    #[test]
    fn fingerprint_follows_the_files_that_would_be_indexed() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        std::fs::write(d.join("a.md"), "one").unwrap();
        let before = fingerprint(d);
        // Excluded and hidden folders do not count.
        std::fs::create_dir_all(d.join("target/debug")).unwrap();
        std::fs::write(d.join("target/debug/x.rs"), "generated").unwrap();
        std::fs::create_dir_all(d.join(".git")).unwrap();
        std::fs::write(d.join(".git/HEAD"), "ref").unwrap();
        assert_eq!(fingerprint(d), before);
        // A new real file does.
        std::fs::write(d.join("b.md"), "two").unwrap();
        assert_ne!(fingerprint(d), before);
    }

    #[test]
    fn helpers() {
        assert!(wildcard("*.min.js", "app.min.js") && !wildcard("*.min.js", "app.js"));
        assert!(wildcard("target", "target") && !wildcard("target", "targets"));
        assert_eq!(timestamp(3725.5), "01:02:05");
        assert_eq!(clip("abcdef", 3), "abc…");
        assert_eq!(fnv1a(b"same"), fnv1a(b"same"));
    }
}
