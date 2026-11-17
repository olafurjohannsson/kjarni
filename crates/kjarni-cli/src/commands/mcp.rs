//! `kjarni mcp`: Kjarni's capabilities as tools for AI agents, over the Model
//! Context Protocol on stdio.
//!
//! An agent (Claude Code, Cursor, Claude Desktop, any MCP client) starts this
//! as a subprocess and calls tools: semantic search over a folder of documents
//! or of photos, speech to text, classification, reranking and similarity, all
//! on the user's machine.
//!
//! The protocol is newline-delimited JSON-RPC 2.0. Only the part of MCP a tool
//! server needs is implemented: initialize, ping, tools/list and tools/call.
//! Models load on first use and stay loaded for the session, so only the first
//! call of each kind pays for it.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use kjarni::agent::tools::media;
use kjarni::classifier::Classifier;
use kjarni::common::{DownloadPolicy, ensure_model_downloaded};
use kjarni::embedder::Embedder;
use kjarni::folder_search::FolderSearch;
use kjarni::reranker::Reranker;
use kjarni::transcriber::Transcriber;
use kjarni::{ImageIndex, ModelType, SearchMode};
use serde_json::{Value, json};
use tokio::io::AsyncBufReadExt;

/// Protocol versions this server speaks, newest first. A client asking for
/// one of these gets it back; anything else gets the newest.
const PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// The embedding model behind `search_folder`. Small and fast, which matters
/// more than the last few points of quality when an agent is waiting.
const SEARCH_MODEL: &str = kjarni::folder_search::DEFAULT_MODEL;
const RERANK_MODEL: &str = "minilm-l6-v2-cross-encoder";
const TRANSCRIBE_MODEL: &str = media::TRANSCRIBE_MODEL;
const CLASSIFY_MODEL: &str = "distilbert-sentiment";
const IMAGE_MODEL: ModelType = media::IMAGE_MODEL;

const INSTRUCTIONS: &str = "Kjarni runs AI models locally on the user's machine; nothing leaves it. \
Use search_folder to find passages in a folder of documents or code by meaning, not just by \
exact words; the first search of a folder builds an index, later ones reuse it until files \
change. Use search_images to find photos in a folder by describing what is in them. Use transcribe for audio or video files, classify for sentiment, emotion or toxicity, \
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
    folders: Folders,
    embedder: Option<Embedder>,
    reranker: Option<Reranker>,
    transcriber: Option<Transcriber>,
    images: Option<ImageIndex>,
    classifiers: HashMap<String, Classifier>,
}

/// `search_folder`'s searches, reranked on request.
struct Folders(FolderSearch);

impl Default for Folders {
    fn default() -> Self {
        Self(FolderSearch::new().reranker(RERANK_MODEL))
    }
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
            "search_images" => self.search_images(&args).await,
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

        let found = self
            .folders
            .0
            .search_with(&folder, query, top_k, mode, rerank)
            .await?;

        let mut text = String::new();
        if let Some(update) = &found.update {
            text.push_str(&format!(
                "({} for this search.)\n\n",
                update.describe(&found.indexed_folder)
            ));
        }
        if found.results.is_empty() {
            text.push_str("No matching passages.");
        }
        for (i, hit) in found.results.iter().enumerate() {
            text.push_str(&format!(
                "{}. {}  (score {:.3})\n{}\n\n",
                i + 1,
                hit.source.display(),
                hit.score,
                clip(hit.text.trim(), 800)
            ));
        }
        Ok(text.trim_end().to_string())
    }

    async fn search_images(&mut self, args: &Value) -> Result<String> {
        let folder = PathBuf::from(string(args, "path")?);
        let folder = folder
            .canonicalize()
            .with_context(|| format!("cannot open {}", folder.display()))?;
        if !folder.is_dir() {
            bail!("{} is not a folder", folder.display());
        }
        let query = string(args, "query")?;
        let top_k = args
            .get("top_k")
            .and_then(Value::as_u64)
            .unwrap_or(5)
            .clamp(1, 50) as usize;

        if self.images.is_none() {
            let dir = ensure_model_downloaded(IMAGE_MODEL, None, DownloadPolicy::IfMissing, true)
                .await
                .context("downloading CLIP")?;
            self.images = Some(ImageIndex::load(&dir).context("loading CLIP")?);
        }
        let images = self.images.as_mut().expect("loaded above");

        // Seconds per hundred photos on the CPU the first time: keep it off
        // the runtime's other threads, as transcribe does.
        tokio::task::block_in_place(|| media::search_photos(images, &folder, query, top_k))
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

        Ok(media::transcript_text(&result))
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
            "description": "Find the passages in a folder of documents or code (txt, md, source code, json, yaml, html, ...) that best match a query, by meaning as well as by keyword. Runs locally. The first search of a folder indexes it, which takes seconds to minutes depending on its size; later searches reuse the index until files change, and a folder inside an indexed one uses that index. Build output, dependencies and hidden folders are skipped. A folder with more than 10,000 documents is refused with its largest subfolders, to search one of those instead.",
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
            "name": "search_images",
            "title": "Find photos by description",
            "description": "Find the photos in a folder that best match a description, such as \"a dog on a beach\" or \"receipt from a restaurant\", with CLIP, locally. Photos are never uploaded. JPEG and PNG, subfolders included; iPhone HEIC photos are not read. The first search of a folder embeds every photo, about 20 ms each on a desktop CPU; later searches reuse that until files change. Scores are only meaningful as a ranking.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Absolute path of the folder of photos." },
                    "query": { "type": "string", "description": "What the photo shows, in natural language." },
                    "top_k": { "type": "integer", "minimum": 1, "maximum": 50, "description": "How many photos to return. Default 5." }
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
    use std::path::Path;

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
                "search_images",
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

    /// A second search after files change updates the index rather than
    /// rebuilding it, and finds what was written since.
    #[tokio::test(flavor = "multi_thread")]
    async fn search_folder_follows_edits_to_the_folder() {
        let dir = tempfile::tempdir().unwrap();
        let notes = dir.path().canonicalize().unwrap();
        std::fs::write(
            notes.join("boat.md"),
            "The ferry leaves the harbour at nine.",
        )
        .unwrap();
        std::fs::write(notes.join("bread.md"), "Knead the dough and let it rise.").unwrap();
        let mut server = Server::default();
        let search = |q: &str| json!({ "path": notes, "query": q, "top_k": 1 });

        let first = server
            .search_folder(&search("when does the ferry go"))
            .await
            .unwrap();
        assert!(first.starts_with("(Indexed "), "{first}");
        assert!(first.contains("boat.md"), "{first}");

        let unchanged = server.search_folder(&search("baking")).await.unwrap();
        assert!(unchanged.starts_with("1. "), "{unchanged}");

        std::fs::write(
            notes.join("bread.md"),
            "Rye bread needs a sourdough starter.",
        )
        .unwrap();
        std::fs::write(notes.join("bike.md"), "Pump the bicycle tyres every week.").unwrap();
        let updated = server
            .search_folder(&search("sourdough starter"))
            .await
            .unwrap();
        assert!(updated.contains("1 new, 1 changed, 0 removed"), "{updated}");
        assert!(
            updated.contains("bread.md") && updated.contains("Rye bread"),
            "{updated}"
        );

        std::fs::remove_file(notes.join("boat.md")).unwrap();
        let removed = server
            .search_folder(&search("ferry harbour"))
            .await
            .unwrap();
        assert!(removed.contains("0 new, 0 changed, 1 removed"), "{removed}");
        assert!(!removed.contains("boat.md"), "{removed}");
    }

    /// Four test photos, one subject each. Each description must put its own
    /// photo first, and a second search reuses the saved vectors.
    #[tokio::test(flavor = "multi_thread")]
    async fn search_images_finds_photos_by_description() {
        let photos = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/clip/photos");
        let photos = photos.canonicalize().unwrap();
        let mut server = Server::default();
        for (query, expected) in [
            ("a red car", "red_car.png"),
            ("snow on a mountain", "snowy_mountain.png"),
            ("a sunny beach", "beach_holiday.png"),
            ("trees in a forest", "deep_forest.png"),
        ] {
            let text = server
                .search_images(&json!({ "path": photos, "query": query, "top_k": 1 }))
                .await
                .unwrap();
            let first = text.lines().last().unwrap();
            assert!(
                first.starts_with("1. ") && first.contains(expected),
                "{query}: {text}"
            );
        }
        let again = server
            .search_images(&json!({ "path": photos, "query": "a red car" }))
            .await
            .unwrap();
        assert!(!again.contains("Indexed"), "{again}");
        assert_eq!(again.lines().count(), 4, "{again}");
    }

    #[test]
    fn helpers() {
        assert_eq!(clip("abcdef", 3), "abc…");
    }
}
