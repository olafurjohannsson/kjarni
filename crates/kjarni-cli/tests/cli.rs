//! The `kjarni` binary as people run it: arguments in, text and an exit code
//! out. These use the tiny checkpoints in kjarni-models, so nothing downloads.

use std::path::PathBuf;
use std::process::{Command, Output};

fn kjarni(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kjarni"))
        .args(args)
        .output()
        .expect("run kjarni")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn testdata(model: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../kjarni-models/src/models")
        .join(model)
        .join("testdata")
}

#[test]
fn help_and_version() {
    let help = kjarni(&["--help"]);
    assert!(help.status.success());
    for command in [
        "chat", "generate", "serve", "agent", "index", "search", "inspect",
    ] {
        assert!(stdout(&help).contains(command), "--help lists {command}");
    }
    let version = kjarni(&["--version"]);
    assert!(version.status.success());
    assert!(stdout(&version).contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn unknown_commands_and_flags_fail_with_a_message() {
    let o = kjarni(&["nonsense"]);
    assert!(!o.status.success());
    assert!(
        stderr(&o).contains("unrecognized subcommand 'nonsense'"),
        "{}",
        stderr(&o)
    );
    let o = kjarni(&["generate", "--no-such-flag"]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("--no-such-flag"), "{}", stderr(&o));
}

#[test]
fn model_search_and_info_read_the_registry() {
    let o = kjarni(&["model", "search", "qwen3.5"]);
    assert!(o.status.success(), "{}", stderr(&o));
    for name in ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b"] {
        assert!(stdout(&o).contains(name), "{}", stdout(&o));
    }
    let o = kjarni(&["model", "info", "qwen3.5-0.8b"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(stdout(&o).contains("Hybrid DeltaNet"), "{}", stdout(&o));
    let o = kjarni(&["model", "info", "no-such-model"]);
    assert!(!o.status.success());
}

#[test]
fn inspect_reads_gguf_and_safetensors() {
    let gguf = testdata("qwen3_5").join("tiny.gguf");
    let o = kjarni(&["inspect", gguf.to_str().unwrap()]);
    assert!(o.status.success(), "{}", stderr(&o));
    let out = stdout(&o);
    assert!(out.contains("Format        GGUF"), "{out}");
    assert!(
        out.contains("qwen35.attention.head_count                  4"),
        "{out}"
    );

    let o = kjarni(&["inspect", testdata("phi").to_str().unwrap()]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(stdout(&o).contains("Phi3ForCausalLM"));

    let o = kjarni(&["inspect", "/no/such/model"]);
    assert!(!o.status.success());
}

/// Greedy text from the tiny Mistral, token for token what transformers gives
/// for the same prompt after the start-of-text token.
#[test]
fn generate_from_a_local_folder_matches_transformers() {
    let dir = testdata("mistral");
    let o = kjarni(&[
        "generate",
        "--model-path",
        dir.to_str().unwrap(),
        "--greedy",
        "-n",
        "12",
        "-q",
        "--no-stream",
        "t5 t17 t99 t3 t42 t200 t7 t8",
    ]);
    assert!(o.status.success(), "{}", stderr(&o));
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("expected.json")).unwrap()).unwrap();
    let want: Vec<String> = expected["bos_short_greedy"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| format!("t{}", id.as_u64().unwrap()))
        .collect();
    assert_eq!(stdout(&o).trim(), want.join(" "));
}

/// `generate --gguf` on a registry model reads the GGUF it has and downloads
/// nothing else. It fetched the full-precision safetensors first whenever they
/// were missing: 70 GB for qwen3.5-35b-a3b, none of it read. The cache here
/// holds the tiny Qwen 3.5 GGUF under qwen3.5-0.8b's folder, and the network
/// goes nowhere, so any download fails the run.
#[test]
fn generate_with_gguf_reads_the_cached_gguf_and_downloads_nothing() {
    let cache = tempfile::tempdir().unwrap();
    let folder = cache.path().join("Qwen_Qwen3.5-0.8B");
    std::fs::create_dir(&folder).unwrap();
    let tiny = testdata("qwen3_5");
    std::fs::copy(tiny.join("tiny.gguf"), folder.join("model.gguf")).unwrap();
    std::fs::copy(tiny.join("tokenizer.json"), folder.join("tokenizer.json")).unwrap();
    std::fs::copy(tiny.join("config.json"), folder.join("config.json")).unwrap();

    let o = Command::new(env!("CARGO_BIN_EXE_kjarni"))
        .args([
            "generate",
            "--model",
            "qwen3.5-0.8b",
            "--gguf",
            "--greedy",
            "-n",
            "4",
            "-q",
            "--no-stream",
            "t5 t17 t99",
        ])
        .env("KJARNI_CACHE_DIR", cache.path())
        .env("HTTPS_PROXY", "http://127.0.0.1:9")
        .env("HTTP_PROXY", "http://127.0.0.1:9")
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(stdout(&o).split_whitespace().count(), 4, "{}", stdout(&o));
    let files: Vec<_> = std::fs::read_dir(&folder)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    assert!(
        !files.iter().any(|f| f.contains("safetensors")),
        "{files:?}"
    );
}

/// A stand-in for the Hugging Face Hub on a local port: answers each path
/// from a table, 404 otherwise, and records what was asked for.
fn fake_hub(
    routes: std::collections::HashMap<String, Vec<u8>>,
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let asked = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let log = asked.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut first = String::new();
            let _ = reader.read_line(&mut first);
            let mut line = String::new();
            while reader.read_line(&mut line).is_ok() && !line.trim().is_empty() {
                line.clear();
            }
            let path = first.split_whitespace().nth(1).unwrap_or("");
            let path = path.split('?').next().unwrap_or("").to_string();
            log.lock().unwrap().push(path.clone());
            let (status, body) = match routes.get(&path) {
                Some(body) => (200, body.clone()),
                None => (404, Vec::new()),
            };
            let head = format!(
                "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&body);
        }
    });
    (endpoint, asked)
}

/// `--model org/name` downloads a Hugging Face repository, safetensors or GGUF,
/// and generates exactly what `--model-path` on the same files does. A GGUF
/// repository takes its tokenizer and config from its base model, and a repo
/// that is not a decoder is turned down before its weights are fetched.
#[test]
fn generate_from_a_hugging_face_repository() {
    let tiny = testdata("qwen3_5");
    let read = |f: &str| std::fs::read(tiny.join(f)).unwrap();
    let api = |files: &[&str], base: Option<&str>| {
        let siblings: Vec<String> = files
            .iter()
            .map(|f| format!("{{\"rfilename\": \"{f}\", \"size\": 1}}"))
            .collect();
        let card = base
            .map(|b| format!(", \"cardData\": {{\"base_model\": \"{b}\"}}"))
            .unwrap_or_default();
        format!(
            "{{\"sha\": \"1\", \"siblings\": [{}]{card}}}",
            siblings.join(", ")
        )
        .into_bytes()
    };
    let routes = std::collections::HashMap::from([
        (
            "/api/models/org/tiny-qwen/revision/main".to_string(),
            api(
                &[
                    "config.json",
                    "tokenizer.json",
                    "generation_config.json",
                    "model.safetensors",
                ],
                None,
            ),
        ),
        (
            "/org/tiny-qwen/resolve/main/config.json".to_string(),
            read("config.json"),
        ),
        (
            "/org/tiny-qwen/resolve/main/tokenizer.json".to_string(),
            read("tokenizer.json"),
        ),
        (
            "/org/tiny-qwen/resolve/main/generation_config.json".to_string(),
            read("generation_config.json"),
        ),
        (
            "/org/tiny-qwen/resolve/main/model.safetensors".to_string(),
            read("model.safetensors"),
        ),
        (
            "/api/models/org/tiny-qwen-GGUF/revision/main".to_string(),
            api(&["tiny-F32.gguf"], Some("org/tiny-qwen")),
        ),
        (
            "/org/tiny-qwen-GGUF/resolve/main/tiny-F32.gguf".to_string(),
            read("tiny.gguf"),
        ),
        (
            "/api/models/org/encoder/revision/main".to_string(),
            api(
                &["config.json", "tokenizer.json", "model.safetensors"],
                None,
            ),
        ),
        (
            "/org/encoder/resolve/main/config.json".to_string(),
            b"{\"architectures\": [\"BertModel\"], \"model_type\": \"bert\"}".to_vec(),
        ),
        (
            "/org/encoder/resolve/main/tokenizer.json".to_string(),
            read("tokenizer.json"),
        ),
    ]);
    let (endpoint, asked) = fake_hub(routes);
    let cache = tempfile::tempdir().unwrap();
    let run = |model_args: &[&str]| {
        let mut args = vec!["generate"];
        args.extend_from_slice(model_args);
        args.extend_from_slice(&["--greedy", "-n", "4", "-q", "--no-stream", "t5 t17 t99"]);
        Command::new(env!("CARGO_BIN_EXE_kjarni"))
            .args(&args)
            .env("KJARNI_CACHE_DIR", cache.path())
            .env("HF_ENDPOINT", &endpoint)
            .env("NO_PROXY", "127.0.0.1")
            .env_remove("HF_TOKEN")
            .output()
            .unwrap()
    };

    let local = run(&["--model-path", tiny.to_str().unwrap()]);
    assert!(local.status.success(), "{}", stderr(&local));
    assert_eq!(
        stdout(&local).split_whitespace().count(),
        4,
        "{}",
        stdout(&local)
    );

    let hub = run(&["--model", "https://huggingface.co/org/tiny-qwen"]);
    assert!(hub.status.success(), "{}", stderr(&hub));
    assert_eq!(stdout(&hub), stdout(&local));
    assert!(
        cache
            .path()
            .join("org_tiny-qwen/model.safetensors")
            .is_file()
    );

    let gguf = run(&["--model", "org/tiny-qwen-GGUF"]);
    assert!(gguf.status.success(), "{}", stderr(&gguf));
    assert_eq!(stdout(&gguf), stdout(&local));
    let folder = cache.path().join("org_tiny-qwen-GGUF/tiny-F32");
    assert!(folder.join("tiny-F32.gguf").is_file() && folder.join("tokenizer.json").is_file());

    let encoder = run(&["--model", "org/encoder"]);
    assert!(!encoder.status.success());
    assert!(
        stderr(&encoder).contains("org/encoder cannot be used here"),
        "{}",
        stderr(&encoder)
    );
    assert!(
        !asked
            .lock()
            .unwrap()
            .iter()
            .any(|p| p == "/org/encoder/resolve/main/model.safetensors"),
        "{:?}",
        asked.lock().unwrap()
    );

    // Downloaded once: with the Hub gone, the cached folder still loads.
    let before = asked.lock().unwrap().len();
    let again = run(&["--model", "org/tiny-qwen"]);
    assert!(again.status.success(), "{}", stderr(&again));
    assert_eq!(stdout(&again), stdout(&local));
    assert_eq!(asked.lock().unwrap().len(), before);
}

/// `--model-path` loads the folder whatever `--model` says: `embed` and
/// `classify` used to turn away any `--model` not in the registry first.
/// Needs minilm-l6-v2 downloaded.
#[test]
#[ignore]
fn embed_with_a_path_ignores_an_unregistered_model_name() {
    let folder = kjarni::ModelType::from_cli_name("minilm-l6-v2")
        .unwrap()
        .cache_dir(&kjarni::common::default_cache_dir());
    let text = "The lighthouse keeper found a bottle.";
    let by_path = kjarni(&[
        "embed",
        "--model",
        "someone/not-in-the-registry",
        "--model-path",
        folder.to_str().unwrap(),
        "-q",
        text,
    ]);
    assert!(by_path.status.success(), "{}", stderr(&by_path));
    let by_name = kjarni(&["embed", "--model", "minilm-l6-v2", "-q", text]);
    assert_eq!(stdout(&by_path), stdout(&by_name));
}
