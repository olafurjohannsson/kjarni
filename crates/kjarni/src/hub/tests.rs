use super::*;

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};

fn hub(repo: &str, revision: &str, file: Option<&str>, quant: Option<&str>) -> HubRef {
    HubRef {
        repo: repo.to_string(),
        revision: revision.to_string(),
        file: file.map(str::to_string),
        quant: quant.map(str::to_string),
    }
}

#[test]
fn hub_references_are_links_or_org_slash_name() {
    let cases = [
        (
            "Qwen/Qwen3.5-9B",
            Some(hub("Qwen/Qwen3.5-9B", "main", None, None)),
        ),
        (
            "https://huggingface.co/Qwen/Qwen3.5-9B",
            Some(hub("Qwen/Qwen3.5-9B", "main", None, None)),
        ),
        (
            "huggingface.co/Qwen/Qwen3.5-9B/",
            Some(hub("Qwen/Qwen3.5-9B", "main", None, None)),
        ),
        (
            "https://hf.co/Qwen/Qwen3.5-9B?library=transformers",
            Some(hub("Qwen/Qwen3.5-9B", "main", None, None)),
        ),
        (
            "https://huggingface.co/Qwen/Qwen3.5-9B/tree/v1.0",
            Some(hub("Qwen/Qwen3.5-9B", "v1.0", None, None)),
        ),
        (
            "unsloth/Qwen3.5-9B-GGUF:Q8_0",
            Some(hub("unsloth/Qwen3.5-9B-GGUF", "main", None, Some("Q8_0"))),
        ),
        (
            "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/main/Qwen3.5-9B-Q6_K.gguf",
            Some(hub(
                "unsloth/Qwen3.5-9B-GGUF",
                "main",
                Some("Qwen3.5-9B-Q6_K.gguf"),
                None,
            )),
        ),
        (
            "https://huggingface.co/org/repo/resolve/abc123/sub/model.gguf",
            Some(hub("org/repo", "abc123", Some("sub/model.gguf"), None)),
        ),
        // Registry names, paths and other sites are not Hub references.
        ("qwen3.5-4b", None),
        ("models/qwen/extra", None),
        ("./org/name", None),
        ("https://example.com/org/name", None),
        ("https://huggingface.co/datasets/org/name", None),
        ("https://huggingface.co/spaces/org/name", None),
        ("https://huggingface.co/org", None),
        ("org/name:", None),
        ("file:///org/name", None),
    ];
    for (text, want) in cases {
        assert_eq!(HubRef::parse(text), want, "{text}");
    }
}

#[test]
fn quantisations_are_read_from_file_names() {
    assert!(has_quant("Qwen3.5-9B-Q4_K_M", "q4_k_m"));
    assert!(has_quant("llama-2-7b.Q4_K_M", "Q4_K_M"));
    assert!(!has_quant("Qwen3.5-9B-UD-Q4_K_XL", "Q4_K_M"));
    assert!(!has_quant("Qwen3.5-9B-IQ4_K_M", "Q4_K_M"));
    assert_eq!(quant_rank("x-Q4_K_M"), Some(0));
    assert_eq!(quant_rank("x-Q8_0"), Some(5));
    assert_eq!(quant_rank("x-IQ4_XS"), None);
    assert!(is_split("Big-Q4_K_M-00001-of-00003.gguf"));
    assert!(!is_split("Qwen3.5-9B-Q4_K_M.gguf"));
}

fn listing(files: &[&str], base: Option<&str>) -> Listing {
    Listing {
        files: files.iter().map(|f| (f.to_string(), Some(10))).collect(),
        base_model: base.map(str::to_string),
        sha: None,
    }
}

/// From an unsloth GGUF repository's real file list: Q4_K_M is picked over
/// the IQ, UD and mmproj files, a named quantisation or file wins, and a
/// repository with both kinds of weights gives safetensors unless GGUF is
/// preferred.
#[test]
fn the_weights_picked_from_a_repository() {
    let unsloth = listing(
        &[
            "Qwen3.5-9B-BF16.gguf",
            "Qwen3.5-9B-IQ4_XS.gguf",
            "Qwen3.5-9B-Q4_K_M.gguf",
            "Qwen3.5-9B-Q8_0.gguf",
            "Qwen3.5-9B-UD-Q4_K_XL.gguf",
            "mmproj-F16.gguf",
        ],
        Some("Qwen/Qwen3.5-9B"),
    );
    let gguf = |f: &str| Plan::Gguf {
        file: f.to_string(),
    };
    let repo = "unsloth/Qwen3.5-9B-GGUF";
    assert_eq!(
        plan(&hub(repo, "main", None, None), &unsloth, false).unwrap(),
        gguf("Qwen3.5-9B-Q4_K_M.gguf")
    );
    assert_eq!(
        plan(&hub(repo, "main", None, Some("q8_0")), &unsloth, false).unwrap(),
        gguf("Qwen3.5-9B-Q8_0.gguf")
    );
    let missing = plan(&hub(repo, "main", None, Some("Q2_K")), &unsloth, false).unwrap_err();
    assert!(
        missing.to_string().contains("has no Q2_K GGUF"),
        "{missing}"
    );
    assert_eq!(
        plan(
            &hub(repo, "main", Some("Qwen3.5-9B-BF16.gguf"), None),
            &unsloth,
            false
        )
        .unwrap(),
        gguf("Qwen3.5-9B-BF16.gguf")
    );

    let both = listing(
        &["config.json", "model.safetensors", "model-Q4_K_M.gguf"],
        None,
    );
    let org = hub("org/both", "main", None, None);
    assert_eq!(
        plan(&org, &both, false).unwrap(),
        Plan::Safetensors { sharded: false }
    );
    assert_eq!(plan(&org, &both, true).unwrap(), gguf("model-Q4_K_M.gguf"));

    let sharded = listing(&["model.safetensors.index.json"], None);
    assert_eq!(
        plan(&org, &sharded, false).unwrap(),
        Plan::Safetensors { sharded: true }
    );

    let pytorch = listing(&["config.json", "pytorch_model.bin"], None);
    let err = plan(&org, &pytorch, false).unwrap_err().to_string();
    assert!(err.contains("PyTorch .bin"), "{err}");

    let only_iq = listing(&["m-IQ4_XS.gguf"], None);
    let err = plan(&org, &only_iq, false).unwrap_err().to_string();
    assert!(
        err.contains("m-IQ4_XS.gguf") && err.contains(":QUANT"),
        "{err}"
    );
}

#[test]
fn a_listing_reads_files_sizes_and_the_base_model() {
    let json = json!({
        "sha": "abc",
        "cardData": {"base_model": ["Qwen/Qwen3.5-9B"]},
        "siblings": [
            {"rfilename": "a.gguf", "size": 5},
            {"rfilename": "model-00001-of-00002.safetensors", "size": 7},
            {"rfilename": "model-00002-of-00002.safetensors", "size": 8},
            {"rfilename": "sub/x.safetensors", "size": 100},
        ]
    });
    let l = Listing::from_json(&json);
    assert_eq!(l.base_model.as_deref(), Some("Qwen/Qwen3.5-9B"));
    assert_eq!(l.sha.as_deref(), Some("abc"));
    assert_eq!(l.weights_size(None), Some(15));
    assert_eq!(l.weights_size(Some("a.gguf")), Some(5));
    let single = Listing::from_json(&json!({"cardData": {"base_model": "org/base"}}));
    assert_eq!(single.base_model.as_deref(), Some("org/base"));
}

/// A local stand-in for the Hub: answers each path from a table and records
/// what was asked for.
struct FakeHub {
    endpoint: String,
    asked: Arc<Mutex<Vec<String>>>,
}

impl FakeHub {
    fn start(routes: HashMap<String, (u16, Vec<u8>)>) -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let asked = Arc::new(Mutex::new(Vec::new()));
        let log = asked.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                if reader.read_line(&mut first).is_err() {
                    continue;
                }
                let mut line = String::new();
                while reader.read_line(&mut line).is_ok() && line.trim() != "" {
                    line.clear();
                }
                let path = first.split_whitespace().nth(1).unwrap_or("");
                let path = path.split('?').next().unwrap_or("").to_string();
                log.lock().unwrap().push(path.clone());
                let (status, body) = routes.get(&path).cloned().unwrap_or((404, Vec::new()));
                let head = format!(
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
            }
        });
        Self { endpoint, asked }
    }

    fn asked(&self) -> Vec<String> {
        self.asked.lock().unwrap().clone()
    }
}

fn api(repo: &str, files: &[&str], base: Option<&str>) -> (String, (u16, Vec<u8>)) {
    let siblings: Vec<Value> = files
        .iter()
        .map(|f| json!({"rfilename": f, "size": 4}))
        .collect();
    let mut body = json!({"sha": "deadbeef", "siblings": siblings});
    if let Some(base) = base {
        body["cardData"] = json!({"base_model": base});
    }
    (
        format!("/api/models/{repo}/revision/main"),
        (200, body.to_string().into_bytes()),
    )
}

fn file(repo: &str, name: &str, body: &str) -> (String, (u16, Vec<u8>)) {
    (
        format!("/{repo}/resolve/main/{name}"),
        (200, body.as_bytes().to_vec()),
    )
}

fn options(hub: &FakeHub, cache: &Path) -> FetchOptions {
    FetchOptions {
        quiet: true,
        cache_dir: Some(cache.to_path_buf()),
        endpoint: Some(hub.endpoint.clone()),
        ..FetchOptions::default()
    }
}

#[tokio::test]
async fn a_safetensors_repository_downloads_once_and_then_loads_offline() {
    let repo = "org/tiny";
    let files = [
        "config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "1_Pooling/config.json",
        "model.safetensors",
        "README.md",
        "pytorch_model.bin",
    ];
    let fake = FakeHub::start(HashMap::from([
        api(repo, &files, None),
        file(repo, "config.json", "{\"architectures\": []}"),
        file(repo, "tokenizer.json", "{}"),
        file(repo, "chat_template.jinja", "{{ x }}"),
        file(repo, "1_Pooling/config.json", "{\"pooling\": 1}"),
        file(repo, "model.safetensors", "weights"),
    ]));
    let cache = tempfile::tempdir().unwrap();
    let opts = options(&fake, cache.path());

    let dir = resolve("https://huggingface.co/org/tiny", &opts)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(dir, cache.path().join("org_tiny"));
    assert_eq!(
        std::fs::read_to_string(dir.join("model.safetensors")).unwrap(),
        "weights"
    );
    assert_eq!(
        std::fs::read_to_string(dir.join("1_Pooling/config.json")).unwrap(),
        "{\"pooling\": 1}"
    );
    assert!(dir.join("chat_template.jinja").is_file());
    assert!(!dir.join("README.md").exists() && !dir.join("pytorch_model.bin").exists());
    let marker: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(MARKER)).unwrap()).unwrap();
    assert_eq!(marker["repo"], "org/tiny");
    assert_eq!(marker["sha"], "deadbeef");

    // A second resolve, offline, asks nothing of the Hub.
    let before = fake.asked().len();
    let offline = FetchOptions {
        offline: true,
        ..opts.clone()
    };
    assert_eq!(resolve("org/tiny", &offline).await.unwrap(), Some(dir));
    assert_eq!(fake.asked().len(), before);

    // Registry names and missing downloads offline are told apart.
    assert_eq!(resolve("minilm-l6-v2", &offline).await.unwrap(), None);
    let err = resolve("org/other", &offline)
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("not downloaded"), "{err}");
}

#[tokio::test]
async fn a_gguf_repository_takes_its_tokenizer_from_its_base_model() {
    let fake = FakeHub::start(HashMap::from([
        api(
            "unsloth/M-GGUF",
            &["M-Q4_K_M.gguf", "M-Q8_0.gguf", "mmproj-F16.gguf"],
            Some("org/M"),
        ),
        file("unsloth/M-GGUF", "M-Q4_K_M.gguf", "gguf-q4"),
        file("unsloth/M-GGUF", "M-Q8_0.gguf", "gguf-q8"),
        api(
            "org/M",
            &[
                "config.json",
                "tokenizer.json",
                "generation_config.json",
                "model.safetensors",
            ],
            None,
        ),
        file("org/M", "config.json", "{}"),
        file("org/M", "tokenizer.json", "{\"t\": 1}"),
        file("org/M", "generation_config.json", "{}"),
    ]));
    let cache = tempfile::tempdir().unwrap();
    let opts = options(&fake, cache.path());

    let dir = resolve("unsloth/M-GGUF", &opts).await.unwrap().unwrap();
    assert_eq!(dir, cache.path().join("unsloth_M-GGUF").join("M-Q4_K_M"));
    assert_eq!(
        std::fs::read_to_string(dir.join("M-Q4_K_M.gguf")).unwrap(),
        "gguf-q4"
    );
    assert_eq!(
        std::fs::read_to_string(dir.join("tokenizer.json")).unwrap(),
        "{\"t\": 1}"
    );
    assert!(dir.join("generation_config.json").is_file());
    assert!(
        !fake
            .asked()
            .iter()
            .any(|p| p.contains("model.safetensors") || p.contains("mmproj"))
    );

    // Another quantisation sits beside it, and each is found again offline.
    let q8 = resolve("unsloth/M-GGUF:Q8_0", &opts)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(q8, cache.path().join("unsloth_M-GGUF").join("M-Q8_0"));
    let offline = FetchOptions {
        offline: true,
        ..opts
    };
    assert_eq!(
        resolve("unsloth/M-GGUF", &offline).await.unwrap(),
        Some(dir)
    );
    assert_eq!(
        resolve("unsloth/M-GGUF:q8_0", &offline).await.unwrap(),
        Some(q8)
    );
}

/// A folder the caller cannot use is turned down before the weights are
/// downloaded.
#[tokio::test]
async fn a_failed_check_stops_before_the_weights() {
    let repo = "org/encoder";
    let fake = FakeHub::start(HashMap::from([
        api(
            repo,
            &["config.json", "tokenizer.json", "model.safetensors"],
            None,
        ),
        file(repo, "config.json", "{}"),
        file(repo, "tokenizer.json", "{}"),
        file(repo, "model.safetensors", "weights"),
    ]));
    let cache = tempfile::tempdir().unwrap();
    let opts = FetchOptions {
        check: Some(|_| Err("BertModel is not a decoder".to_string())),
        ..options(&fake, cache.path())
    };
    let err = resolve(repo, &opts).await.unwrap_err().to_string();
    assert_eq!(
        err,
        "org/encoder cannot be used here: BertModel is not a decoder"
    );
    assert!(
        !fake
            .asked()
            .iter()
            .any(|p| p.ends_with("model.safetensors"))
    );
    assert!(!cache.path().join("org_encoder").join(MARKER).exists());
}

#[tokio::test]
async fn a_missing_or_gated_repository_says_how_to_get_in() {
    let fake = FakeHub::start(HashMap::from([(
        "/api/models/meta-llama/Gated/revision/main".to_string(),
        (401, b"{}".to_vec()),
    )]));
    let cache = tempfile::tempdir().unwrap();
    let err = resolve("meta-llama/Gated", &options(&fake, cache.path()))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.starts_with(
            "meta-llama/Gated was not found on the Hugging Face Hub, or it is private or gated"
        ),
        "{err}"
    );
    assert!(err.contains("HF_TOKEN"), "{err}");

    let fake = FakeHub::start(HashMap::from([api(
        "org/no-tokenizer",
        &["config.json", "model.safetensors"],
        None,
    )]));
    let err = resolve("org/no-tokenizer", &options(&fake, cache.path()))
        .await
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "org/no-tokenizer has no tokenizer.json, which Kjarni needs to load a model"
    );
}
