//! Models straight from the Hugging Face Hub.
//!
//! Wherever a model name is taken, a Hub repository works too:
//!
//! - `Qwen/Qwen3.5-9B` or `https://huggingface.co/Qwen/Qwen3.5-9B`, a
//!   safetensors repository;
//! - `unsloth/Qwen3.5-9B-GGUF`, a GGUF repository, read as Q4_K_M unless
//!   `unsloth/Qwen3.5-9B-GGUF:Q8_0` names another quantisation;
//! - a link to one file, `https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/main/Qwen3.5-9B-Q6_K.gguf`.
//!
//! The files are downloaded once into the model cache and the folder is loaded
//! like any `--model-path` folder. A finished download leaves a marker, so later
//! runs find the folder without asking the Hub anything, offline included. A
//! GGUF repository rarely carries `config.json` and `tokenizer.json`; they come
//! from the repository its model card names as `base_model`.
//!
//! `HF_TOKEN` is sent when set, for gated and private repositories, and
//! `HF_ENDPOINT` points at a mirror.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use kjarni_transformers::models::{
    ModelType, download_file, download_sharded_weights, get_default_cache_dir,
};

/// Written into a folder once everything in it has arrived.
const MARKER: &str = ".kjarni-hub.json";

/// GGUF quantisations Kjarni reads, best first: the one picked when a
/// repository has several and none was asked for.
const PREFERRED_QUANTS: [&str; 9] = [
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "BF16", "F16", "F32",
];

/// Small files a model folder may need beside its weights.
const METADATA: [&str; 12] = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "preprocessor_config.json",
    "sentence_bert_config.json",
    "modules.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
];

/// A repository on the Hub, and optionally one file or quantisation in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubRef {
    /// `org/name`.
    pub repo: String,
    /// A branch, tag or commit; `main` unless a link names another.
    pub revision: String,
    /// One file in the repository, from a `blob` or `resolve` link.
    pub file: Option<String>,
    /// A GGUF quantisation, from `org/name:Q8_0`.
    pub quant: Option<String>,
}

impl HubRef {
    /// Reads a Hub reference: a `huggingface.co` or `hf.co` link, or a bare
    /// `org/name` with an optional `:QUANT`. Anything else is not one.
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        let (rest, is_url) = match text.split_once("://") {
            Some(("https" | "http", rest)) => (rest, true),
            Some(_) => return None,
            None => (text, false),
        };
        let rest = rest.split(['?', '#']).next().unwrap_or("");
        let mut parts: Vec<&str> = rest.split('/').filter(|p| !p.is_empty()).collect();
        let on_hub = matches!(
            parts.first(),
            Some(&("huggingface.co" | "www.huggingface.co" | "hf.co"))
        );
        if on_hub {
            parts.remove(0);
        } else if is_url || parts.len() != 2 {
            return None;
        }
        if parts.len() < 2 {
            return None;
        }

        let (name, quant) = match parts[1].split_once(':') {
            Some((name, quant)) if parts.len() == 2 && valid_part(quant) => {
                (name, Some(quant.to_string()))
            }
            Some(_) => return None,
            None => (parts[1], None),
        };
        let org = parts[0];
        // Datasets and Spaces live under these, and are not models.
        if matches!(org, "datasets" | "spaces" | "docs" | "api" | "models")
            || !valid_part(org)
            || !valid_part(name)
        {
            return None;
        }

        let (revision, file) = match parts.get(2).copied() {
            None => ("main".to_string(), None),
            Some("tree") if parts.len() == 4 => (parts[3].to_string(), None),
            Some("blob" | "resolve") if parts.len() >= 5 => {
                (parts[3].to_string(), Some(parts[4..].join("/")))
            }
            Some(_) => return None,
        };
        Some(Self {
            repo: format!("{org}/{name}"),
            revision,
            file,
            quant,
        })
    }
}

fn valid_part(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 96
        && !s.starts_with(['.', '-'])
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// A check run on the folder once its small files are in and before the
/// weights download: an error stops there, with its reason.
pub type FolderCheck = fn(&Path) -> std::result::Result<(), String>;

/// How to fetch a [`HubRef`].
#[derive(Debug, Clone, Default)]
pub struct FetchOptions {
    /// Take a GGUF file when the repository has both GGUF and safetensors.
    pub prefer_gguf: bool,
    /// No progress output.
    pub quiet: bool,
    /// The model cache; the default cache when unset.
    pub cache_dir: Option<PathBuf>,
    /// Only use what is already downloaded.
    pub offline: bool,
    /// The Hub's address; `HF_ENDPOINT`, or huggingface.co, when unset.
    pub endpoint: Option<String>,
    /// Whether the folder suits the caller, checked before the weights download.
    pub check: Option<FolderCheck>,
}

impl FetchOptions {
    fn endpoint(&self) -> String {
        self.endpoint
            .clone()
            .or_else(|| std::env::var("HF_ENDPOINT").ok())
            .filter(|e| !e.is_empty())
            .unwrap_or_else(|| "https://huggingface.co".to_string())
            .trim_end_matches('/')
            .to_string()
    }
}

/// The folder for `model` when it names a Hub repository rather than a
/// registry model or a local path, fetched if need be; `None` otherwise.
pub async fn resolve(model: &str, options: &FetchOptions) -> Result<Option<PathBuf>> {
    if ModelType::from_cli_name(model).is_some() || Path::new(model).exists() {
        return Ok(None);
    }
    match HubRef::parse(model) {
        Some(hub) => fetch(&hub, options).await.map(Some),
        None => Ok(None),
    }
}

/// For a builder: the folder for `model` when no path was given and it names
/// a Hub repository, fetched unless the download policy is `Never`.
pub(crate) async fn builder_folder(
    model: &str,
    model_path: &Option<PathBuf>,
    cache_dir: &Option<PathBuf>,
    policy: crate::common::DownloadPolicy,
    load_config: &Option<crate::common::LoadConfig>,
    quiet: bool,
    check: Option<FolderCheck>,
) -> Result<Option<PathBuf>> {
    if model_path.is_some() {
        return Ok(None);
    }
    let prefer_gguf = load_config.clone().is_some_and(|c| c.into_inner().use_gguf);
    let options = FetchOptions {
        prefer_gguf,
        quiet,
        cache_dir: cache_dir.clone(),
        offline: matches!(policy, crate::common::DownloadPolicy::Never),
        endpoint: None,
        check,
    };
    resolve(model, &options).await
}

/// Downloads `hub` into the model cache, unless it is there already, and
/// returns the folder to load.
pub async fn fetch(hub: &HubRef, options: &FetchOptions) -> Result<PathBuf> {
    let cache = options
        .cache_dir
        .clone()
        .unwrap_or_else(get_default_cache_dir);
    let repo_dir = cache.join(folder_name(hub));
    if let Some(done) = cached(hub, &repo_dir, options.prefer_gguf) {
        return Ok(done);
    }
    if options.offline {
        bail!(
            "{} is not downloaded, and downloads are turned off",
            hub.repo
        );
    }

    let endpoint = options.endpoint();
    let listing = Listing::get(&endpoint, &hub.repo, &hub.revision).await?;
    let plan = plan(hub, &listing, options.prefer_gguf)?;
    let resolve_url = |repo: &str, file: &str| {
        format!(
            "{endpoint}/{repo}/resolve/{}/{file}",
            hub.revision.replace('/', "%2F")
        )
    };

    match plan {
        Plan::Safetensors { sharded } => {
            let dir = repo_dir;
            require_metadata(&hub.repo, &listing)?;
            announce(options, hub, &listing.weights_size(None), &dir);
            fetch_metadata(&dir, &hub.repo, &listing, &resolve_url, options.quiet).await?;
            run_check(options, hub, &dir)?;
            if sharded {
                let index = "model.safetensors.index.json";
                download_sharded_weights(&dir, &resolve_url(&hub.repo, index), options.quiet)
                    .await?;
            } else {
                download_file(
                    &dir,
                    "model.safetensors",
                    &resolve_url(&hub.repo, "model.safetensors"),
                    options.quiet,
                )
                .await?;
            }
            finish(&dir, hub, &listing, None, None)?;
            Ok(dir)
        }
        Plan::Gguf { file } => {
            let file_name = file.rsplit('/').next().unwrap_or(&file).to_string();
            let dir = repo_dir.join(file_name.trim_end_matches(".gguf"));
            // The tokenizer and config: the repository's own, or its base model's.
            let (meta_repo, meta_listing, base) = if listing.has("config.json")
                && listing.has("tokenizer.json")
            {
                (hub.repo.clone(), listing.clone(), None)
            } else {
                let base = listing.base_model.clone().ok_or_else(|| {
                    anyhow!(
                        "{} has GGUF weights but no tokenizer.json and config.json, and its \
                         model card names no base_model to take them from. Put the GGUF in a \
                         folder with the original model's tokenizer.json and config.json and \
                         pass that folder with --model-path.",
                        hub.repo
                    )
                })?;
                let base_listing = Listing::get(&endpoint, &base, "main")
                    .await
                    .map_err(|e| anyhow!("{} takes its tokenizer from {base}: {e}", hub.repo))?;
                (base.clone(), base_listing, Some(base))
            };
            require_metadata(&meta_repo, &meta_listing)?;
            announce(options, hub, &listing.weights_size(Some(&file)), &dir);
            let meta_url = |repo: &str, f: &str| format!("{endpoint}/{repo}/resolve/main/{f}");
            if base.is_some() {
                fetch_metadata(&dir, &meta_repo, &meta_listing, &meta_url, options.quiet).await?;
            } else {
                fetch_metadata(&dir, &meta_repo, &meta_listing, &resolve_url, options.quiet)
                    .await?;
            }
            run_check(options, hub, &dir)?;
            download_file(
                &dir,
                &file_name,
                &resolve_url(&hub.repo, &file),
                options.quiet,
            )
            .await?;
            finish(&dir, hub, &listing, Some(&file), base.as_deref())?;
            Ok(dir)
        }
    }
}

/// The finished download that answers `hub` in `cache_dir` (the default
/// cache when `None`), without asking the Hub anything.
pub fn downloaded(hub: &HubRef, prefer_gguf: bool, cache_dir: Option<&Path>) -> Option<PathBuf> {
    let cache = cache_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(get_default_cache_dir);
    cached(hub, &cache.join(folder_name(hub)), prefer_gguf)
}

/// `org_name`, as the registry names its folders, with `@revision` when it is
/// not `main`.
fn folder_name(hub: &HubRef) -> String {
    let mut name = hub.repo.replace('/', "_");
    if hub.revision != "main" {
        name.push('@');
        name.push_str(&hub.revision.replace('/', "_"));
    }
    name
}

/// A finished download that answers `hub`, if there is one.
fn cached(hub: &HubRef, repo_dir: &Path, prefer_gguf: bool) -> Option<PathBuf> {
    let done = |dir: &Path| dir.join(MARKER).is_file();
    if let Some(file) = hub.file.as_deref().filter(|f| f.ends_with(".gguf")) {
        let name = file.rsplit('/').next().unwrap_or(file);
        let dir = repo_dir.join(name.trim_end_matches(".gguf"));
        return done(&dir).then_some(dir);
    }
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(repo_dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir() && done(p))
        .collect();
    ggufs.sort();
    if let Some(quant) = &hub.quant {
        return ggufs.into_iter().find(|d| {
            d.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| has_quant(n, quant))
        });
    }
    if !prefer_gguf && done(repo_dir) {
        return Some(repo_dir.to_path_buf());
    }
    let best = ggufs.into_iter().min_by_key(|d| {
        quant_rank(d.file_name().and_then(|n| n.to_str()).unwrap_or(""))
            .unwrap_or(PREFERRED_QUANTS.len())
    });
    best.or_else(|| done(repo_dir).then(|| repo_dir.to_path_buf()))
}

/// What a repository's file list says about it.
#[derive(Debug, Clone, Default)]
struct Listing {
    files: Vec<(String, Option<u64>)>,
    base_model: Option<String>,
    sha: Option<String>,
}

impl Listing {
    async fn get(endpoint: &str, repo: &str, revision: &str) -> Result<Self> {
        let url = format!(
            "{endpoint}/api/models/{repo}/revision/{}?blobs=true",
            revision.replace('/', "%2F")
        );
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(60))
            .build()?;
        let mut request = client.get(&url);
        let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
        if let Some(token) = &token {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .map_err(|e| anyhow!("could not reach the Hugging Face Hub for {repo}: {e}"))?;
        let status = response.status().as_u16();
        if matches!(status, 401 | 403 | 404) {
            let hint = if token.is_some() {
                "HF_TOKEN is set but does not give access to it"
            } else {
                "for a gated model, accept its license on huggingface.co and set HF_TOKEN"
            };
            bail!(
                "{repo} was not found on the Hugging Face Hub, or it is private or gated; {hint}"
            );
        }
        if !(200..300).contains(&status) {
            bail!("the Hugging Face Hub answered {status} for {repo}");
        }
        let body = response.text().await?;
        Ok(Self::from_json(&serde_json::from_str::<Value>(&body)?))
    }

    fn from_json(json: &Value) -> Self {
        let files = json["siblings"]
            .as_array()
            .map(|s| {
                s.iter()
                    .filter_map(|f| {
                        Some((f["rfilename"].as_str()?.to_string(), f["size"].as_u64()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let base = &json["cardData"]["base_model"];
        let base_model = base
            .as_str()
            .or_else(|| {
                base.as_array()
                    .and_then(|a| a.first())
                    .and_then(|b| b.as_str())
            })
            .map(str::to_string);
        Self {
            files,
            base_model,
            sha: json["sha"].as_str().map(str::to_string),
        }
    }

    fn has(&self, name: &str) -> bool {
        self.files.iter().any(|(f, _)| f == name)
    }

    /// The download size of the weights: one GGUF file, or every safetensors
    /// file at the top of the repository.
    fn weights_size(&self, gguf: Option<&str>) -> Option<u64> {
        let sizes = self.files.iter().filter(|(f, _)| match gguf {
            Some(file) => f == file,
            None => !f.contains('/') && f.ends_with(".safetensors"),
        });
        sizes.map(|(_, s)| *s).sum()
    }
}

/// Which weights to fetch.
#[derive(Debug, PartialEq, Eq)]
enum Plan {
    Safetensors { sharded: bool },
    Gguf { file: String },
}

fn plan(hub: &HubRef, listing: &Listing, prefer_gguf: bool) -> Result<Plan> {
    let ggufs: Vec<&str> = listing
        .files
        .iter()
        .map(|(f, _)| f.as_str())
        .filter(|f| f.ends_with(".gguf"))
        .collect();

    if let Some(file) = hub.file.as_deref().filter(|f| f.ends_with(".gguf")) {
        if !listing.has(file) {
            bail!("{} has no file {file}", hub.repo);
        }
        if is_split(file) {
            bail!("{file} is one part of a split GGUF, which Kjarni does not read yet");
        }
        return Ok(Plan::Gguf {
            file: file.to_string(),
        });
    }
    let usable: Vec<&str> = ggufs
        .iter()
        .copied()
        .filter(|f| {
            let name = f.rsplit('/').next().unwrap_or(f);
            !name.starts_with("mmproj") && !is_split(f)
        })
        .collect();
    if let Some(quant) = &hub.quant {
        return usable
            .iter()
            .find(|f| has_quant(f.trim_end_matches(".gguf"), quant))
            .map(|f| Plan::Gguf {
                file: f.to_string(),
            })
            .ok_or_else(|| {
                anyhow!(
                    "{} has no {quant} GGUF; it has: {}",
                    hub.repo,
                    names_or_none(&usable)
                )
            });
    }

    let sharded = listing.has("model.safetensors.index.json");
    let safetensors = sharded || listing.has("model.safetensors");
    let best_gguf = usable
        .iter()
        .filter_map(|f| quant_rank(f.trim_end_matches(".gguf")).map(|r| (r, *f)))
        .min()
        .map(|(_, f)| Plan::Gguf {
            file: f.to_string(),
        });
    match (safetensors, best_gguf) {
        (true, Some(gguf)) if prefer_gguf => Ok(gguf),
        (true, _) => Ok(Plan::Safetensors { sharded }),
        (false, Some(gguf)) => Ok(gguf),
        (false, None) if !usable.is_empty() => bail!(
            "{} has no GGUF in a format Kjarni reads ({}); it has: {}. Name one with \
             {}:QUANT or a link to the file.",
            hub.repo,
            PREFERRED_QUANTS.join(", "),
            names_or_none(&usable),
            hub.repo
        ),
        (false, None) => bail!(
            "{} has no weights Kjarni reads: it needs model.safetensors, a sharded \
             model.safetensors.index.json, or a GGUF file (PyTorch .bin and ONNX files are not read)",
            hub.repo
        ),
    }
}

fn names_or_none(files: &[&str]) -> String {
    if files.is_empty() {
        "none".to_string()
    } else {
        files.join(", ")
    }
}

/// Whether a file name ends in `quant`, as `name-Q4_K_M` or `name.Q4_K_M`.
fn has_quant(stem: &str, quant: &str) -> bool {
    let (stem, quant) = (stem.to_ascii_lowercase(), quant.to_ascii_lowercase());
    stem.strip_suffix(&quant)
        .is_some_and(|rest| rest.ends_with(['-', '.', '_']))
}

/// Where `stem`'s quantisation stands in [`PREFERRED_QUANTS`], if it is one.
fn quant_rank(stem: &str) -> Option<usize> {
    PREFERRED_QUANTS.iter().position(|q| has_quant(stem, q))
}

/// `name-00001-of-00003.gguf`.
fn is_split(file: &str) -> bool {
    let stem = file.trim_end_matches(".gguf");
    let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
    parts.len() == 4
        && parts[1] == "of"
        && [parts[0], parts[2]]
            .iter()
            .all(|n| n.len() == 5 && n.chars().all(|c| c.is_ascii_digit()))
}

fn require_metadata(repo: &str, listing: &Listing) -> Result<()> {
    for needed in ["config.json", "tokenizer.json"] {
        if !listing.has(needed) {
            bail!("{repo} has no {needed}, which Kjarni needs to load a model");
        }
    }
    Ok(())
}

async fn fetch_metadata(
    dir: &Path,
    repo: &str,
    listing: &Listing,
    url: &impl Fn(&str, &str) -> String,
    quiet: bool,
) -> Result<()> {
    for name in METADATA.iter().filter(|n| listing.has(n)) {
        if let Some(parent) = Path::new(name).parent() {
            tokio::fs::create_dir_all(dir.join(parent)).await?;
        }
        download_file(dir, name, &url(repo, name), quiet).await?;
    }
    Ok(())
}

fn run_check(options: &FetchOptions, hub: &HubRef, dir: &Path) -> Result<()> {
    match options.check.map(|check| check(dir)) {
        Some(Err(reason)) => bail!("{} cannot be used here: {reason}", hub.repo),
        _ => Ok(()),
    }
}

fn announce(options: &FetchOptions, hub: &HubRef, size: &Option<u64>, dir: &Path) {
    if options.quiet {
        return;
    }
    let size = size
        .map(|b| format!(", {:.1} GB", b as f64 / 1e9))
        .unwrap_or_default();
    eprintln!(
        "Downloading {} from Hugging Face{size}\n  into {}",
        hub.repo,
        dir.display()
    );
}

fn finish(
    dir: &Path,
    hub: &HubRef,
    listing: &Listing,
    file: Option<&str>,
    base: Option<&str>,
) -> Result<()> {
    let record = json!({
        "repo": hub.repo,
        "revision": hub.revision,
        "sha": listing.sha,
        "file": file,
        "base_model": base,
    });
    std::fs::write(dir.join(MARKER), serde_json::to_string_pretty(&record)?)?;
    Ok(())
}

#[cfg(test)]
mod tests;
