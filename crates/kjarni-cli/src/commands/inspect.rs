//! `kjarni inspect` -- print everything a model file says about itself.
//!
//! Written for the job of adding a new architecture. To write a `Config` and a tensor
//! layout you need three things: the hyperparameters, the exact tensor names, and the
//! shapes and dtypes behind them. A GGUF carries all of that as metadata; a safetensors
//! checkpoint carries it in `config.json` plus the tensor index. This prints whichever
//! it finds, in a form you can read next to the struct you are filling in.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use kjarni::{DType, GgufLoader, SafeTensorsLoader, WeightLoader};

pub async fn run(path: &str) -> Result<()> {
    let target = resolve(path)?;

    if target.is_file() && target.extension().and_then(|e| e.to_str()) == Some("gguf") {
        return inspect_gguf(&target);
    }
    if target.is_dir() {
        // A directory holding a single GGUF is still a GGUF model.
        if let Some(gguf) = first_gguf(&target)? {
            return inspect_gguf(&gguf);
        }
        return inspect_safetensors(&target);
    }
    if target.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        // A lone shard: inspect the directory that holds it, so the index and
        // config.json come along too.
        let dir = target.parent().unwrap_or(Path::new("."));
        return inspect_safetensors(dir);
    }

    Err(anyhow!(
        "don't know how to inspect {}: expected a .gguf, a .safetensors, or a directory",
        target.display()
    ))
}

/// Accepts a path, or the name of something already in the local cache.
fn resolve(path: &str) -> Result<PathBuf> {
    let direct = PathBuf::from(path);
    if direct.exists() {
        return Ok(direct);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let cached = PathBuf::from(home).join(".cache/kjarni").join(path);
        if cached.exists() {
            return Ok(cached);
        }
    }
    Err(anyhow!(
        "'{path}' is not a path, and nothing by that name is in ~/.cache/kjarni"
    ))
}

fn first_gguf(dir: &Path) -> Result<Option<PathBuf>> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .collect();
    found.sort();
    Ok(found.into_iter().next())
}

/// One line per value, short enough to keep the columns readable.
///
/// Vocabularies and merge tables run to hundreds of thousands of entries and chat
/// templates are multi-line Jinja; neither says anything about the architecture, and
/// both wreck the layout if printed whole.
fn render(v: &serde_json::Value, max: usize) -> String {
    let flat = match v {
        // A short array is usually the answer to something ("architectures",
        // "eos_token_id"), so show it. A long one is a vocabulary.
        serde_json::Value::Array(a) if a.len() > 8 => return format!("[{} items]", a.len()),
        serde_json::Value::Array(a) => a
            .iter()
            .map(|x| render(x, max))
            .collect::<Vec<_>>()
            .join(", "),
        serde_json::Value::String(s) => s.replace('\n', " ").replace('\r', ""),
        other => other.to_string(),
    };
    let flat = if matches!(v, serde_json::Value::Array(_)) {
        format!("[{flat}]")
    } else {
        flat
    };
    if flat.chars().count() > max {
        let cut: String = flat.chars().take(max).collect();
        format!("{cut}...")
    } else {
        flat
    }
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    format!("{v:.2} {}", UNITS[u])
}

fn inspect_gguf(path: &Path) -> Result<()> {
    let size = std::fs::metadata(path)?.len();
    let loader = GgufLoader::new(path)?;

    println!("File          {}", path.display());
    println!("Format        GGUF, {}", human_bytes(size));
    println!("Architecture  {}", loader.architecture);
    println!();

    println!("Metadata");
    for (k, v) in loader.metadata() {
        // Token lists and merge tables run to hundreds of thousands of entries and
        // tell you nothing about the architecture.
        println!("  {k:<44} {}", render(v, 60));
    }
    println!();

    let mut names: Vec<String> = loader
        .tensor_names()
        .into_iter()
        .map(String::from)
        .collect();
    names.sort();
    let infos: BTreeMap<String, (DType, Vec<usize>)> = names
        .iter()
        .filter_map(|n| loader.tensor_info(n).map(|i| (n.clone(), i)))
        .collect();
    print_tensors(&infos);
    Ok(())
}

fn inspect_safetensors(dir: &Path) -> Result<()> {
    println!("File          {}", dir.display());
    println!("Format        safetensors");

    let config = dir.join("config.json");
    if config.exists() {
        let text = std::fs::read_to_string(&config)?;
        println!();
        println!("config.json");
        match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(serde_json::Value::Object(map)) => {
                for (k, v) in &map {
                    println!("  {k:<44} {}", render(v, 80));
                }
            }
            _ => println!("{text}"),
        }
    } else {
        println!();
        println!("(no config.json next to the weights)");
    }
    println!();

    let loader = SafeTensorsLoader::new(dir)?;
    let mut names: Vec<String> = loader
        .tensor_names()
        .into_iter()
        .map(String::from)
        .collect();
    names.sort();
    let mut infos: BTreeMap<String, (DType, Vec<usize>)> = BTreeMap::new();
    for n in &names {
        let view = loader.get_raw(n)?;
        infos.insert(n.clone(), (view.dtype, view.shape.clone()));
    }
    print_tensors(&infos);
    Ok(())
}

/// Prints the tensor table with the per-layer pattern collapsed.
///
/// A 3B model has a few hundred tensors and all but a handful are the same dozen names
/// repeated once per layer. Printing them in full buries the thing you are looking for,
/// which is the naming pattern and the shapes. So any name containing a number is
/// folded to `{i}` and reported once with its layer count, and a shape that varies
/// across layers is flagged rather than silently showing only the first.
fn print_tensors(infos: &BTreeMap<String, (DType, Vec<usize>)>) {
    let mut per_layer: BTreeMap<String, (usize, Vec<(DType, Vec<usize>)>)> = BTreeMap::new();
    let mut singles: Vec<(&String, &(DType, Vec<usize>))> = Vec::new();

    for (name, info) in infos {
        let templated: String = name
            .split('.')
            .map(|part| {
                if part.parse::<u32>().is_ok() {
                    "{i}"
                } else {
                    part
                }
            })
            .collect::<Vec<_>>()
            .join(".");

        if templated == *name {
            singles.push((name, info));
        } else {
            let e = per_layer.entry(templated).or_insert((0, Vec::new()));
            e.0 += 1;
            e.1.push(info.clone());
        }
    }

    // Column width from the data: HF names like `post_attention_layernorm` overflow
    // any fixed guess, and a broken column is harder to scan than a wide one.
    let width = per_layer
        .keys()
        .map(|k| k.len())
        .chain(singles.iter().map(|(n, _)| n.len()))
        .max()
        .unwrap_or(40)
        .max(24);

    println!("Tensors ({} total)", infos.len());

    if !per_layer.is_empty() {
        println!();
        println!("  per layer");
        for (pattern, (count, variants)) in &per_layer {
            let first = &variants[0];
            let uniform = variants.iter().all(|v| v == first);
            let shape = format!("{:?}", first.1);
            if uniform {
                println!(
                    "    {pattern:<width$} {shape:<20} {:<6} x{count}",
                    format!("{:?}", first.0)
                );
            } else {
                // Mixed precision across layers is normal in a "_M" quantisation and
                // is exactly the sort of thing a config needs to account for.
                let mut kinds: Vec<String> =
                    variants.iter().map(|v| format!("{:?}", v.0)).collect();
                kinds.sort();
                kinds.dedup();
                println!(
                    "    {pattern:<width$} {shape:<20} {:<6} x{count}  (varies)",
                    kinds.join("/")
                );
            }
        }
    }

    if !singles.is_empty() {
        println!();
        println!("  single");
        for (name, (dtype, shape)) in singles {
            println!("    {name:<width$} {:<20} {dtype:?}", format!("{shape:?}"));
        }
    }
}
