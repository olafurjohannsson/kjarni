use anyhow::{Result, anyhow};
use std::io::{self, Read};
use std::path::Path;

use kjarni::ModelType;

/// Resolve input from: direct text, file path, or stdin
///
/// Rules:
/// - If input is None, read from stdin
/// - If input looks like a file path and exists, read the file
/// - Otherwise, treat input as literal text
pub fn resolve_input(input: Option<&str>) -> Result<String> {
    match input {
        Some(text) => {
            // Check if it's a file path
            let path = Path::new(text);
            if path.exists() && path.is_file() {
                std::fs::read_to_string(path)
                    .map_err(|e| anyhow!("Failed to read file '{}': {}", text, e))
            } else {
                Ok(text.to_string())
            }
        }
        None => {
            // Read from stdin
            let stdin = io::stdin();
            let mut buffer = String::new();
            stdin.lock().read_to_string(&mut buffer)?;

            if buffer.is_empty() {
                return Err(anyhow!(
                    "No input provided. Pass text as argument, a file path, or pipe via stdin."
                ));
            }

            Ok(buffer)
        }
    }
}

/// Resolve a CLI model name, or fail with a message that suggests near matches.
///
/// Every command should go through this before building anything. Previously only
/// `generate` validated up front; `chat`, `classify`, `embed`, `translate` and
/// `summarize` passed the name straight to their builder, so a typo surfaced as
/// whatever the builder happened to say. For chat that was
/// `Failed to initialize chat: <name>`, which does not even reveal that the name
/// was the problem, while `generate` offered a "did you mean?" for the same typo.
///
/// `arch_hint` is the `kjarni model list --arch <hint>` value for the command, so
/// the error can point at the right subset of the registry.
pub fn resolve_model(name: &str, arch_hint: Option<&str>) -> anyhow::Result<ModelType> {
    ModelType::from_cli_name(name)
        .ok_or_else(|| anyhow::anyhow!(model_not_found_error(name, arch_hint)))
}

/// Create a helpful error message with "did you mean?" suggestions
pub fn model_not_found_error(name: &str, arch_hint: Option<&str>) -> String {
    let mut msg = format!("Unknown model: '{}'.", name);

    if let Some(arch) = arch_hint {
        msg.push_str(&format!(
            " Run 'kjarni model list --arch {}' to see available models.",
            arch
        ));
    } else {
        msg.push_str(" Run 'kjarni model list' to see available models.");
    }

    let suggestions = ModelType::find_similar(name);
    if !suggestions.is_empty() {
        msg.push_str("\n\nDid you mean?");
        for (suggestion, _) in suggestions {
            msg.push_str(&format!("\n  - {}", suggestion));
        }
    }

    msg
}

/// Refuses a GPU run whose weights cannot fit in free VRAM.
///
/// Without this the loader uploads until the device is lost, which on Linux takes
/// the terminal down with it rather than printing anything. `WgpuContext` has a
/// `GpuMemoryInfo::available_memory` field for exactly this check, but
/// `query_gpu_memory` is stubbed out and always returns `None`, so the library
/// has no number to gate on. Probing the environment belongs up here anyway.
///
/// Best effort by design: if free VRAM cannot be determined the run proceeds, on
/// the grounds that a missing probe should not block a load that would have
/// worked.
///
/// `prefer_gguf` is the `--gguf` flag: the cache can hold both formats, and only the one
/// that will load counts.
pub fn check_gpu_capacity(model_type: kjarni::ModelType, prefer_gguf: bool) -> anyhow::Result<()> {
    let Some(free_bytes) = free_vram_bytes() else {
        return Ok(());
    };

    let dir = model_type.cache_dir(&kjarni::registry::cache_dir());
    let files: Vec<(String, u64)> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let size = e.metadata().ok()?.len();
            Some((e.file_name().to_string_lossy().into_owned(), size))
        })
        .collect();
    let weights = weights_to_load(&files, prefer_gguf);

    if weights.bytes == 0 {
        return Ok(());
    }

    // Weights are not the whole story: activations, the KV cache and the
    // attention scores all live in VRAM too. A fifth is a rough allowance, and
    // deliberately generous rather than exact, since the failure it prevents is
    // a hard crash and the cost of a false refusal is one flag.
    let needed = weights.bytes + weights.bytes / 5;
    if needed <= free_bytes {
        return Ok(());
    }

    let gb = |b: u64| b as f64 / 1_073_741_824.0;
    let name = model_type.cli_name();
    let quantized_hint = if weights.is_gguf {
        String::new()
    } else {
        format!(
            "The quantized build is roughly a quarter the size:\n    \
             kjarni model download {name} --gguf\n\n"
        )
    };
    Err(anyhow::anyhow!(
        "{name} needs about {:.1} GB of VRAM ({:.1} GB of weights plus working \
         memory) and only {:.1} GB is free.\n\n\
         {quantized_hint}Or run on the CPU by dropping --gpu.",
        gb(needed),
        gb(weights.bytes),
        gb(free_bytes)
    ))
}

/// The weight files a load will upload.
#[derive(Debug, PartialEq, Eq)]
struct WeightsToLoad {
    bytes: u64,
    is_gguf: bool,
}

/// Picks one format from a cache directory's files: GGUF for `--gguf` and otherwise
/// safetensors, or whichever is present when only one is.
fn weights_to_load(files: &[(String, u64)], prefer_gguf: bool) -> WeightsToLoad {
    let total = |suffix: &str| -> u64 {
        files
            .iter()
            .filter(|(name, _)| name.ends_with(suffix))
            .map(|(_, size)| size)
            .sum()
    };
    let (gguf, safetensors) = (total(".gguf"), total(".safetensors"));
    let is_gguf = gguf > 0 && (prefer_gguf || safetensors == 0);
    WeightsToLoad {
        bytes: if is_gguf { gguf } else { safetensors },
        is_gguf,
    }
}

/// Free VRAM in bytes, or `None` when it cannot be determined.
fn free_vram_bytes() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    // One line per GPU; the first is the one wgpu picks by default.
    let mib: u64 = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()?;
    Some(mib * 1_048_576)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sizes of a cached DeepSeek-R1 8B holding both formats.
    fn both_formats() -> Vec<(String, u64)> {
        vec![
            ("config.json".to_string(), 826),
            (
                "model-00001-of-000002.safetensors".to_string(),
                8_667_826_246,
            ),
            (
                "model-00002-of-000002.safetensors".to_string(),
                7_392_730_108,
            ),
            ("model.gguf".to_string(), 4_920_737_216),
            ("tokenizer.json".to_string(), 9_084_480),
        ]
    }

    #[test]
    fn gguf_flag_counts_only_the_gguf_file() {
        assert_eq!(
            weights_to_load(&both_formats(), true),
            WeightsToLoad {
                bytes: 4_920_737_216,
                is_gguf: true
            }
        );
    }

    #[test]
    fn without_the_flag_only_safetensors_count() {
        assert_eq!(
            weights_to_load(&both_formats(), false),
            WeightsToLoad {
                bytes: 16_060_556_354,
                is_gguf: false
            }
        );
    }

    #[test]
    fn the_only_format_present_counts_whatever_the_flag() {
        let gguf_only = vec![("model.gguf".to_string(), 100)];
        let expected = WeightsToLoad {
            bytes: 100,
            is_gguf: true,
        };
        assert_eq!(weights_to_load(&gguf_only, false), expected);

        let safetensors_only = vec![("model.safetensors".to_string(), 100)];
        let expected = WeightsToLoad {
            bytes: 100,
            is_gguf: false,
        };
        assert_eq!(weights_to_load(&safetensors_only, true), expected);
    }

    #[test]
    fn no_weights_count_as_nothing() {
        let files = vec![("config.json".to_string(), 826)];
        assert_eq!(weights_to_load(&files, true).bytes, 0);
    }
}
