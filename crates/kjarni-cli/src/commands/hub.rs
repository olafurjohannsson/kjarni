//! A Hugging Face repository given as `--model`, fetched before the command
//! runs: `kjarni chat --model Qwen/Qwen3.5-9B`.

use anyhow::Result;
use kjarni::hub::{FetchOptions, FolderCheck, resolve};

use kjarni_cli::{Commands, DEFAULT_GENERATE_MODEL};

/// When a command's `--model` names a Hugging Face repository and no
/// `--model-path` was given, downloads the repository into the model cache,
/// if it is not there yet, and points `--model-path` at its folder.
pub async fn resolve_models(command: &mut Commands) -> Result<()> {
    let decoder: Option<FolderCheck> = Some(kjarni::generator::decoder_folder_check);
    match command {
        Commands::Generate {
            model,
            model_path,
            gguf,
            quiet,
            ..
        } => {
            if let Some(dir) = fetch(model, model_path, *gguf, *quiet, decoder).await? {
                *model_path = Some(dir);
                // The folder names its own architecture; `generate` still wants
                // a registry name in `--model`, which a folder leaves unread.
                *model = DEFAULT_GENERATE_MODEL.to_string();
            }
        }
        Commands::Chat {
            model,
            model_path,
            gguf,
            quiet,
            ..
        } => set(model, model_path, *gguf, *quiet, decoder).await?,
        Commands::Agent {
            model,
            model_path,
            quiet,
            ..
        } => set(model, model_path, false, *quiet, decoder).await?,
        Commands::Summarize {
            model,
            model_path,
            quiet,
            ..
        }
        | Commands::Embed {
            model,
            model_path,
            quiet,
            ..
        }
        | Commands::Transcribe {
            model,
            model_path,
            quiet,
            ..
        }
        | Commands::Classify {
            model,
            model_path,
            quiet,
            ..
        }
        | Commands::Ner {
            model,
            model_path,
            quiet,
            ..
        }
        | Commands::Rerank {
            model,
            model_path,
            quiet,
            ..
        } => set(model, model_path, false, *quiet, None).await?,
        Commands::Translate {
            model: Some(model),
            model_path,
            quiet,
            ..
        } => set(model, model_path, false, *quiet, None).await?,
        _ => {}
    }
    Ok(())
}

async fn set(
    model: &str,
    model_path: &mut Option<String>,
    prefer_gguf: bool,
    quiet: bool,
    check: Option<FolderCheck>,
) -> Result<()> {
    if let Some(dir) = fetch(model, model_path, prefer_gguf, quiet, check).await? {
        *model_path = Some(dir);
    }
    Ok(())
}

async fn fetch(
    model: &str,
    model_path: &Option<String>,
    prefer_gguf: bool,
    quiet: bool,
    check: Option<FolderCheck>,
) -> Result<Option<String>> {
    if model_path.is_some() {
        return Ok(None);
    }
    let options = FetchOptions {
        prefer_gguf,
        quiet,
        check,
        ..FetchOptions::default()
    };
    let dir = resolve(model, &options).await?;
    Ok(dir.map(|d| d.to_string_lossy().into_owned()))
}
