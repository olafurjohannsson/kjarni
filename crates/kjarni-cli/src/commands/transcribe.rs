use anyhow::Result;

pub async fn run(
    file: &str,
    model: &str,
    model_path: Option<&str>,
    language: Option<&str>,
) -> Result<()> {
    eprintln!("TODO: transcribe");
    eprintln!("  model: {}", model);
    eprintln!("  model_path: {:?}", model_path);
    eprintln!("  file: {}", file);
    eprintln!("  language: {:?}", language);
    Ok(())
}