use anyhow::Result;

pub async fn run(model: &str, model_path: Option<&str>, mode: &str, gpu: bool) -> Result<()> {
    eprintln!("TODO: repl");
    eprintln!("  model: {}", model);
    eprintln!("  model_path: {:?}", model_path);
    eprintln!("  mode: {}", mode);
    eprintln!("  gpu: {}", gpu);
    Ok(())
}