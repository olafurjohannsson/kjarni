use anyhow::Result;
use super::util::resolve_input;

pub async fn run(
    input: Option<&str>,
    model: &str,
    model_path: Option<&str>,
    src: Option<&str>,
    dst: Option<&str>,
) -> Result<()> {
    let text = resolve_input(input)?;

    eprintln!("TODO: translate");
    eprintln!("  model: {}", model);
    eprintln!("  model_path: {:?}", model_path);
    eprintln!("  src: {:?}", src);
    eprintln!("  dst: {:?}", dst);
    eprintln!("  input length: {} chars", text.len());
    Ok(())
}