pub mod chatml;
pub mod deepseek;
#[cfg(not(target_arch = "wasm32"))]
pub mod jinja;
pub mod llama3;
pub mod mistral;
pub mod phi3;
pub mod templates;
