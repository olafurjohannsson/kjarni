//! Running a shell command in the working directory.

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Map, Value as Json, json};

use super::files::Workspace;
use crate::agent::tool::{Access, Tool, ToolSpec, first_line, string_arg, usize_arg};

/// Seconds a command may run when the call gives no timeout.
const DEFAULT_TIMEOUT_SECS: usize = 120;

/// Bytes of each output stream kept: the start and the end of a long output.
const KEEP_BYTES: usize = 6000;

pub struct RunCommand(pub Arc<Workspace>);

#[async_trait]
impl Tool for RunCommand {
    fn spec(&self) -> ToolSpec {
        let shell = if cfg!(windows) { "cmd" } else { "sh" };
        ToolSpec::new(
            "run_command",
            format!(
                "Run a command with {shell} in the working directory and return its exit code and output."
            ),
            json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command line to run."},
                    "timeout_seconds": {"type": "integer", "description": "Stop the command after this many seconds. Default 120."}
                },
                "required": ["command"]
            }),
        )
    }

    fn access(&self) -> Access {
        Access::Execute
    }

    fn summary(&self, args: &Map<String, Json>) -> String {
        let command = args.get("command").and_then(Json::as_str).unwrap_or("?");
        format!("run_command {}", first_line(command))
    }

    async fn call(&self, args: &Map<String, Json>) -> anyhow::Result<String> {
        let command = string_arg(args, "command")?;
        let timeout = usize_arg(args, "timeout_seconds")?.unwrap_or(DEFAULT_TIMEOUT_SECS);
        let mut cmd = if cfg!(windows) {
            let mut c = tokio::process::Command::new("cmd");
            c.arg("/C").arg(command);
            c
        } else {
            let mut c = tokio::process::Command::new("sh");
            c.arg("-c").arg(command);
            c
        };
        cmd.current_dir(self.0.root())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let child = cmd.spawn()?;
        let output = match tokio::time::timeout(
            Duration::from_secs(timeout as u64),
            child.wait_with_output(),
        )
        .await
        {
            Ok(result) => result?,
            Err(_) => {
                anyhow::bail!("the command did not finish within {timeout} seconds and was stopped")
            }
        };
        let mut out = match output.status.code() {
            Some(code) => format!("exit code {code}\n"),
            None => "stopped by a signal\n".to_string(),
        };
        let stdout = clip(&String::from_utf8_lossy(&output.stdout));
        let stderr = clip(&String::from_utf8_lossy(&output.stderr));
        if !stdout.trim().is_empty() {
            out.push_str(&format!("stdout:\n{}\n", stdout.trim_end()));
        }
        if !stderr.trim().is_empty() {
            out.push_str(&format!("stderr:\n{}\n", stderr.trim_end()));
        }
        if stdout.trim().is_empty() && stderr.trim().is_empty() {
            out.push_str("(no output)\n");
        }
        Ok(out)
    }
}

/// A long output's start and end, with how much was left out between them.
pub(crate) fn clip(text: &str) -> String {
    if text.len() <= 2 * KEEP_BYTES {
        return text.to_string();
    }
    let mut head = KEEP_BYTES;
    while !text.is_char_boundary(head) {
        head -= 1;
    }
    let mut tail = text.len() - KEEP_BYTES;
    while !text.is_char_boundary(tail) {
        tail += 1;
    }
    format!(
        "{}\n[... {} bytes left out ...]\n{}",
        &text[..head],
        tail - head,
        &text[tail..]
    )
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn args(v: Json) -> Map<String, Json> {
        v.as_object().unwrap().clone()
    }

    #[tokio::test]
    async fn returns_exit_code_and_both_streams() {
        let dir = tempfile::tempdir().unwrap();
        let tool = RunCommand(Arc::new(Workspace::new(dir.path())));
        let out = tool
            .call(&args(json!({"command": "echo out; echo err >&2; exit 3"})))
            .await
            .unwrap();
        assert_eq!(out, "exit code 3\nstdout:\nout\nstderr:\nerr\n");
    }

    #[tokio::test]
    async fn runs_in_the_working_directory_and_times_out() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker"), "").unwrap();
        let tool = RunCommand(Arc::new(Workspace::new(dir.path())));
        let out = tool.call(&args(json!({"command": "ls"}))).await.unwrap();
        assert_eq!(out, "exit code 0\nstdout:\nmarker\n");
        let err = tool
            .call(&args(json!({"command": "sleep 5", "timeout_seconds": 1})))
            .await
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "the command did not finish within 1 seconds and was stopped"
        );
    }

    #[test]
    fn long_output_keeps_head_and_tail() {
        let text = "x".repeat(KEEP_BYTES * 3);
        let clipped = clip(&text);
        assert!(clipped.contains(&format!("[... {} bytes left out ...]", KEEP_BYTES)));
        assert_eq!(
            clipped.len(),
            2 * KEEP_BYTES + format!("\n[... {} bytes left out ...]\n", KEEP_BYTES).len()
        );
    }
}
