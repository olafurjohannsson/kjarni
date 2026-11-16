//! The `kjarni` binary as people run it: arguments in, text and an exit code
//! out. These use the tiny checkpoints in kjarni-models, so nothing downloads.

use std::path::PathBuf;
use std::process::{Command, Output};

fn kjarni(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_kjarni"))
        .args(args)
        .output()
        .expect("run kjarni")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn testdata(model: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../kjarni-models/src/models")
        .join(model)
        .join("testdata")
}

#[test]
fn help_and_version() {
    let help = kjarni(&["--help"]);
    assert!(help.status.success());
    for command in [
        "chat", "generate", "serve", "agent", "index", "search", "inspect",
    ] {
        assert!(stdout(&help).contains(command), "--help lists {command}");
    }
    let version = kjarni(&["--version"]);
    assert!(version.status.success());
    assert!(stdout(&version).contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn unknown_commands_and_flags_fail_with_a_message() {
    let o = kjarni(&["nonsense"]);
    assert!(!o.status.success());
    assert!(
        stderr(&o).contains("unrecognized subcommand 'nonsense'"),
        "{}",
        stderr(&o)
    );
    let o = kjarni(&["generate", "--no-such-flag"]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("--no-such-flag"), "{}", stderr(&o));
}

#[test]
fn model_search_and_info_read_the_registry() {
    let o = kjarni(&["model", "search", "qwen3.5"]);
    assert!(o.status.success(), "{}", stderr(&o));
    for name in ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b"] {
        assert!(stdout(&o).contains(name), "{}", stdout(&o));
    }
    let o = kjarni(&["model", "info", "qwen3.5-0.8b"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(stdout(&o).contains("Hybrid DeltaNet"), "{}", stdout(&o));
    let o = kjarni(&["model", "info", "no-such-model"]);
    assert!(!o.status.success());
}

#[test]
fn inspect_reads_gguf_and_safetensors() {
    let gguf = testdata("qwen3_5").join("tiny.gguf");
    let o = kjarni(&["inspect", gguf.to_str().unwrap()]);
    assert!(o.status.success(), "{}", stderr(&o));
    let out = stdout(&o);
    assert!(out.contains("Format        GGUF"), "{out}");
    assert!(
        out.contains("qwen35.attention.head_count                  4"),
        "{out}"
    );

    let o = kjarni(&["inspect", testdata("phi").to_str().unwrap()]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(stdout(&o).contains("Phi3ForCausalLM"));

    let o = kjarni(&["inspect", "/no/such/model"]);
    assert!(!o.status.success());
}

/// Greedy text from the tiny Mistral, token for token what transformers gives
/// for the same prompt after the start-of-text token.
#[test]
fn generate_from_a_local_folder_matches_transformers() {
    let dir = testdata("mistral");
    let o = kjarni(&[
        "generate",
        "--model-path",
        dir.to_str().unwrap(),
        "--greedy",
        "-n",
        "12",
        "-q",
        "--no-stream",
        "t5 t17 t99 t3 t42 t200 t7 t8",
    ]);
    assert!(o.status.success(), "{}", stderr(&o));
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("expected.json")).unwrap()).unwrap();
    let want: Vec<String> = expected["bos_short_greedy"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| format!("t{}", id.as_u64().unwrap()))
        .collect();
    assert_eq!(stdout(&o).trim(), want.join(" "));
}
