use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("kjarni-cli").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Usage:"));
}

#[test]
fn test_cli_version() {
    let mut cmd = Command::cargo_bin("kjarni-cli").unwrap();
    cmd.arg("--version")
        .assert()
        .success();
}

// Test argument parsing failure
#[test]
fn test_invalid_command() {
    let mut cmd = Command::cargo_bin("kjarni-cli").unwrap();
    cmd.arg("not-a-command")
        .assert()
        .failure();
}