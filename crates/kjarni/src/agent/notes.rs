//! Project notes: a Markdown file in a project that tells an agent how the
//! project builds, tests and is laid out, read into the system prompt.

use std::path::{Path, PathBuf};

/// The file names read as notes, in order of preference within one folder:
/// `KJARNI.md` for notes meant for Kjarni alone, `AGENTS.md` for notes any
/// coding agent reads.
pub const NOTES_FILES: [&str; 2] = ["KJARNI.md", "AGENTS.md"];

/// Bytes of notes put in the prompt at most, across all files.
pub const MAX_NOTES_BYTES: usize = 16_000;

/// The notes found for a working folder.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectNotes {
    /// The files read, outermost first.
    pub files: Vec<PathBuf>,
    /// Their text as the prompt shows it, each under its path.
    pub text: String,
}

/// The notes for `workdir`: the notes file of each folder from the root of
/// its repository (the nearest folder holding `.git`) down to `workdir`, or
/// of `workdir` alone outside a repository. Notes past [`MAX_NOTES_BYTES`]
/// are cut, with a line saying how to read the rest.
pub fn project_notes(workdir: &Path) -> Option<ProjectNotes> {
    let mut folders = Vec::new();
    let mut repo = false;
    for folder in workdir.ancestors() {
        folders.push(folder);
        if folder.join(".git").exists() {
            repo = true;
            break;
        }
    }
    if !repo {
        folders.truncate(1);
    }
    folders.reverse();

    let mut files = Vec::new();
    let mut text = String::new();
    let mut left = MAX_NOTES_BYTES;
    for folder in folders {
        let Some((path, body)) = NOTES_FILES.iter().find_map(|name| {
            let path = folder.join(name);
            let body = std::fs::read_to_string(&path).ok()?;
            Some((path, body))
        }) else {
            continue;
        };
        let body = body.trim();
        if body.is_empty() {
            continue;
        }
        let shown = path
            .strip_prefix(workdir)
            .unwrap_or(&path)
            .display()
            .to_string();
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(&format!("From {shown}:\n"));
        if body.len() <= left {
            text.push_str(body);
            left -= body.len();
        } else {
            let mut cut = left;
            while !body.is_char_boundary(cut) {
                cut -= 1;
            }
            text.push_str(body[..cut].trim_end());
            text.push_str(&format!(
                "\n[The rest of {shown} is not shown; read_file shows all of it.]"
            ));
            left = 0;
        }
        files.push(path);
    }
    (!files.is_empty()).then_some(ProjectNotes { files, text })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn reads_the_repository_root_then_the_working_folder() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("repo");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        write(&dir.path().join("AGENTS.md"), "outside the repository");
        write(&root.join("AGENTS.md"), "Build with cargo build.\n");
        write(&root.join("crates/web/KJARNI.md"), "Run npm test here.");
        write(
            &root.join("crates/web/AGENTS.md"),
            "not read: KJARNI.md wins",
        );
        let work = root.join("crates/web");

        let notes = project_notes(&work).unwrap();
        assert_eq!(
            notes.files,
            [root.join("AGENTS.md"), work.join("KJARNI.md")]
        );
        assert_eq!(
            notes.text,
            format!(
                "From {}:\nBuild with cargo build.\n\nFrom KJARNI.md:\nRun npm test here.",
                root.join("AGENTS.md").display()
            )
        );
    }

    #[test]
    fn outside_a_repository_only_the_working_folder_counts() {
        let dir = tempfile::tempdir().unwrap();
        write(&dir.path().join("AGENTS.md"), "parent notes");
        let work = dir.path().join("notes");
        std::fs::create_dir_all(&work).unwrap();
        assert_eq!(project_notes(&work), None);

        write(&work.join("AGENTS.md"), "  \n");
        assert_eq!(project_notes(&work), None, "an empty file is no notes");
    }

    #[test]
    fn long_notes_are_cut_on_a_character_boundary() {
        let dir = tempfile::tempdir().unwrap();
        // Two-byte characters, so the limit falls inside one.
        let body = "ð".repeat(MAX_NOTES_BYTES);
        write(&dir.path().join("AGENTS.md"), &body);

        let notes = project_notes(dir.path()).unwrap();
        let expected = format!(
            "From AGENTS.md:\n{}\n[The rest of AGENTS.md is not shown; read_file shows all of it.]",
            "ð".repeat(MAX_NOTES_BYTES / 2)
        );
        assert_eq!(notes.text, expected);
    }
}
