//! The daemon's log file. A daemon started at login has no terminal, so it
//! sends its output to `logs/daemon.log` in its folder, the same file on
//! every system, and `kjarni daemon logs` reads it back.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};

/// Past this size the log is moved to `daemon.log.1` when the daemon starts,
/// so two runs' worth is kept at most.
const ROTATE_AT: u64 = 10 * 1024 * 1024;

pub fn log_path(daemon_dir: &Path) -> PathBuf {
    daemon_dir.join("logs").join("daemon.log")
}

/// Moves a log past [`ROTATE_AT`] aside, replacing the previous old one.
fn rotate(path: &Path, limit: u64) -> std::io::Result<()> {
    if std::fs::metadata(path).is_ok_and(|m| m.len() > limit) {
        let mut old = path.as_os_str().to_owned();
        old.push(".1");
        std::fs::rename(path, old)?;
    }
    Ok(())
}

/// Sends everything this process prints to `path` from here on, appended.
pub fn redirect_output(path: &Path) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    rotate(path, ROTATE_AT).with_context(|| format!("rotating {}", path.display()))?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    writeln!(
        file,
        "\n--- kjarni daemon {} starting, process {} ---",
        env!("CARGO_PKG_VERSION"),
        std::process::id()
    )?;
    point_output_at(file)
}

#[cfg(unix)]
fn point_output_at(file: File) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    // SAFETY: dup2 onto the standard descriptors; `fd` is open, and stays
    // open because the file is leaked below.
    let failed = unsafe { libc::dup2(fd, 1) < 0 || libc::dup2(fd, 2) < 0 };
    if failed {
        return Err(std::io::Error::last_os_error()).context("redirecting output to the log");
    }
    std::mem::forget(file);
    Ok(())
}

#[cfg(windows)]
fn point_output_at(file: File) -> Result<()> {
    use std::os::windows::io::IntoRawHandle;
    use windows_sys::Win32::System::Console::{
        FreeConsole, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE, SetStdHandle,
    };
    // Rust's stdout and stderr look the handle up on every write, so
    // swapping it here moves all later output. The handle is never closed.
    let handle = file.into_raw_handle();
    // SAFETY: `handle` is a valid, open file handle owned from here on by
    // the process's standard handles.
    unsafe {
        if SetStdHandle(STD_OUTPUT_HANDLE, handle) == 0
            || SetStdHandle(STD_ERROR_HANDLE, handle) == 0
        {
            return Err(std::io::Error::last_os_error()).context("redirecting output to the log");
        }
        // The logon task starts the daemon in a console window; with the
        // output going to the log, the window has nothing to show.
        FreeConsole();
    }
    Ok(())
}

/// The last `n` lines of `text`.
fn last_lines(text: &str, n: usize) -> &str {
    if n == 0 {
        return "";
    }
    let trimmed = text.strip_suffix('\n').unwrap_or(text);
    match trimmed.match_indices('\n').rev().nth(n - 1) {
        Some((i, _)) => &text[i + 1..],
        None => text,
    }
}

/// Prints the last `lines` of the log and, with `follow`, keeps printing
/// what is added, through a restart's rotation too, until interrupted.
pub fn show(path: &Path, lines: usize, follow: bool) -> Result<()> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !follow => {
            println!(
                "No log yet at {}. The daemon writes one when it is installed with `kjarni daemon install`.",
                path.display()
            );
            return Ok(());
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            while !path.exists() {
                std::thread::sleep(Duration::from_millis(500));
            }
            File::open(path)?
        }
        Err(e) => return Err(e).with_context(|| format!("opening {}", path.display())),
    };
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    let mut out = std::io::stdout().lock();
    out.write_all(last_lines(&text, lines).as_bytes())?;
    out.flush()?;
    if !follow {
        return Ok(());
    }
    let mut offset = text.len() as u64;
    let mut buffer = Vec::new();
    loop {
        std::thread::sleep(Duration::from_millis(500));
        let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        if len < offset {
            // Rotated at a restart: the new file starts from the beginning.
            file = match File::open(path) {
                Ok(f) => f,
                Err(_) => continue,
            };
            offset = 0;
        }
        if len > offset {
            file.seek(SeekFrom::Start(offset))?;
            buffer.clear();
            file.read_to_end(&mut buffer)?;
            offset += buffer.len() as u64;
            out.write_all(&buffer)?;
            out.flush()?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_lines_counts_from_the_end() {
        assert_eq!(last_lines("a\nb\nc\n", 2), "b\nc\n");
        assert_eq!(last_lines("a\nb\nc", 2), "b\nc");
        assert_eq!(last_lines("a\nb\n", 5), "a\nb\n");
        assert_eq!(last_lines("a\nb\n", 0), "");
        assert_eq!(last_lines("", 3), "");
    }

    #[test]
    fn a_large_log_is_moved_aside() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("daemon.log");
        std::fs::write(&log, "0123456789").unwrap();
        rotate(&log, 100).unwrap();
        assert!(log.exists(), "a small log stays");
        rotate(&log, 5).unwrap();
        assert!(!log.exists());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("daemon.log.1")).unwrap(),
            "0123456789"
        );
        std::fs::write(&log, "second").unwrap();
        rotate(&log, 5).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("daemon.log.1")).unwrap(),
            "second",
            "the newer log replaces the older one"
        );
    }
}
