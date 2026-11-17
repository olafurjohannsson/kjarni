//! The startup entry that runs the daemon at login, as the user who
//! installed it: a systemd user unit on Linux, a LaunchAgent on macOS, and a
//! Task Scheduler logon task on Windows. None of them needs admin rights.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};

/// The systemd user unit's name.
pub const SYSTEMD_UNIT: &str = "kjarni.service";
/// The LaunchAgent's label, and its file's name without `.plist`.
pub const LAUNCHD_LABEL: &str = "ai.kjarni.daemon";
/// The Task Scheduler task's name.
pub const TASK_NAME: &str = "Kjarni";

/// What the startup entry runs.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub program: PathBuf,
    pub args: Vec<String>,
    /// Variables the daemon needs set, such as `KJARNI_DAEMON_DIR` when the
    /// installer had it set. Windows tasks cannot carry these.
    pub env: Vec<(String, String)>,
    /// Where the daemon's output goes before it opens its own log.
    pub log: PathBuf,
}

// Built on every system so the tests check it everywhere.
#[cfg_attr(any(target_os = "macos", windows), allow(dead_code))]
/// A systemd unit argument: quoted, with systemd's specifiers (`%`) and
/// variable expansion (`$`) escaped.
fn systemd_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '%' => out.push_str("%%"),
            '$' => out.push_str("$$"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// Built on every system so the tests check it everywhere.
#[cfg_attr(any(target_os = "macos", windows), allow(dead_code))]
/// The systemd user unit for `entry`.
pub fn systemd_unit(entry: &Entry) -> String {
    let mut exec = systemd_quote(&entry.program.to_string_lossy());
    for arg in &entry.args {
        exec.push(' ');
        exec.push_str(&systemd_quote(arg));
    }
    let mut unit = String::from(
        "[Unit]\n\
         Description=Kjarni daemon, a local AI assistant\n\
         \n\
         [Service]\n",
    );
    for (key, value) in &entry.env {
        unit.push_str(&format!(
            "Environment={}\n",
            systemd_quote(&format!("{key}={value}"))
        ));
    }
    unit.push_str(&format!(
        "ExecStart={exec}\n\
         Restart=on-failure\n\
         RestartSec=5\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n"
    ));
    unit
}

// Built on every system so the tests check it everywhere.
#[cfg_attr(not(any(target_os = "macos", windows)), allow(dead_code))]
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

// Built on every system so the tests check it everywhere.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
/// The LaunchAgent for `entry`. It starts at login and restarts after a
/// crash, but not after a clean stop.
pub fn launchd_plist(entry: &Entry) -> String {
    let string = |s: &str| format!("<string>{}</string>", xml_escape(s));
    let mut args = format!("\t\t{}\n", string(&entry.program.to_string_lossy()));
    for arg in &entry.args {
        args.push_str(&format!("\t\t{}\n", string(arg)));
    }
    let mut env = String::new();
    if !entry.env.is_empty() {
        env.push_str("\t<key>EnvironmentVariables</key>\n\t<dict>\n");
        for (key, value) in &entry.env {
            env.push_str(&format!(
                "\t\t<key>{}</key>\n\t\t{}\n",
                xml_escape(key),
                string(value)
            ));
        }
        env.push_str("\t</dict>\n");
    }
    let log = string(&entry.log.to_string_lossy());
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
         <plist version=\"1.0\">\n\
         <dict>\n\
         \t<key>Label</key>\n\
         \t<string>{LAUNCHD_LABEL}</string>\n\
         \t<key>ProgramArguments</key>\n\
         \t<array>\n\
         {args}\
         \t</array>\n\
         {env}\
         \t<key>RunAtLoad</key>\n\
         \t<true/>\n\
         \t<key>KeepAlive</key>\n\
         \t<dict>\n\
         \t\t<key>SuccessfulExit</key>\n\
         \t\t<false/>\n\
         \t</dict>\n\
         \t<key>ThrottleInterval</key>\n\
         \t<integer>10</integer>\n\
         \t<key>StandardOutPath</key>\n\
         \t{log}\n\
         \t<key>StandardErrorPath</key>\n\
         \t{log}\n\
         </dict>\n\
         </plist>\n"
    )
}

// Built on every system so the tests check it everywhere.
#[cfg_attr(not(windows), allow(dead_code))]
/// One argument quoted for a Windows command line, as `CommandLineToArgvW`
/// splits it.
fn windows_quote(arg: &str) -> String {
    if !arg.is_empty() && !arg.contains([' ', '\t', '"']) {
        return arg.to_string();
    }
    let mut out = String::from("\"");
    let mut backslashes = 0;
    for c in arg.chars() {
        match c {
            '\\' => backslashes += 1,
            '"' => {
                out.push_str(&"\\".repeat(backslashes * 2 + 1));
                out.push('"');
                backslashes = 0;
            }
            c => {
                out.push_str(&"\\".repeat(backslashes));
                out.push(c);
                backslashes = 0;
            }
        }
    }
    out.push_str(&"\\".repeat(backslashes * 2));
    out.push('"');
    out
}

// Built on every system so the tests check it everywhere.
#[cfg_attr(not(windows), allow(dead_code))]
/// The Task Scheduler task for `entry`, run at `user`'s logon as that user,
/// with no time limit, restarted after a crash, and on battery too.
pub fn task_xml(entry: &Entry, user: &str) -> String {
    let arguments: Vec<String> = entry.args.iter().map(|a| windows_quote(a)).collect();
    let user = xml_escape(user);
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-16\"?>\n\
         <Task version=\"1.2\" xmlns=\"http://schemas.microsoft.com/windows/2004/02/mit/task\">\n\
         \x20 <RegistrationInfo>\n\
         \x20   <Description>Kjarni daemon, a local AI assistant</Description>\n\
         \x20 </RegistrationInfo>\n\
         \x20 <Triggers>\n\
         \x20   <LogonTrigger>\n\
         \x20     <Enabled>true</Enabled>\n\
         \x20     <UserId>{user}</UserId>\n\
         \x20   </LogonTrigger>\n\
         \x20 </Triggers>\n\
         \x20 <Principals>\n\
         \x20   <Principal id=\"Author\">\n\
         \x20     <UserId>{user}</UserId>\n\
         \x20     <LogonType>InteractiveToken</LogonType>\n\
         \x20     <RunLevel>LeastPrivilege</RunLevel>\n\
         \x20   </Principal>\n\
         \x20 </Principals>\n\
         \x20 <Settings>\n\
         \x20   <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n\
         \x20   <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\n\
         \x20   <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n\
         \x20   <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>\n\
         \x20   <RestartOnFailure>\n\
         \x20     <Interval>PT1M</Interval>\n\
         \x20     <Count>3</Count>\n\
         \x20   </RestartOnFailure>\n\
         \x20   <Enabled>true</Enabled>\n\
         \x20 </Settings>\n\
         \x20 <Actions Context=\"Author\">\n\
         \x20   <Exec>\n\
         \x20     <Command>{}</Command>\n\
         \x20     <Arguments>{}</Arguments>\n\
         \x20   </Exec>\n\
         \x20 </Actions>\n\
         </Task>\n",
        xml_escape(&windows_quote(&entry.program.to_string_lossy())),
        xml_escape(&arguments.join(" ")),
    )
}

/// Runs a system tool, returning its output, or an error with what it said.
fn tool(program: &str, args: &[&str]) -> Result<String> {
    let out = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("running {program}"))?;
    if !out.status.success() {
        let said = String::from_utf8_lossy(&out.stderr);
        let said = if said.trim().is_empty() {
            String::from_utf8_lossy(&out.stdout).into_owned()
        } else {
            said.into_owned()
        };
        bail!("`{program} {}` failed: {}", args.join(" "), said.trim());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Where the startup entry is described, for the systems that use a file.
pub fn entry_path() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        dirs::home_dir().map(|h| {
            h.join("Library/LaunchAgents")
                .join(format!("{LAUNCHD_LABEL}.plist"))
        })
    } else if cfg!(windows) {
        None
    } else {
        dirs::config_dir().map(|c| c.join("systemd/user").join(SYSTEMD_UNIT))
    }
}

/// Where the entry is, in words, for messages.
pub fn describe() -> String {
    match entry_path() {
        Some(path) => path.display().to_string(),
        None => format!("Task Scheduler task \"{TASK_NAME}\""),
    }
}

pub fn is_installed() -> bool {
    if cfg!(windows) {
        tool("schtasks", &["/Query", "/TN", TASK_NAME]).is_ok()
    } else {
        entry_path().is_some_and(|p| p.exists())
    }
}

#[cfg(target_os = "macos")]
fn launchd_target() -> String {
    // SAFETY: getuid has no failure modes and touches no memory.
    let uid = unsafe { libc::getuid() };
    format!("gui/{uid}/{LAUNCHD_LABEL}")
}

#[cfg(target_os = "macos")]
fn launchd_domain() -> String {
    // SAFETY: as in launchd_target.
    let uid = unsafe { libc::getuid() };
    format!("gui/{uid}")
}

#[cfg_attr(windows, allow(dead_code))]
fn write_entry_file(text: &str) -> Result<PathBuf> {
    let path = entry_path().ok_or_else(|| anyhow!("no home folder to install into"))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Writes the startup entry, replacing any earlier one, and starts the daemon
/// (restarting it if it was running).
pub fn install(entry: &Entry) -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        // A loaded agent keeps its old settings until unloaded.
        let _ = tool("launchctl", &["bootout", &launchd_target()]);
        let path = write_entry_file(&launchd_plist(entry))?;
        tool(
            "launchctl",
            &["bootstrap", &launchd_domain(), &path.to_string_lossy()],
        )?;
        Ok(())
    }
    #[cfg(windows)]
    {
        let user = match (std::env::var("USERDOMAIN"), std::env::var("USERNAME")) {
            (Ok(domain), Ok(name)) => format!("{domain}\\{name}"),
            (_, Ok(name)) => name,
            _ => bail!("USERNAME is not set; cannot tell whose logon starts the daemon"),
        };
        // schtasks reads the task's XML from a file, in UTF-16.
        let path = entry
            .log
            .parent()
            .map(|d| d.join("kjarni-task.xml"))
            .ok_or_else(|| anyhow!("the log file has no folder"))?;
        let mut bytes = vec![0xFF, 0xFE];
        for unit in task_xml(entry, &user).encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
        let created = tool(
            "schtasks",
            &[
                "/Create",
                "/TN",
                TASK_NAME,
                "/XML",
                &path.to_string_lossy(),
                "/F",
            ],
        );
        let _ = std::fs::remove_file(&path);
        created?;
        let _ = tool("schtasks", &["/End", "/TN", TASK_NAME]);
        tool("schtasks", &["/Run", "/TN", TASK_NAME])?;
        Ok(())
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        write_entry_file(&systemd_unit(entry))?;
        tool("systemctl", &["--user", "daemon-reload"])?;
        tool("systemctl", &["--user", "enable", SYSTEMD_UNIT])?;
        tool("systemctl", &["--user", "restart", SYSTEMD_UNIT])?;
        Ok(())
    }
}

/// Stops the daemon and removes the startup entry. False if there was none.
pub fn uninstall() -> Result<bool> {
    if !is_installed() {
        return Ok(false);
    }
    #[cfg(target_os = "macos")]
    {
        let _ = tool("launchctl", &["bootout", &launchd_target()]);
    }
    #[cfg(windows)]
    {
        let _ = tool("schtasks", &["/End", "/TN", TASK_NAME]);
        tool("schtasks", &["/Delete", "/TN", TASK_NAME, "/F"])?;
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        let _ = tool("systemctl", &["--user", "disable", "--now", SYSTEMD_UNIT]);
    }
    if let Some(path) = entry_path() {
        std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        let _ = tool("systemctl", &["--user", "daemon-reload"]);
    }
    Ok(true)
}

fn not_installed() -> anyhow::Error {
    anyhow!(
        "the daemon is not installed; run `kjarni daemon install`, or `kjarni daemon run` to run it in this terminal"
    )
}

pub fn start() -> Result<()> {
    if !is_installed() {
        return Err(not_installed());
    }
    #[cfg(target_os = "macos")]
    {
        tool("launchctl", &["kickstart", &launchd_target()])?;
    }
    #[cfg(windows)]
    {
        tool("schtasks", &["/Run", "/TN", TASK_NAME])?;
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        tool("systemctl", &["--user", "start", SYSTEMD_UNIT])?;
    }
    Ok(())
}

pub fn stop() -> Result<()> {
    if !is_installed() {
        return Err(not_installed());
    }
    #[cfg(target_os = "macos")]
    {
        // A clean exit on SIGTERM, which KeepAlive does not restart.
        tool("launchctl", &["kill", "SIGTERM", &launchd_target()])?;
    }
    #[cfg(windows)]
    {
        tool("schtasks", &["/End", "/TN", TASK_NAME])?;
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        tool("systemctl", &["--user", "stop", SYSTEMD_UNIT])?;
    }
    Ok(())
}

/// Linux: whether the user's services keep running after they log out.
/// `None` where that does not apply or cannot be told.
pub fn linger() -> Option<bool> {
    if cfg!(any(target_os = "macos", windows)) {
        return None;
    }
    let user = std::env::var("USER").ok()?;
    let out = tool("loginctl", &["show-user", &user, "--property=Linger"]).ok()?;
    Some(out.trim() == "Linger=yes")
}

/// Linux: keeps the user's services running after logout, or stops doing so.
pub fn set_linger(on: bool) -> Result<()> {
    tool(
        "loginctl",
        &[if on {
            "enable-linger"
        } else {
            "disable-linger"
        }],
    )
    .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry() -> Entry {
        Entry {
            program: PathBuf::from("/opt/my apps/kjarni"),
            args: vec![
                "daemon".into(),
                "run".into(),
                "-C".into(),
                "/home/ada/50% \"done\"".into(),
            ],
            env: vec![("KJARNI_DAEMON_DIR".into(), "/tmp/d$x".into())],
            log: PathBuf::from("/home/ada/.local/share/kjarni/daemon/logs/daemon.log"),
        }
    }

    #[test]
    fn the_systemd_unit_quotes_every_argument() {
        let unit = systemd_unit(&entry());
        assert!(unit.contains(
            "ExecStart=\"/opt/my apps/kjarni\" \"daemon\" \"run\" \"-C\" \"/home/ada/50%% \\\"done\\\"\"\n"
        ));
        assert!(unit.contains("Environment=\"KJARNI_DAEMON_DIR=/tmp/d$$x\"\n"));
        assert!(unit.contains("Restart=on-failure\n"));
        assert!(unit.contains("WantedBy=default.target\n"));
    }

    #[test]
    fn the_launch_agent_escapes_xml_and_restarts_only_after_a_crash() {
        let plist = launchd_plist(&entry());
        assert!(plist.contains("<string>/opt/my apps/kjarni</string>"));
        assert!(plist.contains("<string>/home/ada/50% &quot;done&quot;</string>"));
        assert!(plist.contains("<key>KJARNI_DAEMON_DIR</key>\n\t\t<string>/tmp/d$x</string>"));
        assert!(plist.contains("<key>SuccessfulExit</key>\n\t\t<false/>"));
        assert!(plist.contains("<key>RunAtLoad</key>\n\t<true/>"));
    }

    #[test]
    fn windows_arguments_survive_command_line_splitting() {
        assert_eq!(windows_quote("run"), "run");
        assert_eq!(windows_quote(""), "\"\"");
        assert_eq!(
            windows_quote(r"C:\Users\Ada Lovelace\kjarni"),
            r#""C:\Users\Ada Lovelace\kjarni""#
        );
        // Backslashes before a quote, or before the closing quote, double.
        assert_eq!(windows_quote(r#"a "b""#), r#""a \"b\"""#);
        assert_eq!(windows_quote(r"C:\my dir\"), r#""C:\my dir\\""#);
    }

    #[test]
    fn the_task_runs_at_the_users_logon_without_a_time_limit() {
        let mut e = entry();
        e.program = PathBuf::from(r"C:\Program Files\Kjarni\kjarni.exe");
        let xml = task_xml(&e, r"DESK\ada");
        assert!(xml.contains(r"<UserId>DESK\ada</UserId>"));
        assert!(xml.contains("<LogonType>InteractiveToken</LogonType>"));
        // The default limit of three days would stop the daemon.
        assert!(xml.contains("<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>"));
        assert!(xml.contains(r"<Command>&quot;C:\Program Files\Kjarni\kjarni.exe&quot;</Command>"));
        assert!(xml.contains(
            r"<Arguments>daemon run -C &quot;/home/ada/50% \&quot;done\&quot;&quot;</Arguments>"
        ));
    }
}
