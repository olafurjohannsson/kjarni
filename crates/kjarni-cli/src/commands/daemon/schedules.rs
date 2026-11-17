//! Schedules: saved prompts the daemon runs by itself at set times, each in
//! a new conversation. Nobody is there to approve a call, so a schedule lists
//! the tools it may use, and any other call that needs approval is declined.
//! When a run ends, a notification can go to the user's phone through an
//! ntfy server the user chose.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Datelike, Local, NaiveTime, TimeZone, Utc, Weekday};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::broadcast::error::RecvError;

use super::api::DaemonState;
use super::conversations::{Answers, EventKind, Live};

/// The tools that ask before they run, so the ones a schedule lists to let
/// them run unattended. The rest only read, and always run.
pub const ASKING_TOOLS: &[&str] = &[
    "fetch_url",
    "write_file",
    "edit_file",
    "run_command",
    "remember",
    "forget",
];

const MIN_EVERY_MINUTES: u32 = 5;
const MAX_EVERY_MINUTES: u32 = 7 * 24 * 60;
/// How often the daemon looks for schedules that are due.
const TICK: Duration = Duration::from_secs(20);
/// ntfy takes messages up to 4096 bytes.
const MAX_NOTIFICATION_BYTES: usize = 3800;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Day {
    Mon,
    Tue,
    Wed,
    Thu,
    Fri,
    Sat,
    Sun,
}

impl Day {
    fn of(day: Weekday) -> Day {
        match day {
            Weekday::Mon => Day::Mon,
            Weekday::Tue => Day::Tue,
            Weekday::Wed => Day::Wed,
            Weekday::Thu => Day::Thu,
            Weekday::Fri => Day::Fri,
            Weekday::Sat => Day::Sat,
            Weekday::Sun => Day::Sun,
        }
    }
}

/// When a schedule runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum When {
    /// At `time` (HH:MM, the computer's local time) every day, or only on
    /// `days` when it lists any.
    Daily {
        time: String,
        #[serde(default)]
        days: Vec<Day>,
    },
    /// Every `minutes` minutes.
    Every { minutes: u32 },
}

fn parse_time(text: &str) -> Result<NaiveTime> {
    NaiveTime::parse_from_str(text.trim(), "%H:%M")
        .map_err(|_| anyhow!("'{text}' is not a time like 07:30"))
}

impl When {
    fn check(&self) -> Result<()> {
        match self {
            When::Daily { time, .. } => {
                parse_time(time)?;
            }
            When::Every { minutes } => {
                if !(MIN_EVERY_MINUTES..=MAX_EVERY_MINUTES).contains(minutes) {
                    bail!(
                        "a schedule can run at most every {MIN_EVERY_MINUTES} minutes and must run at least once a week"
                    );
                }
            }
        }
        Ok(())
    }

    /// The first time after `after` that this runs.
    pub fn next_after<Tz: TimeZone>(&self, after: &DateTime<Tz>) -> DateTime<Tz> {
        match self {
            When::Every { minutes } => {
                after.clone() + chrono::Duration::minutes(i64::from(*minutes))
            }
            When::Daily { time, days } => {
                let at = parse_time(time).unwrap_or(NaiveTime::MIN);
                let zone = after.timezone();
                let first = after.date_naive();
                for offset in 0..=7 {
                    let date = first + chrono::Days::new(offset);
                    if !days.is_empty() && !days.contains(&Day::of(date.weekday())) {
                        continue;
                    }
                    let local = date.and_time(at);
                    // A time the clock skips when it moves forward runs an
                    // hour later.
                    let Some(time) = zone.from_local_datetime(&local).earliest().or_else(|| {
                        zone.from_local_datetime(&(local + chrono::Duration::hours(1)))
                            .earliest()
                    }) else {
                        continue;
                    };
                    if time > *after {
                        return time;
                    }
                }
                after.clone() + chrono::Duration::days(1)
            }
        }
    }
}

/// What the user decides about a schedule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spec {
    pub name: String,
    /// The message each run starts with.
    pub prompt: String,
    pub when: When,
    /// Tools from [`ASKING_TOOLS`] that may run without asking.
    #[serde(default)]
    pub allow: Vec<String>,
    /// Send a notification when a run ends.
    #[serde(default = "yes")]
    pub notify: bool,
    #[serde(default = "yes")]
    pub enabled: bool,
}

fn yes() -> bool {
    true
}

impl Spec {
    pub fn check(&self) -> Result<()> {
        let name = self.name.trim();
        if name.is_empty() || name.chars().count() > 80 {
            bail!("a schedule needs a name of at most 80 characters");
        }
        if self.prompt.trim().is_empty() {
            bail!("a schedule needs a message to run");
        }
        if self.prompt.chars().count() > 8000 {
            bail!("a schedule's message can be at most 8000 characters");
        }
        self.when.check()?;
        for tool in &self.allow {
            if !ASKING_TOOLS.contains(&tool.as_str()) {
                bail!(
                    "'{tool}' is not a tool a schedule can allow; these are {}",
                    ASKING_TOOLS.join(", ")
                );
            }
        }
        Ok(())
    }
}

/// A saved schedule and what happened when it last ran.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Schedule {
    pub id: String,
    #[serde(flatten)]
    pub spec: Spec,
    /// When it was made, in unix seconds. The first run is the first time
    /// after this.
    pub created: i64,
    #[serde(default)]
    pub last_run: Option<i64>,
    #[serde(default)]
    pub last_conversation: Option<String>,
    /// `running`, `done`, `error` or `stopped`.
    #[serde(default)]
    pub last_result: Option<String>,
}

impl Schedule {
    /// When it runs next. A run missed while the computer was off is due at
    /// once, and only once, however many were missed.
    pub fn next_run<Tz: TimeZone>(&self, zone: &Tz) -> DateTime<Tz> {
        let from = self.last_run.unwrap_or(self.created);
        let from = DateTime::<Utc>::from_timestamp(from, 0)
            .unwrap_or_default()
            .with_timezone(zone);
        self.spec.when.next_after(&from)
    }

    pub fn due<Tz: TimeZone>(&self, now: &DateTime<Tz>) -> bool {
        self.spec.enabled && self.next_run(&now.timezone()) <= *now
    }
}

/// Where notifications go.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Notify {
    /// The ntfy topic's full address, such as `https://ntfy.sh/some-topic`.
    pub url: String,
    /// An access token, for a server or topic that needs one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// The address the user opens the daemon at, so a notification links
    /// to the run's conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_url: Option<String>,
}

/// The server to publish to and the topic, from a topic's full address.
pub fn split_topic(url: &str) -> Result<(String, String)> {
    let parsed = reqwest::Url::parse(url.trim())
        .map_err(|_| anyhow!("'{url}' is not an address like https://ntfy.sh/your-topic"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        bail!("'{url}' is not an http or https address");
    }
    let mut segments: Vec<&str> = parsed
        .path_segments()
        .map(|s| s.filter(|p| !p.is_empty()).collect())
        .unwrap_or_default();
    let Some(topic) = segments.pop() else {
        bail!("'{url}' names no topic; add one, as in https://ntfy.sh/your-topic");
    };
    let valid = topic.len() <= 64
        && topic
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !valid {
        bail!("'{topic}' is not an ntfy topic: use letters, digits, - and _");
    }
    let mut server = parsed.clone();
    server.set_query(None);
    server.set_fragment(None);
    let base: String = segments.iter().map(|s| format!("{s}/")).collect();
    server.set_path(&format!("/{base}"));
    Ok((server.to_string(), topic.to_string()))
}

/// Whether `url` is an http or https address to open the daemon at.
pub fn check_open_url(url: &str) -> Result<()> {
    match reqwest::Url::parse(url) {
        Ok(u) if matches!(u.scheme(), "http" | "https") && u.host_str().is_some() => Ok(()),
        _ => bail!("'{url}' is not an http or https address"),
    }
}

/// `text`, cut to fit in a notification.
fn clip(text: &str) -> String {
    if text.len() <= MAX_NOTIFICATION_BYTES {
        return text.to_string();
    }
    let mut end = MAX_NOTIFICATION_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", text[..end].trim_end())
}

/// Publishes a notification. `conversation`, with an `open_url`, makes
/// tapping it open that conversation.
pub async fn send_notification(
    notify: &Notify,
    title: &str,
    message: &str,
    conversation: Option<&str>,
) -> Result<()> {
    let (server, topic) = split_topic(&notify.url)?;
    let message = if message.trim().is_empty() {
        "Finished."
    } else {
        message.trim()
    };
    let mut body = json!({
        "topic": topic,
        "title": title,
        "message": clip(message),
        "markdown": true,
    });
    if let (Some(base), Some(id)) = (&notify.open_url, conversation) {
        body["click"] = json!(format!("{}/c/{id}", base.trim_end_matches('/')));
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;
    let mut request = client
        .post(&server)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.to_string());
    if let Some(token) = &notify.token {
        request = request.bearer_auth(token);
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("could not reach the ntfy server at {server}"))?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        let text: String = text.chars().take(200).collect();
        bail!("the ntfy server answered {status}: {}", text.trim());
    }
    Ok(())
}

#[derive(Default, Serialize, Deserialize)]
struct Saved {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    notify: Option<Notify>,
    #[serde(default)]
    schedules: Vec<Schedule>,
}

/// The schedules and notification settings, kept in one JSON file.
pub struct Store {
    path: PathBuf,
    saved: Mutex<Saved>,
    /// Schedules with a run in progress.
    running: Mutex<HashSet<String>>,
}

fn new_id() -> String {
    let mut bytes = [0u8; 6];
    getrandom::fill(&mut bytes).expect("the system's random source");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

impl Store {
    /// Reads `path`, or starts empty when it does not exist yet.
    pub fn open(path: PathBuf) -> Result<Self> {
        let saved = match std::fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text)
                .with_context(|| format!("reading the schedules in {}", path.display()))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Saved::default(),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        Ok(Self {
            path,
            saved: Mutex::new(saved),
            running: Mutex::new(HashSet::new()),
        })
    }

    /// Writes the file whole, through a temporary file so a crash leaves
    /// the old one. Only the user may read it: it can hold an access token.
    fn write(&self, saved: &Saved) -> Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let temp = self.path.with_extension("json.tmp");
        std::fs::write(&temp, serde_json::to_string_pretty(saved)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&temp, std::fs::Permissions::from_mode(0o600))?;
        }
        std::fs::rename(&temp, &self.path)
            .with_context(|| format!("saving {}", self.path.display()))
    }

    /// Changes the saved state with `change` and writes it.
    fn change<T>(&self, change: impl FnOnce(&mut Saved) -> T) -> Result<T> {
        let mut saved = self.saved.lock().expect("schedules lock");
        let out = change(&mut saved);
        self.write(&saved)?;
        Ok(out)
    }

    pub fn list(&self) -> Vec<Schedule> {
        self.saved.lock().expect("schedules lock").schedules.clone()
    }

    pub fn get(&self, id: &str) -> Option<Schedule> {
        self.list().into_iter().find(|s| s.id == id)
    }

    pub fn create(&self, spec: Spec, now: i64) -> Result<Schedule> {
        spec.check()?;
        let schedule = Schedule {
            id: new_id(),
            spec,
            created: now,
            last_run: None,
            last_conversation: None,
            last_result: None,
        };
        self.change(|s| s.schedules.push(schedule.clone()))?;
        Ok(schedule)
    }

    /// Replaces schedule `id`'s settings. `None` when there is no such
    /// schedule.
    pub fn update(&self, id: &str, spec: Spec) -> Result<Option<Schedule>> {
        spec.check()?;
        self.change(|s| {
            let found = s.schedules.iter_mut().find(|x| x.id == id)?;
            found.spec = spec;
            Some(found.clone())
        })
    }

    pub fn delete(&self, id: &str) -> Result<bool> {
        self.change(|s| {
            let before = s.schedules.len();
            s.schedules.retain(|x| x.id != id);
            s.schedules.len() != before
        })
    }

    fn record(&self, id: &str, change: impl FnOnce(&mut Schedule)) -> Result<()> {
        self.change(|s| {
            if let Some(found) = s.schedules.iter_mut().find(|x| x.id == id) {
                change(found);
            }
        })
    }

    pub fn notify(&self) -> Option<Notify> {
        self.saved.lock().expect("schedules lock").notify.clone()
    }

    pub fn set_notify(&self, notify: Option<Notify>) -> Result<()> {
        self.change(|s| s.notify = notify)
    }

    pub fn is_running(&self, id: &str) -> bool {
        self.running.lock().expect("running lock").contains(id)
    }

    fn any_running(&self) -> bool {
        !self.running.lock().expect("running lock").is_empty()
    }

    /// The schedule that has waited longest to run, if any is due.
    pub fn next_due<Tz: TimeZone>(&self, now: &DateTime<Tz>) -> Option<Schedule> {
        self.list()
            .into_iter()
            .filter(|s| s.due(now) && !self.is_running(&s.id))
            .min_by_key(|s| s.next_run(&now.timezone()))
    }
}

/// Why a schedule could not start.
#[derive(Debug)]
pub enum StartError {
    NotFound,
    Running,
    Failed(anyhow::Error),
}

/// Starts schedule `id` now, in a new conversation, and returns that
/// conversation's id. The run goes on in the background.
pub fn start(state: &Arc<DaemonState>, id: &str) -> Result<String, StartError> {
    let store = &state.schedules;
    let schedule = store.get(id).ok_or(StartError::NotFound)?;
    if !store
        .running
        .lock()
        .expect("running lock")
        .insert(id.to_string())
    {
        return Err(StartError::Running);
    }
    let now = Utc::now().timestamp();
    let started = (|| -> Result<Arc<Live>> {
        let live = state.conversations.create()?;
        let allowed: HashSet<String> = schedule.spec.allow.iter().cloned().collect();
        live.send_with(&schedule.spec.prompt, Answers::Allowed(Arc::new(allowed)))
            .map_err(|e| anyhow!("could not start the run: {e:?}"))?;
        Ok(live)
    })();
    let live = match started {
        Ok(live) => live,
        Err(e) => {
            store.running.lock().expect("running lock").remove(id);
            // Recorded as a run, so a failing schedule waits for its next
            // time instead of retrying at every tick.
            let _ = store.record(id, |s| {
                s.last_run = Some(now);
                s.last_result = Some("error".into());
            });
            return Err(StartError::Failed(e));
        }
    };
    let conversation = live.id.clone();
    if let Err(e) = store.record(id, |s| {
        s.last_run = Some(now);
        s.last_conversation = Some(conversation.clone());
        s.last_result = Some("running".into());
    }) {
        eprintln!(
            "Could not save that schedule '{}' ran: {e:#}",
            schedule.spec.name
        );
    }
    let state = state.clone();
    let id = id.to_string();
    tokio::spawn(async move { finish(state, id, schedule, live).await });
    Ok(conversation)
}

/// Waits for a run to end, records how, and sends its notification.
async fn finish(state: Arc<DaemonState>, id: String, schedule: Schedule, live: Arc<Live>) {
    let end = wait_for_end(&live).await;
    let (result, message) = match end {
        EventKind::Done { text, .. } => ("done", text),
        EventKind::Stopped => ("stopped", "The run was stopped.".to_string()),
        EventKind::Error { message } => ("error", format!("The run failed: {message}")),
        _ => ("error", "The run ended without an answer.".to_string()),
    };
    let store = &state.schedules;
    if let Err(e) = store.record(&id, |s| s.last_result = Some(result.into())) {
        eprintln!(
            "Could not save how schedule '{}' ended: {e:#}",
            schedule.spec.name
        );
    }
    store.running.lock().expect("running lock").remove(&id);
    if !schedule.spec.notify {
        return;
    }
    if let Some(notify) = store.notify()
        && let Err(e) =
            send_notification(&notify, &schedule.spec.name, &message, Some(&live.id)).await
    {
        eprintln!(
            "Could not send the notification for schedule '{}': {e:#}",
            schedule.spec.name
        );
    }
}

/// The event that ended the conversation's current turn.
async fn wait_for_end(live: &Live) -> EventKind {
    loop {
        let (past, mut rx) = live.follow(0);
        if let Some(end) = past.into_iter().find(|e| e.kind.ends_turn()) {
            return end.kind;
        }
        loop {
            match rx.recv().await {
                Ok(e) if e.kind.ends_turn() => return e.kind,
                Ok(_) => {}
                // Missed some; the turn's log has them.
                Err(RecvError::Lagged(_)) => break,
                Err(RecvError::Closed) => {
                    return EventKind::Error {
                        message: "the conversation closed".into(),
                    };
                }
            }
        }
    }
}

/// Runs schedules as they come due, one at a time: they share the model
/// with everything else the daemon does.
pub async fn run(state: Arc<DaemonState>) {
    let mut tick = tokio::time::interval(TICK);
    loop {
        tick.tick().await;
        if state.schedules.any_running() {
            continue;
        }
        let Some(due) = state.schedules.next_due(&Local::now()) else {
            continue;
        };
        match start(&state, &due.id) {
            Ok(conversation) => eprintln!(
                "Schedule '{}' is running, in conversation {conversation}.",
                due.spec.name
            ),
            Err(StartError::Failed(e)) => {
                eprintln!("Schedule '{}' could not run: {e:#}", due.spec.name)
            }
            Err(StartError::NotFound | StartError::Running) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;

    fn at(text: &str) -> DateTime<FixedOffset> {
        DateTime::parse_from_rfc3339(text).unwrap()
    }

    fn daily(time: &str, days: &[Day]) -> When {
        When::Daily {
            time: time.into(),
            days: days.to_vec(),
        }
    }

    #[test]
    fn daily_runs_at_the_next_matching_time() {
        // 2026-11-17 is a Tuesday.
        let tuesday_morning = at("2026-11-17T06:00:00+00:00");
        let when = daily("07:30", &[]);
        assert_eq!(
            when.next_after(&tuesday_morning),
            at("2026-11-17T07:30:00+00:00")
        );
        // Exactly at the time is not after it.
        assert_eq!(
            when.next_after(&at("2026-11-17T07:30:00+00:00")),
            at("2026-11-18T07:30:00+00:00")
        );
        let weekends = daily("09:00", &[Day::Sat, Day::Sun]);
        assert_eq!(
            weekends.next_after(&tuesday_morning),
            at("2026-11-21T09:00:00+00:00")
        );
        // The time is the zone's own, not UTC.
        let new_york = at("2026-11-17T06:00:00-05:00");
        assert_eq!(when.next_after(&new_york), at("2026-11-17T07:30:00-05:00"));
        let every = When::Every { minutes: 90 };
        assert_eq!(
            every.next_after(&tuesday_morning),
            at("2026-11-17T07:30:00+00:00")
        );
    }

    fn schedule(when: When, created: &str, last_run: Option<&str>) -> Schedule {
        Schedule {
            id: "a".into(),
            spec: Spec {
                name: "Digest".into(),
                prompt: "Summarize".into(),
                when,
                allow: vec![],
                notify: true,
                enabled: true,
            },
            created: at(created).timestamp(),
            last_run: last_run.map(|t| at(t).timestamp()),
            last_conversation: None,
            last_result: None,
        }
    }

    #[test]
    fn a_missed_run_is_due_once_and_a_new_schedule_waits_for_its_time() {
        let s = schedule(daily("07:00", &[]), "2026-11-17T08:00:00+00:00", None);
        // Made after today's 07:00: the first run is tomorrow's.
        assert!(!s.due(&at("2026-11-17T20:00:00+00:00")));
        assert!(s.due(&at("2026-11-18T07:00:00+00:00")));

        // Ran on Monday, computer off until Thursday noon: due once...
        let s = schedule(
            daily("07:00", &[]),
            "2026-11-01T00:00:00+00:00",
            Some("2026-11-16T07:00:05+00:00"),
        );
        assert!(s.due(&at("2026-11-19T12:00:00+00:00")));
        // ...and after that run, not again until Friday.
        let s = schedule(
            daily("07:00", &[]),
            "2026-11-01T00:00:00+00:00",
            Some("2026-11-19T12:00:00+00:00"),
        );
        assert!(!s.due(&at("2026-11-19T23:00:00+00:00")));
        assert!(s.due(&at("2026-11-20T07:00:00+00:00")));

        let mut off = s.clone();
        off.spec.enabled = false;
        assert!(!off.due(&at("2026-11-25T07:00:00+00:00")));
    }

    #[test]
    fn specs_are_checked() {
        let mut spec = schedule(daily("07:00", &[]), "2026-11-17T00:00:00+00:00", None).spec;
        assert!(spec.check().is_ok());
        for bad_time in ["7", "25:00", "07:60", "seven"] {
            spec.when = daily(bad_time, &[]);
            assert!(spec.check().is_err(), "{bad_time}");
        }
        spec.when = When::Every { minutes: 1 };
        assert!(spec.check().is_err());
        spec.when = When::Every { minutes: 60 };
        spec.allow = vec!["read_file".into()];
        let e = spec.check().unwrap_err().to_string();
        assert!(e.contains("fetch_url"), "{e}");
        spec.allow = vec!["fetch_url".into()];
        spec.name = "  ".into();
        assert!(spec.check().is_err());
    }

    #[test]
    fn ntfy_addresses_split_into_server_and_topic() {
        assert_eq!(
            split_topic("https://ntfy.sh/kjarni-a1b2").unwrap(),
            ("https://ntfy.sh/".into(), "kjarni-a1b2".into())
        );
        assert_eq!(
            split_topic(" http://box.lan:8080/ntfy/alerts/?x=1 ").unwrap(),
            ("http://box.lan:8080/ntfy/".into(), "alerts".into())
        );
        for bad in [
            "ntfy.sh/topic",
            "https://ntfy.sh/",
            "ftp://ntfy.sh/topic",
            "https://ntfy.sh/a b",
            "https://ntfy.sh/a.b",
        ] {
            assert!(split_topic(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn long_notifications_are_cut_on_a_character_boundary() {
        let text = "þ".repeat(MAX_NOTIFICATION_BYTES);
        let cut = clip(&text);
        assert!(cut.len() <= MAX_NOTIFICATION_BYTES + "…".len());
        assert!(cut.ends_with('…'));
        assert_eq!(clip("short"), "short");
    }

    #[test]
    fn the_store_keeps_schedules_and_settings_across_restarts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("schedules.json");
        let store = Store::open(path.clone()).unwrap();
        let spec = schedule(
            daily("07:00", &[Day::Mon]),
            "2026-11-17T00:00:00+00:00",
            None,
        )
        .spec;
        let made = store.create(spec.clone(), 100).unwrap();
        store
            .set_notify(Some(Notify {
                url: "https://ntfy.sh/t".into(),
                token: Some("secret".into()),
                open_url: None,
            }))
            .unwrap();

        let reopened = Store::open(path.clone()).unwrap();
        assert_eq!(reopened.get(&made.id).unwrap().spec, spec);
        assert_eq!(reopened.notify().unwrap().token.as_deref(), Some("secret"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        assert!(reopened.delete(&made.id).unwrap());
        assert!(!reopened.delete(&made.id).unwrap());
        assert!(Store::open(path).unwrap().list().is_empty());
    }
}
