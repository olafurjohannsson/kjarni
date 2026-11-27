//! The chat model the daemon runs, which the user can change while it runs.
//!
//! A change waits until no conversation is answering and holds new messages
//! back meanwhile. It downloads the new model if needed, unloads the old one
//! so both are never in memory at once, and loads the new one. When the load
//! fails the old model is loaded again. The choice is saved, so the next
//! start loads it too.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use kjarni::Conversation;
use kjarni::agent::{AgentModel, TextStream};
use serde::{Deserialize, Serialize};

use super::conversations::Conversations;

/// The models the agent can run: those whose chat template has a tool-call
/// format, smallest first.
pub const CHOICES: [&str; 6] = [
    "qwen3.5-0.8b",
    "qwen3.5-2b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
];

/// What the daemon remembers between starts, in `settings.json` in its
/// folder.
#[derive(Serialize, Deserialize, Debug, Default, PartialEq)]
pub struct Settings {
    pub model: Option<String>,
    /// Whether the model reasons before answering.
    #[serde(default)]
    pub thinking: bool,
}

impl Settings {
    /// The saved settings; the defaults when there are none or they cannot be
    /// read.
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

pub fn settings_path(daemon_dir: &Path) -> PathBuf {
    daemon_dir.join("settings.json")
}

/// The model to load at start: the one given with `--model`, else the saved
/// one, else the default.
pub fn starting_model(given: Option<&str>, saved: &Settings) -> String {
    given
        .map(str::to_string)
        .or_else(|| saved.model.clone())
        .unwrap_or_else(|| kjarni_cli::DEFAULT_AGENT_MODEL.to_string())
}

/// Downloads, frees and loads models for a change. The daemon's runs real
/// models; the tests' scripted ones.
#[async_trait]
pub trait Loader: Send + Sync {
    /// Whether `name` is downloaded in the format the loader loads.
    fn downloaded(&self, name: &str) -> bool;
    /// Bytes of `name` on disk so far, to show a download's progress.
    fn bytes_on_disk(&self, name: &str) -> u64;
    async fn download(&self, name: &str) -> Result<()>;
    /// Waits for the generations still running on `name` to end, and lets
    /// go of it.
    async fn unload(&self, name: &str);
    async fn load(&self, name: &str) -> Result<Arc<dyn AgentModel>>;
}

/// The agent's model, in a slot a change swaps. Empty while a change has
/// unloaded the old model and not yet loaded the new one; messages are held
/// back then, so nothing asks it for a reply.
#[derive(Default)]
pub struct Slot(RwLock<Option<Arc<dyn AgentModel>>>);

impl Slot {
    pub fn new(model: Arc<dyn AgentModel>) -> Self {
        Self(RwLock::new(Some(model)))
    }

    fn get(&self) -> Result<Arc<dyn AgentModel>> {
        self.0
            .read()
            .expect("model slot lock")
            .clone()
            .ok_or_else(|| anyhow!("no model is loaded; Kjarni is switching models"))
    }

    fn set(&self, model: Option<Arc<dyn AgentModel>>) {
        *self.0.write().expect("model slot lock") = model;
    }
}

#[async_trait]
impl AgentModel for Slot {
    fn render(&self, conversation: &Conversation) -> Result<String> {
        self.get()?.render(conversation)
    }

    async fn stream(&self, prompt: String) -> Result<TextStream> {
        self.get()?.stream(prompt).await
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        self.get()?.count_tokens(text)
    }

    fn context_size(&self) -> usize {
        self.get().map_or(0, |m| m.context_size())
    }

    fn name(&self) -> String {
        self.get().map_or_else(|_| String::new(), |m| m.name())
    }
}

/// Where a change is.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Phase {
    Idle,
    Downloading {
        model: String,
    },
    Loading {
        model: String,
    },
    /// The last change failed; `model` is the one that would not load.
    Failed {
        model: String,
        message: String,
    },
}

/// A change could not start.
#[derive(Debug, PartialEq)]
pub enum SwitchError {
    Unknown,
    Same,
    Switching,
    Busy,
}

pub struct Models {
    slot: Arc<Slot>,
    loader: Arc<dyn Loader>,
    /// The model loaded, or last loaded when a change left none.
    current: Mutex<String>,
    phase: Mutex<Phase>,
    settings: PathBuf,
    /// The daemon was started with `--model` or `--model-path`, which the
    /// next start loads again whatever is chosen here.
    pub pinned: Option<String>,
    /// Whether the model reasons before answering, which conversations read
    /// at each turn.
    thinking: Arc<AtomicBool>,
}

impl Models {
    pub fn new(
        slot: Arc<Slot>,
        loader: Arc<dyn Loader>,
        current: String,
        settings: PathBuf,
        pinned: Option<String>,
        thinking: Arc<AtomicBool>,
    ) -> Self {
        Self {
            slot,
            loader,
            current: Mutex::new(current),
            phase: Mutex::new(Phase::Idle),
            settings,
            pinned,
            thinking,
        }
    }

    pub fn thinking(&self) -> bool {
        self.thinking.load(Ordering::Relaxed)
    }

    /// Turns reasoning before answering on or off from the next turn, and
    /// saves the choice.
    pub fn set_thinking(&self, on: bool) -> Result<()> {
        self.thinking.store(on, Ordering::Relaxed);
        let mut saved = Settings::load(&self.settings);
        saved.thinking = on;
        saved.save(&self.settings)
    }

    pub fn current(&self) -> String {
        self.current.lock().expect("current model lock").clone()
    }

    pub fn phase(&self) -> Phase {
        self.phase.lock().expect("phase lock").clone()
    }

    /// The models the user can pick, with whether each is downloaded.
    pub fn choices(&self) -> Vec<(&'static str, bool)> {
        CHOICES
            .iter()
            .map(|name| (*name, self.loader.downloaded(name)))
            .collect()
    }

    /// Bytes of the model being downloaded so far, during a download.
    pub fn download_progress(&self) -> Option<u64> {
        match self.phase() {
            Phase::Downloading { model } => Some(self.loader.bytes_on_disk(&model)),
            _ => None,
        }
    }

    /// Starts changing to `name`. The change goes on in the background;
    /// [`phase`](Self::phase) follows it.
    pub fn switch(
        self: &Arc<Self>,
        name: &str,
        conversations: &Arc<Conversations>,
    ) -> Result<(), SwitchError> {
        let name = CHOICES
            .iter()
            .find(|c| **c == name)
            .ok_or(SwitchError::Unknown)?
            .to_string();
        {
            let mut phase = self.phase.lock().expect("phase lock");
            if matches!(*phase, Phase::Downloading { .. } | Phase::Loading { .. }) {
                return Err(SwitchError::Switching);
            }
            if name == self.current() && !matches!(*phase, Phase::Failed { .. }) {
                return Err(SwitchError::Same);
            }
            *phase = Phase::Downloading {
                model: name.clone(),
            };
        }
        // Downloading needs no pause, so the hold starts at the load; checked
        // now too, so a busy daemon is told at once.
        if conversations.any_busy() {
            *self.phase.lock().expect("phase lock") = Phase::Idle;
            return Err(SwitchError::Busy);
        }
        let models = self.clone();
        let conversations = conversations.clone();
        tokio::spawn(async move { models.run_switch(name, conversations).await });
        Ok(())
    }

    async fn run_switch(&self, name: String, conversations: Arc<Conversations>) {
        let set = |p: Phase| *self.phase.lock().expect("phase lock") = p;
        let fail = |message: String| {
            eprintln!("Could not switch to {name}: {message}");
            set(Phase::Failed {
                model: name.clone(),
                message,
            });
        };
        if !self.loader.downloaded(&name) {
            eprintln!("Downloading {name}...");
            if let Err(e) = self.loader.download(&name).await {
                return fail(format!("{e:#}"));
            }
        }
        // Wait for running turns to end, then hold new ones back.
        let hold = loop {
            match conversations.hold() {
                Some(hold) => break hold,
                None => tokio::time::sleep(std::time::Duration::from_millis(250)).await,
            }
        };
        set(Phase::Loading {
            model: name.clone(),
        });
        let old = self.current();
        eprintln!("Switching from {old} to {name}...");
        self.slot.set(None);
        self.loader.unload(&old).await;
        match self.loader.load(&name).await {
            Ok(model) => {
                self.slot.set(Some(model));
                *self.current.lock().expect("current model lock") = name.clone();
                let mut saved = Settings::load(&self.settings);
                saved.model = Some(name.clone());
                if let Err(e) = saved.save(&self.settings) {
                    eprintln!("Could not save the model choice: {e:#}");
                }
                eprintln!("Now running {name}.");
                set(Phase::Idle);
            }
            Err(e) => {
                let message = format!("{e:#}");
                match self.loader.load(&old).await {
                    Ok(model) => self.slot.set(Some(model)),
                    Err(again) => eprintln!("Could not load {old} again either: {again:#}"),
                }
                fail(message);
            }
        }
        drop(hold);
    }
}
