//! The model an agent thinks with.

use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use kjarni_transformers::Conversation;

use crate::chat::Chat;

/// Text as the model generates it, piece by piece.
pub type TextStream = Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send>>;

/// What the agent needs from a model: a prompt for a conversation with tools,
/// and a reply to a prompt.
#[async_trait]
pub trait AgentModel: Send + Sync {
    /// The prompt for `conversation`, tools included.
    fn render(&self, conversation: &Conversation) -> anyhow::Result<String>;

    /// The reply to `prompt`, as it is generated.
    async fn stream(&self, prompt: String) -> anyhow::Result<TextStream>;

    /// The number of tokens `text` encodes to.
    fn count_tokens(&self, text: &str) -> anyhow::Result<usize>;

    /// The most tokens the model can attend to.
    fn context_size(&self) -> usize;

    /// The model's name, for display.
    fn name(&self) -> String;
}

#[async_trait]
impl AgentModel for Chat {
    fn render(&self, conversation: &Conversation) -> anyhow::Result<String> {
        Ok(self.render_conversation(conversation)?)
    }

    async fn stream(&self, prompt: String) -> anyhow::Result<TextStream> {
        let stream = self.generate_stream(prompt, Default::default()).await?;
        Ok(Box::pin(stream.map(|r| r.map_err(anyhow::Error::from))))
    }

    fn count_tokens(&self, text: &str) -> anyhow::Result<usize> {
        Ok(Chat::count_tokens(self, text)?)
    }

    fn context_size(&self) -> usize {
        Chat::context_size(self)
    }

    fn name(&self) -> String {
        self.model_name().to_string()
    }
}
