//! A reply that opens with `<think>`, split into its reasoning and its answer,
//! which the API returns as `reasoning_content` and `content`, as vLLM does.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

#[derive(Debug, Default, PartialEq, Eq)]
enum State {
    /// Not yet known whether the reply opens with `<think>`.
    #[default]
    Start,
    Reasoning,
    Answer,
}

/// Splits text as it streams in. Text that may be the start of `<think>` or
/// `</think>` is held until the next piece settles it.
#[derive(Debug, Default)]
pub struct ReasoningSplit {
    state: State,
    held: String,
    /// The reasoning has started: its leading newlines are dropped until then.
    reasoning: bool,
    /// The answer has started, likewise.
    answering: bool,
}

/// What one piece of text adds to the reasoning and to the answer.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Split {
    pub reasoning: String,
    pub answer: String,
}

impl ReasoningSplit {
    pub fn push(&mut self, text: &str) -> Split {
        self.held.push_str(text);
        let mut out = Split::default();
        loop {
            match self.state {
                State::Start => {
                    let trimmed = self.held.trim_start();
                    if let Some(rest) = trimmed.strip_prefix(OPEN) {
                        self.held = rest.trim_start_matches('\n').to_string();
                        self.state = State::Reasoning;
                    } else if OPEN.starts_with(trimmed) {
                        return out;
                    } else {
                        self.state = State::Answer;
                    }
                }
                State::Reasoning if !self.reasoning => {
                    let text = self.held.trim_start_matches('\n');
                    if text.is_empty() {
                        self.held.clear();
                        return out;
                    }
                    self.held = text.to_string();
                    self.reasoning = true;
                }
                State::Reasoning => match self.held.find(CLOSE) {
                    Some(at) => {
                        out.reasoning
                            .push_str(self.held[..at].trim_end_matches('\n'));
                        self.held = self.held[at + CLOSE.len()..].to_string();
                        self.state = State::Answer;
                    }
                    None => {
                        // Newlines may be the ones before `</think>`, which are dropped.
                        let keep = partial_suffix(&self.held, CLOSE);
                        let cut = self.held[..self.held.len() - keep]
                            .trim_end_matches('\n')
                            .len();
                        out.reasoning.push_str(&self.held[..cut]);
                        self.held.drain(..cut);
                        return out;
                    }
                },
                State::Answer => {
                    let text = std::mem::take(&mut self.held);
                    let text = if self.answering {
                        &text[..]
                    } else {
                        text.trim_start_matches('\n')
                    };
                    self.answering |= !text.is_empty();
                    out.answer.push_str(text);
                    return out;
                }
            }
        }
    }

    /// What is still held once the reply has ended: reasoning cut off before
    /// its `</think>`, or a start too short to tell.
    pub fn finish(&mut self) -> Split {
        let held = std::mem::take(&mut self.held);
        match self.state {
            State::Reasoning => Split {
                reasoning: held,
                answer: String::new(),
            },
            _ => Split {
                reasoning: String::new(),
                answer: held,
            },
        }
    }

    /// The whole of `text`, split.
    pub fn all(text: &str) -> Split {
        let mut split = Self::default();
        let mut out = split.push(text);
        let rest = split.finish();
        out.reasoning.push_str(&rest.reasoning);
        out.answer.push_str(&rest.answer);
        out
    }
}

/// The length of the longest end of `text` that `marker` starts with.
fn partial_suffix(text: &str, marker: &str) -> usize {
    (1..marker.len().min(text.len() + 1))
        .rev()
        .find(|&n| {
            text.is_char_boundary(text.len() - n) && marker.starts_with(&text[text.len() - n..])
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(reasoning: &str, answer: &str) -> Split {
        Split {
            reasoning: reasoning.into(),
            answer: answer.into(),
        }
    }

    #[test]
    fn a_whole_reply_splits_at_the_closing_tag() {
        assert_eq!(
            ReasoningSplit::all("<think>\nTwo and two.\n</think>\n\nFour."),
            split("Two and two.", "Four.")
        );
        // No think block: all answer.
        assert_eq!(ReasoningSplit::all("Four."), split("", "Four."));
        // Cut off while reasoning: all reasoning.
        assert_eq!(ReasoningSplit::all("<think>\nHmm"), split("Hmm", ""));
    }

    #[test]
    fn tags_split_across_pieces_are_held_until_settled() {
        let mut s = ReasoningSplit::default();
        let mut got = Split::default();
        for piece in [
            "<thi",
            "nk>\nTwo",
            " and two.\n</th",
            "ink>",
            "\n\nFo",
            "ur.",
        ] {
            let out = s.push(piece);
            got.reasoning.push_str(&out.reasoning);
            got.answer.push_str(&out.answer);
        }
        let rest = s.finish();
        got.reasoning.push_str(&rest.reasoning);
        got.answer.push_str(&rest.answer);
        assert_eq!(got, split("Two and two.", "Four."));
        // The opening arrives alone and the model's first token is a newline.
        let mut s = ReasoningSplit::default();
        assert_eq!(s.push("<think>\n"), split("", ""));
        assert_eq!(s.push("\nThinking"), split("Thinking", ""));
        assert_eq!(s.push("\n\nmore"), split("\n\nmore", ""));
        // Nothing of "</th" leaked into the reasoning while it was held.
        let mut s = ReasoningSplit::default();
        assert_eq!(s.push("<think>\nab</th"), split("ab", ""));
        assert_eq!(s.push("x"), split("</thx", ""));
    }

    #[test]
    fn an_answer_that_merely_starts_like_the_tag_is_an_answer() {
        assert_eq!(
            ReasoningSplit::all("<thing> is a word"),
            split("", "<thing> is a word")
        );
        let mut s = ReasoningSplit::default();
        assert_eq!(s.push("<th"), split("", ""));
        assert_eq!(s.push("e end"), split("", "<the end"));
    }
}
