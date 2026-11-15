//! Cutting streamed text at stop sequences, and trimming a chat reply's
//! surrounding whitespace, without waiting for the whole reply.

/// Holds back any tail that could be the start of a stop sequence, so text
/// already sent never has to be taken back.
pub struct StopFilter {
    stops: Vec<String>,
    held: String,
}

impl StopFilter {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            held: String::new(),
        }
    }

    /// Takes the next piece of text. Returns what can be sent now, and whether
    /// a stop sequence was reached, in which case nothing after it is sent.
    pub fn push(&mut self, text: &str) -> (String, bool) {
        self.held.push_str(text);

        let first_stop = self
            .stops
            .iter()
            .filter_map(|stop| self.held.find(stop.as_str()))
            .min();
        if let Some(at) = first_stop {
            let out = self.held[..at].to_string();
            self.held.clear();
            return (out, true);
        }

        let keep = self.partial_stop_len();
        let split = self.held.len() - keep;
        let out = self.held[..split].to_string();
        self.held.drain(..split);
        (out, false)
    }

    /// What is still held back once the text has ended without a stop.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// The length of the longest tail of the held text that a stop sequence
    /// starts with.
    fn partial_stop_len(&self) -> usize {
        let mut longest = 0;
        for stop in &self.stops {
            for (i, _) in self.held.char_indices() {
                let tail = &self.held[i..];
                if tail.len() <= longest {
                    break;
                }
                if stop.starts_with(tail) {
                    longest = tail.len();
                    break;
                }
            }
        }
        longest
    }
}

/// Drops whitespace at the start and end of streamed text, giving the same
/// result as trimming it whole: whitespace is only sent once more text
/// follows it.
#[derive(Default)]
pub struct Trim {
    started: bool,
    pending: String,
}

impl Trim {
    pub fn push(&mut self, text: &str) -> String {
        let text = if self.started {
            text
        } else {
            let rest = text.trim_start();
            if rest.is_empty() {
                return String::new();
            }
            self.started = true;
            rest
        };

        let body = text.trim_end();
        if body.is_empty() {
            self.pending.push_str(text);
            return String::new();
        }
        let mut out = std::mem::take(&mut self.pending);
        out.push_str(body);
        self.pending.push_str(&text[body.len()..]);
        out
    }
}
