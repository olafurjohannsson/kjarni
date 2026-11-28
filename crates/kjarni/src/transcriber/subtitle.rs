//! Subtitle formatting (SRT and WebVTT) for transcription segments.

use super::types::{TranscriptionResult, TranscriptionSegment};

/// Default maximum Unicode characters per subtitle line.
pub const DEFAULT_MAX_LINE_CHARS: usize = 42;

/// Default maximum lines per subtitle cue.
pub const DEFAULT_MAX_LINES: usize = 2;

#[derive(Clone, Copy)]
enum SubtitleFormat {
    Srt,
    Vtt,
}

/// Format timed segments as SubRip (`.srt`) subtitle text.
///
/// Long segments are split on word boundaries into lines of around
/// [`DEFAULT_MAX_LINE_CHARS`] characters and at most [`DEFAULT_MAX_LINES`]
/// lines per cue, with timestamps distributed proportionally by character count.
/// Empty or whitespace-only segments are skipped.
pub fn segments_to_srt(segments: &[TranscriptionSegment]) -> String {
    format_segments(
        segments,
        SubtitleFormat::Srt,
        DEFAULT_MAX_LINE_CHARS,
        DEFAULT_MAX_LINES,
    )
}

/// Format timed segments as WebVTT (`.vtt`) subtitle text.
///
/// Always begins with the `WEBVTT` header. Long segments are split on word
/// boundaries into lines of around [`DEFAULT_MAX_LINE_CHARS`] characters and at
/// most [`DEFAULT_MAX_LINES`] lines per cue, with timestamps distributed
/// proportionally by character count. Empty or whitespace-only segments are
/// skipped, and any `-->` sequence inside cue text is escaped as `-- >`.
pub fn segments_to_vtt(segments: &[TranscriptionSegment]) -> String {
    format_segments(
        segments,
        SubtitleFormat::Vtt,
        DEFAULT_MAX_LINE_CHARS,
        DEFAULT_MAX_LINES,
    )
}

impl TranscriptionResult {
    /// Format this transcription as SubRip (`.srt`) subtitle text.
    ///
    /// Falls back to a single segment spanning `0.0..duration_secs` if
    /// `segments` is empty and `text` is non-empty.
    pub fn to_srt(&self) -> String {
        if self.segments.is_empty() {
            if self.text.trim().is_empty() {
                return String::new();
            }
            let fallback = [fallback_segment(&self.text, self.duration_secs)];
            return segments_to_srt(&fallback);
        }
        segments_to_srt(&self.segments)
    }

    /// Format this transcription as WebVTT (`.vtt`) subtitle text.
    ///
    /// Falls back to a single segment spanning `0.0..duration_secs` if
    /// `segments` is empty and `text` is non-empty.
    pub fn to_vtt(&self) -> String {
        if self.segments.is_empty() {
            if self.text.trim().is_empty() {
                return String::from("WEBVTT\n\n");
            }
            let fallback = [fallback_segment(&self.text, self.duration_secs)];
            return segments_to_vtt(&fallback);
        }
        segments_to_vtt(&self.segments)
    }
}

fn fallback_segment(text: &str, duration_secs: f32) -> TranscriptionSegment {
    let end = if duration_secs.is_finite() {
        duration_secs.max(0.0)
    } else {
        0.0
    };
    TranscriptionSegment {
        start: 0.0,
        end,
        text: text.to_string(),
    }
}

fn format_segments(
    segments: &[TranscriptionSegment],
    format: SubtitleFormat,
    max_line_chars: usize,
    max_lines: usize,
) -> String {
    let mut out = match format {
        SubtitleFormat::Srt => String::new(),
        SubtitleFormat::Vtt => String::from("WEBVTT\n\n"),
    };
    let decimal = match format {
        SubtitleFormat::Srt => ',',
        SubtitleFormat::Vtt => '.',
    };
    let max_line_chars = max_line_chars.max(1);
    let max_lines = max_lines.max(1);

    let mut cue_index = 1usize;
    for segment in segments {
        let cues = build_segment_cues(&segment.text, format, max_line_chars, max_lines);
        if cues.is_empty() {
            continue;
        }

        let start_ms = to_millis(segment.start);
        let end_ms = to_millis(segment.end).max(start_ms);
        let timed = distribute_timestamps(&cues, start_ms, end_ms);

        for (cue_start, cue_end, cue_text) in timed {
            if matches!(format, SubtitleFormat::Srt) {
                out.push_str(&format!("{cue_index}\n"));
                cue_index += 1;
            }
            out.push_str(&format!(
                "{} --> {}\n{}\n\n",
                format_timestamp(cue_start, decimal),
                format_timestamp(cue_end, decimal),
                cue_text
            ));
        }
    }

    out
}

fn build_segment_cues(
    text: &str,
    format: SubtitleFormat,
    max_line_chars: usize,
    max_lines: usize,
) -> Vec<String> {
    let raw_words: Vec<&str> = text.split_whitespace().collect();
    if raw_words.is_empty() {
        return Vec::new();
    }

    let escaped;
    let words: Vec<&str> = match format {
        SubtitleFormat::Srt => raw_words,
        SubtitleFormat::Vtt => {
            let joined = raw_words.join(" ");
            if joined.contains("-->") {
                escaped = joined.replace("-->", "-- >");
                escaped.split_whitespace().collect()
            } else {
                raw_words
            }
        }
    };

    let lines = wrap_words(&words, max_line_chars);
    lines
        .chunks(max_lines)
        .map(|chunk| chunk.join("\n"))
        .collect()
}

fn wrap_words(words: &[&str], max_line_chars: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_chars = 0usize;

    for &word in words {
        let word_chars = word.chars().count();
        if current_chars == 0 {
            current.push_str(word);
            current_chars = word_chars;
        } else if current_chars + 1 + word_chars <= max_line_chars {
            current.push(' ');
            current.push_str(word);
            current_chars += 1 + word_chars;
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
            current_chars = word_chars;
        }
    }

    if current_chars > 0 {
        lines.push(current);
    }

    lines
}

fn distribute_timestamps(cues: &[String], start_ms: u64, end_ms: u64) -> Vec<(u64, u64, &str)> {
    let count = cues.len();
    if count == 1 {
        return vec![(start_ms, end_ms, cues[0].as_str())];
    }

    let weights: Vec<u64> = cues
        .iter()
        .map(|c| (c.chars().count() as u64).max(1))
        .collect();
    let total_weight: u64 = weights.iter().sum();
    let total_dur = end_ms - start_ms;

    let mut out = Vec::with_capacity(count);
    let mut prev_end = start_ms;
    let mut cum_weight = 0u64;

    for (i, cue) in cues.iter().enumerate() {
        let cue_start = prev_end;
        let cue_end = if i + 1 == count {
            end_ms
        } else {
            cum_weight += weights[i];
            let offset = (total_dur as u128 * cum_weight as u128 + (total_weight as u128 / 2))
                / total_weight as u128;
            (start_ms + offset as u64).clamp(cue_start, end_ms)
        };
        out.push((cue_start, cue_end, cue.as_str()));
        prev_end = cue_end;
    }

    out
}

fn to_millis(seconds: f32) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    let ms = (seconds as f64) * 1000.0;
    if ms >= u64::MAX as f64 {
        u64::MAX
    } else {
        ms.round() as u64
    }
}

fn format_timestamp(millis: u64, decimal: char) -> String {
    let ms = millis % 1000;
    let total_secs = millis / 1000;
    let s = total_secs % 60;
    let m = (total_secs / 60) % 60;
    let h = total_secs / 3600;
    format!("{h:02}:{m:02}:{s:02}{decimal}{ms:03}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f32, end: f32, text: &str) -> TranscriptionSegment {
        TranscriptionSegment {
            start,
            end,
            text: text.to_string(),
        }
    }

    #[test]
    fn empty_segments_produce_empty_srt_and_vtt_header() {
        assert_eq!(segments_to_srt(&[]), "");
        assert_eq!(segments_to_vtt(&[]), "WEBVTT\n\n");
    }

    #[test]
    fn short_segments_and_empty_segments_are_skipped_without_gaps_in_numbering() {
        let segs = [
            seg(0.0, 1.5, "  Hello there. "),
            seg(1.5, 2.0, "  \n\t  "),
            seg(61.25, 3725.004, " General\n\tKenobi. "),
        ];
        assert_eq!(
            segments_to_srt(&segs),
            "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n\
             2\n00:01:01,250 --> 01:02:05,004\nGeneral Kenobi.\n\n"
        );
        assert_eq!(
            segments_to_vtt(&segs),
            "WEBVTT\n\n\
             00:00:00.000 --> 00:00:01.500\nHello there.\n\n\
             00:01:01.250 --> 01:02:05.004\nGeneral Kenobi.\n\n"
        );
    }

    #[test]
    fn wraps_long_segment_over_42_chars_into_two_lines() {
        let segs = [seg(
            0.0,
            4.0,
            "This is a moderately long subtitle line that should wrap onto two lines.",
        )];
        assert_eq!(
            segments_to_srt(&segs),
            "1\n00:00:00,000 --> 00:00:04,000\n\
             This is a moderately long subtitle line\n\
             that should wrap onto two lines.\n\n"
        );
    }

    #[test]
    fn splits_very_long_segment_into_multiple_cues_with_proportional_timestamps() {
        // Cue 1: 38 + 1 + 38 = 77 chars; Cue 2: 27 chars -> total 104 chars.
        let segs = [seg(
            10.0,
            20.0,
            "Alpha bravo charlie delta echo foxtrot golf \
             hotel india juliet kilo lima mike november \
             oscar papa quebec.",
        )];
        let srt = segments_to_srt(&segs);
        assert_eq!(
            srt,
            "1\n00:00:10,000 --> 00:00:17,404\n\
             Alpha bravo charlie delta echo foxtrot\n\
             golf hotel india juliet kilo lima mike\n\n\
             2\n00:00:17,404 --> 00:00:20,000\n\
             november oscar papa quebec.\n\n"
        );
        // Verify each line is <= 42 chars
        for block in srt.trim_end().split("\n\n") {
            let lines: Vec<&str> = block.lines().skip(2).collect();
            assert!(lines.len() <= 2);
            for line in lines {
                assert!(line.chars().count() <= 42, "line too long: {line:?}");
            }
        }
    }

    #[test]
    fn keeps_single_word_over_42_chars_intact_on_its_own_line() {
        let long_word = "supercalifragilisticexpialidociousandmorewords";
        assert_eq!(long_word.chars().count(), 46);
        let text = format!("Prefix {long_word} suffix");
        let segs = [seg(0.0, 6.0, &text)];
        let srt = segments_to_srt(&segs);
        assert_eq!(
            srt,
            format!(
                "1\n00:00:00,000 --> 00:00:05,390\n\
                 Prefix\n\
                 {long_word}\n\n\
                 2\n00:00:05,390 --> 00:00:06,000\n\
                 suffix\n\n"
            )
        );
    }

    #[test]
    fn counts_unicode_scalar_characters_not_utf8_bytes() {
        // 41 Unicode scalar values (52 UTF-8 bytes) fits on one line <= 42.
        let line = "Þórður sagði halló við alla í Reykjavík í";
        assert_eq!(line.chars().count(), 41);
        assert!(line.len() > 42);
        let segs = [seg(0.0, 3.0, &format!("{line} dag."))];
        assert_eq!(
            segments_to_srt(&segs),
            format!(
                "1\n00:00:00,000 --> 00:00:03,000\n\
                 {line}\n\
                 dag.\n\n"
            )
        );
    }

    #[test]
    fn handles_timestamp_rounding_non_finite_negative_and_reversed_bounds() {
        let segs = [
            seg(0.0, 0.9996, "Rounding up."),
            seg(f32::NAN, f32::INFINITY, "Non-finite."),
            seg(-2.5, -1.0, "Negative."),
            seg(5.0, 3.0, "Reversed."),
        ];
        assert_eq!(
            segments_to_srt(&segs),
            "1\n00:00:00,000 --> 00:00:01,000\nRounding up.\n\n\
             2\n00:00:00,000 --> 00:00:00,000\nNon-finite.\n\n\
             3\n00:00:00,000 --> 00:00:00,000\nNegative.\n\n\
             4\n00:00:05,000 --> 00:00:05,000\nReversed.\n\n"
        );
    }

    #[test]
    fn vtt_escapes_timing_arrow_in_cue_text() {
        let segs = [seg(0.0, 2.0, "From A --> B and x-->y")];
        assert_eq!(
            segments_to_vtt(&segs),
            "WEBVTT\n\n\
             00:00:00.000 --> 00:00:02.000\n\
             From A -- > B and x-- >y\n\n"
        );
    }

    #[test]
    fn result_falls_back_to_text_when_segments_empty() {
        let result = TranscriptionResult {
            text: "  Just full text. ".into(),
            segments: Vec::new(),
            language: "en".into(),
            duration_secs: 4.5,
        };
        assert_eq!(
            result.to_srt(),
            "1\n00:00:00,000 --> 00:00:04,500\nJust full text.\n\n"
        );
        assert_eq!(
            result.to_vtt(),
            "WEBVTT\n\n00:00:00.000 --> 00:00:04.500\nJust full text.\n\n"
        );

        let empty = TranscriptionResult {
            text: "   ".into(),
            segments: Vec::new(),
            language: "en".into(),
            duration_secs: 1.0,
        };
        assert_eq!(empty.to_srt(), "");
        assert_eq!(empty.to_vtt(), "WEBVTT\n\n");
    }
}
