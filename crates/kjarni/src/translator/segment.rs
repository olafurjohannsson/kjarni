//! Splits text into the pieces a translation model handles well.
//!
//! Translation models are trained on single sentences. Given several at once they end
//! early and leave the rest of the text out, without any error, so each sentence is
//! translated on its own and the whitespace between them is kept.

/// The most tokens given to the model at once. Opus-MT dropped sentences from about 170
/// tokens of input, and a normal sentence is far below this.
pub(super) const MAX_SEGMENT_TOKENS: usize = 128;

/// Words whose full stop does not end a sentence.
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "vs", "no", "nr", "fig", "ca", "bzw", "ggf",
    "sog", "hr", "fr", "inkl", "evtl", "vgl",
];

/// Text to translate, and the whitespace that follows it in the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Segment<'a> {
    pub text: &'a str,
    pub gap: &'a str,
}

/// Splits `text` into sentences, and any sentence over `max_tokens` into smaller parts.
///
/// A line break always ends a segment, so lists and headings keep their layout. There is
/// always at least one segment, even for empty text.
pub(super) fn segments<'a>(
    text: &'a str,
    max_tokens: usize,
    count_tokens: &dyn Fn(&str) -> usize,
) -> Vec<Segment<'a>> {
    let mut out = Vec::new();
    for sentence in sentences(text) {
        fit(
            sentence.text,
            sentence.gap,
            max_tokens,
            count_tokens,
            &mut out,
        );
    }
    out
}

fn sentences(text: &str) -> Vec<Segment<'_>> {
    let text = text.trim();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let byte_at = |i: usize| chars.get(i).map_or(text.len(), |c| c.0);

    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i].1;
        let end = if c == '\n' {
            i
        } else if is_terminator(c) {
            let mut after = i + 1;
            while after < chars.len()
                && (is_terminator(chars[after].1) || is_closer(chars[after].1))
            {
                after += 1;
            }
            if !ends_sentence(&text[start..chars[i].0], c, &chars[after..]) {
                i = after;
                continue;
            }
            after
        } else {
            i += 1;
            continue;
        };

        let mut next = end;
        while next < chars.len() && chars[next].1.is_whitespace() {
            next += 1;
        }
        if next == chars.len() {
            break;
        }
        let sentence = text[start..byte_at(end)].trim_end();
        let gap = &text[start + sentence.len()..byte_at(next)];
        out.push(Segment {
            text: sentence,
            // Sentences in Chinese and Japanese have nothing between them.
            gap: if gap.is_empty() { " " } else { gap },
        });
        start = byte_at(next);
        i = next;
    }
    out.push(Segment {
        text: &text[start..],
        gap: "",
    });
    out
}

/// Whether the terminator `c` ends a sentence, given the sentence so far and what follows
/// the terminator and any closing quotes. A wrong merge costs little, since the result is
/// only a longer segment, so an unclear full stop does not split.
fn ends_sentence(before: &str, c: char, after: &[(usize, char)]) -> bool {
    if matches!(c, '。' | '！' | '？' | '｡') {
        return true;
    }
    // Covers 3.14, example.com and v1.2.
    if after.first().is_some_and(|(_, next)| !next.is_whitespace()) {
        return false;
    }
    if c != '.' {
        return true;
    }
    let next_word = after.iter().map(|(_, c)| *c).find(|c| !c.is_whitespace());
    if next_word.is_some_and(char::is_lowercase) {
        return false;
    }
    let word = before
        .rsplit(|c: char| c.is_whitespace() || is_opener(c))
        .next()
        .unwrap_or_default();
    let is_ordinal =
        word.len() <= 2 && !word.is_empty() && word.chars().all(|c| c.is_ascii_digit());
    let is_initial = word.chars().count() == 1 && word.chars().all(char::is_uppercase);
    !(is_ordinal
        || is_initial
        || word.contains('.')
        || ABBREVIATIONS.contains(&word.to_lowercase().as_str()))
}

fn is_terminator(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…' | '。' | '！' | '？' | '｡')
}

fn is_closer(c: char) -> bool {
    matches!(
        c,
        '"' | '\'' | ')' | ']' | '”' | '’' | '»' | '“' | '」' | '』' | '）'
    )
}

fn is_opener(c: char) -> bool {
    matches!(
        c,
        '"' | '\'' | '(' | '[' | '“' | '‘' | '«' | '„' | '「' | '『' | '（'
    )
}

/// Halves `text` until each part fits. A segment of one character is left as it is.
fn fit<'a>(
    text: &'a str,
    gap: &'a str,
    max_tokens: usize,
    count_tokens: &dyn Fn(&str) -> usize,
    out: &mut Vec<Segment<'a>>,
) {
    if text.chars().nth(1).is_none() || count_tokens(text) <= max_tokens {
        out.push(Segment { text, gap });
        return;
    }
    let at = break_point(text);
    let left = text[..at].trim_end();
    let right = text[at..].trim_start();
    let inner = &text[left.len()..text.len() - right.len()];
    fit(left, inner, max_tokens, count_tokens, out);
    fit(right, gap, max_tokens, count_tokens, out);
}

/// Where to cut `text`, which has at least two characters and no whitespace at either end:
/// after a clause near the middle, otherwise at the space closest to the middle, otherwise
/// at the middle itself.
fn break_point(text: &str) -> usize {
    let middle = text.len() / 2;
    let closest =
        |candidates: Vec<usize>| candidates.into_iter().min_by_key(|at| at.abs_diff(middle));

    let clauses = text
        .char_indices()
        .filter(|(_, c)| matches!(c, ',' | ';' | ':' | '，' | '；' | '：' | '、'))
        .map(|(i, c)| i + c.len_utf8())
        .filter(|at| (text.len() / 4..=text.len() * 3 / 4).contains(at))
        .filter(|at| *at < text.len())
        .collect();
    if let Some(at) = closest(clauses) {
        return at;
    }

    let spaces = text
        .char_indices()
        .filter(|(_, c)| c.is_whitespace())
        .map(|(i, _)| i)
        .collect();
    if let Some(at) = closest(spaces) {
        return at;
    }

    closest(text.char_indices().map(|(i, _)| i).skip(1).collect()).unwrap_or(text.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn texts(text: &str) -> Vec<&str> {
        segments(text, 1000, &words)
            .iter()
            .map(|s| s.text)
            .collect()
    }

    #[test]
    fn splits_sentences_and_keeps_the_space_between_them() {
        let got = segments("Where is it? It is here! Good.", 1000, &words);
        assert_eq!(
            got,
            vec![
                Segment {
                    text: "Where is it?",
                    gap: " "
                },
                Segment {
                    text: "It is here!",
                    gap: " "
                },
                Segment {
                    text: "Good.",
                    gap: ""
                },
            ]
        );
    }

    #[test]
    fn line_breaks_end_a_segment_and_are_kept() {
        let got = segments("Title\n\nFirst line  \nSecond line", 1000, &words);
        assert_eq!(
            got,
            vec![
                Segment {
                    text: "Title",
                    gap: "\n\n"
                },
                Segment {
                    text: "First line",
                    gap: "  \n"
                },
                Segment {
                    text: "Second line",
                    gap: ""
                },
            ]
        );
    }

    #[test]
    fn surrounding_whitespace_is_dropped() {
        assert_eq!(
            segments("  One sentence.\n", 1000, &words),
            vec![Segment {
                text: "One sentence.",
                gap: ""
            }]
        );
    }

    #[test]
    fn empty_text_is_one_empty_segment() {
        assert_eq!(
            segments(" \n ", 1000, &words),
            vec![Segment { text: "", gap: "" }]
        );
    }

    #[test]
    fn full_stops_inside_a_sentence_do_not_split() {
        for text in [
            "Pi is 3.14 and the site is example.com today.",
            "Mr. Smith met Dr. Jones.",
            "J. Smith wrote it.",
            "Use a tool, e.g. this one, or etc. and more.",
            "Das Treffen ist am 3. Mai in Berlin.",
            "Das gilt z.B. Hier und dort.",
        ] {
            assert_eq!(texts(text), vec![text], "{text}");
        }
    }

    #[test]
    fn closing_quotes_stay_with_their_sentence() {
        assert_eq!(
            texts("He said \"Stop.\" Then he left (quickly.) Done."),
            vec!["He said \"Stop.\"", "Then he left (quickly.)", "Done."]
        );
    }

    #[test]
    fn a_year_can_end_a_sentence() {
        assert_eq!(
            texts("It ended in 1999. Then it began."),
            vec!["It ended in 1999.", "Then it began."]
        );
    }

    #[test]
    fn chinese_sentences_split_without_spaces() {
        let got = segments("你好。你好吗？我很好", 1000, &words);
        assert_eq!(
            got,
            vec![
                Segment {
                    text: "你好。",
                    gap: " "
                },
                Segment {
                    text: "你好吗？",
                    gap: " "
                },
                Segment {
                    text: "我很好",
                    gap: ""
                },
            ]
        );
    }

    #[test]
    fn a_long_sentence_is_cut_after_a_clause() {
        let got = segments("one two three, four five six", 4, &words);
        assert_eq!(
            got,
            vec![
                Segment {
                    text: "one two three,",
                    gap: " "
                },
                Segment {
                    text: "four five six",
                    gap: ""
                },
            ]
        );
    }

    #[test]
    fn a_long_sentence_without_clauses_is_cut_at_spaces() {
        let got = segments("a b c d e f g h", 2, &words);
        assert_eq!(got.len(), 4);
        assert!(got.iter().all(|s| words(s.text) <= 2));
        let joined: String = got.iter().flat_map(|s| [s.text, s.gap]).collect();
        assert_eq!(joined, "a b c d e f g h");
    }

    #[test]
    fn text_with_no_spaces_is_cut_on_character_boundaries() {
        let chars = |text: &str| text.chars().count();
        let text = "翻訳翻訳翻訳翻訳翻訳";
        let got = segments(text, 3, &chars);
        assert!(got.iter().all(|s| chars(s.text) <= 3));
        let joined: String = got.iter().flat_map(|s| [s.text, s.gap]).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn no_text_is_lost() {
        let text = "First one. Second, which is rather long, goes on and on and on.\n\n\
                    3. Third item\nLast: done! Really?";
        let got = segments(text, 4, &words);
        let joined: String = got.iter().flat_map(|s| [s.text, s.gap]).collect();
        assert_eq!(joined, text);
    }
}
