//! Text splitting with markdown-aware chunking and preprocessing
use std::collections::HashMap;
use text_splitter::{ChunkConfig, MarkdownSplitter};

/// Configuration for text splitting
#[derive(Debug, Clone)]
pub struct SplitterConfig {
    /// Maximum chunk size in characters
    pub chunk_size: usize,
    /// Overlap between chunks in characters
    pub chunk_overlap: usize,
    /// Whether to clean markdown syntax before chunking
    pub clean_markdown: bool,
}

impl Default for SplitterConfig {
    /// 512 characters, which fits inside the encoder window rather than just past it.
    ///
    /// `minilm-l6-v2` reads 256 tokens, about 900 characters of English. The previous
    /// default of 1000 sat just over that line, so roughly nine chunks in ten lost
    /// their tail before the encoder saw them, with no warning and a normal looking
    /// vector out the other end. mpnet reads 384 tokens and has more room, but 512
    /// is safe for both.
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 100,
            clean_markdown: true,
        }
    }
}

impl SplitterConfig {
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap: chunk_size / 5,
            clean_markdown: true,
        }
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.chunk_size == 0 {
            return Err("chunk_size must be greater than 0");
        }
        if self.chunk_overlap >= self.chunk_size {
            return Err("chunk_overlap must be less than chunk_size");
        }
        Ok(())
    }
}

/// Markdown-aware text splitter
pub struct TextSplitter {
    config: SplitterConfig,
}

impl TextSplitter {
    /// Create a new text splitter with the given configuration
    pub fn new(config: SplitterConfig) -> Self {
        if let Err(e) = config.validate() {
            panic!("Invalid SplitterConfig: {}", e);
        }
        Self { config }
    }

    /// Create a text splitter with default configuration
    pub fn with_defaults() -> Self {
        Self::new(SplitterConfig::default())
    }

    /// Get a reference to the current configuration
    pub fn config(&self) -> &SplitterConfig {
        &self.config
    }

    /// Split text into chunks using markdown-aware boundaries
    pub fn split(&self, text: &str) -> Vec<String> {
        if self.config.clean_markdown {
            self.split_verbatim(&clean_markdown(text))
        } else {
            self.split_verbatim(text)
        }
    }

    /// Split text into chunks without cleaning it first
    pub fn split_verbatim(&self, text: &str) -> Vec<String> {
        if text.trim().is_empty() {
            return vec![];
        }

        let chunk_config = ChunkConfig::new(self.config.chunk_size)
            .with_overlap(self.config.chunk_overlap)
            .expect("validated in TextSplitter::new");

        MarkdownSplitter::new(chunk_config)
            .chunks(text)
            .map(|chunk| chunk.to_string())
            .filter(|chunk| !chunk.trim().is_empty())
            .collect()
    }

    /// Split text into chunks with metadata
    pub fn split_with_metadata(
        &self,
        text: &str,
        base_metadata: HashMap<String, String>,
    ) -> Vec<(String, HashMap<String, String>)> {
        let chunks = self.split(text);
        let total = chunks.len();

        chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let mut metadata = base_metadata.clone();
                metadata.insert("chunk_index".to_string(), i.to_string());
                metadata.insert("total_chunks".to_string(), total.to_string());
                (chunk, metadata)
            })
            .collect()
    }

    /// Estimate the number of chunks for a given text
    pub fn estimate_chunks(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        let effective = self
            .config
            .chunk_size
            .saturating_sub(self.config.chunk_overlap)
            .max(1);
        let char_count = text.chars().count();
        char_count.div_ceil(effective)
    }
}

/// Removes markdown and HTML markup, leaving code spans and fenced blocks as written.
pub fn clean_markdown(text: &str) -> String {
    let text = strip_frontmatter(text);
    let mut result = String::with_capacity(text.len());
    let mut prose = String::new();
    let mut fence: Option<&str> = None;

    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start();
        match fence {
            Some(marker) => {
                result.push_str(line);
                if trimmed.starts_with(marker) {
                    fence = None;
                }
            }
            None if trimmed.starts_with("```") || trimmed.starts_with("~~~") => {
                result.push_str(&clean_prose(&prose));
                prose.clear();
                result.push_str(line);
                fence = Some(&trimmed[..3]);
            }
            None => prose.push_str(line),
        }
    }
    result.push_str(&clean_prose(&prose));

    collapse_whitespace(&result)
}

fn clean_prose(text: &str) -> String {
    let text = strip_noisy_lines(text);
    let mut result = String::with_capacity(text.len());
    let mut rest = text.as_str();

    // Inline code spans: a run of backticks closed by a run of the same length.
    while let Some(open) = rest.find('`') {
        let ticks = rest[open..].chars().take_while(|c| *c == '`').count();
        let after = &rest[open + ticks..];
        let marker = "`".repeat(ticks);
        match after.find(&marker) {
            Some(close) => {
                result.push_str(&clean_inline(&rest[..open]));
                result.push_str(&rest[open..open + ticks + close + ticks]);
                rest = &after[close + ticks..];
            }
            None => {
                result.push_str(&clean_inline(&rest[..open + ticks]));
                rest = after;
            }
        }
    }
    result.push_str(&clean_inline(rest));
    result
}

fn clean_inline(text: &str) -> String {
    let text = clean_wikilinks(text);
    let text = clean_markdown_images(&text);
    let text = clean_markdown_links(&text);
    strip_html_tags(&text)
}

fn strip_frontmatter(text: &str) -> String {
    let trimmed = text.trim_start();
    if !trimmed.starts_with("---") {
        return text.to_string();
    }

    let after_first = &trimmed[3..];
    if let Some(end_pos) = after_first.find("\n---") {
        let rest = &after_first[end_pos + 4..];
        rest.trim_start_matches('\n').to_string()
    } else {
        text.to_string()
    }
}

/// Remove lines that are pure noise for embeddings
fn strip_noisy_lines(text: &str) -> String {
    let mut result = String::with_capacity(text.len());

    for line in text.lines() {
        let trimmed = line.trim();

        // Skip badge/shield image lines
        if trimmed.contains("img.shields.io")
            || trimmed.contains("badgen.net")
            || trimmed.contains("badge.fury.io")
        {
            continue;
        }

        // Skip pure URL lines
        if (trimmed.starts_with("http://") || trimmed.starts_with("https://"))
            && !trimmed.contains(' ')
        {
            continue;
        }

        // Skip empty heading lines (e.g., just "##")
        if trimmed.starts_with('#') && trimmed.trim_start_matches('#').trim().is_empty() {
            continue;
        }

        // Skip horizontal rules
        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            continue;
        }

        result.push_str(line);
        result.push('\n');
    }

    result
}

/// Convert [[wiki-links]] to plain text
///
/// - `[[page]]` → `page`
/// - `[[page|display]]` → `display`
/// - `[[path/to/page]]` → `page`
/// - `[[path/to/page|display]]` → `display`
///
/// Anything else with double brackets, such as `[[1, 2], [3, 4]]`, is left alone.
fn clean_wikilinks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '['
            && chars.get(i + 1) == Some(&'[')
            && let Some(close) = wikilink_end(&chars, i + 2)
        {
            let link_content: String = chars[i + 2..close].iter().collect();
            // Use display text (after |) if present
            let display = if let Some(pipe_pos) = link_content.rfind('|') {
                &link_content[pipe_pos + 1..]
            } else if let Some(slash_pos) = link_content.rfind('/') {
                // Use filename part of path
                &link_content[slash_pos + 1..]
            } else {
                &link_content
            };
            result.push_str(display);
            i = close + 2;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// The index of the `]]` closing a wikilink whose content starts at `start`.
fn wikilink_end(chars: &[char], start: usize) -> Option<usize> {
    for i in start..chars.len() {
        match chars[i] {
            ']' if chars.get(i + 1) == Some(&']') => return Some(i),
            '[' | ']' | '\n' => return None,
            _ => {}
        }
    }
    None
}

/// The index of the `]` closing a bracket opened just before `start`, on the same line.
fn closing_bracket(chars: &[char], start: usize) -> Option<usize> {
    let mut depth = 1;
    for (i, c) in chars.iter().enumerate().skip(start) {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            '\n' => return None,
            _ => {}
        }
    }
    None
}

/// The index of the `)` closing a `(` at `open`, on the same line.
fn closing_paren(chars: &[char], open: usize) -> Option<usize> {
    if chars.get(open) != Some(&'(') {
        return None;
    }
    let mut depth = 0;
    for (i, c) in chars.iter().enumerate().skip(open) {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            '\n' => return None,
            _ => {}
        }
    }
    None
}

/// Convert markdown images to just alt text
///
/// `![alt text](url)` → `alt text`
/// `![](url)` → (removed)
fn clean_markdown_images(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '!'
            && chars.get(i + 1) == Some(&'[')
            && let Some(close) = closing_bracket(&chars, i + 2)
            && let Some(end) = closing_paren(&chars, close + 1)
        {
            let alt: String = chars[i + 2..close].iter().collect();
            result.push_str(alt.trim());
            i = end + 1;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Convert markdown links to just display text
///
/// `[display](url)` → `display`. Brackets not followed by a URL are left alone.
fn clean_markdown_links(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '['
            && let Some(close) = closing_bracket(&chars, i + 1)
            && let Some(end) = closing_paren(&chars, close + 1)
        {
            result.extend(&chars[i + 1..close]);
            i = end + 1;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Strip HTML tags, keep inner text. A `<` only opens a tag when a tag name, `/` or `!`
/// follows and a `>` closes it, so `a < b` is kept.
fn strip_html_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(open) = rest.find('<') {
        result.push_str(&rest[..open]);
        let tag = &rest[open + 1..];
        let opens_tag = tag
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '/' || c == '!');
        let close = tag.find(['<', '>']).filter(|&i| tag.as_bytes()[i] == b'>');

        match (opens_tag, close) {
            (true, Some(close)) => rest = &tag[close + 1..],
            _ => {
                result.push('<');
                rest = tag;
            }
        }
    }
    result.push_str(rest);

    result
}

/// Collapse multiple blank lines into at most two newlines
fn collapse_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut consecutive_newlines = 0;

    for ch in text.chars() {
        if ch == '\n' {
            consecutive_newlines += 1;
            if consecutive_newlines <= 2 {
                result.push(ch);
            }
        } else {
            consecutive_newlines = 0;
            result.push(ch);
        }
    }

    result.trim().to_string()
}

#[cfg(test)]
mod review_regression_tests {
    use super::*;

    #[test]
    fn prose_with_brackets_and_comparisons_survives_cleaning() {
        for (input, expected) in [
            (
                "Prices < 5 dollars.\nMore text.",
                "Prices < 5 dollars.\nMore text.",
            ),
            ("See [note] for details.", "See [note] for details."),
            ("matrix = [[1, 2], [3, 4]]", "matrix = [[1, 2], [3, 4]]"),
            ("Wow![really] yes", "Wow![really] yes"),
            ("a <b>bold</b> word", "a bold word"),
            ("![alt](u.png) [t](http://x) [[p|d]]", "alt t d"),
        ] {
            assert_eq!(clean_markdown(input), expected);
        }
    }

    #[test]
    fn code_in_markdown_is_left_alone() {
        let input = "Use `Vec<String>` or `arr[0]`.\n\n```rust\nlet v = vec![1];\nif a < b {}\n```\n\n<b>done</b>";
        let expected =
            "Use `Vec<String>` or `arr[0]`.\n\n```rust\nlet v = vec![1];\nif a < b {}\n```\n\ndone";
        assert_eq!(clean_markdown(input), expected);
    }

    #[test]
    fn chunk_overlap_repeats_text_between_chunks() {
        let text = (0..400)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let split = |chunk_overlap| {
            TextSplitter::new(SplitterConfig {
                chunk_size: 200,
                chunk_overlap,
                clean_markdown: false,
            })
            .split(&text)
        };
        let (without, with) = (split(0), split(100));

        assert!(with.len() > without.len());
        let tail = without[0].split(' ').next_back().unwrap();
        assert!(with[1].contains(tail));
        assert!(!without[1].contains(tail));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_frontmatter() {
        let input = "---\ntitle: Hello\ntags: [a, b]\n---\n\nActual content here.";
        let result = strip_frontmatter(input);
        assert_eq!(result, "Actual content here.");
    }

    #[test]
    fn test_strip_frontmatter_no_frontmatter() {
        let input = "Just regular text";
        assert_eq!(strip_frontmatter(input), input);
    }

    #[test]
    fn test_strip_frontmatter_unclosed() {
        let input = "---\ntitle: Hello\nNo closing marker";
        assert_eq!(strip_frontmatter(input), input);
    }

    #[test]
    fn test_strip_noisy_lines_badges() {
        let input = "Some text\n![badge](https://img.shields.io/github/stars/foo)\nMore text";
        let result = strip_noisy_lines(input);
        assert!(result.contains("Some text"));
        assert!(result.contains("More text"));
        assert!(!result.contains("shields.io"));
    }

    #[test]
    fn test_strip_noisy_lines_pure_urls() {
        let input = "Text before\nhttps://example.com/some/page\nText after";
        let result = strip_noisy_lines(input);
        assert!(result.contains("Text before"));
        assert!(result.contains("Text after"));
        assert!(!result.contains("https://example.com"));
    }

    #[test]
    fn test_strip_noisy_lines_url_in_sentence() {
        let input = "Visit https://example.com for more info";
        let result = strip_noisy_lines(input);
        // URL in a sentence should be kept
        assert!(result.contains("https://example.com"));
    }

    #[test]
    fn test_clean_wikilinks_simple() {
        assert_eq!(
            clean_wikilinks("See [[My Page]] for details"),
            "See My Page for details"
        );
    }

    #[test]
    fn test_clean_wikilinks_display_text() {
        assert_eq!(
            clean_wikilinks("Check [[path/to/page|the page]]"),
            "Check the page"
        );
    }

    #[test]
    fn test_clean_wikilinks_path() {
        assert_eq!(clean_wikilinks("See [[folder/subfolder/Note]]"), "See Note");
    }

    #[test]
    fn test_clean_markdown_images() {
        assert_eq!(
            clean_markdown_images("Before ![alt text](http://example.com/img.png) after"),
            "Before alt text after"
        );
    }

    #[test]
    fn test_clean_markdown_images_no_alt() {
        assert_eq!(
            clean_markdown_images("Before ![](http://example.com/img.png) after"),
            "Before  after"
        );
    }

    #[test]
    fn test_clean_markdown_links() {
        assert_eq!(
            clean_markdown_links("Click [here](https://example.com) now"),
            "Click here now"
        );
    }

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(strip_html_tags("Hello <b>world</b>!"), "Hello world!");
    }

    #[test]
    fn test_collapse_whitespace() {
        assert_eq!(collapse_whitespace("a\n\n\n\n\nb"), "a\n\nb");
    }

    #[test]
    fn test_clean_markdown_full() {
        let input = r#"---
title: Test Doc
tags: [test]
---

# My Document

Check out [[some/path/Cool Plugin|Cool Plugin]] for more.

![badge](https://img.shields.io/github/stars/user/repo)

Here is a [link](https://example.com) to the docs.

https://standalone-url.com/page

The actual content that matters is here."#;

        let result = clean_markdown(input);

        // Frontmatter removed
        assert!(!result.contains("title: Test Doc"));
        // Badge removed
        assert!(!result.contains("shields.io"));
        // Wiki-link cleaned
        assert!(result.contains("Cool Plugin"));
        assert!(!result.contains("[["));
        // Markdown link cleaned
        assert!(result.contains("link"));
        assert!(!result.contains("https://example.com"));
        // Standalone URL removed
        assert!(!result.contains("standalone-url.com"));
        // Content preserved
        assert!(result.contains("actual content that matters"));
    }

    // ── Splitter Tests ──────────────────────────────────────────

    #[test]
    fn test_split_empty() {
        let splitter = TextSplitter::with_defaults();
        assert!(splitter.split("").is_empty());
    }

    #[test]
    fn test_split_short_text() {
        let splitter = TextSplitter::with_defaults();
        let chunks = splitter.split("Hello world.");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("Hello world"));
    }

    #[test]
    fn test_split_respects_chunk_size() {
        let config = SplitterConfig {
            chunk_size: 100,
            chunk_overlap: 0,
            clean_markdown: false,
        };
        let splitter = TextSplitter::new(config);
        let text = "A ".repeat(200); // 400 chars
        let chunks = splitter.split(&text);

        for chunk in &chunks {
            assert!(chunk.len() <= 200, "Chunk too large: {} chars", chunk.len());
        }
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_split_with_metadata() {
        let splitter = TextSplitter::with_defaults();
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "test.md".to_string());

        let results = splitter.split_with_metadata("Hello world.", meta);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1["source"], "test.md");
        assert_eq!(results[0].1["chunk_index"], "0");
        assert_eq!(results[0].1["total_chunks"], "1");
    }

    #[test]
    fn test_split_markdown_aware() {
        let config = SplitterConfig {
            chunk_size: 200,
            chunk_overlap: 0,
            clean_markdown: false,
        };
        let splitter = TextSplitter::new(config);

        let text = "# Header One\n\nFirst paragraph with some content.\n\n# Header Two\n\nSecond paragraph with different content.";
        let chunks = splitter.split(text);

        // Should split at markdown boundaries, not mid-sentence
        assert!(!chunks.is_empty());
        // Each chunk should be coherent
        for chunk in &chunks {
            assert!(!chunk.trim().is_empty());
        }
    }

    #[test]
    fn test_config_validation() {
        assert!(
            SplitterConfig {
                chunk_size: 0,
                chunk_overlap: 0,
                clean_markdown: true
            }
            .validate()
            .is_err()
        );
        assert!(
            SplitterConfig {
                chunk_size: 100,
                chunk_overlap: 100,
                clean_markdown: true
            }
            .validate()
            .is_err()
        );
        assert!(
            SplitterConfig {
                chunk_size: 100,
                chunk_overlap: 50,
                clean_markdown: true
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn test_estimate_chunks() {
        let splitter = TextSplitter::new(SplitterConfig {
            chunk_size: 100,
            chunk_overlap: 20,
            clean_markdown: true,
        });
        // 400 chars, effective size 80 → ~5 chunks
        let text = "x".repeat(400);
        let estimate = splitter.estimate_chunks(&text);
        assert!((4..=6).contains(&estimate), "estimate was {}", estimate);
    }
}
