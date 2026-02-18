//! Text splitting with markdown-aware chunking and preprocessing
use std::collections::HashMap;
use text_splitter::MarkdownSplitter;

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
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
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
        if text.is_empty() {
            return vec![];
        }

        // Preprocess: clean markdown noise
        let cleaned = if self.config.clean_markdown {
            clean_markdown(text)
        } else {
            text.to_string()
        };

        if cleaned.trim().is_empty() {
            return vec![];
        }

        let splitter = MarkdownSplitter::new(self.config.chunk_size);

        splitter
            .chunks(&cleaned)
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
        let effective = self.config.chunk_size.saturating_sub(self.config.chunk_overlap).max(1);
        let char_count = text.chars().count();
        (char_count + effective - 1) / effective
    }
}

pub fn clean_markdown(text: &str) -> String {
    let text = strip_frontmatter(text);
    let text = strip_noisy_lines(&text);
    let text = clean_wikilinks(&text);
    let text = clean_markdown_images(&text);
    let text = clean_markdown_links(&text);
    let text = strip_html_tags(&text);
    let text = collapse_whitespace(&text);
    text
}

/// Remove YAML frontmatter (--- ... ---)
fn strip_frontmatter(text: &str) -> String {
    let trimmed = text.trim_start();
    if !trimmed.starts_with("---") {
        return text.to_string();
    }

    // Find the closing ---
    let after_first = &trimmed[3..];
    if let Some(end_pos) = after_first.find("\n---") {
        // Skip past the closing --- and any trailing newline
        let rest = &after_first[end_pos + 4..];
        let rest = rest.strip_prefix('\n').unwrap_or(rest);
        rest.to_string()
    } else {
        // No closing --- found, return as-is
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
fn clean_wikilinks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Look for [[
        if i + 1 < len && chars[i] == '[' && chars[i + 1] == '[' {
            i += 2; // skip [[

            // Collect everything until ]]
            let mut link_content = String::new();
            let mut found_close = false;

            while i + 1 < len {
                if chars[i] == ']' && chars[i + 1] == ']' {
                    i += 2; // skip ]]
                    found_close = true;
                    break;
                }
                link_content.push(chars[i]);
                i += 1;
            }

            if found_close {
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
            } else {
                // Malformed — output as-is
                result.push_str("[[");
                result.push_str(&link_content);
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Convert markdown images to just alt text
///
/// `![alt text](url)` → `alt text`
/// `![](url)` → (removed)
fn clean_markdown_images(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if i + 1 < len && chars[i] == '!' && chars[i + 1] == '[' {
            i += 2; // skip ![

            // Collect alt text until ]
            let mut alt = String::new();
            while i < len && chars[i] != ']' {
                alt.push(chars[i]);
                i += 1;
            }

            if i < len {
                i += 1; // skip ]
            }

            // Skip (url) if present
            if i < len && chars[i] == '(' {
                i += 1;
                let mut depth = 1;
                while i < len && depth > 0 {
                    if chars[i] == '(' {
                        depth += 1;
                    } else if chars[i] == ')' {
                        depth -= 1;
                    }
                    i += 1;
                }
            }

            // Keep alt text if non-empty
            if !alt.trim().is_empty() {
                result.push_str(alt.trim());
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Convert markdown links to just display text
///
/// `[display](url)` → `display`
fn clean_markdown_links(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // [display](url) — but not ![image](url) which is already handled
        if chars[i] == '[' && (i == 0 || chars[i - 1] != '!') {
            i += 1; // skip [

            // Collect display text until ]
            let mut display = String::new();
            let mut depth = 1;
            while i < len && depth > 0 {
                if chars[i] == '[' {
                    depth += 1;
                } else if chars[i] == ']' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                display.push(chars[i]);
                i += 1;
            }

            if i < len {
                i += 1; // skip ]
            }

            // Check if followed by (url)
            if i < len && chars[i] == '(' {
                i += 1; // skip (
                let mut paren_depth = 1;
                while i < len && paren_depth > 0 {
                    if chars[i] == '(' {
                        paren_depth += 1;
                    } else if chars[i] == ')' {
                        paren_depth -= 1;
                    }
                    i += 1;
                }
                // Output just the display text
                result.push_str(&display);
            } else {
                // Not a link, output as-is
                result.push('[');
                result.push_str(&display);
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Strip HTML tags, keep inner text
fn strip_html_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_tag = false;

    for ch in text.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(ch);
        }
    }

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
        assert_eq!(clean_wikilinks("See [[My Page]] for details"), "See My Page for details");
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
        assert_eq!(
            clean_wikilinks("See [[folder/subfolder/Note]]"),
            "See Note"
        );
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
        assert!(chunks.len() >= 1);
        // Each chunk should be coherent
        for chunk in &chunks {
            assert!(!chunk.trim().is_empty());
        }
    }

    #[test]
    fn test_config_validation() {
        assert!(SplitterConfig { chunk_size: 0, chunk_overlap: 0, clean_markdown: true }.validate().is_err());
        assert!(SplitterConfig { chunk_size: 100, chunk_overlap: 100, clean_markdown: true }.validate().is_err());
        assert!(SplitterConfig { chunk_size: 100, chunk_overlap: 50, clean_markdown: true }.validate().is_ok());
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
        assert!(estimate >= 4 && estimate <= 6, "estimate was {}", estimate);
    }
}