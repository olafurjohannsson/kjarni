use colored::*;

/// Render a progress bar with color based on score
/// score: 0.0 - 1.0
/// width: total bar width in chars
pub fn score_bar(score: f32, width: usize) -> ColoredString {
    let filled = ((score * width as f32).round() as usize).min(width);
    let empty = width - filled;

    let bar = format!(
        "{}{}",
        "█".repeat(filled),
        "░".repeat(empty),
    );

    colorize_by_score(&bar, score)
}

/// Colorize text based on score threshold
pub fn colorize_by_score(text: &str, score: f32) -> ColoredString {
    if score >= 0.8 {
        text.green()
    } else if score >= 0.5 {
        text.yellow()
    } else if score >= 0.3 {
        text.truecolor(255, 165, 0) // orange
    } else {
        text.red()
    }
}

/// Format a score as percentage with color
pub fn score_pct(score: f32) -> ColoredString {
    let pct = format!("{:>5.1}%", score * 100.0);
    colorize_by_score(&pct, score)
}

/// Format a rank number with dimmed style
pub fn rank_label(rank: usize) -> ColoredString {
    format!("{:>3}.", rank).dimmed()
}

/// Truncate text and replace newlines, with dimmed style
pub fn snippet(text: &str, max_len: usize) -> ColoredString {
    let clean = text.replace('\n', " ").replace('\r', "");
    let truncated = if clean.len() > max_len {
        format!("{}…", &clean[..max_len - 1])
    } else {
        clean
    };
    truncated.dimmed()
}

/// Horizontal separator
pub fn separator(width: usize) -> ColoredString {
    "─".repeat(width).dimmed()
}

/// Interpret similarity score as human-readable label
pub fn similarity_label(score: f32) -> ColoredString {
    let label = if score > 0.9 {
        "near identical"
    } else if score > 0.7 {
        "highly similar"
    } else if score > 0.5 {
        "moderately similar"
    } else if score > 0.3 {
        "somewhat related"
    } else {
        "unrelated"
    };
    colorize_by_score(label, score)
}