//! Find named entities: people, places, organizations and the like.

use anyhow::{Result, anyhow};
use colored::*;
use kjarni::ner::{Aggregation, Entity, EntityRecognizer};
use serde_json::json;

pub async fn run(
    input: &[String],
    model: &str,
    model_path: Option<&str>,
    aggregation: &str,
    format: &str,
    gpu: bool,
    quiet: bool,
) -> Result<()> {
    let aggregation = parse_aggregation(aggregation)?;
    if model_path.is_none() {
        let _ = crate::commands::util::resolve_model(model, Some("entity recognizer"))?;
    }

    let text = if input.is_empty() {
        crate::commands::util::resolve_input(None)?
    } else {
        input.join(" ")
    };
    // One text per line, as classify reads them.
    let texts: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if texts.is_empty() {
        return Err(anyhow!("No input text provided."));
    }

    let mut builder = match model_path {
        Some(path) => EntityRecognizer::from_path(path),
        None => EntityRecognizer::builder(model),
    };
    builder = builder.aggregation(aggregation).quiet(quiet);
    if gpu {
        builder = builder.gpu();
    }
    let ner = builder.build().await?;
    let results = ner.recognize_batch(&texts).await?;

    print!("{}", format_results(&texts, &results, format)?);
    Ok(())
}

fn parse_aggregation(s: &str) -> Result<Aggregation> {
    match s.to_lowercase().as_str() {
        "first" => Ok(Aggregation::First),
        "simple" => Ok(Aggregation::Simple),
        "max" => Ok(Aggregation::Max),
        "average" => Ok(Aggregation::Average),
        other => Err(anyhow!(
            "Unknown aggregation '{other}'. Use first, simple, max or average."
        )),
    }
}

fn entity_json(e: &Entity) -> serde_json::Value {
    json!({
        "label": e.label,
        "text": e.text,
        "start": e.start,
        "end": e.end,
        "score": e.score,
    })
}

fn format_results(texts: &[&str], results: &[Vec<Entity>], format: &str) -> Result<String> {
    let mut out = String::new();
    match format {
        "json" => {
            let items: Vec<_> = texts
                .iter()
                .zip(results)
                .map(|(t, es)| json!({"text": t, "entities": es.iter().map(entity_json).collect::<Vec<_>>()}))
                .collect();
            let value = if items.len() == 1 {
                items.into_iter().next().unwrap()
            } else {
                serde_json::Value::Array(items)
            };
            out.push_str(&serde_json::to_string_pretty(&value)?);
            out.push('\n');
        }
        "jsonl" => {
            for (t, es) in texts.iter().zip(results) {
                let line =
                    json!({"text": t, "entities": es.iter().map(entity_json).collect::<Vec<_>>()});
                out.push_str(&serde_json::to_string(&line)?);
                out.push('\n');
            }
        }
        "text" => {
            for (i, (t, es)) in texts.iter().zip(results).enumerate() {
                if texts.len() > 1 {
                    if i > 0 {
                        out.push('\n');
                    }
                    out.push_str(&format!("  {}\n", format!("\"{}\"", t).dimmed()));
                }
                if es.is_empty() {
                    out.push_str(&format!("  {}\n", "no entities".dimmed()));
                    continue;
                }
                let width = es
                    .iter()
                    .map(|e| e.label.chars().count())
                    .max()
                    .unwrap_or(0);
                for e in es {
                    out.push_str(&format!(
                        "  {:<width$}  {}  {}\n",
                        e.label.bold(),
                        e.text,
                        format!("{:.1}%", e.score * 100.0).dimmed(),
                        width = width
                    ));
                }
            }
        }
        other => {
            return Err(anyhow!(
                "Unknown format '{other}'. Use text, json or jsonl."
            ));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(label: &str, text: &str, start: usize, end: usize, score: f32) -> Entity {
        Entity {
            label: label.into(),
            text: text.into(),
            start,
            end,
            score,
        }
    }

    #[test]
    fn jsonl_writes_one_line_per_text_with_character_spans() {
        let texts = ["Þórður býr í Reykjavík.", "nothing here"];
        let results = vec![
            vec![
                entity("PER", "Þórður", 0, 6, 0.99),
                entity("LOC", "Reykjavík", 13, 22, 0.98),
            ],
            vec![],
        ];
        let out = format_results(&texts, &results, "jsonl").unwrap();
        let lines: Vec<serde_json::Value> = out
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["entities"][1]["text"], "Reykjavík");
        assert_eq!(lines[0]["entities"][1]["start"], 13);
        assert_eq!(lines[1]["entities"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn json_of_one_text_is_an_object_not_a_list() {
        let out = format_results(&["a"], &[vec![]], "json").unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert!(v.is_object());
    }

    #[test]
    fn text_lists_each_entity_with_its_score() {
        colored::control::set_override(false);
        let out =
            format_results(&["x"], &[vec![entity("ORG", "Apple", 0, 5, 0.997)]], "text").unwrap();
        assert_eq!(out, "  ORG  Apple  99.7%\n");
    }

    #[test]
    fn aggregation_names_are_checked() {
        assert_eq!(parse_aggregation("First").unwrap(), Aggregation::First);
        assert!(parse_aggregation("word").is_err());
    }

    #[test]
    fn an_unknown_format_is_an_error() {
        assert!(format_results(&["a"], &[vec![]], "csv").is_err());
    }
}
