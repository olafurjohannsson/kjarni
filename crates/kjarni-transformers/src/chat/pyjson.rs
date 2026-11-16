//! JSON written as Python's `json.dumps` writes it, which is what chat
//! templates put in a prompt when they serialize tool definitions.

use serde_json::Value as Json;

pub(crate) struct PyJson {
    pub ensure_ascii: bool,
    pub indent: Option<String>,
    pub item_sep: String,
    pub key_sep: String,
    pub sort_keys: bool,
}

impl PyJson {
    /// `json.dumps(v, ensure_ascii=False)`: one line, `", "` and `": "`
    /// between items, keys in the map's order.
    pub fn dumps(v: &Json) -> String {
        let style = Self {
            ensure_ascii: false,
            indent: None,
            item_sep: ", ".into(),
            key_sep: ": ".into(),
            sort_keys: false,
        };
        let mut out = String::new();
        style.write(v, 0, &mut out);
        out
    }

    pub fn write(&self, v: &Json, level: usize, out: &mut String) {
        match v {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Number(n) => out.push_str(&py_number(n)),
            Json::String(s) => self.string(s, out),
            Json::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&self.item_sep);
                    }
                    self.newline(level + 1, out);
                    self.write(item, level + 1, out);
                }
                self.newline(level, out);
                out.push(']');
            }
            Json::Object(map) => {
                if map.is_empty() {
                    out.push_str("{}");
                    return;
                }
                let mut entries: Vec<(&String, &Json)> = map.iter().collect();
                if self.sort_keys {
                    entries.sort_by(|a, b| a.0.cmp(b.0));
                }
                out.push('{');
                for (i, (k, item)) in entries.into_iter().enumerate() {
                    if i > 0 {
                        out.push_str(&self.item_sep);
                    }
                    self.newline(level + 1, out);
                    self.string(k, out);
                    out.push_str(&self.key_sep);
                    self.write(item, level + 1, out);
                }
                self.newline(level, out);
                out.push('}');
            }
        }
    }

    fn newline(&self, level: usize, out: &mut String) {
        if let Some(indent) = &self.indent {
            out.push('\n');
            for _ in 0..level {
                out.push_str(indent);
            }
        }
    }

    fn string(&self, s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{8}' => out.push_str("\\b"),
                '\u{c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c if self.ensure_ascii && (c as u32) > 0x7e => {
                    let mut buf = [0u16; 2];
                    for unit in c.encode_utf16(&mut buf) {
                        out.push_str(&format!("\\u{unit:04x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
}

/// A number as Python writes it: integers as they are, floats by `repr`.
fn py_number(n: &serde_json::Number) -> String {
    if n.is_f64() {
        let f = n.as_f64().unwrap_or(0.0);
        if f.is_finite() && f == f.trunc() && f.abs() < 1e16 {
            format!("{f:.1}")
        } else {
            let s = format!("{f:?}");
            // Rust writes 1e16 where Python writes 1e+16.
            match s.split_once('e') {
                Some((m, e)) if !e.starts_with('-') => format!("{m}e+{e}"),
                _ => s,
            }
        }
    } else {
        n.to_string()
    }
}
