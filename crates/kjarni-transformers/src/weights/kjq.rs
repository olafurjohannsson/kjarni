//! The `.kjq` container format.
//!
//! A single self-contained file holding a model's config, tokenizer and tensors,
//! with the tensors optionally stored as int8 plus a per-tensor scale. It exists
//! because shipping a model to a browser over the network is dominated by transfer
//! size: `all-MiniLM-L6-v2` is 88 MB as f32 safetensors and 23 MB as `.kjq`, and it
//! arrives as one request instead of three.
//!
//! Quantization here is a *transport* concern, not a runtime one. Tensors are
//! dequantized back to f32 while unpacking, so nothing downstream needs int8
//! kernels and numerical behaviour matches the f32 model exactly to within the
//! quantization error already baked into the file.
//!
//! ```text
//! magic     "KJQ1"                     4 bytes
//! config    u32 length + JSON
//! tokenizer u32 length + JSON
//! tensors   u32 count, then per tensor:
//!             u32 name length + name
//!             u32 rank + u32 per dimension
//!             u8  quantized flag
//!             if quantized: f32 scale + i8 per element
//!             else:         f32 per element
//! ```

use anyhow::{Result, anyhow};

/// A `.kjq` file unpacked into the pieces the loaders expect.
#[derive(Debug)]
pub struct KjqUnpacked {
    /// Tensors re-encoded as safetensors bytes.
    pub safetensors: Vec<u8>,
    /// The model's `config.json`.
    pub config_json: String,
    /// The model's `tokenizer.json`.
    pub tokenizer_json: String,
}

fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32> {
    if *cursor + 4 > data.len() {
        return Err(anyhow!("truncated .kjq: expected a u32 at offset {cursor}"));
    }
    let v = u32::from_le_bytes([
        data[*cursor],
        data[*cursor + 1],
        data[*cursor + 2],
        data[*cursor + 3],
    ]);
    *cursor += 4;
    Ok(v)
}

fn read_str(data: &[u8], cursor: &mut usize, what: &str) -> Result<String> {
    let len = read_u32(data, cursor)? as usize;
    if *cursor + len > data.len() {
        return Err(anyhow!("truncated .kjq: {what} claims {len} bytes"));
    }
    let s = std::str::from_utf8(&data[*cursor..*cursor + len])
        .map_err(|e| anyhow!("invalid UTF-8 in {what}: {e}"))?
        .to_string();
    *cursor += len;
    Ok(s)
}

/// Unpack a `.kjq` file.
///
/// Tensors are dequantized to f32 and re-encoded as safetensors so the result can
/// go through the same loading path as any other model. That costs one extra copy
/// of the weights in memory; a zero-copy `WeightLoader` reading `.kjq` directly
/// would avoid it, and can replace this without changing callers.
pub fn unpack(data: &[u8]) -> Result<KjqUnpacked> {
    use safetensors::Dtype;
    use safetensors::tensor::TensorView;

    let mut cursor = 0usize;

    if data.len() < 4 || &data[0..4] != b"KJQ1" {
        return Err(anyhow!(
            "not a .kjq file: expected magic \"KJQ1\", found {:?}",
            &data[..data.len().min(4)]
        ));
    }
    cursor += 4;

    let config_json = read_str(data, &mut cursor, "config")?;
    let tokenizer_json = read_str(data, &mut cursor, "tokenizer")?;

    let num_tensors = read_u32(data, &mut cursor)? as usize;

    // Names and shapes are kept alongside a flat byte buffer so the safetensors
    // views can borrow from one allocation rather than one per tensor.
    let mut meta: Vec<(String, Vec<usize>, usize, usize)> = Vec::with_capacity(num_tensors);
    let mut bytes: Vec<u8> = Vec::new();

    for i in 0..num_tensors {
        let name = read_str(data, &mut cursor, &format!("tensor {i} name"))?;

        let rank = read_u32(data, &mut cursor)? as usize;
        let mut shape = Vec::with_capacity(rank);
        let mut numel = 1usize;
        for _ in 0..rank {
            let dim = read_u32(data, &mut cursor)? as usize;
            numel = numel
                .checked_mul(dim)
                .ok_or_else(|| anyhow!("tensor '{name}' has an implausible element count"))?;
            shape.push(dim);
        }

        if cursor >= data.len() {
            return Err(anyhow!(
                "truncated .kjq: tensor '{name}' has no storage flag"
            ));
        }
        let quantized = data[cursor] != 0;
        cursor += 1;

        let start = bytes.len();

        if quantized {
            if cursor + 4 > data.len() {
                return Err(anyhow!("truncated .kjq: tensor '{name}' has no scale"));
            }
            let scale = f32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            cursor += 4;

            if cursor + numel > data.len() {
                return Err(anyhow!(
                    "truncated .kjq: tensor '{name}' needs {numel} bytes, {} remain",
                    data.len() - cursor
                ));
            }
            bytes.reserve(numel * 4);
            for k in 0..numel {
                let q = data[cursor + k] as i8;
                bytes.extend_from_slice(&(q as f32 * scale).to_le_bytes());
            }
            cursor += numel;
        } else {
            let byte_len = numel * 4;
            if cursor + byte_len > data.len() {
                return Err(anyhow!(
                    "truncated .kjq: tensor '{name}' needs {byte_len} bytes, {} remain",
                    data.len() - cursor
                ));
            }
            bytes.extend_from_slice(&data[cursor..cursor + byte_len]);
            cursor += byte_len;
        }

        meta.push((name, shape, start, bytes.len()));
    }

    let views: Vec<(&str, TensorView<'_>)> = meta
        .iter()
        .map(|(name, shape, start, end)| {
            TensorView::new(Dtype::F32, shape.clone(), &bytes[*start..*end])
                .map(|v| (name.as_str(), v))
                .map_err(|e| anyhow!("tensor '{name}' rejected by safetensors: {e}"))
        })
        .collect::<Result<_>>()?;

    let safetensors = safetensors::serialize(views, &None)
        .map_err(|e| anyhow!("failed to encode tensors as safetensors: {e}"))?;

    Ok(KjqUnpacked {
        safetensors,
        config_json,
        tokenizer_json,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a `.kjq` byte stream, mirroring whatever the exporter writes.
    fn build(
        config: &str,
        tokenizer: &str,
        tensors: &[(&str, Vec<usize>, TensorPayload)],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"KJQ1");
        out.extend_from_slice(&(config.len() as u32).to_le_bytes());
        out.extend_from_slice(config.as_bytes());
        out.extend_from_slice(&(tokenizer.len() as u32).to_le_bytes());
        out.extend_from_slice(tokenizer.as_bytes());
        out.extend_from_slice(&(tensors.len() as u32).to_le_bytes());

        for (name, shape, payload) in tensors {
            out.extend_from_slice(&(name.len() as u32).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for d in shape {
                out.extend_from_slice(&(*d as u32).to_le_bytes());
            }
            match payload {
                TensorPayload::F32(v) => {
                    out.push(0);
                    for f in v {
                        out.extend_from_slice(&f.to_le_bytes());
                    }
                }
                TensorPayload::Q8 { scale, values } => {
                    out.push(1);
                    out.extend_from_slice(&scale.to_le_bytes());
                    for q in values {
                        out.push(*q as u8);
                    }
                }
            }
        }
        out
    }

    enum TensorPayload {
        F32(Vec<f32>),
        Q8 { scale: f32, values: Vec<i8> },
    }

    fn tensor_f32(unpacked: &KjqUnpacked, name: &str) -> Vec<f32> {
        let st = safetensors::SafeTensors::deserialize(&unpacked.safetensors).unwrap();
        let view = st.tensor(name).unwrap();
        view.data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    #[test]
    fn unpacks_config_and_tokenizer() {
        let bytes = build(r#"{"hidden_size":8}"#, r#"{"version":"1.0"}"#, &[]);
        let out = unpack(&bytes).unwrap();

        assert_eq!(out.config_json, r#"{"hidden_size":8}"#);
        assert_eq!(out.tokenizer_json, r#"{"version":"1.0"}"#);
    }

    #[test]
    fn passes_unquantized_tensors_through_unchanged() {
        let values = vec![1.5f32, -2.25, 0.0, 4.75];
        let bytes = build(
            "{}",
            "{}",
            &[("a", vec![2, 2], TensorPayload::F32(values.clone()))],
        );

        let out = unpack(&bytes).unwrap();
        assert_eq!(tensor_f32(&out, "a"), values);
    }

    #[test]
    fn dequantizes_int8_tensors_by_their_scale() {
        // The point of the format: int8 on the wire, f32 in memory.
        let scale = 0.5f32;
        let bytes = build(
            "{}",
            "{}",
            &[(
                "w",
                vec![4],
                TensorPayload::Q8 {
                    scale,
                    values: vec![2, -4, 0, 127],
                },
            )],
        );

        let out = unpack(&bytes).unwrap();
        assert_eq!(tensor_f32(&out, "w"), vec![1.0, -2.0, 0.0, 63.5]);
    }

    #[test]
    fn preserves_shape_and_multiple_tensors() {
        let bytes = build(
            "{}",
            "{}",
            &[
                ("first", vec![2, 3], TensorPayload::F32(vec![0.0; 6])),
                (
                    "second",
                    vec![3],
                    TensorPayload::Q8 {
                        scale: 1.0,
                        values: vec![1, 2, 3],
                    },
                ),
            ],
        );

        let out = unpack(&bytes).unwrap();
        let st = safetensors::SafeTensors::deserialize(&out.safetensors).unwrap();

        assert_eq!(st.tensor("first").unwrap().shape(), &[2, 3]);
        assert_eq!(st.tensor("second").unwrap().shape(), &[3]);
        assert_eq!(tensor_f32(&out, "second"), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn rejects_a_file_that_is_not_kjq() {
        let err = unpack(b"NOPE and then some").unwrap_err().to_string();
        assert!(err.contains("KJQ1"), "unhelpful error: {err}");
    }

    #[test]
    fn rejects_truncation_rather_than_reading_past_the_end() {
        let full = build(
            "{}",
            "{}",
            &[("w", vec![4], TensorPayload::F32(vec![1.0, 2.0, 3.0, 4.0]))],
        );

        // Chop the last tensor's data in half.
        let truncated = &full[..full.len() - 8];
        let err = unpack(truncated).unwrap_err().to_string();
        assert!(err.contains("truncated"), "unhelpful error: {err}");
    }
}
