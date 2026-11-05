//! SentencePiece `.model` files, the vocabulary of XLM-RoBERTa, CamemBERT, T5
//! and Llama 2 style tokenizers.
//!
//! A `.model` file is a protobuf `ModelProto`. Only the fields transformers'
//! converters read are decoded; the rest are skipped by wire type, so a file
//! written by a newer SentencePiece still loads.

use anyhow::{Result, anyhow, bail};

/// The pieces of a SentencePiece model and the settings the converters use.
#[derive(Debug, Clone)]
pub(crate) struct SpmModel {
    pub pieces: Vec<Piece>,
    pub model_type: ModelType,
    pub byte_fallback: bool,
    pub unk_id: i32,
    pub unk_piece: String,
    /// The normalization table, in the format `tokenizers`' `Precompiled` reads.
    pub precompiled_charsmap: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Piece {
    pub piece: String,
    pub score: f32,
    pub kind: PieceType,
}

/// `TrainerSpec.ModelType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelType {
    Unigram,
    Bpe,
    Other(u64),
}

/// `SentencePiece.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Other(u64),
}

impl PieceType {
    fn from_wire(v: u64) -> Self {
        match v {
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            other => Self::Other(other),
        }
    }
}

impl SpmModel {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut model = SpmModel {
            pieces: Vec::new(),
            model_type: ModelType::Unigram,
            byte_fallback: false,
            unk_id: 0,
            unk_piece: "<unk>".into(),
            precompiled_charsmap: Vec::new(),
        };
        let mut r = Reader::new(bytes);
        while let Some((field, value)) = r.next_field()? {
            match (field, value) {
                (1, Wire::Bytes(b)) => model.pieces.push(parse_piece(b)?),
                (2, Wire::Bytes(b)) => parse_trainer_spec(b, &mut model)?,
                (3, Wire::Bytes(b)) => parse_normalizer_spec(b, &mut model)?,
                _ => {}
            }
        }
        if model.pieces.is_empty() {
            bail!("the SentencePiece model has no pieces");
        }
        Ok(model)
    }
}

fn parse_piece(bytes: &[u8]) -> Result<Piece> {
    let mut piece = Piece {
        piece: String::new(),
        score: 0.0,
        kind: PieceType::Normal,
    };
    let mut r = Reader::new(bytes);
    while let Some((field, value)) = r.next_field()? {
        match (field, value) {
            (1, Wire::Bytes(b)) => {
                piece.piece = String::from_utf8(b.to_vec())
                    .map_err(|_| anyhow!("a SentencePiece piece is not UTF-8"))?
            }
            (2, Wire::Fixed32(v)) => piece.score = f32::from_bits(v),
            (3, Wire::Varint(v)) => piece.kind = PieceType::from_wire(v),
            _ => {}
        }
    }
    Ok(piece)
}

fn parse_trainer_spec(bytes: &[u8], model: &mut SpmModel) -> Result<()> {
    let mut r = Reader::new(bytes);
    while let Some((field, value)) = r.next_field()? {
        match (field, value) {
            (3, Wire::Varint(v)) => {
                model.model_type = match v {
                    1 => ModelType::Unigram,
                    2 => ModelType::Bpe,
                    other => ModelType::Other(other),
                }
            }
            (35, Wire::Varint(v)) => model.byte_fallback = v != 0,
            // int32: negative values are sign-extended to 64 bits on the wire.
            (40, Wire::Varint(v)) => model.unk_id = v as i64 as i32,
            (45, Wire::Bytes(b)) => model.unk_piece = String::from_utf8_lossy(b).into_owned(),
            _ => {}
        }
    }
    Ok(())
}

fn parse_normalizer_spec(bytes: &[u8], model: &mut SpmModel) -> Result<()> {
    let mut r = Reader::new(bytes);
    while let Some((field, value)) = r.next_field()? {
        if let (2, Wire::Bytes(b)) = (field, value) {
            model.precompiled_charsmap = b.to_vec();
        }
    }
    Ok(())
}

enum Wire<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self
                .bytes
                .get(self.pos)
                .ok_or_else(|| anyhow!("truncated SentencePiece model"))?;
            self.pos += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        bail!("malformed varint in SentencePiece model")
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| anyhow!("truncated SentencePiece model"))?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn next_field(&mut self) -> Result<Option<(u64, Wire<'a>)>> {
        if self.pos >= self.bytes.len() {
            return Ok(None);
        }
        let key = self.varint()?;
        let value = match key & 7 {
            0 => Wire::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Wire::Fixed64
            }
            2 => {
                let len = usize::try_from(self.varint()?)?;
                Wire::Bytes(self.take(len)?)
            }
            5 => {
                let b = self.take(4)?;
                Wire::Fixed32(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
            t => bail!("unsupported protobuf wire type {t} in SentencePiece model"),
        };
        Ok(Some((key >> 3, value)))
    }
}

/// The merges transformers' `generate_merges` derives from a BPE model's
/// pieces: every split of a piece into two pieces of `vocab`, ordered by the
/// merged piece's score, highest first, then by the lengths of the halves.
///
/// `vocab` maps each piece to its id in the `.model` file; `scored` is the
/// converter's vocabulary in id order.
pub(crate) fn generate_merges(
    vocab: &std::collections::HashMap<String, u32>,
    scored: &[(String, f64)],
) -> Vec<(String, String)> {
    // dict(vocab_scores): a repeated piece keeps its first position and last score.
    let mut order: Vec<&str> = Vec::new();
    let mut score: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
    for (piece, s) in scored {
        if score.insert(piece.as_str(), *s).is_none() {
            order.push(piece.as_str());
        }
    }

    let mut merges: Vec<(&str, &str, f64)> = Vec::new();
    for merged in order {
        let s = score[merged];
        let mut local: Vec<(&str, &str)> = merged
            .char_indices()
            .skip(1)
            .map(|(i, _)| merged.split_at(i))
            .filter(|(l, r)| vocab.contains_key(*l) && vocab.contains_key(*r))
            .collect();
        local.sort_by_key(|(l, r)| (vocab[*l], vocab[*r]));
        merges.extend(local.into_iter().map(|(l, r)| (l, r, s)));
    }
    // Python's sorted(..., reverse=True) is stable: ties keep their order.
    merges.sort_by(|a, b| {
        let key = |m: &(&str, &str, f64)| (m.2, m.0.chars().count(), m.1.chars().count());
        let (ka, kb) = (key(a), key(b));
        kb.0.total_cmp(&ka.0)
            .then(kb.1.cmp(&ka.1))
            .then(kb.2.cmp(&ka.2))
    });
    merges
        .into_iter()
        .map(|(l, r, _)| (l.to_owned(), r.to_owned()))
        .collect()
}

/// Writes the protobuf of a `.model` file, for tests.
#[cfg(test)]
pub(crate) mod proto {
    pub fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let byte = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    pub fn bytes_field(field: u64, b: &[u8], out: &mut Vec<u8>) {
        varint(field << 3 | 2, out);
        varint(b.len() as u64, out);
        out.extend_from_slice(b);
    }

    pub fn piece(text: &str, score: f32, kind: u64) -> Vec<u8> {
        let mut p = Vec::new();
        bytes_field(1, text.as_bytes(), &mut p);
        varint(2 << 3 | 5, &mut p);
        p.extend_from_slice(&score.to_le_bytes());
        varint(3 << 3, &mut p);
        varint(kind, &mut p);
        p
    }

    /// A Unigram model of `pieces`, (text, score, type), with no
    /// normalization table.
    pub fn unigram_model(pieces: &[(&str, f32, u64)]) -> Vec<u8> {
        let mut m = Vec::new();
        for (text, score, kind) in pieces {
            bytes_field(1, &piece(text, *score, *kind), &mut m);
        }
        let mut trainer = Vec::new();
        varint(3 << 3, &mut trainer);
        varint(1, &mut trainer);
        bytes_field(2, &trainer, &mut m);
        m
    }
}

#[cfg(test)]
mod tests {
    use super::proto::*;
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn parses_pieces_and_the_settings_converters_read() {
        let mut m = Vec::new();
        bytes_field(1, &piece("<unk>", 0.0, 2), &mut m);
        bytes_field(1, &piece("▁hi", -1.5, 1), &mut m);
        let mut trainer = Vec::new();
        varint(3 << 3, &mut trainer);
        varint(2, &mut trainer);
        varint(35 << 3, &mut trainer);
        varint(1, &mut trainer);
        // unk_id = -1 is a ten-byte varint.
        varint(40 << 3, &mut trainer);
        varint(-1i64 as u64, &mut trainer);
        // A field the reader does not know, of each wire type, is skipped.
        varint(99 << 3 | 1, &mut trainer);
        trainer.extend_from_slice(&[0; 8]);
        bytes_field(2, &trainer, &mut m);
        let mut norm = Vec::new();
        bytes_field(2, &[1, 2, 3], &mut norm);
        bytes_field(3, &norm, &mut m);

        let model = SpmModel::parse(&m).unwrap();
        assert_eq!(model.pieces.len(), 2);
        assert_eq!(model.pieces[1].piece, "▁hi");
        assert_eq!(model.pieces[1].score, -1.5);
        assert_eq!(model.pieces[0].kind, PieceType::Unknown);
        assert_eq!(model.model_type, ModelType::Bpe);
        assert!(model.byte_fallback);
        assert_eq!(model.unk_id, -1);
        assert_eq!(model.precompiled_charsmap, vec![1, 2, 3]);
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_panic() {
        let mut m = Vec::new();
        bytes_field(1, &piece("abc", 0.0, 1), &mut m);
        assert!(SpmModel::parse(&m[..m.len() - 3]).is_err());
        assert!(SpmModel::parse(&[]).is_err());
    }

    /// Expected merges from transformers 4.57's `generate_merges` on the same input.
    #[test]
    fn merges_follow_scores_then_lengths_like_transformers() {
        let pieces = ["<unk>", "a", "b", "c", "ab", "bc", "abc", "▁", "▁a"];
        let vocab: HashMap<String, u32> = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.to_string(), i as u32))
            .collect();
        let scored: Vec<(String, f64)> = pieces
            .iter()
            .zip([0.0, -1.0, -2.0, -3.0, -0.5, -0.5, -0.25, -4.0, -0.75])
            .map(|(p, s)| (p.to_string(), s))
            .collect();
        let merges = generate_merges(&vocab, &scored);
        let got: Vec<String> = merges.iter().map(|(l, r)| format!("{l} {r}")).collect();
        assert_eq!(got, ["ab c", "a bc", "a b", "b c", "▁ a"]);
    }
}
