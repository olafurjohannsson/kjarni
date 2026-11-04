//! Load a model folder's tokenizer, from `tokenizer.json` or the older files.
//!
//! Many models on the Hugging Face Hub predate `tokenizer.json` and ship only
//! the files their Python tokenizer reads: `vocab.json` and `merges.txt` for the
//! byte-level BPE of GPT-2, RoBERTa and BART, or `vocab.txt` for the WordPiece
//! of BERT. transformers builds a fast tokenizer from those files with its
//! `convert_slow_tokenizer` converters; [`load_tokenizer`] builds the same one,
//! so the token ids match `AutoTokenizer` for the same folder.
//!
//! When `tokenizer.json` exists it is used exactly as written. The other files
//! are read only when it is missing.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::byte_level::ByteLevel as ByteLevelProcessor;
use tokenizers::processors::roberta::RobertaProcessing;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Tokenizer};

/// The tokenizer files a model folder can have, `tokenizer.json` first.
pub const TOKENIZER_FILES: [&str; 3] = ["tokenizer.json", "vocab.json", "vocab.txt"];

/// Whether `dir` has a file a tokenizer can be loaded from.
pub fn has_tokenizer(dir: &Path) -> bool {
    dir.join("tokenizer.json").exists()
        || (dir.join("vocab.json").exists() && dir.join("merges.txt").exists())
        || dir.join("vocab.txt").exists()
}

/// Load the tokenizer of the model in `dir`.
///
/// Uses `tokenizer.json` when there is one. Without it, builds the tokenizer
/// transformers would build from `vocab.json` and `merges.txt` (GPT-2, RoBERTa,
/// BART, Longformer) or from `vocab.txt` (BERT, DistilBERT, ELECTRA, MPNet),
/// reading its settings from `tokenizer_config.json`, `special_tokens_map.json`,
/// `added_tokens.json` and `config.json`. Any other tokenizer is an error rather
/// than a guess, since a wrong guess gives wrong ids without failing.
pub fn load_tokenizer(dir: &Path) -> Result<Tokenizer> {
    let json = dir.join("tokenizer.json");
    if json.exists() {
        return Tokenizer::from_file(&json)
            .map_err(|e| anyhow!("Failed to load tokenizer {}: {e}", json.display()));
    }

    let settings = Settings::read(dir)?;
    let has_bpe = dir.join("vocab.json").exists() && dir.join("merges.txt").exists();
    let has_wordpiece = dir.join("vocab.txt").exists();
    if !has_bpe && !has_wordpiece {
        bail!(
            "No tokenizer in {}: it needs tokenizer.json, vocab.json with merges.txt, \
             or vocab.txt",
            dir.display()
        );
    }

    let kind = settings.kind().ok_or_else(|| {
        anyhow!(
            "{} has no tokenizer.json, and Kjarni can only build one for GPT-2, RoBERTa, \
             BART, Longformer, BERT, DistilBERT, ELECTRA and MPNet tokenizers, not {}. \
             Save one with transformers: \
             AutoTokenizer.from_pretrained(path).save_pretrained(path)",
            dir.display(),
            settings.describe()
        )
    })?;

    let mut tokenizer = match kind {
        Kind::Gpt2 | Kind::Roberta if has_bpe => build_bpe(dir, kind, &settings)?,
        Kind::Bert | Kind::MpNet if has_wordpiece => build_wordpiece(dir, kind, &settings)?,
        _ => bail!(
            "{} has no tokenizer.json, and the files a {} tokenizer is built from are missing",
            dir.display(),
            settings.describe()
        ),
    };
    add_tokens(&mut tokenizer, dir, kind, &settings)?;
    Ok(tokenizer)
}

/// The tokenizers [`load_tokenizer`] can build, each after one converter in
/// transformers' `convert_slow_tokenizer.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// `GPT2Converter`.
    Gpt2,
    /// `RobertaConverter`, which BART and Longformer use too.
    Roberta,
    /// `BertConverter`, which DistilBERT and ELECTRA use too.
    Bert,
    /// `MPNetConverter`.
    MpNet,
}

impl Kind {
    fn from_tokenizer_class(class: &str) -> Option<Self> {
        match class.trim_end_matches("Fast") {
            "GPT2Tokenizer" => Some(Self::Gpt2),
            "RobertaTokenizer" | "BartTokenizer" | "LongformerTokenizer" => Some(Self::Roberta),
            "BertTokenizer" | "DistilBertTokenizer" | "ElectraTokenizer" => Some(Self::Bert),
            "MPNetTokenizer" => Some(Self::MpNet),
            _ => None,
        }
    }

    fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "gpt2" => Some(Self::Gpt2),
            "roberta" | "bart" | "longformer" => Some(Self::Roberta),
            "bert" | "distilbert" | "electra" => Some(Self::Bert),
            "mpnet" => Some(Self::MpNet),
            _ => None,
        }
    }

    /// The special tokens the Python tokenizer's `__init__` defaults to, for
    /// each name `tokenizer_config.json` may leave out.
    fn default_special(self, name: &str) -> Option<&'static str> {
        match (self, name) {
            (Self::Gpt2, "bos_token" | "eos_token" | "unk_token") => Some("<|endoftext|>"),
            (Self::Roberta, "bos_token" | "cls_token") => Some("<s>"),
            (Self::Roberta, "eos_token" | "sep_token") => Some("</s>"),
            (Self::Roberta, "unk_token") => Some("<unk>"),
            (Self::Roberta, "pad_token") => Some("<pad>"),
            (Self::Roberta, "mask_token") => Some("<mask>"),
            (Self::Bert, "unk_token") => Some("[UNK]"),
            (Self::Bert, "sep_token") => Some("[SEP]"),
            (Self::Bert, "pad_token") => Some("[PAD]"),
            (Self::Bert, "cls_token") => Some("[CLS]"),
            (Self::Bert, "mask_token") => Some("[MASK]"),
            (Self::MpNet, "bos_token" | "cls_token") => Some("<s>"),
            (Self::MpNet, "eos_token" | "sep_token") => Some("</s>"),
            (Self::MpNet, "unk_token") => Some("[UNK]"),
            (Self::MpNet, "pad_token") => Some("<pad>"),
            (Self::MpNet, "mask_token") => Some("<mask>"),
            _ => None,
        }
    }
}

/// The special-token names, in the order transformers adds them.
const SPECIAL_NAMES: [&str; 7] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
];

/// What `tokenizer_config.json`, `special_tokens_map.json` and `config.json`
/// say, merged with the later file taking precedence, as transformers does.
struct Settings {
    tokenizer: serde_json::Map<String, Value>,
    model_type: Option<String>,
}

impl Settings {
    fn read(dir: &Path) -> Result<Self> {
        let mut tokenizer = read_object(&dir.join("tokenizer_config.json"))?;
        for (k, v) in read_object(&dir.join("special_tokens_map.json"))? {
            tokenizer.insert(k, v);
        }
        let model_type = read_object(&dir.join("config.json"))?
            .get("model_type")
            .and_then(Value::as_str)
            .map(str::to_owned);
        Ok(Self {
            tokenizer,
            model_type,
        })
    }

    fn tokenizer_class(&self) -> Option<&str> {
        self.tokenizer
            .get("tokenizer_class")
            .and_then(Value::as_str)
    }

    /// The tokenizer class decides; `config.json`'s model type is the fallback,
    /// as in `AutoTokenizer`.
    fn kind(&self) -> Option<Kind> {
        match self.tokenizer_class() {
            Some(class) => Kind::from_tokenizer_class(class),
            None => self.model_type.as_deref().and_then(Kind::from_model_type),
        }
    }

    fn describe(&self) -> String {
        match (self.tokenizer_class(), &self.model_type) {
            (Some(class), _) => format!("tokenizer class {class}"),
            (None, Some(t)) => format!("model type {t}"),
            (None, None) => "an unnamed tokenizer (no tokenizer_class or model_type)".into(),
        }
    }

    fn flag(&self, key: &str, default: bool) -> bool {
        self.tokenizer
            .get(key)
            .and_then(Value::as_bool)
            .unwrap_or(default)
    }

    /// `strip_accents` is three-valued: absent or null follows lowercasing.
    fn strip_accents(&self) -> Option<bool> {
        self.tokenizer.get("strip_accents").and_then(Value::as_bool)
    }

    /// A special token as an `AddedToken`, from the config or the class default.
    fn special(&self, kind: Kind, name: &str) -> Option<AddedToken> {
        let mut token = match self.tokenizer.get(name) {
            Some(Value::Null) => return None,
            Some(v) => added_token(v, true)?,
            None => AddedToken::from(kind.default_special(name)?, true),
        };
        // RoBERTa, BART, Longformer and MPNet make a mask token given as a plain
        // string strip the space before it, so "a <mask>" ends in one token.
        if name == "mask_token"
            && matches!(kind, Kind::Roberta | Kind::MpNet)
            && !matches!(self.tokenizer.get(name), Some(Value::Object(_)))
        {
            token = token.lstrip(true);
        }
        Some(token)
    }

    fn special_content(&self, kind: Kind, name: &str) -> Option<String> {
        self.special(kind, name).map(|t| t.content)
    }
}

fn read_object(path: &Path) -> Result<serde_json::Map<String, Value>> {
    if !path.exists() {
        return Ok(Default::default());
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    match serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))? {
        Value::Object(map) => Ok(map),
        _ => bail!("{} is not a JSON object", path.display()),
    }
}

/// An `AddedToken` from its JSON form: a plain string, or an object with
/// `content` and the flags transformers saves.
fn added_token(v: &Value, special: bool) -> Option<AddedToken> {
    match v {
        Value::String(s) => Some(AddedToken::from(s.clone(), special)),
        Value::Object(o) => {
            let content = o.get("content")?.as_str()?;
            let flag = |k: &str| o.get(k).and_then(Value::as_bool);
            let special = flag("special").unwrap_or(special);
            let mut t = AddedToken::from(content, special)
                .single_word(flag("single_word").unwrap_or(false))
                .lstrip(flag("lstrip").unwrap_or(false))
                .rstrip(flag("rstrip").unwrap_or(false));
            if let Some(n) = flag("normalized") {
                t = t.normalized(n);
            }
            Some(t)
        }
        _ => None,
    }
}

fn build_bpe(dir: &Path, kind: Kind, settings: &Settings) -> Result<Tokenizer> {
    let vocab = dir.join("vocab.json");
    let merges = dir.join("merges.txt");
    let bpe = BPE::from_file(&path_str(&vocab)?, &path_str(&merges)?)
        .continuing_subword_prefix(String::new())
        .end_of_word_suffix(String::new())
        .fuse_unk(false)
        .build()
        .map_err(|e| {
            anyhow!(
                "Failed to read {} and {}: {e}",
                vocab.display(),
                merges.display()
            )
        })?;

    let add_prefix_space = settings.flag("add_prefix_space", false);
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(add_prefix_space, true, true)));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

    match kind {
        Kind::Roberta => {
            let sep = settings
                .special_content(kind, "sep_token")
                .unwrap_or_default();
            let cls = settings
                .special_content(kind, "cls_token")
                .unwrap_or_default();
            let sep_id = vocab_id(&tokenizer, &sep)?;
            let cls_id = vocab_id(&tokenizer, &cls)?;
            tokenizer.with_post_processor(Some(
                RobertaProcessing::new((sep, sep_id), (cls, cls_id))
                    .trim_offsets(true)
                    .add_prefix_space(add_prefix_space),
            ));
        }
        _ if settings.flag("add_bos_token", false) => {
            let bos = settings
                .special_content(kind, "bos_token")
                .unwrap_or_default();
            let bos_id = vocab_id(&tokenizer, &bos)?;
            tokenizer.with_post_processor(Some(
                TemplateProcessing::builder()
                    .try_single(format!("{bos}:0 $A:0"))
                    .map_err(|e| anyhow!(e))?
                    .try_pair(format!("{bos}:0 $A:0 $B:1"))
                    .map_err(|e| anyhow!(e))?
                    .special_tokens(vec![(bos, bos_id)])
                    .build()?,
            ));
        }
        _ => {
            tokenizer.with_post_processor(Some(ByteLevelProcessor::default().trim_offsets(false)));
        }
    }
    Ok(tokenizer)
}

fn build_wordpiece(dir: &Path, kind: Kind, settings: &Settings) -> Result<Tokenizer> {
    let unk = settings
        .special_content(kind, "unk_token")
        .unwrap_or_else(|| "[UNK]".into());
    // Through serde, since the builder takes the crate's own map type.
    let wordpiece: WordPiece = serde_json::from_value(serde_json::json!({
        "vocab": read_vocab_txt(&dir.join("vocab.txt"))?,
        "unk_token": unk,
        "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100,
    }))
    .context("building WordPiece from vocab.txt")?;

    let lowercase = settings.flag("do_lower_case", true);
    let mut tokenizer = Tokenizer::new(wordpiece);
    tokenizer.with_normalizer(Some(BertNormalizer::new(
        true,
        settings.flag("tokenize_chinese_chars", true),
        settings.strip_accents(),
        lowercase,
    )));
    tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
    tokenizer.with_decoder(Some(WordPieceDecoder::new("##".into(), true)));

    let cls = settings
        .special_content(kind, "cls_token")
        .unwrap_or_default();
    let sep = settings
        .special_content(kind, "sep_token")
        .unwrap_or_default();
    let cls_id = vocab_id(&tokenizer, &cls)?;
    let sep_id = vocab_id(&tokenizer, &sep)?;
    let pair = match kind {
        Kind::MpNet => format!("{cls}:0 $A:0 {sep}:0 {sep}:0 $B:1 {sep}:1"),
        _ => format!("{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1"),
    };
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single(format!("{cls}:0 $A:0 {sep}:0"))
            .map_err(|e| anyhow!(e))?
            .try_pair(pair)
            .map_err(|e| anyhow!(e))?
            .special_tokens(vec![(cls, cls_id), (sep, sep_id)])
            .build()?,
    ));
    Ok(tokenizer)
}

/// `vocab.txt` as BERT's Python `load_vocab` reads it: one token per line, its
/// id the line number. Only the line break is removed; the `tokenizers` crate's
/// own reader trims all trailing whitespace, which would turn a vocabulary
/// entry such as an ideographic space into an empty token.
fn read_vocab_txt(path: &Path) -> Result<HashMap<String, u32>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    // Python opens the file in text mode, where "\r\n" and "\r" end lines too.
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut vocab = HashMap::new();
    for (index, token) in text.split_terminator('\n').enumerate() {
        vocab.insert(token.to_owned(), index as u32);
    }
    Ok(vocab)
}

/// Add the special tokens, then `added_tokens.json` and the tokens listed in
/// `tokenizer_config.json`'s `added_tokens_decoder`, as transformers does.
fn add_tokens(
    tokenizer: &mut Tokenizer,
    dir: &Path,
    kind: Kind,
    settings: &Settings,
) -> Result<()> {
    let mut specials: Vec<AddedToken> = SPECIAL_NAMES
        .iter()
        .filter_map(|name| settings.special(kind, name))
        .collect();
    if let Some(Value::Array(extra)) = settings.tokenizer.get("additional_special_tokens") {
        specials.extend(extra.iter().filter_map(|v| added_token(v, true)));
    }

    // added_tokens_decoder maps ids to tokens; added_tokens.json maps tokens to
    // ids. Either way, the ids say the order they were added in.
    let mut by_id: Vec<(u32, AddedToken)> = Vec::new();
    if let Some(Value::Object(decoder)) = settings.tokenizer.get("added_tokens_decoder") {
        for (id, v) in decoder {
            if let (Ok(id), Some(t)) = (id.parse::<u32>(), added_token(v, false)) {
                by_id.push((id, t));
            }
        }
    }
    for (content, id) in read_object(&dir.join("added_tokens.json"))? {
        if let Some(id) = id.as_u64() {
            by_id.push((id as u32, AddedToken::from(content, false)));
        }
    }
    by_id.sort_by_key(|(id, _)| *id);

    tokenizer.add_special_tokens(&specials);
    let (special, normal): (Vec<_>, Vec<_>) =
        by_id.into_iter().map(|(_, t)| t).partition(|t| t.special);
    tokenizer.add_special_tokens(&special);
    tokenizer.add_tokens(&normal);
    Ok(())
}

fn vocab_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or_else(|| anyhow!("special token {token:?} is not in the vocabulary"))
}

fn path_str(path: &Path) -> Result<String> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("path is not UTF-8: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The expected ids are what transformers 4.57's `AutoTokenizer` gives for
    /// the same files.
    const TEXTS: [&str; 3] = ["Hello, Wörld! the", "Hello world!", "Hello <mask>!"];

    const WORDPIECE_VOCAB: &str = "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n,\nw\n##or\n##ld\n\
        !\nwor\n##ld\nHello\nW\n##ö\n##rl\n##d\nö\no\n##rld\nworld\nthe\n";

    const MERGES: &str = "#version: 0.2\nH e\nl l\nHe ll\nHell o\nĠ w\no r\nĠw or\nl d\nĠwor ld\n";

    fn folder(files: &[(&str, String)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, content) in files {
            std::fs::write(dir.path().join(name), content).unwrap();
        }
        dir
    }

    fn ids(dir: &Path) -> Vec<Vec<u32>> {
        let tok = load_tokenizer(dir).unwrap();
        TEXTS
            .iter()
            .map(|t| tok.encode(*t, true).unwrap().get_ids().to_vec())
            .collect()
    }

    fn roberta_vocab() -> String {
        json!({"<s>":0,"<pad>":1,"</s>":2,"<unk>":3,"H":4,"e":5,"l":6,"o":7,"Ġ":8,"w":9,
               "r":10,"d":11,"He":12,"ll":13,"Hell":14,"Hello":15,"Ġw":16,"or":17,
               "Ġwor":18,"ld":19,"Ġworld":20,"!":21,"<mask>":22})
        .to_string()
    }

    #[test]
    fn bert_vocab_txt_lowercases_and_strips_accents_by_default() {
        let dir = folder(&[
            ("vocab.txt", WORDPIECE_VOCAB.into()),
            ("config.json", json!({"model_type": "bert"}).to_string()),
        ]);
        assert_eq!(
            ids(dir.path()),
            vec![
                vec![2, 5, 6, 21, 10, 22, 3],
                vec![2, 5, 21, 10, 3],
                vec![2, 5, 1, 1, 1, 10, 3],
            ]
        );
    }

    #[test]
    fn bert_vocab_txt_keeps_case_when_the_config_says_so() {
        let dir = folder(&[
            ("vocab.txt", WORDPIECE_VOCAB.into()),
            ("config.json", json!({"model_type": "bert"}).to_string()),
            (
                "tokenizer_config.json",
                json!({"do_lower_case": false, "tokenizer_class": "BertTokenizer"}).to_string(),
            ),
        ]);
        assert_eq!(
            ids(dir.path()),
            vec![
                vec![2, 13, 6, 14, 15, 20, 10, 22, 3],
                vec![2, 13, 21, 10, 3],
                vec![2, 13, 1, 1, 1, 10, 3],
            ]
        );
    }

    #[test]
    fn roberta_vocab_json_and_merges_wrap_in_s_and_match_mask() {
        let dir = folder(&[
            ("vocab.json", roberta_vocab()),
            ("merges.txt", MERGES.into()),
            ("config.json", json!({"model_type": "roberta"}).to_string()),
        ]);
        assert_eq!(
            ids(dir.path()),
            vec![
                vec![0, 15, 8, 10, 19, 21, 8, 5, 2],
                vec![0, 15, 20, 21, 2],
                vec![0, 15, 22, 21, 2],
            ]
        );
    }

    #[test]
    fn gpt2_vocab_json_and_merges_add_no_special_tokens() {
        let vocab = json!({"H":0,"e":1,"l":2,"o":3,"Ġ":4,"w":5,"r":6,"d":7,"He":8,"ll":9,
                           "Hell":10,"Hello":11,"Ġw":12,"or":13,"Ġwor":14,"ld":15,
                           "Ġworld":16,"!":17,"<|endoftext|>":18});
        let dir = folder(&[
            ("vocab.json", vocab.to_string()),
            ("merges.txt", MERGES.into()),
            ("config.json", json!({"model_type": "gpt2"}).to_string()),
        ]);
        assert_eq!(
            ids(dir.path()),
            vec![
                vec![11, 4, 6, 15, 17, 4, 1],
                vec![11, 16, 17],
                vec![11, 4, 17]
            ]
        );
    }

    #[test]
    fn tokenizer_json_is_used_when_present_even_beside_older_files() {
        // A tokenizer.json whose only token is "Hello", beside a vocab.txt that
        // would give different ids: the json must win.
        let json = json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "Hello": 7}, "unk_token": "[UNK]"}
        });
        let dir = folder(&[
            ("tokenizer.json", json.to_string()),
            ("vocab.txt", WORDPIECE_VOCAB.into()),
            ("config.json", json!({"model_type": "bert"}).to_string()),
        ]);
        let tok = load_tokenizer(dir.path()).unwrap();
        assert_eq!(tok.encode("Hello there", true).unwrap().get_ids(), &[7, 0]);
    }

    #[test]
    fn an_unknown_tokenizer_class_is_an_error_not_a_guess() {
        let dir = folder(&[
            ("vocab.json", roberta_vocab()),
            ("merges.txt", MERGES.into()),
            (
                "tokenizer_config.json",
                json!({"tokenizer_class": "CLIPTokenizer"}).to_string(),
            ),
        ]);
        let err = load_tokenizer(dir.path()).unwrap_err().to_string();
        assert!(err.contains("CLIPTokenizer"), "{err}");
        assert!(err.contains("save_pretrained"), "{err}");
    }

    #[test]
    fn a_folder_without_tokenizer_files_says_which_files_it_needs() {
        let dir = folder(&[("config.json", json!({"model_type": "bert"}).to_string())]);
        assert!(!has_tokenizer(dir.path()));
        let err = load_tokenizer(dir.path()).unwrap_err().to_string();
        assert!(
            err.contains("tokenizer.json, vocab.json with merges.txt, or vocab.txt"),
            "{err}"
        );
    }

    #[test]
    fn vocab_txt_keeps_whitespace_inside_entries() {
        let dir = folder(&[("vocab.txt", "[UNK]\n\u{3000}\r\nend \nx\n".into())]);
        let vocab = read_vocab_txt(&dir.path().join("vocab.txt")).unwrap();
        assert_eq!(vocab.get("\u{3000}"), Some(&1));
        assert_eq!(vocab.get("end "), Some(&2));
        assert_eq!(vocab.get("x"), Some(&3));
        assert_eq!(vocab.len(), 4);
    }
}
