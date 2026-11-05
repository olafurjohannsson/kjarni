//! Load a model folder's tokenizer, from `tokenizer.json` or the older files.
//!
//! Many models on the Hugging Face Hub predate `tokenizer.json` and ship only
//! the files their Python tokenizer reads: `vocab.json` and `merges.txt` for the
//! byte-level BPE of GPT-2, RoBERTa and BART, `vocab.txt` for the WordPiece of
//! BERT, or a SentencePiece `.model` file for XLM-RoBERTa, CamemBERT, T5 and
//! Llama 2 style tokenizers. transformers builds a fast tokenizer from those
//! files with its `convert_slow_tokenizer` converters; [`load_tokenizer`] builds
//! the same one, so the token ids match `AutoTokenizer` (4.57) for the same
//! folder.
//!
//! When `tokenizer.json` exists it is used exactly as written. The other files
//! are read only when it is missing.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::decoders::sequence::Sequence as DecoderSequence;
use tokenizers::decoders::strip::Strip as StripDecoder;
use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::models::unigram::Unigram;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::normalizers::{
    NormalizerWrapper, Precompiled, Prepend, Replace, Sequence as NormalizerSequence, Strip,
};
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::processors::byte_level::ByteLevel as ByteLevelProcessor;
use tokenizers::processors::roberta::RobertaProcessing;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Tokenizer};

use super::sentencepiece::{self, PieceType, SpmModel};

/// The tokenizer files a model folder can have, `tokenizer.json` first.
/// `vocab.json` is used together with `merges.txt`.
pub const TOKENIZER_FILES: &[&str] = &[
    "tokenizer.json",
    "vocab.json",
    "vocab.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "tokenizer.model",
];

/// The SentencePiece files, by the name each Python tokenizer class gives it.
const SPM_FILES: [&str; 3] = ["sentencepiece.bpe.model", "spiece.model", "tokenizer.model"];

/// Whether `dir` has a file a tokenizer can be loaded from.
pub fn has_tokenizer(dir: &Path) -> bool {
    dir.join("tokenizer.json").exists()
        || (dir.join("vocab.json").exists() && dir.join("merges.txt").exists())
        || dir.join("vocab.txt").exists()
        || spm_file(dir).is_some()
}

fn spm_file(dir: &Path) -> Option<std::path::PathBuf> {
    SPM_FILES.iter().map(|f| dir.join(f)).find(|p| p.exists())
}

/// The tokenizer a model loader uses: [`load_tokenizer`], with a published
/// Unigram `tokenizer.json` made to drop whitespace as transformers does (see
/// [`super::match_sentencepiece_whitespace`]). A tokenizer built from a
/// `.model` file already matches transformers and is used as built.
pub fn load_model_tokenizer(dir: &Path) -> Result<Tokenizer> {
    let mut tokenizer = load_tokenizer(dir)?;
    if dir.join("tokenizer.json").exists() {
        super::match_sentencepiece_whitespace(&mut tokenizer);
    }
    Ok(tokenizer)
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
    let spm = spm_file(dir);
    if !has_bpe && !has_wordpiece && spm.is_none() {
        bail!(
            "No tokenizer in {}: it needs tokenizer.json, vocab.json with merges.txt, \
             vocab.txt, or a SentencePiece .model file",
            dir.display()
        );
    }

    let kind = settings.kind().ok_or_else(|| {
        anyhow!(
            "{} has no tokenizer.json, and Kjarni can only build one for GPT-2, RoBERTa, \
             BART, Longformer, BERT, DistilBERT, ELECTRA, MPNet, XLM-RoBERTa, CamemBERT, \
             T5 and Llama tokenizers, not {}. Save one with transformers: \
             AutoTokenizer.from_pretrained(path).save_pretrained(path)",
            dir.display(),
            settings.describe()
        )
    })?;

    let mut tokenizer = match (kind, &spm) {
        (Kind::Gpt2 | Kind::Roberta, _) if has_bpe => build_bpe(dir, kind, &settings)?,
        (Kind::Bert | Kind::MpNet, _) if has_wordpiece => build_wordpiece(dir, kind, &settings)?,
        (Kind::XlmRoberta | Kind::Camembert | Kind::T5 | Kind::Llama, Some(file)) => {
            build_sentencepiece(file, kind, &settings)?
        }
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
    /// `XLMRobertaConverter`: Unigram SentencePiece.
    XlmRoberta,
    /// `CamembertConverter`: Unigram SentencePiece.
    Camembert,
    /// `T5Converter`, which mT5 uses too: Unigram SentencePiece.
    T5,
    /// `LlamaConverter`, which Mistral and Phi-3 use too: SentencePiece BPE.
    Llama,
}

impl Kind {
    fn from_tokenizer_class(class: &str) -> Option<Self> {
        match class.trim_end_matches("Fast") {
            "GPT2Tokenizer" => Some(Self::Gpt2),
            "RobertaTokenizer" | "BartTokenizer" | "LongformerTokenizer" => Some(Self::Roberta),
            "BertTokenizer" | "DistilBertTokenizer" | "ElectraTokenizer" => Some(Self::Bert),
            "MPNetTokenizer" => Some(Self::MpNet),
            "XLMRobertaTokenizer" => Some(Self::XlmRoberta),
            "CamembertTokenizer" => Some(Self::Camembert),
            "T5Tokenizer" | "MT5Tokenizer" => Some(Self::T5),
            "LlamaTokenizer" => Some(Self::Llama),
            _ => None,
        }
    }

    fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "gpt2" => Some(Self::Gpt2),
            "roberta" | "bart" | "longformer" => Some(Self::Roberta),
            "bert" | "distilbert" | "electra" => Some(Self::Bert),
            "mpnet" => Some(Self::MpNet),
            "xlm-roberta" => Some(Self::XlmRoberta),
            "camembert" => Some(Self::Camembert),
            "t5" | "mt5" => Some(Self::T5),
            "llama" | "mistral" | "phi3" => Some(Self::Llama),
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
            (Self::XlmRoberta | Self::Camembert, "bos_token" | "cls_token") => Some("<s>"),
            (Self::XlmRoberta | Self::Camembert, "eos_token" | "sep_token") => Some("</s>"),
            (Self::XlmRoberta | Self::Camembert, "unk_token") => Some("<unk>"),
            (Self::XlmRoberta | Self::Camembert, "pad_token") => Some("<pad>"),
            (Self::XlmRoberta | Self::Camembert, "mask_token") => Some("<mask>"),
            (Self::T5, "eos_token") => Some("</s>"),
            (Self::T5, "unk_token") => Some("<unk>"),
            (Self::T5, "pad_token") => Some("<pad>"),
            (Self::Llama, "bos_token") => Some("<s>"),
            (Self::Llama, "eos_token") => Some("</s>"),
            (Self::Llama, "unk_token") => Some("<unk>"),
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
        // RoBERTa, BART, Longformer, MPNet, XLM-RoBERTa and CamemBERT make a
        // mask token given as a plain
        // string strip the space before it, so "a <mask>" ends in one token.
        if name == "mask_token"
            && matches!(
                kind,
                Kind::Roberta | Kind::MpNet | Kind::XlmRoberta | Kind::Camembert
            )
            && !matches!(self.tokenizer.get(name), Some(Value::Object(_)))
        {
            token = token.lstrip(true);
        }
        Some(token)
    }

    fn special_content(&self, kind: Kind, name: &str) -> Option<String> {
        self.special(kind, name).map(|t| t.content)
    }

    /// A flag of the Python tokenizer that `null` in the config turns off,
    /// where leaving it out keeps `default`.
    fn nullable_flag(&self, key: &str, default: bool) -> bool {
        match self.tokenizer.get(key) {
            None => default,
            Some(v) => v.as_bool().unwrap_or(false),
        }
    }

    /// `legacy`, which only T5 and Llama tokenizers have; unset or null means
    /// true, the behaviour from before the setting existed.
    fn legacy(&self, kind: Kind) -> bool {
        match kind {
            Kind::T5 | Kind::Llama => self
                .tokenizer
                .get("legacy")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            _ => true,
        }
    }

    /// `add_prefix_space` as the converter reads it from the Python tokenizer:
    /// XLM-RoBERTa and CamemBERT have no such attribute and always add it.
    fn spm_add_prefix_space(&self, kind: Kind) -> bool {
        match kind {
            Kind::T5 | Kind::Llama => self.nullable_flag("add_prefix_space", true),
            _ => true,
        }
    }

    /// The special tokens the Python class adds when the config lists none.
    fn default_additional_special(&self, kind: Kind) -> Vec<String> {
        match kind {
            Kind::Camembert => ["<s>NOTUSED", "</s>NOTUSED", "<unk>NOTUSED"]
                .map(String::from)
                .to_vec(),
            Kind::T5 => (0..self.t5_extra_ids())
                .map(|i| format!("<extra_id_{i}>"))
                .collect(),
            _ => Vec::new(),
        }
    }

    fn t5_extra_ids(&self) -> usize {
        self.tokenizer
            .get("extra_ids")
            .and_then(Value::as_u64)
            .unwrap_or(100) as usize
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

/// The tokenizer transformers' `SpmConverter` subclasses build from a
/// SentencePiece `.model` file, with the post-processor the fast tokenizer
/// class sets.
fn build_sentencepiece(file: &Path, kind: Kind, settings: &Settings) -> Result<Tokenizer> {
    let bytes = std::fs::read(file).with_context(|| format!("reading {}", file.display()))?;
    let spm = SpmModel::parse(&bytes).with_context(|| format!("parsing {}", file.display()))?;
    let pieces = &spm.pieces;
    let scored_piece = |p: &sentencepiece::Piece| (p.piece.clone(), f64::from(p.score));
    let from = |n: usize| pieces.get(n..).unwrap_or(&[]).iter().map(scored_piece);
    let zero = |t: &str| (t.to_owned(), 0.0);

    // Each converter's `vocab()`: the fairseq models put their own special
    // tokens first, so their ids differ from the .model file's.
    let (scored, unk_id): (Vec<(String, f64)>, usize) = match kind {
        Kind::XlmRoberta => {
            let mut v: Vec<_> = ["<s>", "<pad>", "</s>", "<unk>"].map(zero).to_vec();
            v.extend(from(3));
            v.push(zero("<mask>"));
            (v, 3)
        }
        Kind::Camembert => {
            let mut v: Vec<_> = ["<s>NOTUSED", "<pad>", "</s>NOTUSED", "<unk>"]
                .map(zero)
                .to_vec();
            v.push(("<unk>NOTUSED".into(), -100.0));
            v.extend(from(1));
            v.push(zero("<mask>"));
            (v, 3)
        }
        Kind::T5 => {
            let mut v: Vec<_> = from(0).collect();
            v.extend(
                (0..settings.t5_extra_ids())
                    .rev()
                    .map(|i| (format!("<extra_id_{i}>"), 0.0)),
            );
            (v, usize::try_from(spm.unk_id).unwrap_or(0))
        }
        Kind::Llama => {
            let mut v: Vec<_> = pieces.iter().take(3).map(|p| zero(&p.piece)).collect();
            v.extend(from(3));
            (v, 0)
        }
        _ => unreachable!("not a SentencePiece kind"),
    };

    let byte_fallback = kind == Kind::Llama;
    let mut tokenizer = match spm.model_type {
        sentencepiece::ModelType::Unigram => Tokenizer::new(
            Unigram::from(scored, Some(unk_id), byte_fallback).map_err(|e| anyhow!(e))?,
        ),
        sentencepiece::ModelType::Bpe => {
            let file_vocab: HashMap<String, u32> = pieces
                .iter()
                .enumerate()
                .map(|(i, p)| (p.piece.clone(), i as u32))
                .collect();
            let merges = sentencepiece::generate_merges(&file_vocab, &scored);
            let vocab: HashMap<&str, u32> = scored
                .iter()
                .enumerate()
                .map(|(i, (w, _))| (w.as_str(), i as u32))
                .collect();
            // Through serde, since the builder takes the crate's own map type.
            let bpe: BPE = serde_json::from_value(serde_json::json!({
                "vocab": vocab,
                "merges": merges,
                "unk_token": spm.unk_piece,
                "fuse_unk": true,
                "byte_fallback": byte_fallback,
            }))
            .context("building BPE from the SentencePiece model")?;
            Tokenizer::new(bpe)
        }
        sentencepiece::ModelType::Other(t) => bail!(
            "{} is a SentencePiece model of type {t}; only Unigram and BPE models can be read",
            file.display()
        ),
    };

    // Control and user-defined pieces become added tokens, so text holding
    // them is never split; control pieces are special.
    let file_tokens: Vec<AddedToken> = pieces
        .iter()
        .filter(|p| matches!(p.kind, PieceType::Control | PieceType::UserDefined))
        .map(|p| AddedToken::from(p.piece.clone(), p.kind == PieceType::Control).normalized(false))
        .collect();
    tokenizer.add_tokens(&file_tokens);

    let add_prefix_space = settings.spm_add_prefix_space(kind);
    let legacy = settings.legacy(kind);
    let prepend_scheme = match (add_prefix_space, legacy) {
        (false, _) => PrependScheme::Never,
        (true, true) => PrependScheme::Always,
        (true, false) => PrependScheme::First,
    };

    if kind == Kind::Llama {
        if legacy {
            let mut seq: Vec<NormalizerWrapper> = Vec::new();
            if add_prefix_space {
                seq.push(Prepend::new("▁".into()).into());
            }
            seq.push(Replace::new(" ", "▁").map_err(|e| anyhow!(e))?.into());
            tokenizer.with_normalizer(Some(NormalizerSequence::new(seq)));
        } else {
            tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', prepend_scheme, false)));
        }
        let mut decoders: Vec<DecoderWrapper> = vec![
            DecoderWrapper::Replace(Replace::new("▁", " ").map_err(|e| anyhow!(e))?),
            ByteFallback::new().into(),
            Fuse::new().into(),
        ];
        if add_prefix_space {
            decoders.push(StripDecoder::new(' ', 1, 0).into());
        }
        tokenizer.with_decoder(Some(DecoderSequence::new(decoders)));
    } else {
        let mut seq: Vec<NormalizerWrapper> = Vec::new();
        if !spm.precompiled_charsmap.is_empty() {
            let precompiled = Precompiled::from(&spm.precompiled_charsmap).map_err(|e| {
                anyhow!("reading the normalization table of {}: {e}", file.display())
            })?;
            seq.push(precompiled.into());
        }
        seq.push(Strip::new(false, true).into());
        seq.push(
            Replace::new(
                tokenizers::normalizers::replace::ReplacePattern::Regex(" {2,}".into()),
                "▁",
            )
            .map_err(|e| anyhow!(e))?
            .into(),
        );
        tokenizer.with_normalizer(Some(NormalizerSequence::new(seq)));
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', prepend_scheme, true)));
        tokenizer.with_decoder(Some(Metaspace::new('▁', prepend_scheme, true)));
    }

    let template = |single: String, pair: String, specials: Vec<(String, u32)>| {
        TemplateProcessing::builder()
            .try_single(single)
            .map_err(|e| anyhow!(e))?
            .try_pair(pair)
            .map_err(|e| anyhow!(e))?
            .special_tokens(specials)
            .build()
            .map_err(anyhow::Error::from)
    };
    let post = match kind {
        Kind::XlmRoberta | Kind::Camembert => {
            let s = vocab_id(&tokenizer, "<s>")?;
            let e = vocab_id(&tokenizer, "</s>")?;
            template(
                "<s> $A </s>".into(),
                "<s> $A </s> </s> $B </s>".into(),
                vec![("<s>".into(), s), ("</s>".into(), e)],
            )?
        }
        Kind::T5 => {
            let e = vocab_id(&tokenizer, "</s>")?;
            template(
                "$A </s>".into(),
                "$A </s> $B </s>".into(),
                vec![("</s>".into(), e)],
            )?
        }
        _ => {
            // LlamaTokenizerFast.update_post_processor.
            let add_bos = settings.flag("add_bos_token", true);
            let add_eos = settings.flag("add_eos_token", false);
            let mut specials = Vec::new();
            let mut wrap = |name: &str, on: bool| -> Result<Option<String>> {
                if !on {
                    return Ok(None);
                }
                let token = settings
                    .special_content(kind, name)
                    .ok_or_else(|| anyhow!("{name} is null, but the tokenizer is set to add it"))?;
                specials.push((token.clone(), vocab_id(&tokenizer, &token)?));
                Ok(Some(token))
            };
            let bos = wrap("bos_token", add_bos)?;
            let eos = wrap("eos_token", add_eos)?;
            let part = |t: usize| {
                let pre = bos
                    .as_ref()
                    .map(|b| format!("{b}:{t} "))
                    .unwrap_or_default();
                let post = eos
                    .as_ref()
                    .map(|e| format!(" {e}:{t}"))
                    .unwrap_or_default();
                (pre, post)
            };
            let ((b0, e0), (b1, e1)) = (part(0), part(1));
            let single = format!("{b0}$A:0{e0}");
            let pair = format!("{single} {b1}$B:1{e1}");
            specials.dedup();
            template(single, pair, specials)?
        }
    };
    tokenizer.with_post_processor(Some(post));
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
    match settings.tokenizer.get("additional_special_tokens") {
        Some(Value::Array(extra)) => {
            specials.extend(extra.iter().filter_map(|v| added_token(v, true)))
        }
        _ => specials.extend(
            settings
                .default_additional_special(kind)
                .into_iter()
                .map(|t| AddedToken::from(t, true)),
        ),
    }

    // added_tokens_decoder maps ids to tokens; added_tokens.json maps tokens to
    // ids. Either way, the ids say the order they were added in. The older
    // added_tokens.json only fills ids the decoder leaves out, and makes a token
    // special when it is one of the special tokens.
    let mut by_id: std::collections::BTreeMap<u32, AddedToken> = Default::default();
    if let Some(Value::Object(decoder)) = settings.tokenizer.get("added_tokens_decoder") {
        for (id, v) in decoder {
            if let (Ok(id), Some(t)) = (id.parse::<u32>(), added_token(v, false)) {
                by_id.insert(id, t);
            }
        }
    }
    for (content, id) in read_object(&dir.join("added_tokens.json"))? {
        let Some(id) = id.as_u64().map(|id| id as u32) else {
            continue;
        };
        if by_id.contains_key(&id) || by_id.values().any(|t| t.content == content) {
            continue;
        }
        let special = specials.iter().any(|t| t.content == content);
        by_id.insert(id, AddedToken::from(content, special));
    }

    tokenizer.add_special_tokens(&specials);
    let (special, normal): (Vec<_>, Vec<_>) = by_id.into_values().partition(|t| t.special);
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
    use tokenizers::PreTokenizerWrapper;

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

    /// An XLM-RoBERTa folder with only `sentencepiece.bpe.model`.
    fn xlmr_spm_folder() -> tempfile::TempDir {
        use super::super::sentencepiece::proto::unigram_model;
        let model = unigram_model(&[
            ("<unk>", 0.0, 2),
            ("<s>", 0.0, 3),
            ("</s>", 0.0, 3),
            ("▁hello", -1.0, 1),
            ("▁world", -1.0, 1),
            ("▁", -2.0, 1),
            ("h", -3.0, 1),
            ("e", -3.0, 1),
            ("l", -3.0, 1),
            ("o", -3.0, 1),
            ("w", -3.0, 1),
            ("r", -3.0, 1),
            ("d", -3.0, 1),
        ]);
        let dir = folder(&[(
            "tokenizer_config.json",
            json!({"tokenizer_class": "XLMRobertaTokenizer"}).to_string(),
        )]);
        std::fs::write(dir.path().join("sentencepiece.bpe.model"), model).unwrap();
        dir
    }

    /// A converted tokenizer is already what transformers builds, so
    /// `load_model_tokenizer` leaves it alone; the whitespace patch is for a
    /// published tokenizer.json only. Patched, a converted XLM-R tokenizer gave
    /// NER spans one character off.
    #[test]
    fn load_model_tokenizer_patches_only_a_published_tokenizer_json() {
        let spm = xlmr_spm_folder();
        let converted = load_model_tokenizer(spm.path()).unwrap();
        assert_eq!(
            converted.to_string(false).unwrap(),
            load_tokenizer(spm.path())
                .unwrap()
                .to_string(false)
                .unwrap()
        );
        assert!(matches!(
            converted.get_pre_tokenizer(),
            Some(PreTokenizerWrapper::Metaspace(_))
        ));
        // fairseq ids: <s> <pad> </s> <unk>, then the file's pieces shifted by
        // one. As in transformers, "▁world" spans the space it stands for.
        let enc = converted.encode("hello world", true).unwrap();
        assert_eq!(enc.get_ids(), [0, 4, 5, 2]);
        assert_eq!(enc.get_offsets(), [(0, 0), (0, 5), (5, 11), (0, 0)]);

        let published = folder(&[("tokenizer.json", converted.to_string(false).unwrap())]);
        let patched = load_model_tokenizer(published.path()).unwrap();
        assert!(matches!(
            patched.get_pre_tokenizer(),
            Some(PreTokenizerWrapper::Sequence(_))
        ));
        assert!(matches!(
            load_tokenizer(published.path())
                .unwrap()
                .get_pre_tokenizer(),
            Some(PreTokenizerWrapper::Metaspace(_))
        ));
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
            err.contains(
                "tokenizer.json, vocab.json with merges.txt, vocab.txt, or a SentencePiece .model file"
            ),
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
