//! `pytorch_model.bin`, the weights `torch.save` writes, read without Python.
//!
//! Two layouts exist. Since PyTorch 1.6 the file is a zip archive: the state
//! dict pickled in `<archive>/data.pkl`, each storage stored uncompressed in
//! `<archive>/data/<key>`. Before that, and in many models still on the Hugging
//! Face Hub, it is a run of pickles: a magic number, a protocol version, system
//! information, the state dict, the list of storage keys, and then each
//! storage's element count and bytes in that order.
//!
//! A pickle is a program, and loading one with Python's `pickle` can run any
//! code it names. This reader runs none: it knows the opcodes `torch.save`
//! emits, builds plain values, and accepts only the globals a state dict is
//! made of (`OrderedDict`, the tensor and parameter rebuild functions and the
//! storage types). Anything else is an error, so a file built to do harm fails
//! to load instead of doing it.

use std::any::Any;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;

use crate::tensor::DType;
use crate::tensor::raw_tensor::TensorView;
use crate::weights::WeightLoader;
use crate::weights::mmap_cache::get_or_create_mmap;

/// The weights of a `pytorch_model.bin`, or of the shards a
/// `pytorch_model.bin.index.json` lists.
pub struct PyTorchLoader {
    files: Vec<Arc<Mmap>>,
    tensors: HashMap<String, Entry>,
}

/// Where a tensor's elements are: a storage in one of `files`, and the
/// offset, shape and strides of the tensor in it, counted in elements.
#[derive(Debug, Clone)]
struct Entry {
    file: usize,
    /// Byte range of the whole storage in the file.
    storage: std::ops::Range<usize>,
    dtype: Element,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

/// A storage's element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Element {
    F32,
    F16,
    BF16,
    F64,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Element {
    fn from_storage_name(name: &str) -> Option<Self> {
        Some(match name {
            "FloatStorage" => Self::F32,
            "HalfStorage" => Self::F16,
            "BFloat16Storage" => Self::BF16,
            "DoubleStorage" => Self::F64,
            "LongStorage" => Self::I64,
            "IntStorage" => Self::I32,
            "ShortStorage" => Self::I16,
            "CharStorage" => Self::I8,
            "ByteStorage" => Self::U8,
            "BoolStorage" => Self::Bool,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 | Self::I16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }
}

impl PyTorchLoader {
    /// Reads `pytorch_model.bin`, or its shards, in `dir`; or the file `path`.
    pub fn new(path: &Path) -> Result<Self> {
        let files: Vec<PathBuf> = if path.is_file() {
            vec![path.to_path_buf()]
        } else if path.join("pytorch_model.bin.index.json").exists() {
            let index_path = path.join("pytorch_model.bin.index.json");
            let index: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&index_path)?)
                    .with_context(|| format!("cannot parse {}", index_path.display()))?;
            let mut shards: Vec<String> = index["weight_map"]
                .as_object()
                .ok_or_else(|| anyhow!("{} has no weight_map", index_path.display()))?
                .values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            shards.sort();
            shards.dedup();
            shards.into_iter().map(|s| path.join(s)).collect()
        } else {
            vec![path.join("pytorch_model.bin")]
        };

        let mut loader = Self {
            files: Vec::new(),
            tensors: HashMap::new(),
        };
        for file in files {
            let mmap = get_or_create_mmap(&file)
                .with_context(|| format!("cannot open {}", file.display()))?;
            let index = loader.files.len();
            let tensors = read_file(&mmap, index)
                .with_context(|| format!("cannot read PyTorch weights {}", file.display()))?;
            loader.tensors.extend(tensors);
            loader.files.push(mmap);
        }
        Ok(loader)
    }

    /// The names of the tensors, in no particular order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

impl WeightLoader for PyTorchLoader {
    fn get_raw(&self, name: &str) -> Result<TensorView<'_>> {
        let entry = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("tensor '{}' not found in model", name))?;
        let storage = &self.files[entry.file][entry.storage.clone()];
        let size = entry.dtype.size();
        let numel: usize = entry.shape.iter().product();

        let bytes: Cow<'_, [u8]> = if is_contiguous(&entry.shape, &entry.stride) {
            let start = entry.offset * size;
            let end = start + numel * size;
            Cow::Borrowed(
                storage
                    .get(start..end)
                    .ok_or_else(|| anyhow!("tensor '{name}' runs past its storage"))?,
            )
        } else {
            Cow::Owned(gather(storage, entry, size).with_context(|| format!("tensor '{name}'"))?)
        };

        // No kernel takes f16, f64 or integer weights: floats reach the model
        // as f32, as safetensors f16 does.
        let (bytes, dtype) = match entry.dtype {
            Element::F32 => (bytes, DType::F32),
            Element::BF16 => (bytes, DType::BF16),
            Element::F16 => {
                let widened: Vec<f32> = crate::weights::cast_or_copy::<half::f16>(&bytes)
                    .into_iter()
                    .map(half::f16::to_f32)
                    .collect();
                (
                    Cow::Owned(bytemuck::cast_slice(&widened).to_vec()),
                    DType::F32,
                )
            }
            Element::F64 => {
                let narrowed: Vec<f32> = bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect();
                (
                    Cow::Owned(bytemuck::cast_slice(&narrowed).to_vec()),
                    DType::F32,
                )
            }
            other => bail!("tensor '{name}' holds {other:?} values, not floating point weights"),
        };
        Ok(TensorView {
            name: name.to_string(),
            bytes,
            shape: entry.shape.clone(),
            dtype,
        })
    }

    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn release(&self, bytes: &[u8]) {
        for mmap in &self.files {
            crate::weights::mmap_cache::release_pages(mmap, bytes);
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn is_contiguous(shape: &[usize], stride: &[usize]) -> bool {
    let mut expected = 1;
    for (&dim, &s) in shape.iter().zip(stride).rev() {
        if dim != 1 && s != expected {
            return false;
        }
        expected *= dim;
    }
    true
}

/// The elements of a strided tensor, copied out in row-major order.
fn gather(storage: &[u8], entry: &Entry, size: usize) -> Result<Vec<u8>> {
    let numel: usize = entry.shape.iter().product();
    let mut out = Vec::with_capacity(numel * size);
    let mut index = vec![0usize; entry.shape.len()];
    for _ in 0..numel {
        let element = entry.offset
            + index
                .iter()
                .zip(&entry.stride)
                .map(|(i, s)| i * s)
                .sum::<usize>();
        let start = element * size;
        out.extend_from_slice(
            storage
                .get(start..start + size)
                .ok_or_else(|| anyhow!("runs past its storage"))?,
        );
        for d in (0..index.len()).rev() {
            index[d] += 1;
            if index[d] < entry.shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
    Ok(out)
}

/// The tensors of one file, either layout.
fn read_file(bytes: &[u8], file: usize) -> Result<HashMap<String, Entry>> {
    if bytes.starts_with(b"PK\x03\x04") {
        read_zip(bytes, file)
    } else {
        read_legacy(bytes, file)
    }
}

/// A tensor as the state dict pickle describes it: which storage, and where
/// in it.
#[derive(Debug, Clone)]
struct TensorRef {
    key: String,
    dtype: Element,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

/// The tensors of a state dict, by name. Parameters nested in modules come
/// out under the same dotted names `state_dict()` gives them.
fn tensors_of(value: Value) -> Result<Vec<(String, TensorRef)>> {
    let Value::Dict(items) = value else {
        bail!("the pickle is not a state dict but {}", value.kind());
    };
    let mut out = Vec::new();
    for (k, v) in items {
        let Value::Str(name) = k else { continue };
        match v {
            Value::Tensor(t) => out.push((name, t)),
            // Some checkpoints wrap the weights, as {"state_dict": {...}}.
            nested @ Value::Dict(_) => {
                for (inner, t) in tensors_of(nested)? {
                    out.push((format!("{name}.{inner}"), t));
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// The legacy layout: pickles one after another, then the storages.
fn read_legacy(bytes: &[u8], file: usize) -> Result<HashMap<String, Entry>> {
    const MAGIC: i128 = 0x1950a86a20f9469cfc6c;
    let mut pos = 0;
    let magic = Unpickler::new(&bytes[pos..]).load(&mut pos)?;
    if !matches!(magic, Value::Int(m) if m == MAGIC) {
        bail!(
            "not a PyTorch file: it is neither a zip archive nor starts with torch's magic number"
        );
    }
    let version = Unpickler::new(&bytes[pos..]).load(&mut pos)?;
    if !matches!(version, Value::Int(1001)) {
        bail!("unsupported legacy PyTorch format version {version:?}");
    }
    let sys_info = Unpickler::new(&bytes[pos..]).load(&mut pos)?;
    if let Value::Dict(items) = &sys_info {
        for (k, v) in items {
            if matches!(k, Value::Str(s) if s == "little_endian") && matches!(v, Value::Bool(false))
            {
                bail!("the file was saved on a big-endian machine");
            }
        }
    }
    let mut unpickler = Unpickler::new(&bytes[pos..]);
    let state = unpickler.load(&mut pos)?;
    let element_of = std::mem::take(&mut unpickler.storages);
    let keys = Unpickler::new(&bytes[pos..]).load(&mut pos)?;
    let Value::List(keys) = keys else {
        bail!("expected the list of storage keys, found {}", keys.kind());
    };
    let tensors = tensors_of(state)?;

    // Each storage: its element count as a little-endian i64, then its bytes.
    let mut storages: HashMap<String, std::ops::Range<usize>> = HashMap::new();
    for key in keys {
        let Value::Str(key) = key else {
            bail!("a storage key is {}, not a string", key.kind());
        };
        let count = bytes
            .get(pos..pos + 8)
            .ok_or_else(|| anyhow!("the file ends before storage {key}"))?;
        let numel = usize::try_from(i64::from_le_bytes(count.try_into().unwrap()))?;
        pos += 8;
        let size = element_of
            .get(key.as_str())
            .ok_or_else(|| anyhow!("storage {key} has no type"))?
            .size();
        let end = pos + numel * size;
        if end > bytes.len() {
            bail!("storage {key} runs past the end of the file");
        }
        storages.insert(key, pos..end);
        pos = end;
    }
    entries(tensors, &storages, file)
}

/// The zip layout: `data.pkl` and one stored entry per storage.
fn read_zip(bytes: &[u8], file: usize) -> Result<HashMap<String, Entry>> {
    let zip = Zip::parse(bytes)?;
    let pickle_name = zip
        .names()
        .find(|n| n.ends_with("/data.pkl") || *n == "data.pkl")
        .ok_or_else(|| anyhow!("the archive has no data.pkl"))?
        .to_string();
    let prefix = pickle_name.trim_end_matches("data.pkl").to_string();
    let pickle = zip.get(bytes, &pickle_name)?;
    let state = Unpickler::new(pickle).load(&mut 0)?;
    let tensors = tensors_of(state)?;

    let mut storages = HashMap::new();
    for (_, t) in &tensors {
        let name = format!("{prefix}data/{}", t.key);
        let range = zip.range(&name)?;
        storages.insert(t.key.clone(), range);
    }
    entries(tensors, &storages, file)
}

fn entries(
    tensors: Vec<(String, TensorRef)>,
    storages: &HashMap<String, std::ops::Range<usize>>,
    file: usize,
) -> Result<HashMap<String, Entry>> {
    let mut out = HashMap::new();
    for (name, t) in tensors {
        let storage = storages
            .get(&t.key)
            .ok_or_else(|| anyhow!("tensor '{name}' names storage {} the file lacks", t.key))?
            .clone();
        out.insert(
            name,
            Entry {
                file,
                storage,
                dtype: t.dtype,
                offset: t.offset,
                shape: t.shape,
                stride: t.stride,
            },
        );
    }
    Ok(out)
}

/// The entries of a zip archive whose members are stored uncompressed, as
/// `torch.save` writes them.
struct Zip {
    /// Name to (local header offset, size, compression method).
    members: Vec<(String, usize, usize, u16)>,
}

impl Zip {
    fn parse(bytes: &[u8]) -> Result<Self> {
        let u16_at = |p: usize| -> Result<u16> {
            Ok(u16::from_le_bytes(
                bytes
                    .get(p..p + 2)
                    .ok_or_else(|| anyhow!("truncated zip"))?
                    .try_into()?,
            ))
        };
        let u32_at = |p: usize| -> Result<u32> {
            Ok(u32::from_le_bytes(
                bytes
                    .get(p..p + 4)
                    .ok_or_else(|| anyhow!("truncated zip"))?
                    .try_into()?,
            ))
        };
        let u64_at = |p: usize| -> Result<u64> {
            Ok(u64::from_le_bytes(
                bytes
                    .get(p..p + 8)
                    .ok_or_else(|| anyhow!("truncated zip"))?
                    .try_into()?,
            ))
        };

        // The end of central directory record, within the last 64 KiB.
        let search_from = bytes.len().saturating_sub(22 + 65535);
        let eocd = (search_from..=bytes.len().saturating_sub(22))
            .rev()
            .find(|&p| bytes[p..].starts_with(b"PK\x05\x06"))
            .ok_or_else(|| anyhow!("no zip end of central directory"))?;
        let mut count = u16_at(eocd + 10)? as u64;
        let mut cd_offset = u32_at(eocd + 16)? as u64;
        // Zip64: the real values are in the zip64 end of central directory.
        if (count == 0xFFFF || cd_offset == 0xFFFF_FFFF) && eocd >= 20 {
            let locator = eocd - 20;
            if bytes[locator..].starts_with(b"PK\x06\x07") {
                let record = usize::try_from(u64_at(locator + 8)?)?;
                if !bytes[record..].starts_with(b"PK\x06\x06") {
                    bail!("bad zip64 end of central directory");
                }
                count = u64_at(record + 32)?;
                cd_offset = u64_at(record + 48)?;
            }
        }

        let mut members = Vec::new();
        let mut p = usize::try_from(cd_offset)?;
        for _ in 0..count {
            if !bytes.get(p..).is_some_and(|b| b.starts_with(b"PK\x01\x02")) {
                bail!("bad zip central directory entry");
            }
            let method = u16_at(p + 10)?;
            let mut size = u32_at(p + 24)? as u64;
            let name_len = u16_at(p + 28)? as usize;
            let extra_len = u16_at(p + 30)? as usize;
            let comment_len = u16_at(p + 32)? as usize;
            let mut local = u32_at(p + 42)? as u64;
            let name = String::from_utf8_lossy(
                bytes
                    .get(p + 46..p + 46 + name_len)
                    .ok_or_else(|| anyhow!("truncated zip"))?,
            )
            .into_owned();
            // Zip64 extended information: present values replace 0xFFFFFFFF.
            let mut e = p + 46 + name_len;
            let extra_end = e + extra_len;
            while e + 4 <= extra_end {
                let id = u16_at(e)?;
                let len = u16_at(e + 2)? as usize;
                if id == 1 {
                    let mut q = e + 4;
                    if u32_at(p + 24)? == 0xFFFF_FFFF {
                        size = u64_at(q)?;
                        q += 8;
                    }
                    if u32_at(p + 20)? == 0xFFFF_FFFF {
                        q += 8;
                    }
                    if u32_at(p + 42)? == 0xFFFF_FFFF {
                        local = u64_at(q)?;
                    }
                }
                e += 4 + len;
            }
            members.push((
                name,
                usize::try_from(local)?,
                usize::try_from(size)?,
                method,
            ));
            p = extra_end + comment_len;
        }

        // The data starts after each member's local header.
        let mut resolved = Vec::with_capacity(members.len());
        for (name, local, size, method) in members {
            if !bytes
                .get(local..)
                .is_some_and(|b| b.starts_with(b"PK\x03\x04"))
            {
                bail!("bad zip local header for {name}");
            }
            let start = local + 30 + u16_at(local + 26)? as usize + u16_at(local + 28)? as usize;
            resolved.push((name, start, size, method));
        }
        Ok(Self { members: resolved })
    }

    fn names(&self) -> impl Iterator<Item = &str> {
        self.members.iter().map(|(n, ..)| n.as_str())
    }

    fn range(&self, name: &str) -> Result<std::ops::Range<usize>> {
        let (_, start, size, method) = self
            .members
            .iter()
            .find(|(n, ..)| n == name)
            .ok_or_else(|| anyhow!("the archive has no {name}"))?;
        if *method != 0 {
            bail!("{name} is compressed; torch.save stores its members uncompressed");
        }
        Ok(*start..start + size)
    }

    fn get<'a>(&self, bytes: &'a [u8], name: &str) -> Result<&'a [u8]> {
        let range = self.range(name)?;
        bytes
            .get(range)
            .ok_or_else(|| anyhow!("{name} runs past the end of the archive"))
    }
}

/// The values a state dict pickle builds.
#[derive(Debug, Clone)]
enum Value {
    None,
    Bool(bool),
    Int(i128),
    /// A float or a bytes object: read past, never needed for the tensors.
    Float,
    Str(String),
    Bytes,
    Tuple(Vec<Value>),
    List(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    /// A module-level name, `module.name`, from the allowed set.
    Global(&'static str),
    /// A storage, by its persistent id: its key and element type.
    Storage(String, Element),
    Tensor(TensorRef),
    Mark,
}

impl Value {
    fn kind(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Bool(_) => "a bool",
            Self::Int(_) => "an int",
            Self::Float => "a float",
            Self::Str(_) => "a string",
            Self::Bytes => "bytes",
            Self::Tuple(_) => "a tuple",
            Self::List(_) => "a list",
            Self::Dict(_) => "a dict",
            Self::Global(_) => "a global",
            Self::Storage(..) => "a storage",
            Self::Tensor(_) => "a tensor",
            Self::Mark => "a mark",
        }
    }

    fn as_usize(&self) -> Result<usize> {
        match self {
            Self::Int(i) => usize::try_from(*i).map_err(|_| anyhow!("{i} is not a size")),
            other => bail!("expected an int, found {}", other.kind()),
        }
    }

    fn as_sizes(&self) -> Result<Vec<usize>> {
        match self {
            Self::Tuple(v) | Self::List(v) => v.iter().map(Value::as_usize).collect(),
            other => bail!("expected a tuple of sizes, found {}", other.kind()),
        }
    }
}

/// The globals a state dict may name. Every other global is refused.
const ALLOWED_GLOBALS: &[&str] = &[
    "collections.OrderedDict",
    "torch._utils._rebuild_tensor",
    "torch._utils._rebuild_tensor_v2",
    "torch._utils._rebuild_parameter",
    "torch.FloatStorage",
    "torch.HalfStorage",
    "torch.BFloat16Storage",
    "torch.DoubleStorage",
    "torch.LongStorage",
    "torch.IntStorage",
    "torch.ShortStorage",
    "torch.CharStorage",
    "torch.ByteStorage",
    "torch.BoolStorage",
];

/// Runs one pickle, with only the opcodes and globals a state dict needs.
struct Unpickler<'a> {
    bytes: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: HashMap<u32, Value>,
    /// Every storage a persistent id named, with its element type.
    storages: HashMap<String, Element>,
}

impl<'a> Unpickler<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            stack: Vec::new(),
            memo: HashMap::new(),
            storages: HashMap::new(),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| anyhow!("the pickle ends early"))?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn line(&mut self) -> Result<&'a str> {
        let rest = &self.bytes[self.pos..];
        let end = rest
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| anyhow!("the pickle ends early"))?;
        self.pos += end + 1;
        std::str::from_utf8(&rest[..end]).map_err(|_| anyhow!("a pickle name is not UTF-8"))
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack
            .pop()
            .ok_or_else(|| anyhow!("the pickle pops an empty stack"))
    }

    /// The values above the topmost mark, which is removed.
    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let mark = self
            .stack
            .iter()
            .rposition(|v| matches!(v, Value::Mark))
            .ok_or_else(|| anyhow!("the pickle has no mark to pop to"))?;
        let items = self.stack.split_off(mark + 1);
        self.stack.pop();
        Ok(items)
    }

    fn top_mut(&mut self) -> Result<&mut Value> {
        self.stack
            .last_mut()
            .ok_or_else(|| anyhow!("the pickle's stack is empty"))
    }

    fn global(&self, module: &str, name: &str) -> Result<Value> {
        let full = format!("{module}.{name}");
        ALLOWED_GLOBALS
            .iter()
            .find(|g| **g == full)
            .map(|g| Value::Global(g))
            .ok_or_else(|| {
                anyhow!("refusing to load {full}: a PyTorch state dict does not need it, and loading it could run code")
            })
    }

    /// Runs the pickle, and advances `*pos` by the bytes it took.
    fn load(&mut self, pos: &mut usize) -> Result<Value> {
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    let proto = self.u8()?;
                    if proto > 5 {
                        bail!("pickle protocol {proto} is newer than this reader");
                    }
                }
                0x95 => {
                    self.u64()?; // FRAME: the frame's length, not needed.
                }
                b'.' => {
                    *pos += self.pos;
                    return self.pop();
                }
                b'(' => self.stack.push(Value::Mark),
                b'}' => self.stack.push(Value::Dict(Vec::new())),
                b']' => self.stack.push(Value::List(Vec::new())),
                b')' => self.stack.push(Value::Tuple(Vec::new())),
                b'N' => self.stack.push(Value::None),
                0x88 => self.stack.push(Value::Bool(true)),
                0x89 => self.stack.push(Value::Bool(false)),
                b'J' => {
                    let v = i32::from_le_bytes(self.take(4)?.try_into()?);
                    self.stack.push(Value::Int(v.into()));
                }
                b'K' => {
                    let v = self.u8()?;
                    self.stack.push(Value::Int(v.into()));
                }
                b'M' => {
                    let v = self.u16()?;
                    self.stack.push(Value::Int(v.into()));
                }
                0x8a => {
                    let n = self.u8()? as usize;
                    let digits = self.take(n)?;
                    if n > 16 {
                        bail!("a pickled int of {n} bytes is too large");
                    }
                    let mut buf = [0u8; 16];
                    buf[..n].copy_from_slice(digits);
                    if n > 0 && digits[n - 1] & 0x80 != 0 {
                        buf[n..].fill(0xff);
                    }
                    self.stack.push(Value::Int(i128::from_le_bytes(buf)));
                }
                b'G' => {
                    self.take(8)?;
                    self.stack.push(Value::Float);
                }
                b'X' => {
                    let n = self.u32()? as usize;
                    let s = std::str::from_utf8(self.take(n)?)?.to_string();
                    self.stack.push(Value::Str(s));
                }
                0x8c => {
                    let n = self.u8()? as usize;
                    let s = std::str::from_utf8(self.take(n)?)?.to_string();
                    self.stack.push(Value::Str(s));
                }
                0x8d => {
                    let n = usize::try_from(self.u64()?)?;
                    let s = std::str::from_utf8(self.take(n)?)?.to_string();
                    self.stack.push(Value::Str(s));
                }
                b'U' => {
                    let n = self.u8()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                b'T' => {
                    let n = self.u32()? as usize;
                    let s = String::from_utf8_lossy(self.take(n)?).into_owned();
                    self.stack.push(Value::Str(s));
                }
                b'C' => {
                    let n = self.u8()? as usize;
                    self.take(n)?;
                    self.stack.push(Value::Bytes);
                }
                b'B' => {
                    let n = self.u32()? as usize;
                    self.take(n)?;
                    self.stack.push(Value::Bytes);
                }
                b'q' => {
                    let i = self.u8()? as u32;
                    let v = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| anyhow!("empty stack"))?;
                    self.memo.insert(i, v);
                }
                b'r' => {
                    let i = self.u32()?;
                    let v = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| anyhow!("empty stack"))?;
                    self.memo.insert(i, v);
                }
                0x94 => {
                    let i = self.memo.len() as u32;
                    let v = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| anyhow!("empty stack"))?;
                    self.memo.insert(i, v);
                }
                b'h' => {
                    let i = self.u8()? as u32;
                    let v = self
                        .memo
                        .get(&i)
                        .cloned()
                        .ok_or_else(|| anyhow!("memo {i} is unset"))?;
                    self.stack.push(v);
                }
                b'j' => {
                    let i = self.u32()?;
                    let v = self
                        .memo
                        .get(&i)
                        .cloned()
                        .ok_or_else(|| anyhow!("memo {i} is unset"))?;
                    self.stack.push(v);
                }
                b'c' => {
                    let module = self.line()?.to_string();
                    let name = self.line()?.to_string();
                    let g = self.global(&module, &name)?;
                    self.stack.push(g);
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let (Value::Str(module), Value::Str(name)) = (module, name) else {
                        bail!("STACK_GLOBAL needs two strings");
                    };
                    let g = self.global(&module, &name)?;
                    self.stack.push(g);
                }
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items));
                }
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a]));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b]));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b, c]));
                }
                b'l' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::List(items));
                }
                b'd' => {
                    let items = self.pop_mark()?;
                    let pairs = pairs(items)?;
                    self.stack.push(Value::Dict(pairs));
                }
                b'a' => {
                    let v = self.pop()?;
                    match self.top_mut()? {
                        Value::List(l) => l.push(v),
                        other => bail!("APPEND to {}", other.kind()),
                    }
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    match self.top_mut()? {
                        Value::List(l) => l.extend(items),
                        other => bail!("APPENDS to {}", other.kind()),
                    }
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    match self.top_mut()? {
                        Value::Dict(d) => d.push((k, v)),
                        other => bail!("SETITEM on {}", other.kind()),
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    let new = pairs(items)?;
                    match self.top_mut()? {
                        Value::Dict(d) => d.extend(new),
                        other => bail!("SETITEMS on {}", other.kind()),
                    }
                }
                b'Q' => {
                    let pid = self.pop()?;
                    let storage = persistent_storage(pid)?;
                    if let Value::Storage(key, element) = &storage {
                        self.storages.insert(key.clone(), *element);
                    }
                    self.stack.push(storage);
                }
                b'R' => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let value = reduce(callable, args)?;
                    self.stack.push(value);
                }
                0x81 => {
                    let args = self.pop()?;
                    let class = self.pop()?;
                    let value = reduce(class, args)?;
                    self.stack.push(value);
                }
                b'b' => {
                    // BUILD: object state, such as a state dict's `_metadata`.
                    // Not needed to find the tensors.
                    self.pop()?;
                }
                other => bail!(
                    "pickle opcode {other:#04x} at byte {} is not one torch.save writes",
                    self.pos - 1
                ),
            }
        }
    }
}

fn pairs(items: Vec<Value>) -> Result<Vec<(Value, Value)>> {
    if !items.len().is_multiple_of(2) {
        bail!("a pickled dict has an odd number of items");
    }
    let mut out = Vec::with_capacity(items.len() / 2);
    let mut it = items.into_iter();
    while let (Some(k), Some(v)) = (it.next(), it.next()) {
        out.push((k, v));
    }
    Ok(out)
}

/// A persistent id: `("storage", storage_type, key, location, numel, ...)`.
fn persistent_storage(pid: Value) -> Result<Value> {
    let Value::Tuple(parts) = pid else {
        bail!("a persistent id is {}, not a tuple", pid.kind());
    };
    let mut parts = parts.into_iter();
    if !matches!(parts.next(), Some(Value::Str(s)) if s == "storage") {
        bail!("a persistent id that is not a storage");
    }
    let element = match parts.next() {
        Some(Value::Global(g)) => g
            .strip_prefix("torch.")
            .and_then(Element::from_storage_name)
            .ok_or_else(|| anyhow!("{g} is not a storage type"))?,
        other => bail!("a storage of type {other:?}"),
    };
    let key = match parts.next() {
        Some(Value::Str(k)) => k,
        Some(Value::Int(i)) => i.to_string(),
        other => bail!("a storage key {other:?}"),
    };
    // Legacy files may follow (location, numel) with a view into another
    // storage, which torch has not written since 0.4.
    let _location = parts.next();
    let _numel = parts.next();
    if let Some(view) = parts.next()
        && !matches!(view, Value::None)
    {
        bail!("storage {key} is a view into another storage, which this reader does not support");
    }
    Ok(Value::Storage(key, element))
}

fn reduce(callable: Value, args: Value) -> Result<Value> {
    let Value::Global(name) = callable else {
        bail!("calling {}, which is not a function", callable.kind());
    };
    let args = match args {
        Value::Tuple(a) => a,
        other => bail!("{name} called with {}, not a tuple", other.kind()),
    };
    match name {
        "collections.OrderedDict" => Ok(Value::Dict(Vec::new())),
        "torch._utils._rebuild_tensor" | "torch._utils._rebuild_tensor_v2" => {
            let mut args = args.into_iter();
            let (Some(storage), Some(offset), Some(shape), Some(stride)) =
                (args.next(), args.next(), args.next(), args.next())
            else {
                bail!("{name} needs storage, offset, size and stride");
            };
            let Value::Storage(key, dtype) = storage else {
                bail!("{name} rebuilds from {}, not a storage", storage.kind());
            };
            let shape = shape.as_sizes()?;
            let stride = stride.as_sizes()?;
            if shape.len() != stride.len() {
                bail!(
                    "a tensor of {} dimensions with {} strides",
                    shape.len(),
                    stride.len()
                );
            }
            Ok(Value::Tensor(TensorRef {
                key,
                dtype,
                offset: offset.as_usize()?,
                shape,
                stride,
            }))
        }
        // nn.Parameter(tensor, requires_grad, hooks): the tensor.
        "torch._utils._rebuild_parameter" => args
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("{name} without a tensor")),
        other => bail!("{other} cannot be called here"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pickle that names `os.system` is refused before anything is called.
    #[test]
    fn a_global_outside_the_state_dict_set_is_refused() {
        // protocol 2: GLOBAL 'os system', SHORT_BINUNICODE 'echo', TUPLE1, REDUCE, STOP
        let mut p = vec![0x80, 2];
        p.extend_from_slice(b"cos\nsystem\n");
        p.extend_from_slice(&[0x8c, 4]);
        p.extend_from_slice(b"echo");
        p.extend_from_slice(&[0x85, b'R', b'.']);
        let err = Unpickler::new(&p).load(&mut 0).unwrap_err().to_string();
        assert!(err.contains("refusing to load os.system"), "{err}");
    }

    /// The same through STACK_GLOBAL, which protocol 4 uses.
    #[test]
    fn a_stack_global_outside_the_set_is_refused_too() {
        let mut p = vec![0x80, 4];
        for s in ["builtins", "eval"] {
            p.extend_from_slice(&[0x8c, s.len() as u8]);
            p.extend_from_slice(s.as_bytes());
        }
        p.extend_from_slice(&[0x93, b'.']);
        let err = Unpickler::new(&p).load(&mut 0).unwrap_err().to_string();
        assert!(err.contains("refusing to load builtins.eval"), "{err}");
    }

    #[test]
    fn an_opcode_torch_save_does_not_write_is_an_error() {
        // INST, which builds an arbitrary class instance.
        let p = [0x80, 2, b'i', b'o', b's', b'\n', b's', b'\n', b'.'];
        assert!(Unpickler::new(&p).load(&mut 0).is_err());
    }

    #[test]
    fn a_truncated_pickle_is_an_error_not_a_panic() {
        let p = [0x80, 2, b'X', 9, 0, 0, 0, b'a'];
        assert!(Unpickler::new(&p).load(&mut 0).is_err());
        assert!(Unpickler::new(&[]).load(&mut 0).is_err());
    }

    #[test]
    fn long1_reads_negative_and_wide_ints() {
        // LONG1 of the legacy magic number, then LONG1 -2.
        let mut p = vec![0x80, 2, 0x8a, 10];
        p.extend_from_slice(&0x1950a86a20f9469cfc6c_i128.to_le_bytes()[..10]);
        p.push(b'.');
        assert!(matches!(
            Unpickler::new(&p).load(&mut 0).unwrap(),
            Value::Int(0x1950a86a20f9469cfc6c)
        ));
        let p = [0x80, 2, 0x8a, 1, 0xfe, b'.'];
        assert!(matches!(
            Unpickler::new(&p).load(&mut 0).unwrap(),
            Value::Int(-2)
        ));
    }

    #[test]
    fn a_transposed_tensor_is_gathered_in_row_major_order() {
        // Storage 0..6 as f32; the tensor is its [2, 3] view transposed to [3, 2].
        let storage: Vec<u8> = (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let entry = Entry {
            file: 0,
            storage: 0..storage.len(),
            dtype: Element::F32,
            offset: 0,
            shape: vec![3, 2],
            stride: vec![1, 3],
        };
        assert!(!is_contiguous(&entry.shape, &entry.stride));
        let out = gather(&storage, &entry, 4).unwrap();
        let values: Vec<f32> = out
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        assert!(is_contiguous(&[2, 3], &[3, 1]));
        assert!(is_contiguous(&[1, 3], &[99, 1]));
    }
}
