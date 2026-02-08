<p align="center">
  <img src="assets/logo.svg" width="80" height="92" alt="Kjarni Logo">
</p>

<h1 align="center">kjarni</h1>

<p align="center">
  <strong>The SQLite of AI</strong><br>
  <sub>AI inference for software engineers who don't care about ML.<br>
  One library. No Python. No CUDA. No containers. Just add and go.</sub>
</p>

<p align="center">
  <a href="https://crates.io/crates/kjarni"><img src="https://img.shields.io/crates/v/kjarni.svg" alt="Crates.io"></a>
  <a href="https://www.nuget.org/packages/Kjarni"><img src="https://img.shields.io/nuget/v/Kjarni.svg" alt="NuGet"></a>
  <a href="https://pypi.org/project/kjarni"><img src="https://img.shields.io/pypi/v/kjarni.svg" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/kjarni"><img src="https://img.shields.io/npm/v/kjarni.svg" alt="npm"></a>
  <a href="https://github.com/olafurjohannsson/kjarni/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

You're a C# developer and you need to classify emails. You're a Go developer and you want local embeddings for search. You don't care about tokenizers, attention masks, or pooling strategies — you just want it to work.

That's Kjarni. A self-contained inference engine with a task-level API: `Classifier`, not `BertForSequenceClassification`. Add one dependency, pick a model by name, and call a method. Kjarni handles the rest.

## Install

```bash
# CLI
cargo install kjarni-cli

# Rust
cargo add kjarni

# Python
pip install kjarni

# C# / .NET
dotnet add package Kjarni
```

## 30-Second Demo

```bash
# Classify sentiment
echo "I love this product" | kjarni classify
# POSITIVE  99.8%  ████████████████████████████████████████

# Translate to German
kjarni translate -i "The future of AI is local" --src en --dst de
# "Die Zukunft der KI ist lokal"

# Chat with a local LLM
kjarni chat llama3.2-3b-instruct
# > What is recursion?
# Recursion is when a function calls itself...

# Transcribe audio
kjarni transcribe meeting.mp3 --timestamps
# [00:00.000 --> 00:12.480] We need to finalize the Q4 roadmap...

# Summarize a document
cat article.txt | kjarni summarize
# AI has rapidly become a central component of modern software...

# Build a search index and query it
kjarni index create ./docs --name my-docs
kjarni search my-docs "password reset" --top-k 5
```

Models are downloaded automatically on first use and cached locally.

## Capabilities

| Capability | What it does | Models |
|---|---|---|
| **Chat** | Conversational AI with streaming | Llama, Mistral, Qwen, Phi, DeepSeek |
| **Generation** | Text completion from base models | GPT-2, Llama |
| **Embeddings** | Dense vectors for semantic search | Nomic, BGE, MiniLM |
| **Search** | Vector + BM25 hybrid retrieval | HNSW, BM25 |
| **Reranking** | Cross-encoder relevance scoring | MiniLM Cross-Encoder |
| **Classification** | Sentiment, emotion, toxicity | DistilBERT, RoBERTa, BERT |
| **Translation** | Neural machine translation | Flan-T5 |
| **Transcription** | Speech-to-text with timestamps | Whisper |
| **Summarization** | Condense long documents | BART, T5 |

## API Examples

### Classification

<table>
<tr><th>Rust</th><th>C#</th></tr>
<tr>
<td>

```rust
use kjarni::Classifier;

let clf = Classifier::new("distilbert-sentiment")?;
let result = clf.classify("I love this!")?;

println!("{}: {:.1}%", result.label, result.score * 100.0);
// POSITIVE: 99.8%
```

</td>
<td>

```csharp
using Kjarni;

var clf = new Classifier("distilbert-sentiment");
var result = clf.Classify("I love this!");

Console.WriteLine($"{result.Label}: {result.Score:P1}");
// POSITIVE: 99.8%
```

</td>
</tr>
<tr><th>Python</th><th>CLI</th></tr>
<tr>
<td>

```python
from kjarni import Classifier

clf = Classifier("distilbert-sentiment")
result = clf.classify("I love this!")

print(f"{result.label}: {result.score:.1%}")
# POSITIVE: 99.8%
```

</td>
<td>

```bash
echo "I love this!" | kjarni classify
# POSITIVE  99.8%  ██████████████████████████████████████
```

</td>
</tr>
</table>

### Embeddings

<table>
<tr><th>Rust</th><th>C#</th></tr>
<tr>
<td>

```rust
use kjarni::Embedder;

let emb = Embedder::new("nomic-embed-text")?;
let vec = emb.embed("semantic search")?;

println!("Dimension: {}", vec.len()); // 768
```

</td>
<td>

```csharp
using Kjarni;

var emb = new Embedder("nomic-embed-text");
var vec = emb.Embed("semantic search");

Console.WriteLine($"Dimension: {vec.Length}"); // 768
```

</td>
</tr>
<tr><th>Python</th><th>CLI</th></tr>
<tr>
<td>

```python
from kjarni import Embedder

emb = Embedder("nomic-embed-text")
vec = emb.embed("semantic search")

print(f"Dimension: {len(vec)}")  # 768
```

</td>
<td>

```bash
kjarni embed "semantic search" -m nomic-embed-text
# 0.1622 0.0428 0.0673 0.2243 -0.1236 ...
```

</td>
</tr>
</table>

### Chat

<table>
<tr><th>Rust</th><th>C#</th></tr>
<tr>
<td>

```rust
use kjarni::Chat;

let mut chat = Chat::new("llama3.2-3b-instruct")?;
chat.system("You are helpful.");

for token in chat.stream("What is Rust?")? {
    print!("{}", token);
}
```

</td>
<td>

```csharp
using Kjarni;

var chat = new Chat("llama3.2-3b-instruct");
chat.System("You are helpful.");

await foreach (var token in chat.Stream("What is Rust?"))
    Console.Write(token);
```

</td>
</tr>
<tr><th>Python</th><th>CLI</th></tr>
<tr>
<td>

```python
from kjarni import Chat

chat = Chat("llama3.2-3b-instruct")
chat.system("You are helpful.")

for token in chat.stream("What is Rust?"):
    print(token, end="", flush=True)
```

</td>
<td>

```bash
kjarni chat llama3.2-3b-instruct \
  -s "You are helpful."
```

</td>
</tr>
</table>

### Search & RAG

<table>
<tr><th>Rust</th><th>C#</th></tr>
<tr>
<td>

```rust
use kjarni::{Embedder, Indexer};

let emb = Embedder::new("nomic-embed-text")?;
let mut idx = Indexer::new(emb.dim());

for doc in &docs {
    idx.add(emb.embed(&doc.text)?, doc.id);
}

let results = idx.search(emb.embed("query")?, 10);
```

</td>
<td>

```csharp
using Kjarni;

var emb = new Embedder("nomic-embed-text");
var idx = new Indexer(emb.Dim);

foreach (var doc in docs)
    idx.Add(emb.Embed(doc.Text), doc.Id);

var results = idx.Search(emb.Embed("query"), k: 10);
```

</td>
</tr>
<tr><th>Python</th><th>CLI</th></tr>
<tr>
<td>

```python
from kjarni import Embedder, Indexer

emb = Embedder("nomic-embed-text")
idx = Indexer(emb.dimension)

for doc in docs:
    idx.add(emb.embed(doc.text), doc.id)

results = idx.search(emb.embed("query"), k=10)
```

</td>
<td>

```bash
kjarni index create ./docs --name my-docs
kjarni search my-docs "password reset" -k 10
```

</td>
</tr>
</table>

### Translation

<table>
<tr><th>Rust</th><th>C#</th></tr>
<tr>
<td>

```rust
use kjarni::Translator;

let tr = Translator::new("flan-t5-large")?;
let out = tr.translate("Hello", "en", "is")?;

println!("{}", out); // "Halló"
```

</td>
<td>

```csharp
using Kjarni;

var tr = new Translator("flan-t5-large");
var result = tr.Translate("Hello", "en", "is");

Console.WriteLine(result); // "Halló"
```

</td>
</tr>
<tr><th>Python</th><th>CLI</th></tr>
<tr>
<td>

```python
from kjarni import Translator

tr = Translator("flan-t5-large")
result = tr.translate("Hello", src="en", dst="is")

print(result)  # "Halló"
```

</td>
<td>

```bash
kjarni translate -i "Hello" --src en --dst is
# "Halló"
```

</td>
</tr>
</table>

### Transcription

<table>
<tr><th>Rust</th><th>C#</th></tr>
<tr>
<td>

```rust
use kjarni::Transcriber;

let whisper = Transcriber::new("whisper-small")?;
let result = whisper.transcribe("meeting.mp3")?;

for seg in result.segments {
    println!("[{:?}] {}", seg.start, seg.text);
}
```

</td>
<td>

```csharp
using Kjarni;

var whisper = new Transcriber("whisper-small");
var result = whisper.Transcribe("meeting.mp3");

foreach (var seg in result.Segments)
    Console.WriteLine($"[{seg.Start}] {seg.Text}");
```

</td>
</tr>
<tr><th>Python</th><th>CLI</th></tr>
<tr>
<td>

```python
from kjarni import Transcriber

whisper = Transcriber("whisper-small")
result = whisper.transcribe("meeting.mp3")

for seg in result.segments:
    print(f"[{seg.start}] {seg.text}")
```

</td>
<td>

```bash
kjarni transcribe meeting.mp3 --timestamps
# [00:00.000 --> 00:12.480] We need to finalize...
```

</td>
</tr>
</table>

### Summarization

<table>
<tr><th>Rust</th><th>C#</th></tr>
<tr>
<td>

```rust
use kjarni::Summarizer;

let sum = Summarizer::new("bart-large-cnn")?;
let summary = sum.summarize(&long_text)?;

println!("{}", summary);
```

</td>
<td>

```csharp
using Kjarni;

var sum = new Summarizer("bart-large-cnn");
var summary = sum.Summarize(longText);

Console.WriteLine(summary);
```

</td>
</tr>
<tr><th>Python</th><th>CLI</th></tr>
<tr>
<td>

```python
from kjarni import Summarizer

summ = Summarizer("bart-large-cnn")
summary = summ.summarize(long_text)

print(summary)
```

</td>
<td>

```bash
cat article.txt | kjarni summarize
```

</td>
</tr>
</table>

## Model Registry

28 curated, production-tested models. Downloaded on first use, cached at `~/.cache/kjarni`.

```bash
kjarni model list                     # See all available models
kjarni model download nomic-embed-text # Download a specific model
kjarni model info llama3.2-3b-instruct # Show model details
```

<details>
<summary><strong>Full model list</strong></summary>

**Chat / LLM**

| Model | Parameters | Format | Notes |
|---|---|---|---|
| `qwen2.5-0.5b-instruct` | 490M | GGUF | Tiny, fast structured output |
| `qwen2.5-1.5b-instruct` | 1.5B | GGUF | Balanced edge model |
| `llama3.2-1b-instruct` | 1.2B | GGUF | Meta's official edge model |
| `llama3.2-3b-instruct` | 3.2B | GGUF | Best balance of speed and quality |
| `phi3.5-mini` | 3.8B | GGUF | Microsoft's reasoning model |
| `mistral-7b` | 7.2B | GGUF | Reliable workhorse |
| `llama3.1-8b-instruct` | 8.0B | GGUF | Production standard |
| `deepseek-r1-8b` | 8.0B | GGUF | State-of-the-art reasoning |

**Embeddings**

| Model | Parameters | Notes |
|---|---|---|
| `minilm-l6-v2` | 22M | Fastest, good for prototyping |
| `mpnet-base-v2` | 110M | High quality general purpose |
| `nomic-embed-text` | 137M | Modern standard, 8K context |
| `bge-m3` | 567M | Multilingual, state-of-the-art |
| `distilbert-base` | 66M | Lightweight DistilBERT |

**Classifiers**

| Model | Parameters | Notes |
|---|---|---|
| `distilbert-sentiment` | 66M | Binary positive/negative |
| `roberta-sentiment` | 125M | 3-class, social media tuned |
| `bert-sentiment-multilingual` | 168M | 5-star rating, 6 languages |
| `roberta-emotions` | 125M | 28 emotions, multi-label |
| `distilroberta-emotion` | 82M | 7 basic emotions |
| `toxic-bert` | 110M | 6 toxicity categories |

**Seq2Seq / Translation / Summarization**

| Model | Parameters | Notes |
|---|---|---|
| `flan-t5-base` | 250M | General purpose text-to-text |
| `flan-t5-large` | 780M | Higher quality translation |
| `distilbart-cnn` | 306M | Fast summarization |
| `bart-large-cnn` | 406M | High quality summarization |

**Transcription**

| Model | Parameters | Notes |
|---|---|---|
| `whisper-small` | 244M | Fast, good accuracy |
| `whisper-large-v3` | 1.5B | Highest accuracy |

**Rerankers**

| Model | Parameters | Notes |
|---|---|---|
| `minilm-l6-v2-cross-encoder` | 22M | Fast passage reranking |

</details>

## How It Works

Kjarni is a native inference engine written in Rust. It loads SafeTensors and GGUF model weights directly, runs its own tokenizers, and executes transformer architectures (encoder, decoder, encoder-decoder) without any external ML framework.

- **CPU**: Optimized with SIMD intrinsics (AVX2/AVX-512 on x86, NEON on ARM)
- **GPU**: Optional acceleration via WebGPU (no CUDA required)
- **Quantization**: GGUF Q4/Q8 for running large models on modest hardware
- **Precision**: f32, f16, bf16 support

### Platforms

| Platform | Architectures | Status |
|---|---|---|
| Linux | x64, ARM64 | ✅ |
| macOS | Apple Silicon, Intel | ✅ |
| Windows | x64, ARM64 | ✅ |

### Language Bindings

| Language | Package | Status |
|---|---|---|
| Rust | [`kjarni`](https://crates.io/crates/kjarni) | ✅ Native |
| C# / .NET | [`Kjarni`](https://www.nuget.org/packages/Kjarni) | ✅ Available |
| Python | [`kjarni`](https://pypi.org/project/kjarni) | ✅ Available |
| JavaScript | [`kjarni`](https://www.npmjs.com/package/kjarni) | 🚧 WASM, coming soon |
| Go | | 🚧 Coming soon |
| C / C++ | `kjarni.h` | 🚧 Coming soon |

## What Kjarni is NOT

Kjarni solves **one thing**: running pretrained models in production apps.

- Not a training framework — use PyTorch
- Not a fine-tuning toolkit — use Hugging Face
- Not an LLM orchestration layer — use LangChain
- Not a research tool — use transformers

If you need to train models or build agent pipelines, use the right tool for the job.

## Why Kjarni Exists

I wanted to use transformer models in my C# and Rust projects. Every path led to pain: Python interop with version conflicts, Docker containers for simple classification, cloud APIs with latency and privacy concerns, or linking C++ libraries with bespoke build systems.

Kjarni is the library I wished existed. One dependency, a task-level API that hides the ML complexity, and models that just work out of the box. The same principles that made SQLite ubiquitous — self-contained, zero-config, cross-platform, offline-first — applied to AI inference.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <a href="https://kjarni.ai">kjarni.ai</a> · <a href="https://docs.kjarni.ai">Docs</a> · <a href="https://github.com/olafurjohannsson/kjarni/issues">Issues</a>
</p>