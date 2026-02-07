<p align="center">
  <img src="assets/logo.svg" width="80" height="92" alt="Kjarni Logo">
</p>

<h1 align="center">kjarni</h1>

<p align="center">
  <strong>The SQLite of AI</strong><br>
  <sub>A single library for AI inference. No Python. No CUDA. No containers. Just works.</sub>
</p>

<p align="center">
  <a href="https://crates.io/crates/kjarni"><img src="https://img.shields.io/crates/v/kjarni.svg" alt="Crates.io"></a>
  <a href="https://www.nuget.org/packages/Kjarni"><img src="https://img.shields.io/nuget/v/Kjarni.svg" alt="NuGet"></a>
  <a href="https://pypi.org/project/kjarni"><img src="https://img.shields.io/pypi/v/kjarni.svg" alt="PyPI"></a>
  <a href="https://github.com/wyvern/kjarni/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

---

Kjarni is a self-contained AI inference engine. Embeddings, chat, translation, transcription, classification, search, reranking, and summarization — all in one library with zero external dependencies.

**Why Kjarni?** I wanted to use transformer models in C# and Rust without Python bindings, Docker containers, or linking llama.cpp. Kjarni is the library I wished existed: add one dependency and start inferencing.

## Install

```bash
# CLI
cargo install kjarni

# Rust
cargo add kjarni

# Python
pip install kjarni

# C# / .NET
dotnet add package Kjarni
```

## Quick Start

```bash
# Classify sentiment
echo "I love this product" | kjarni classify
# POSITIVE  99.8%  ████████████████████████████████████████

# Translate to Icelandic
kjarni translate -i "Hello, how are you?" --dst is
# "Halló, hvernig hefur þú það?"

# Chat with Llama
kjarni chat llama3.2-3b-instruct
# > What is recursion?
# Recursion is when a function calls itself...

# Transcribe audio
kjarni transcribe meeting.mp3
# "We need to finalize the Q4 roadmap by Friday..."
```

## Capabilities

| Capability | What it does | Models |
|------------|--------------|--------|
| **Chat** | Conversational AI with streaming | Llama, Mistral, Qwen, Phi |
| **Generation** | Text completion | GPT-2, Llama (base) |
| **Embeddings** | Dense vectors for semantic search | Nomic, BGE, MiniLM, E5 |
| **Search** | Vector + keyword hybrid search | HNSW, BM25 |
| **Reranking** | Cross-encoder relevance scoring | MiniLM, BGE Reranker |
| **Classification** | Sentiment, emotion, toxicity | DistilBERT, RoBERTa |
| **Translation** | 100+ language pairs | NLLB, Flan-T5 |
| **Transcription** | Speech-to-text with timestamps | Whisper |
| **Summarization** | Condense long documents | BART, T5 |

## Examples

### Chat

<table>
<tr><th>CLI</th><th>Rust</th></tr>
<tr>
<td>

```bash
kjarni chat llama3.2-3b-instruct \
  -s "You are a helpful assistant"
```

</td>
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
</tr>
<tr><th>Python</th><th>C#</th></tr>
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

```csharp
using Kjarni;

var chat = new Chat("llama3.2-3b-instruct");
chat.System("You are helpful.");

await foreach (var token in chat.Stream("What is Rust?"))
    Console.Write(token);
```

</td>
</tr>
</table>

### Embeddings

<table>
<tr><th>CLI</th><th>Rust</th></tr>
<tr>
<td>

```bash
kjarni embed "semantic search" \
  -m nomic-embed-text \
  --format json
```

</td>
<td>

```rust
use kjarni::Embedder;

let emb = Embedder::new("nomic-embed-text")?;
let vec = emb.embed("semantic search")?;

println!("Dimension: {}", vec.len()); // 768
```

</td>
</tr>
<tr><th>Python</th><th>C#</th></tr>
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

```csharp
using Kjarni;

var emb = new Embedder("nomic-embed-text");
var vec = emb.Embed("semantic search");

Console.WriteLine($"Dimension: {vec.Length}");
```

</td>
</tr>
</table>

### Classification

<table>
<tr><th>CLI</th><th>Rust</th></tr>
<tr>
<td>

```bash
echo "I love this!" | kjarni classify
# POSITIVE  99.8%

kjarni classify "The server is down" \
  --labels "urgent,normal,low"
```

</td>
<td>

```rust
use kjarni::Classifier;

let clf = Classifier::new("distilbert-sentiment")?;
let result = clf.classify("I love this!")?;

println!("{}: {:.1}%", result.label, result.score * 100.0);
```

</td>
</tr>
<tr><th>Python</th><th>C#</th></tr>
<tr>
<td>

```python
from kjarni import Classifier

clf = Classifier("distilbert-sentiment")
result = clf.classify("I love this!")

print(f"{result.label}: {result.score:.1%}")
```

</td>
<td>

```csharp
using Kjarni;

var clf = new Classifier("distilbert-sentiment");
var result = clf.Classify("I love this!");

Console.WriteLine($"{result.Label}: {result.Score:P1}");
```

</td>
</tr>
</table>

### Translation

<table>
<tr><th>CLI</th><th>Rust</th></tr>
<tr>
<td>

```bash
kjarni translate \
  -i "Hello, how are you?" \
  --src en --dst is
# "Halló, hvernig hefur þú það?"
```

</td>
<td>

```rust
use kjarni::Translator;

let tr = Translator::new("flan-t5-large")?;
let result = tr.translate("Hello", "en", "is")?;

println!("{}", result); // "Halló"
```

</td>
</tr>
<tr><th>Python</th><th>C#</th></tr>
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

```csharp
using Kjarni;

var tr = new Translator("flan-t5-large");
var result = tr.Translate("Hello", "en", "is");

Console.WriteLine(result); // "Halló"
```

</td>
</tr>
</table>

### Transcription

<table>
<tr><th>CLI</th><th>Rust</th></tr>
<tr>
<td>

```bash
kjarni transcribe meeting.mp3

# With timestamps
kjarni transcribe meeting.mp3 --timestamps
```

</td>
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
</tr>
<tr><th>Python</th><th>C#</th></tr>
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

```csharp
using Kjarni;

var whisper = new Transcriber("whisper-small");
var result = whisper.Transcribe("meeting.mp3");

foreach (var seg in result.Segments)
    Console.WriteLine($"[{seg.Start}] {seg.Text}");
```

</td>
</tr>
</table>

### Search & Rerank

<table>
<tr><th>CLI</th><th>Rust</th></tr>
<tr>
<td>

```bash
# Build an index
kjarni index create ./docs --name my-docs

# Search
kjarni search my-docs "password reset" -k 10

# Rerank results
kjarni rerank "best language" \
  "Python is easy" "Rust is fast" "Go is simple"
```

</td>
<td>

```rust
use kjarni::{Embedder, Indexer, Reranker};

let emb = Embedder::new("nomic-embed-text")?;
let mut idx = Indexer::new(emb.dim());

for doc in &docs {
    idx.add(emb.embed(&doc.text)?, doc.id);
}

let results = idx.search(emb.embed("query")?, 10);
```

</td>
</tr>
</table>

### Summarization

<table>
<tr><th>CLI</th><th>Rust</th></tr>
<tr>
<td>

```bash
cat article.txt | kjarni summarize

kjarni summarize paper.pdf --max-length 100
```

</td>
<td>

```rust
use kjarni::Summarizer;

let sum = Summarizer::new("bart-large-cnn")?;
let summary = sum.summarize(&long_text)?;

println!("{}", summary);
```

</td>
</tr>
<tr><th>Python</th><th>C#</th></tr>
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

```csharp
using Kjarni;

var sum = new Summarizer("bart-large-cnn");
var summary = sum.Summarize(longText);

Console.WriteLine(summary);
```

</td>
</tr>
</table>

## Supported Models

Models are downloaded on first use and cached locally (~/.cache/kjarni).

| Category | Models |
|----------|--------|
| **Chat** | `qwen2.5-0.5b-instruct`, `qwen2.5-1.5b-instruct`, `llama3.2-1b-instruct`, `llama3.2-3b-instruct`, `phi3.5-mini`, `mistral-7b`, `llama3.1-8b-instruct`, `deepseek-r1-8b` |
| **Generation** | `gpt2`, `distilgpt2` |
| **Embeddings** | `minilm-l6-v2`, `mpnet-base-v2`, `nomic-embed-text`, `bge-m3` |
| **Rerankers** | `minilm-l6-v2-cross-encoder` |
| **Classifiers** | `distilbert-sentiment`, `roberta-sentiment`, `bert-sentiment-multilingual`, `roberta-emotions`, `distilroberta-emotion`, `toxic-bert` |
| **Summarization** | `distilbart-cnn`, `bart-large-cnn` |
| **Translation** | `flan-t5-base`, `flan-t5-large` |
| **Transcription** | `whisper-small`, `whisper-large-v3` |

See [full model registry](https://docs.kjarni.ai/models) for all options.

## Technical Details

### Precision & Quantization

Kjarni supports multiple precision modes for balancing speed, memory, and quality:

| Format | Memory | Speed | Quality | Use case |
|--------|--------|-------|---------|----------|
| **f32** | Baseline | Baseline | Best | When accuracy matters most |
| **f16** | 50% | Faster | Excellent | Default for most models |
| **bf16** | 50% | Faster | Excellent | Better for training-derived weights |
| **GGUF Q8** | 25% | Fast | Very good | Good balance |
| **GGUF Q4** | 12.5% | Fastest | Good | Maximum efficiency |

```bash
# Use specific dtype
kjarni chat llama3.2-3b-instruct --dtype bf16

# Use quantized GGUF
kjarni chat llama3.2-3b-instruct --quantization q4
```

### Platforms

| Platform | Architectures | Status |
|----------|---------------|--------|
| Linux | x64, ARM64 | ✅ |
| macOS | Apple Silicon, Intel | ✅ |
| Windows | x64, ARM64 | ✅ |

### Language Bindings

| Language | Package | Status |
|----------|---------|--------|
| Rust | `kjarni` | ✅ Native |
| Python | `kjarni` | ✅ Available |
| C# / .NET | `Kjarni` | ✅ Available |
| Go | `kjarni-go` | 🚧 Coming soon |
| C++ | `kjarni.h` | 🚧 Coming soon |

### Execution Backends

- **CPU**: Optimized with SIMD (AVX2/AVX-512, NEON)
- **GPU**: Optional acceleration via WebGPU

## What Kjarni is NOT

Kjarni is focused on **inference**, not everything:

- ❌ Not a training framework (use PyTorch)
- ❌ Not a fine-tuning toolkit (use Hugging Face)
- ❌ Not an LLM orchestration layer (use LangChain)
- ❌ Not a research tool (use transformers)

**Kjarni solves one thing:** running pretrained models in production apps.

## Philosophy

Kjarni follows the same principles that made SQLite ubiquitous:

1. **Self-contained** — Single library, no external dependencies
2. **Zero configuration** — Download a model, start inferencing
3. **Cross-platform** — Same code runs everywhere
4. **Offline-first** — No API keys, no cloud, no network required
5. **Private by default** — Your data never leaves your machine

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <a href="https://kjarni.ai">Website</a> •
  <a href="https://docs.kjarni.ai">Documentation</a> •
  <a href="https://github.com/wyvern/kjarni/issues">Issues</a>
</p>