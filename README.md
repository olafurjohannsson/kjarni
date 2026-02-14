# Kjarni

A native library for running machine learning models.

Kjarni compiles to a single shared library and runs locally, without
Python, containers, external services, or GPU requirements. It also
ships as a command-line tool that reads from stdin, writes JSON, and
pipes like any UNIX tool.

C# bindings are available via [NuGet](https://www.nuget.org/packages/Kjarni).
The CLI is available as a [binary download](https://github.com/olafurjohannsson/kjarni/releases).

The name is Icelandic [ˈkʰjartnɪ]. It means "core."

## Install

**C# / .NET:**

```bash
dotnet add package Kjarni
```

**CLI (Linux x64):**

```bash
curl -fsSL https://kjarni.ai/install.sh | sh
```

## Quick Start

**C#:**

```csharp
using Kjarni;

using var classifier = new Classifier("roberta-sentiment");
Console.WriteLine(classifier.Classify("I love this product!"));
// positive (98.5%)
```

**CLI:**

```bash
$ kjarni classify "I love this product!"
positive   98.50%  ██████████████████████████████████████

$ echo "Terrible quality" | kjarni classify
negative   94.08%  █████████████████████████████████████
```

Models download on first use and are cached locally. No setup or configuration required.

## CLI

The CLI supports the same capabilities as the C# library. It reads
from arguments or stdin, and outputs human-readable tables or JSON.

```bash
# Sentiment analysis
$ kjarni classify "Best purchase ever"
positive   98.50%  ██████████████████████████████████████

# Pipe from stdin
$ echo "I hate mondays" | kjarni classify --model toxic-bert
toxic        2.79%  █
insult       1.14%
obscene      0.69%
severe       0.10%
identity     0.08%
threat       0.06%

# JSON output for scripting
$ echo "Great service" | kjarni classify --format json | jq '.label'
"positive"

# Semantic similarity
$ kjarni similarity "doctor" "physician"
0.8598

# Index a folder and search it
$ kjarni index create my-docs --inputs ./docs/
✓ Indexed 51 documents (186 KB)

$ kjarni search my-docs "password reset"
0.0325  account.txt   To reset your password, click...
0.0159  faq.txt       How do I change my login cred...

# Generate embeddings
$ kjarni embed "hello world" --format json | jq '.dimensions'
384
```

## Examples

### Embeddings

**Kjarni:**

```csharp
using var embedder = new Embedder("minilm-l6-v2");
float[] vector = embedder.Encode("Hello world");
Console.WriteLine(string.Join(", ", vector[..5]));
// -0.034477282, 0.03102318, 0.006734989, 0.02610899, -0.03936202
```

**sentence-transformers:**

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
vector = model.encode("Hello world", normalize_embeddings=True)
print(vector[:5])
# [-0.03447726  0.03102319  0.00673499  0.02610895 -0.03936201]
```

### Toxicity Detection

```csharp
using var classifier = new Classifier("toxic-bert");
Console.WriteLine(classifier.Classify("You are an idiot").ToDetailedString());
```

```
             toxic   98.61%  ███████████████████████████████████████
            insult   96.00%  ██████████████████████████████████████
           obscene   75.64%  ██████████████████████████████
      severe_toxic    4.56%  █
     identity_hate    1.41%  
```

### Semantic Similarity

```csharp
using var embedder = new Embedder("minilm-l6-v2");
Console.WriteLine(embedder.Similarity("doctor", "physician"));
// 0.8598132
```

### Index & Search

```bash
mkdir -p docs
```

**docs/returns.txt:**
```
Our return policy allows customers to return any unused item within 30 days of purchase for a full refund. Items must be in their original packaging. Shipping costs are non-refundable.
```

**docs/shipping.txt:**
```
We ship to all 50 US states and internationally to over 40 countries. Standard shipping takes 5-7 business days. Express shipping is available for an additional fee.
```

```csharp
using var indexer = new Indexer(model: "minilm-l6-v2", quiet: true);
indexer.Create("my_index", new[] { "docs/" });

using var searcher = new Searcher(
    model: "minilm-l6-v2",
    rerankerModel: "minilm-l6-v2-cross-encoder");

var results = searcher.Search("my_index", "how do returns work?",
    mode: SearchMode.Hybrid);

foreach (var r in results)
    Console.WriteLine($"  {r.Score:F4}: {r.Text}");
```

```
  1.3282: Our return policy allows customers to return any unused item
          within 30 days of purchase for a full refund. Items must be in
          their original packaging. Shipping costs are non-refundable.

-11.0939: We ship to all 50 US states and internationally to over 40
          countries. Standard shipping takes 5-7 business days. Express
          shipping is available for an additional fee.
```

Search modes: `Semantic` (vector similarity), `Keyword` (BM25), `Hybrid` (both).

## Models

| Task | Model | Size |
|------|-------|------|
| Sentiment (3-class) | `roberta-sentiment` | 125MB |
| Sentiment (multilingual) | `bert-sentiment-multilingual` | 168MB |
| Sentiment (binary) | `distilbert-sentiment` | 66MB |
| Emotion (7-class) | `distilroberta-emotion` | 82MB |
| Emotion (28-class) | `roberta-emotions` | 125MB |
| Toxicity | `toxic-bert` | 110MB |
| Embeddings | `minilm-l6-v2` | 90MB |
| Embeddings | `mpnet-base-v2` | 420MB |
| Reranking | `minilm-l6-v2-cross-encoder` | 90MB |

Models download on first use. The engine also supports Llama, Qwen2, Mistral, Phi-3, T5, BART, and Whisper.

Bindings and APIs for these models are intentionally not exposed in
the initial release and will ship in a future version.

## Configuration

### Cache Directory

Default locations:
- **Linux:** `~/.cache/kjarni`
- **Windows:** `%LOCALAPPDATA%\kjarni`

Override with `KJARNI_CACHE_DIR` or the constructor parameter:

```csharp
using var embedder = new Embedder("minilm-l6-v2", cacheDir: "/my/models");
```

### HuggingFace Token

For gated models:

```bash
export HF_TOKEN=hf_your_token_here
```

### Quiet Mode

```csharp
using var embedder = new Embedder("minilm-l6-v2", quiet: true);
```

## Platform Support

| Platform | CPU | GPU |
|----------|-----|-----|
| Linux x64 | Yes | Yes (Vulkan) |
| Windows x64 | Yes | Yes (DX12/Vulkan) |
| macOS ARM64 | Planned | Planned (Metal) |

GPU inference uses WebGPU. CUDA is not required.

```csharp
using var embedder = new Embedder("minilm-l6-v2", device: "gpu");
```

## How It Works

Kjarni does not wrap ONNX, LibTorch, or any external inference engine.
The runtime is written in Rust. The only system dependency is glibc 2.17.

- Hand-tuned SIMD kernels (AVX2/FMA, NEON)
- Custom WGSL compute shaders for GPU
- Zero-copy model loading via mmap
- BF16 compute path
- Quantization: Q4, Q6, Q8

## Why

Adding machine learning to an application usually means Python, CUDA,
containers, or an external service. Kjarni is a single `.so` or `.dll`
that runs as part of your application, with predictable behavior and
no external infrastructure.

## Project Structure

```
crates/
├── kjarni/              # High-level API
├── kjarni-transformers/  # Engine — models, kernels, GPU shaders
├── kjarni-ffi/          # C ABI + language bindings
│   └── bindings/
│       └── csharp/      # NuGet package
├── kjarni-cli/          # Command-line tool
└── kjarni-examples/     # Rust examples
```

## Building from Source

```bash
cargo build --release -p kjarni-ffi
cargo build --release -p kjarni-cli
cargo test
```

## License

MIT or Apache-2.0, at your option.
