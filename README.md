# Kjarni

A native library for running machine learning models.

Kjarni is designed to be linked directly into your application.
It compiles to a single shared library and runs locally, without
Python, containers, external services, or GPU requirements.

C# bindings are available today via [NuGet](https://www.nuget.org/packages/Kjarni).
Go, Rust, Python, and C++ bindings are planned.

The name is Icelandic [ˈkʰjartnɪ]. It means "core."

## Install

```bash
dotnet add package Kjarni
```

## Quick Start

```csharp
using Kjarni;

using var classifier = new Classifier("roberta-sentiment");
Console.WriteLine(classifier.Classify("I love this product!"));
// positive (98.5%)
```

Models download on first use and are cached locally. No setup or configuration required.

## Examples

Same models, same results.

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

### Three-Class Sentiment

```csharp
using var classifier = new Classifier("roberta-sentiment");
var result = classifier.Classify("Terrible quality, broke after one day.");
Console.WriteLine(result.ToJson());
```

```json
{
  "label": "negative",
  "score": 0.9408,
  "predictions": [
    {"label": "negative", "score": 0.9408},
    {"label": "neutral", "score": 0.0509},
    {"label": "positive", "score": 0.0083}
  ]
}
```

### Multilingual Sentiment

```csharp
using var classifier = new Classifier("bert-sentiment-multilingual");
var result = classifier.Classify("Esta es la peor compra que he hecho.");
Console.WriteLine(result.ToJson());
```

```json
{
  "label": "1 star",
  "score": 0.9407,
  "predictions": [
    {"label": "1 star", "score": 0.9407},
    {"label": "2 stars", "score": 0.0514},
    {"label": "3 stars", "score": 0.0060},
    {"label": "5 stars", "score": 0.0015},
    {"label": "4 stars", "score": 0.0005}
  ]
}
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

### Emotion Detection

```csharp
using var classifier = new Classifier("distilroberta-emotion");
var result = classifier.Classify("I just got promoted!");
Console.WriteLine(result.ToJson());
```

```json
{
  "label": "surprise",
  "score": 0.5066,
  "predictions": [
    {"label": "surprise", "score": 0.5066},
    {"label": "anger", "score": 0.2376},
    {"label": "joy", "score": 0.0980},
    {"label": "neutral", "score": 0.0664},
    {"label": "disgust", "score": 0.0658},
    {"label": "sadness", "score": 0.0221},
    {"label": "fear", "score": 0.0035}
  ]
}
```

### Semantic Similarity

```csharp
using var embedder = new Embedder("minilm-l6-v2");
Console.WriteLine(embedder.Similarity("doctor", "physician"));
// 0.8598132
```

### Semantic Search

```csharp
using var embedder = new Embedder("minilm-l6-v2");

var docs = new[] {
    "How do I reset my password?",
    "What is your refund policy?",
    "Do you ship internationally?",
};
var vectors = embedder.EncodeBatch(docs);
var query = embedder.Encode("I need to change my login credentials");

for (int i = 0; i < docs.Length; i++)
{
    var score = Embedder.CosineSimilarity(query, vectors[i]);
    Console.WriteLine($"  {score:F4}: {docs[i]}");
}
//  0.5981: How do I reset my password?
// -0.0027: What is your refund policy?
// -0.0451: Do you ship internationally?
```

No keyword overlap between "change my login credentials" and "reset my password."

### Reranking

```csharp
using var reranker = new Reranker();
var results = reranker.Rerank(
    "What is machine learning?",
    new[] {
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "The weather today is sunny.",
    });

foreach (var r in results)
    Console.WriteLine($"  {r.Score:F4}: {r.Document}");
//  10.5139: Machine learning is a subset of artificial intelligence.
//  -5.5301: Deep learning uses neural networks with many layers.
// -11.1001: The weather today is sunny.
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

### GPU

```csharp
using var embedder = new Embedder("minilm-l6-v2", device: "gpu");
```

GPU inference is optional and uses WebGPU.
Vulkan on Linux, DX12/Vulkan on Windows. CUDA is not required.

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
| Linux x64 | Yes | Yes Vulkan |
| Windows x64 | Yes | Yes DX12/Vulkan |
| macOS ARM64 | Planned | Planned (Metal) |

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
cargo test
```

## License

MIT or Apache-2.0, at your option.