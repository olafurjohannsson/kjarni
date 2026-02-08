# Kjarni

AI inference for .NET — embeddings, classification, reranking, and search.  
No Python. No ONNX. No CUDA. One NuGet package.

## Install

```bash
dotnet add package Kjarni
```

## Quick Start

### Classify Text

```csharp
using Kjarni;

using var classifier = new Classifier("distilbert-sentiment");
var result = classifier.Classify("I love this product!");

Console.WriteLine(result);
// POSITIVE (99.9%)
```

### Generate Embeddings

```csharp
using var embedder = new Embedder("minilm-l6-v2");
float[] vector = embedder.Encode("Hello world");

Console.WriteLine($"Dimensions: {vector.Length}");
// Dimensions: 384
```

### Compute Similarity

```csharp
using var embedder = new Embedder("minilm-l6-v2");
float score = embedder.Similarity("cat", "dog");

Console.WriteLine($"Similarity: {score:F4}");
// Similarity: 0.8012
```

### Rerank Search Results

```csharp
using var reranker = new Reranker();

var results = reranker.Rerank(
    "What is machine learning?",
    new[] {
        "Machine learning is a subset of artificial intelligence.",
        "The weather today is sunny.",
        "Deep learning uses neural networks with many layers.",
    });

foreach (var r in results)
    Console.WriteLine($"[{r.Index}] {r.Score:F4} {r.Document}");
```

### Batch Embeddings

```csharp
using var embedder = new Embedder("minilm-l6-v2");
float[][] vectors = embedder.EncodeBatch(new[] {
    "First document",
    "Second document",
    "Third document",
});
```

### Index & Search Documents (RAG)

```csharp
// Index a folder of documents
using var indexer = new Indexer(model: "minilm-l6-v2");
var stats = indexer.Create("my_index", new[] { "docs/" });
Console.WriteLine($"Indexed {stats.DocumentsIndexed} documents");

// Search the index
using var searcher = new Searcher(model: "minilm-l6-v2");
var results = searcher.Search("my_index", "What is machine learning?");

foreach (var r in results)
    Console.WriteLine($"{r.Score:F4}: {r.Text[..60]}...");
```

## Supported Models

| Task | Model | Description |
|------|-------|-------------|
| Embeddings | `minilm-l6-v2` | Fast, 384 dimensions |
| Embeddings | `bge-small-en` | High quality, 384 dimensions |
| Embeddings | `gte-small` | General text, 384 dimensions |
| Classification | `distilbert-sentiment` | Positive/negative sentiment |
| Classification | `roberta-emotion` | 28 emotion labels |
| Reranking | `minilm-l6-v2-cross-encoder` | Cross-encoder reranker |

Models are downloaded automatically on first use and cached locally.

## Configuration

All constructors accept optional parameters:

```csharp
// GPU acceleration
using var embedder = new Embedder("minilm-l6-v2", device: "gpu");

// Custom cache directory
using var classifier = new Classifier("distilbert-sentiment", cacheDir: "/models");

// Suppress download progress
using var reranker = new Reranker(quiet: true);
```

### Indexer Options

```csharp
using var indexer = new Indexer(
    model: "minilm-l6-v2",
    chunkSize: 512,
    chunkOverlap: 50,
    batchSize: 32,
    extensions: new[] { "txt", "md", "pdf" },
    recursive: true
);
```

### Search Options

```csharp
var results = searcher.Search("my_index", "query",
    mode: SearchMode.Hybrid,    // Keyword, Semantic, or Hybrid
    topK: 10,
    threshold: 0.5f
);
```

## How It Works

Kjarni is a native inference engine written in Rust with hand-tuned SIMD kernels (AVX2/FMA on x86, NEON on ARM). It loads HuggingFace models directly from safetensors using memory-mapped I/O — no Python runtime, no ONNX conversion step, no GPU required.

The C# package includes precompiled native libraries for each platform. `dotnet add package Kjarni` is all you need.

## Platform Support

| Platform | Architecture | Status |
|----------|-------------|--------|
| Linux | x64 | ✅ |
| Linux | ARM64 | ✅ |
| Windows | x64 | ✅ |
| macOS | x64 | ✅ |
| macOS | ARM64 (Apple Silicon) | ✅ |

## License

MIT