# Kjarni

AI inference for .NET — embeddings, classification, reranking, and search.  
No Python. No ONNX. No API keys. One NuGet package.

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

### Content Moderation

```csharp
using var classifier = new Classifier("toxic-bert");
var result = classifier.Classify("You are an idiot");
var toxicScore = result.AllScores.First(s => s.Label == "toxic").Score;
Console.WriteLine($"Toxic: {toxicScore > 0.5} ({toxicScore:P0})");
// Toxic: True (72%)
```

### Semantic Search

```csharp
using var embedder = new Embedder("minilm-l6-v2");

// Index your documents
var docs = new[] {
    "Refunds are processed within 5-7 business days.",
    "Shipping takes 3-5 business days.",
    "Premium members get free expedited shipping.",
};
var docVectors = docs.Select(d => embedder.Encode(d)).ToArray();

// Search by meaning — no keyword overlap needed
var query = embedder.Encode("how do I get my money back?");
var best = docVectors
    .Select((v, i) => (Doc: docs[i], Score: Embedder.CosineSimilarity(query, v)))
    .OrderByDescending(r => r.Score)
    .First();

Console.WriteLine($"{best.Score:F3}: {best.Doc}");
// 0.489: Refunds are processed within 5-7 business days.
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
// Similarity: 0.6606
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

## Models

Models are downloaded automatically on first use and cached locally.

| Task | Model | Size | Description |
|------|-------|------|-------------|
| Embeddings | `minilm-l6-v2` | 90MB | Fast, 384 dimensions |
| Embeddings | `mpnet-base-v2` | 420MB | High quality, 768 dimensions |
| Embeddings | `distilbert-base` | 260MB | General purpose, 768 dimensions |
| Classification | `distilbert-sentiment` | 268MB | Positive/negative |
| Classification | `roberta-sentiment` | 499MB | Negative/neutral/positive |
| Classification | `bert-sentiment-multilingual` | 681MB | 5-star rating, 6 languages |
| Classification | `distilroberta-emotion` | 329MB | 7 emotions |
| Classification | `toxic-bert` | 438MB | Toxicity detection, 6 labels |
| Reranking | `minilm-l6-v2-cross-encoder` | 90MB | Passage reranking |

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
| Windows | x64 | ✅ |
| macOS | ARM64 (Apple Silicon) | Planned |
| macOS | x64 | Planned |
| Linux | ARM64 | Planned |

## License

MIT