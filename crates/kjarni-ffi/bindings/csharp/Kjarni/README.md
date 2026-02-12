# Kjarni

A native library for running machine learning models from C#.

Classification, embeddings, reranking, and semantic search.
No Python, no containers, no GPU requirements.

Full documentation on [GitHub](https://github.com/olafurjohannsson/kjarni).

## Quick Start

```csharp
using Kjarni;

using var classifier = new Classifier("roberta-sentiment");
Console.WriteLine(classifier.Classify("I love this product!"));
// positive (98.5%)
```

Models download on first use and are cached locally.

## Classification

```csharp
using var classifier = new Classifier("roberta-sentiment");
Console.WriteLine(classifier.Classify("Terrible quality.").ToJson());
// {"label": "negative", "score": 0.9408, "predictions": [...]}

using var multi = new Classifier("bert-sentiment-multilingual");
Console.WriteLine(multi.Classify("Esta es la peor compra que he hecho."));
// 1 star (94.1%)

using var toxic = new Classifier("toxic-bert");
Console.WriteLine(toxic.Classify("You are an idiot").ToDetailedString());
    //          toxic   98.61%  ███████████████████████████████████████
    //         insult   96.00%  ██████████████████████████████████████
    //        obscene   75.64%  ██████████████████████████████
    //   severe_toxic    4.56%  █
    //  identity_hate    1.41%  

using var emotion = new Classifier("distilroberta-emotion");
Console.WriteLine(emotion.Classify("I just got promoted!"));
// surprise (50.7%)
```

## Embeddings

```csharp
using var embedder = new Embedder("minilm-l6-v2");

float[] vector = embedder.Encode("Hello world");           // 384 dimensions
Console.WriteLine(embedder.Similarity("doctor", "physician")); // 0.8598

var docs = new[] { "How do I reset my password?", "What is your refund policy?" };
var vectors = embedder.EncodeBatch(docs);
var query = embedder.Encode("I need to change my login credentials");
var score = Embedder.CosineSimilarity(query, vectors[0]);  // 0.5981
```

## Reranking

```csharp
using var reranker = new Reranker();
var results = reranker.Rerank("What is machine learning?", new[] {
    "Machine learning is a subset of artificial intelligence.",
    "The weather today is sunny.",
});
//  10.5139: Machine learning is a subset of artificial intelligence.
// -11.1001: The weather today is sunny.
```

## Index & Search

```csharp
using var indexer = new Indexer(model: "minilm-l6-v2", quiet: true);
indexer.Create("my_index", new[] { "docs/" });

using var searcher = new Searcher(
    model: "minilm-l6-v2",
    rerankerModel: "minilm-l6-v2-cross-encoder");
var results = searcher.Search("my_index", "how do returns work?",
    mode: SearchMode.Hybrid);
```

Search modes: `Semantic`, `Keyword` (BM25), `Hybrid`.

## GPU

```csharp
using var embedder = new Embedder("minilm-l6-v2", device: "gpu");
```

WebGPU — Vulkan on Linux, DX12/Vulkan on Windows. CUDA is not required.

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

## Configuration

```csharp
// Custom cache directory
using var embedder = new Embedder("minilm-l6-v2", cacheDir: "/my/models");

// Quiet mode
using var embedder = new Embedder("minilm-l6-v2", quiet: true);
```

Set `KJARNI_CACHE_DIR` to override the default cache location.
Set `HF_TOKEN` for gated models.

## Platform Support

| Platform | CPU | GPU |
|----------|-----|-----|
| Linux x64 | Yes | Yes Vulkan |
| Windows x64 | Yes | Yes DX12/Vulkan |
| macOS ARM64 | Planned | Planned |