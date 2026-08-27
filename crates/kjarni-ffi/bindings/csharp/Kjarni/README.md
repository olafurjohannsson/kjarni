# Kjarni

**Local AI inference for .NET. No Python, no ONNX Runtime, no CUDA, no cloud.**

Embeddings, classification, semantic search, reranking, and local LLM chat — running in your
process against a native library. The only runtime dependency is glibc.

```bash
dotnet add package Kjarni
```

Native binaries for `linux-x64`, `linux-arm64`, `win-x64` and `osx-arm64` ship inside the package
and are selected automatically. There is no second install step, no model server to run, and no
API key.

## Quick start

```csharp
using Kjarni;

using var classifier = new Classifier("roberta-sentiment");
Console.WriteLine(classifier.Classify("I love this product!"));
// positive (98.5%)
```

Models download on first use and are cached locally. After that, nothing touches the network.

## Embeddings

```csharp
using var embedder = new Embedder("minilm-l6-v2");

float[] vector = embedder.Encode("Hello world");                // 384 dimensions
Console.WriteLine(embedder.Similarity("doctor", "physician"));   // 0.8598

var docs = new[] { "How do I reset my password?", "What is your refund policy?" };
var vectors = embedder.EncodeBatch(docs);
var query = embedder.Encode("I need to change my login credentials");
var score = Embedder.CosineSimilarity(query, vectors[0]);        // 0.5981
```

Pass whole collections to `EncodeBatch` rather than looping — Kjarni batches natively.

### Microsoft.Extensions.AI and Semantic Kernel

The companion package implements `IEmbeddingGenerator<string, Embedding<float>>` and
`IChatClient`, so Kjarni drops into the standard .NET AI abstractions — including Semantic
Kernel — with no cloud call and no Ollama daemon:

```bash
dotnet add package Kjarni.Extensions.AI
```

```cs
builder.Services.AddKjarniEmbeddingGenerator("minilm-l6-v2");
builder.Services.AddKjarniChatClient("llama3.2-3b-instruct");
```

See [Kjarni.Extensions.AI](https://www.nuget.org/packages/Kjarni.Extensions.AI).

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

## Chat

Local LLM chat, in-process. No daemon, no API key.

```csharp
using var chat = new Chat("llama3.2-3b-instruct");
Console.WriteLine(chat.Send("Explain retrieval-augmented generation in one sentence."));
```

Streaming, token by token:

```csharp
chat.Stream("Write a haiku about Reykjavík.", token =>
{
    Console.Write(token);
    return true;             // return false to stop generation early
});
```

Multi-turn conversations keep their own history:

```csharp
var convo = chat.Conversation();
convo.Send("My name is Ólafur.");
Console.WriteLine(convo.Send("What is my name?"));   // remembers
convo.Clear();                                        // keeps the system prompt
```

Sampling is configurable via `GenerationConfig`:

```csharp
var config = GenerationConfig.Default() with { Temperature = 0.2f, MaxNewTokens = 512 };
chat.Send("Summarise this changelog.", config);
```

`GenerationConfig.Greedy()` and `GenerationConfig.Creative()` are provided as presets.

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

Cross-encoder reranking scores query and document together, which is markedly more accurate than
comparing embeddings — worth applying to the top ~50 results of a search.

## Index and search

```csharp
using var indexer = new Indexer(model: "minilm-l6-v2", quiet: true);
indexer.Create("my_index", new[] { "docs/" });

using var searcher = new Searcher(
    model: "minilm-l6-v2",
    rerankerModel: "minilm-l6-v2-cross-encoder");

var results = searcher.Search("my_index", "how do returns work?", mode: SearchMode.Hybrid);
```

Search modes: `Semantic`, `Keyword` (BM25), `Hybrid`.

## Models

Sizes are what the model occupies on disk after download.

| Task | Model | Dimensions | On disk |
|------|-------|-----------|---------|
| Embeddings | `minilm-l6-v2` | 384 | 88 MB |
| Embeddings | `mpnet-base-v2` | 768 | 419 MB |
| Embeddings | `nomic-embed-text` | 768 | 523 MB |
| Embeddings (multilingual) | `bge-m3` | 1024 | ~2 GB |
| Reranking | `minilm-l6-v2-cross-encoder` | — | 88 MB |
| Sentiment (binary) | `distilbert-sentiment` | — | 257 MB |
| Sentiment (3-class) | `roberta-sentiment` | — | 479 MB |
| Sentiment (multilingual) | `bert-sentiment-multilingual` | — | 641 MB |
| Emotion (7-class) | `distilroberta-emotion` | — | 317 MB |
| Emotion (28-class) | `roberta-emotions` | — | 478 MB |
| Toxicity | `toxic-bert` | — | 419 MB |

Chat models range from `qwen2.5-0.5b-instruct` up through `llama3.2-3b-instruct`, `phi3.5-mini`,
`mistral-7b` and `deepseek-r1-8b`. Run `kjarni model list` with the CLI for the full registry.

Start with `minilm-l6-v2` for embeddings — at 384 dimensions it is fast on CPU and the quality gap
against much larger models is smaller than people expect.

## GPU

```csharp
using var embedder = new Embedder("minilm-l6-v2", device: "gpu");
```

GPU inference uses WebGPU — Vulkan on Linux, DX12 or Vulkan on Windows, Metal on macOS. CUDA is
not required and is not used.

## Configuration

```csharp
using var embedder = new Embedder("minilm-l6-v2", cacheDir: "/my/models");
using var quiet    = new Embedder("minilm-l6-v2", quiet: true);
```

`KJARNI_CACHE_DIR` overrides the default cache location. `HF_TOKEN` is used for gated models.

## Platform support

| Platform | Shipped | GPU backend |
|----------|---------|-------------|
| Linux x64 | Yes | Vulkan |
| Linux arm64 | Yes | Vulkan |
| Windows x64 | Yes | DX12 / Vulkan |
| macOS arm64 | Yes | Metal |

The native library links only against glibc — no CUDA, no BLAS, no ONNX Runtime, no Python.

## Links

- [Source and issues](https://github.com/olafurjohannsson/kjarni)
- [Kjarni.Extensions.AI](https://www.nuget.org/packages/Kjarni.Extensions.AI) — Microsoft.Extensions.AI provider
- [kjarni.ai](https://kjarni.ai)

MIT licensed.
