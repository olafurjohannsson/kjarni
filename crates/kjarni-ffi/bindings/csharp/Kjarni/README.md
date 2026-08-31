# Kjarni

**Local AI inference for .NET. One package, running inside your process.**

Sentiment analysis, embeddings, semantic search, reranking and local LLM chat, on the CPU you
already have. Add the package, name a model, call a method. Models download on first use and
everything after that works offline.

The package has no dependencies and the native library links only against glibc.

```bash
dotnet add package Kjarni
```

Native binaries for `linux-x64`, `linux-arm64`, `win-x64` and `osx-arm64` ship inside the package
and are selected automatically. Installing the package is the whole setup.

## Quick start

Read the sentiment of a customer message in three lines:

```csharp
using Kjarni;

using var classifier = new Classifier("roberta-sentiment");
Console.WriteLine(classifier.Classify("I love this product!"));
// positive (98.5%)
```

That is the shape of the whole library. Every task is a class you construct with a model name.

## Embeddings and semantic similarity

Find text that means the same thing, even when it shares no words. This is what powers
"related articles", duplicate detection, and matching a question to an FAQ entry.

```csharp
using var embedder = new Embedder("minilm-l6-v2");

float[] vector = embedder.Encode("Hello world");                // 384 dimensions
Console.WriteLine(embedder.Similarity("doctor", "physician"));   // 0.8598

var docs = new[] { "How do I reset my password?", "What is your refund policy?" };
var vectors = embedder.EncodeBatch(docs);
var query = embedder.Encode("I need to change my login credentials");
var score = Embedder.CosineSimilarity(query, vectors[0]);        // 0.5981
```

Pass whole collections to `EncodeBatch` rather than looping. It runs the batch as a single
forward pass, which is substantially faster than the same texts one at a time.

### Microsoft.Extensions.AI and Semantic Kernel

The companion package implements `IEmbeddingGenerator<string, Embedding<float>>` and
`IChatClient`, so Kjarni drops into the standard .NET AI abstractions, Semantic Kernel included.
If your code already targets those interfaces, this is a registration change and nothing else:

```bash
dotnet add package Kjarni.Extensions.AI
```

```cs
builder.Services.AddKjarniEmbeddingGenerator("minilm-l6-v2");
builder.Services.AddKjarniChatClient("llama3.2-3b-instruct");
```

See [Kjarni.Extensions.AI](https://www.nuget.org/packages/Kjarni.Extensions.AI).

## Classification: sentiment, emotion and toxicity

Sort text into categories without training anything. Route support tickets by tone, flag abusive
comments before they post, or measure how customers feel about a release.

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

## Chat and text generation

Run a language model inside your application. Useful when the text cannot leave the building, or
when you want an assistant feature that keeps working without a network.

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

## Reranking: better search results

Your search returns twenty results and the right one is at position eleven. A cross-encoder
rescores the shortlist by reading query and document together, which lifts the right answer to
the top.

```csharp
using var reranker = new Reranker();
var results = reranker.Rerank("What is machine learning?", new[] {
    "Machine learning is a subset of artificial intelligence.",
    "The weather today is sunny.",
});
//  10.5139: Machine learning is a subset of artificial intelligence.
// -11.1001: The weather today is sunny.
```

This is markedly more accurate than comparing embeddings, and slow enough that you want it on a
shortlist rather than a whole corpus. Applying it to the top 50 results is the usual pattern, and
it works just as well on results from Elasticsearch or a SQL query as on Kjarni's own.

## Index and search your own documents

Point it at a directory, then query by keyword, by meaning, or both. The index is a folder on
your disk, so there is no database to run and no service to keep alive.

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

Start with `minilm-l6-v2` for embeddings. At 384 dimensions it is fast on CPU and the quality gap
against much larger models is smaller than people expect. Note that it truncates at 256 tokens, so
chunk long documents rather than feeding them whole.

## GPU

```csharp
using var embedder = new Embedder("minilm-l6-v2", device: "gpu");
```

GPU inference uses WebGPU: Vulkan on Linux, DX12 or Vulkan on Windows, Metal on macOS. It uses
whichever adapter the platform already provides, so there is no toolkit to install.

## Configuration

```csharp
using var embedder = new Embedder("minilm-l6-v2", cacheDir: "/my/models");
using var quiet    = new Embedder("minilm-l6-v2", quiet: true);
```

The `cacheDir` parameter is the reliable way to relocate model storage; pass it per
instance as above. `HF_TOKEN` is read from the environment and is required for gated
Hugging Face repositories, such as any `meta-llama/*` model.

## Platform support

| Platform | Shipped | GPU backend |
|----------|---------|-------------|
| Linux x64 | Yes | Vulkan |
| Linux arm64 | Yes | Vulkan |
| Windows x64 | Yes | DX12 / Vulkan |
| macOS arm64 | Yes | Metal |

The native library links only against glibc, so it runs on anything from a modern distribution back to CentOS 7.

## The same engine elsewhere

Kjarni is one Rust engine behind several packages. If you are building the browser
half of the same product, or want to try it before installing anything:

- **[Live demo](https://kjarni.ai/demo/)**: embeddings, reranking and search running
  in your own browser tab, nothing installed
- **[kjarni-wasm](https://www.npmjs.com/package/kjarni-wasm)** on npm: the same
  engine compiled to WebAssembly, running in a Web Worker
- **[Go](https://pkg.go.dev/github.com/olafurjohannsson/kjarni-go)**,
  **[C++](https://github.com/olafurjohannsson/kjarni/tree/main/crates/kjarni-ffi/examples/cpp)**
  and a CLI that reads stdin and writes JSON
- **[Source and issues](https://github.com/olafurjohannsson/kjarni)**

- **[Kjarni.Extensions.AI](https://www.nuget.org/packages/Kjarni.Extensions.AI)**:
  the Microsoft.Extensions.AI provider
- **[kjarni.ai](https://kjarni.ai)**: guides and quickstarts

MIT licensed.
