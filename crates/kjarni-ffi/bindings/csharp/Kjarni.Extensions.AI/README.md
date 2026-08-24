# Kjarni.Extensions.AI

**Local embeddings for .NET through `Microsoft.Extensions.AI` — no Python, no ONNX, no cloud.**

Implements `IEmbeddingGenerator<string, Embedding<float>>` on top of the [Kjarni](https://www.nuget.org/packages/Kjarni) inference engine, so anything built against the Microsoft.Extensions.AI abstractions — including **Semantic Kernel** — can run embeddings entirely in-process.

```bash
dotnet add package Kjarni.Extensions.AI
```

```csharp
using Kjarni.Extensions.AI;
using Microsoft.Extensions.AI;

using IEmbeddingGenerator<string, Embedding<float>> generator =
    new KjarniEmbeddingGenerator("minilm-l6-v2");

var embeddings = await generator.GenerateAsync(["semantic search, locally"]);
Console.WriteLine(embeddings[0].Vector.Length);   // 384
```

The model downloads on first use and is cached on disk. After that, nothing touches the network.

## Why

Every other `IEmbeddingGenerator` provider calls a hosted service. This one doesn't call anything — inference runs inside your process against a native library. That makes it the option that works when data cannot leave the building: air-gapped deployments, regulated environments, on-premise installs, or a laptop on a plane.

## Dependency injection

```csharp
builder.Services.AddKjarniEmbeddingGenerator("minilm-l6-v2");
```

Registered as a singleton — weights load once, and the generator serializes concurrent calls internally, so one instance is safe to share across requests.

Then inject it anywhere:

```csharp
public class SearchService(IEmbeddingGenerator<string, Embedding<float>> embeddings)
{
    public async Task<ReadOnlyMemory<float>> VectorFor(string text) =>
        (await embeddings.GenerateAsync([text]))[0].Vector;
}
```

## Wrapping an existing Embedder

If you already hold a Kjarni `Embedder` — because you also use it for similarity or search — wrap it rather than loading the model twice:

```csharp
using var embedder = new Embedder("minilm-l6-v2");

var generator = embedder.AsEmbeddingGenerator("minilm-l6-v2");
```

By default the generator **borrows** the embedder: disposing the generator leaves the embedder alive, since you created it. Pass `ownsEmbedder: true` to transfer ownership.

## With Semantic Kernel

Semantic Kernel builds on the Microsoft.Extensions.AI abstractions, so registration is the same:

```csharp
var builder = Kernel.CreateBuilder();
builder.Services.AddKjarniEmbeddingGenerator("minilm-l6-v2");

var kernel = builder.Build();
```

Any SK component that resolves `IEmbeddingGenerator<string, Embedding<float>>` — memory stores, vector connectors, RAG pipelines — now runs against local inference.

## Batching

Pass the whole collection in one call. Kjarni batches natively, and it is substantially faster than looping:

```csharp
string[] documents = File.ReadAllLines("corpus.txt");

var embeddings = await generator.GenerateAsync(documents);
// embeddings[i] corresponds to documents[i]
```

## Models

Any Kjarni embedding model works. `minilm-l6-v2` is the default — smallest and fastest.

| Model | Dimensions | Size |
| ----- | ---------- | ---- |
| `minilm-l6-v2` | 384 | 22 MB |
| `mpnet-base-v2` | 768 | 110 MB |
| `nomic-embed-text` | 768 | 137 MB |
| `bge-m3` (multilingual) | 1024 | 567 MB |

```csharp
var generator = new KjarniEmbeddingGenerator("mpnet-base-v2");
```

## GPU

```csharp
var generator = new KjarniEmbeddingGenerator("minilm-l6-v2", device: "gpu");
```

GPU inference uses WebGPU — Vulkan on Linux, DX12 or Vulkan on Windows, Metal on macOS. CUDA is not required.

## Behavior notes

- **Generation is synchronous under the hood.** Inference is compute-bound, not I/O-bound, so the work runs on the calling thread and returns a completed task. There is no hidden thread-pool hop.
- **One model per instance.** Kjarni loads a single model per embedder. Passing an `EmbeddingGenerationOptions.ModelId` that differs from the instance's model throws `NotSupportedException` rather than silently ignoring it. Construct a second generator instead.
- **Fixed dimensions.** These models don't support Matryoshka truncation, so `EmbeddingGenerationOptions.Dimensions` throws if it doesn't match the model's native size.
- **Metadata** is available through `GetService(typeof(EmbeddingGeneratorMetadata))`, reporting provider name `kjarni`, the model id, and its dimension count.

## License

MIT or Apache-2.0, at your option.
