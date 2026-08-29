# Kjarni.Extensions.AI

**Local embeddings and chat for .NET through `Microsoft.Extensions.AI`, running in your process.**

Implements `IEmbeddingGenerator<string, Embedding<float>>` and `IChatClient` on top of the [Kjarni](https://www.nuget.org/packages/Kjarni) inference engine. Anything already written against the Microsoft.Extensions.AI abstractions, **Semantic Kernel** included, keeps working unchanged: you swap the registration and the model runs locally instead of over the network.

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

## When you would reach for this

Every other `IEmbeddingGenerator` provider is a client for a hosted service. This one loads the model into your process and runs it there, which makes it the option that survives conditions the others cannot: air-gapped deployments, regulated data that cannot leave the building, on-premise installs at customer sites, CI runs with no secrets, or a laptop on a plane.

It is also the cheapest option at volume, since embedding a million documents costs CPU time rather than per-token billing.

## Chat

`IChatClient` runs a local decoder model (Llama, Qwen or Phi) in your process. The model is loaded
directly by your application, so there is nothing listening on a port and nothing to keep running
alongside it.

```cs
using Kjarni.Extensions.AI;
using Microsoft.Extensions.AI;

using IChatClient client = new KjarniChatClient("llama3.2-3b-instruct");

var response = await client.GetResponseAsync("Explain retrieval-augmented generation in one sentence.");
Console.WriteLine(response.Text);
```

Multi-turn works the way `IChatClient` expects: you own the transcript and replay it.

```cs
List<ChatMessage> conversation =
[
    new(ChatRole.System, "You are a terse assistant."),
    new(ChatRole.User,   "My name is Olafur."),
];

var reply = await client.GetResponseAsync(conversation);
conversation.Add(new ChatMessage(ChatRole.Assistant, reply.Text));

conversation.Add(new ChatMessage(ChatRole.User, "What is my name?"));
Console.WriteLine((await client.GetResponseAsync(conversation)).Text);   // Olafur.
```

Streaming:

```cs
await foreach (var update in client.GetStreamingResponseAsync("Write a haiku about Reykjavik."))
    Console.Write(update.Text);
```

Registration is the same shape as embeddings:

```cs
builder.Services.AddKjarniChatClient("llama3.2-3b-instruct");
```

And an existing `Chat` can be wrapped rather than loading the model twice:

```cs
using var chat = new Chat("llama3.2-3b-instruct");
var client = chat.AsChatClient();
```

### Chat models

| Model | Notes |
| ----- | ----- |
| `qwen2.5-0.5b-instruct` | smallest; simple structured tasks |
| `llama3.2-1b-instruct` | fast on modest hardware |
| **`llama3.2-3b-instruct`** | the sweet spot on CPU |
| `phi3.5-mini` | reasoning-leaning, 3.8B |

### What maps, and what doesn't

`ChatOptions` is honoured where Kjarni's sampler has an equivalent: `Temperature` (0 selects
greedy decoding), `TopP`, `TopK` and `MaxOutputTokens`.

Everything else is deliberately explicit rather than silently ignored:

- **`Tools` throws `NotSupportedException`.** Tool calling is not implemented, and accepting the
  option would leave you believing your functions had been offered to the model.
- **`ResponseFormat = Json` throws.** There is no constrained decoding; ask for JSON in the
  prompt and validate the result.
- **`ChatRole.Tool` throws.** Folding tool output in as user text would misrepresent the transcript.
- **`Seed`, `StopSequences`, `FrequencyPenalty` and `PresencePenalty` are ignored.** Mapping a
  frequency penalty onto Kjarni's repetition penalty would change output in ways you did not ask for.

Generation is compute-bound and the native model is not re-entrant, so calls are serialized per
instance: concurrent requests queue rather than overlap. One instance per model, shared as a
singleton, is the intended shape.

## Dependency injection

```csharp
builder.Services.AddKjarniEmbeddingGenerator("minilm-l6-v2");
```

Registered as a singleton, so the weights load once. The generator serializes concurrent calls internally, which makes one instance safe to share across requests.

Then inject it anywhere:

```csharp
public class SearchService(IEmbeddingGenerator<string, Embedding<float>> embeddings)
{
    public async Task<ReadOnlyMemory<float>> VectorFor(string text) =>
        (await embeddings.GenerateAsync([text]))[0].Vector;
}
```

## Wrapping an existing Embedder

If you already hold a Kjarni `Embedder`, because you also use it for similarity or search, wrap it rather than loading the model twice:

```csharp
using Kjarni;
using Microsoft.Extensions.AI;

using var embedder = new Embedder("minilm-l6-v2");

var generator = embedder.AsEmbeddingGenerator("minilm-l6-v2");
```

`AsEmbeddingGenerator` and `AddKjarniEmbeddingGenerator` live in the `Microsoft.Extensions.AI`
namespace, so they appear alongside the abstractions themselves without a further `using` to find.

By default the generator **borrows** the embedder: disposing the generator leaves the embedder alive, since you created it. Pass `ownsEmbedder: true` to transfer ownership.

## With Semantic Kernel

Semantic Kernel builds on the Microsoft.Extensions.AI abstractions, so registration is the same:

```csharp
var builder = Kernel.CreateBuilder();
builder.Services.AddKjarniEmbeddingGenerator("minilm-l6-v2");

var kernel = builder.Build();
```

Any SK component that resolves `IEmbeddingGenerator<string, Embedding<float>>`, such as memory stores, vector connectors and RAG pipelines, now runs against local inference.

## Batching

Pass the whole collection in one call. Kjarni batches natively, and it is substantially faster than looping:

```csharp
string[] documents = File.ReadAllLines("corpus.txt");

var embeddings = await generator.GenerateAsync(documents);
// embeddings[i] corresponds to documents[i]
```

## Models

Any Kjarni embedding model works. `minilm-l6-v2` is the default, being the smallest and fastest. It truncates at 256 tokens, so chunk long documents rather than passing them whole.

| Model | Dimensions | On disk |
| ----- | ---------- | ------- |
| `minilm-l6-v2` | 384 | 88 MB |
| `mpnet-base-v2` | 768 | 419 MB |
| `nomic-embed-text` | 768 | 523 MB |
| `bge-m3` (multilingual) | 1024 | ~2 GB |

```csharp
var generator = new KjarniEmbeddingGenerator("mpnet-base-v2");
```

## GPU

```csharp
var generator = new KjarniEmbeddingGenerator("minilm-l6-v2", device: "gpu");
```

GPU inference uses WebGPU: Vulkan on Linux, DX12 or Vulkan on Windows, Metal on macOS. It uses whichever adapter the platform already provides, so there is no toolkit to install.

## Behavior notes

- **Generation is synchronous under the hood.** Inference is compute-bound, not I/O-bound, so the work runs on the calling thread and returns a completed task. There is no hidden thread-pool hop.
- **One model per instance.** Kjarni loads a single model per embedder. Passing an `EmbeddingGenerationOptions.ModelId` that differs from the instance's model throws `NotSupportedException` rather than silently ignoring it. Construct a second generator instead.
- **Fixed dimensions.** These models don't support Matryoshka truncation, so `EmbeddingGenerationOptions.Dimensions` throws if it doesn't match the model's native size.
- **Metadata** is available through `GetService(typeof(EmbeddingGeneratorMetadata))` and `GetService(typeof(ChatClientMetadata))`, reporting provider name `kjarni` and the model id.
- **Unwrapping.** `GetService(typeof(Embedder))` and `GetService(typeof(Chat))` return the underlying Kjarni objects when you need the native API directly.

## License

MIT.
