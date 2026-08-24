// Local embeddings through Microsoft.Extensions.AI — the abstraction Semantic Kernel builds on.
// No Python, no ONNX, no cloud, no API key. Everything runs in this process.

using Kjarni.Extensions.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

// ─── 1. Direct construction ───────────────────────────────────────

Console.WriteLine("── Direct ──\n");

using IEmbeddingGenerator<string, Embedding<float>> generator =
    new KjarniEmbeddingGenerator("minilm-l6-v2");

var single = await generator.GenerateAsync(["semantic search, running locally"]);

Console.WriteLine($"Dimensions: {single[0].Vector.Length}");
Console.WriteLine($"First 5:    [{string.Join(", ", single[0].Vector.ToArray().Take(5).Select(f => f.ToString("F5")))}]");
Console.WriteLine();

// ─── 2. Provider metadata ─────────────────────────────────────────
// Consumers use this to report which backend is in use.

var metadata = generator.GetService(typeof(EmbeddingGeneratorMetadata)) as EmbeddingGeneratorMetadata;
Console.WriteLine($"Provider:   {metadata?.ProviderName}");
Console.WriteLine($"Model:      {metadata?.DefaultModelId}");
Console.WriteLine($"Dimensions: {metadata?.DefaultModelDimensions}");
Console.WriteLine();

// ─── 3. Batching ──────────────────────────────────────────────────
// Pass the whole collection at once — Kjarni batches natively.

Console.WriteLine("── Batch ──\n");

string[] documents =
[
    "Returns are accepted within 30 days of purchase.",
    "Standard shipping takes 3-5 business days.",
    "To reset your password, use the link on the login page.",
    "Premium members get free express shipping.",
];

var embeddings = await generator.GenerateAsync(documents);
Console.WriteLine($"Encoded {embeddings.Count} documents\n");

// ─── 4. Semantic search over those documents ──────────────────────
// Cosine similarity — the vectors are already L2-normalized, so a dot product suffices.

const string query = "how long until my order arrives?";
var queryVector = (await generator.GenerateAsync([query]))[0].Vector.ToArray();

var ranked = documents
    .Select((doc, i) => (doc, score: Dot(queryVector, embeddings[i].Vector.ToArray())))
    .OrderByDescending(x => x.score);

Console.WriteLine($"Query: \"{query}\"\n");
foreach (var (doc, score) in ranked)
    Console.WriteLine($"  {score:F4}  {doc}");
Console.WriteLine();

// ─── 5. Dependency injection ──────────────────────────────────────
// This is the registration Semantic Kernel and ASP.NET Core apps use.
// Registered as a singleton: weights load once and are shared safely.

Console.WriteLine("── Dependency injection ──\n");

var services = new ServiceCollection();
services.AddKjarniEmbeddingGenerator("minilm-l6-v2");

using var provider = services.BuildServiceProvider();
var injected = provider.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();

var resolved = await injected.GenerateAsync(["resolved from the container"]);
Console.WriteLine($"Resolved generator produced {resolved[0].Vector.Length} dimensions");

static float Dot(float[] a, float[] b)
{
    float sum = 0f;
    for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
    return sum;
}
