// Kjarni Example: AI-Powered Content API
// 
// A production-ready ASP.NET Minimal API with:
//   - Sentiment analysis (product reviews, support tickets)
//   - Content moderation (toxic comment detection)
//   - Semantic search (search docs by meaning, not keywords)
//
// Zero external dependencies. No Python. No cloud APIs. No API keys.
// Just: dotnet add package Kjarni && dotnet run
//
// Usage:
//   dotnet run
//   curl http://localhost:5000/sentiment -d '{"text":"I love this product!"}'
//   curl http://localhost:5000/moderate -d '{"text":"You are an idiot"}'
//   curl http://localhost:5000/search -d '{"query":"how do returns work?"}'

using System.Text.Json;
using Kjarni;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

// ─── Load models once at startup (cached after first download) ───

Console.WriteLine("Loading models...");

var sentiment = new Classifier("distilbert-sentiment", quiet: true);
var toxicity = new Classifier("toxic-bert", quiet: true);
var embedder = new Embedder("minilm-l6-v2", quiet: true);

Console.WriteLine("Models loaded. Starting server...");

// ─── In-memory document store for semantic search ───

var documents = new List<(string Text, float[] Embedding)>();

// Pre-load some sample docs (replace with your own)
var sampleDocs = new[]
{
    "Returns are accepted within 30 days of purchase with a valid receipt.",
    "Shipping takes 3-5 business days for standard delivery.",
    "Premium members get free expedited shipping on all orders.",
    "To cancel your subscription, go to Settings > Account > Cancel Plan.",
    "Our customer support team is available Monday through Friday, 9am to 5pm.",
    "Gift cards can be purchased in denominations of $25, $50, and $100.",
    "We offer a price match guarantee within 14 days of purchase.",
    "Two-factor authentication can be enabled in your security settings.",
    "Refunds are processed within 5-7 business days to your original payment method.",
    "Bulk orders of 100+ units qualify for a 15% wholesale discount.",
};

foreach (var doc in sampleDocs)
    documents.Add((doc, embedder.Encode(doc)));

Console.WriteLine($"Indexed {documents.Count} documents.");

// ─── Endpoints ───

// POST /sentiment — Analyze sentiment of text
// Returns: { label: "POSITIVE", score: 0.9998, all_scores: {...} }
app.MapPost("/sentiment", (TextRequest req) =>
{
    var result = sentiment.Classify(req.Text);
    return Results.Json(new
    {
        label = result.Label,
        score = result.Score,
        all_scores = result.AllScores.ToDictionary(s => s.Label, s => s.Score),
    });
});

// POST /moderate — Check text for toxicity
// Returns: { flagged: true, scores: { toxic: 0.85, insult: 0.12, ... } }
app.MapPost("/moderate", (TextRequest req) =>
{
    var result = toxicity.Classify(req.Text);
    var scores = result.AllScores.ToDictionary(s => s.Label, s => s.Score);
    var flagged = scores.TryGetValue("toxic", out var toxicScore) && toxicScore > 0.5f;

    return Results.Json(new
    {
        flagged,
        toxic_score = toxicScore,
        scores,
        recommendation = flagged ? "block" : "allow",
    });
});

// POST /sentiment/batch — Analyze multiple texts at once
// Body: { "texts": ["Great!", "Terrible.", "It's okay."] }
app.MapPost("/sentiment/batch", (BatchTextRequest req) =>
{
    var results = req.Texts.Select(text =>
    {
        var result = sentiment.Classify(text);
        return new { text, label = result.Label, score = result.Score };
    });

    return Results.Json(new { results });
});

// POST /search — Semantic search over documents
// Returns: top matching documents ranked by meaning similarity
app.MapPost("/search", (SearchRequest req) =>
{
    var queryEmbedding = embedder.Encode(req.Query);
    var topK = req.TopK ?? 3;

    var results = documents
        .Select((doc, index) => new
        {
            text = doc.Text,
            score = Embedder.CosineSimilarity(queryEmbedding, doc.Embedding),
            index,
        })
        .OrderByDescending(r => r.score)
        .Take(topK);

    return Results.Json(new { query = req.Query, results });
});

// POST /index — Add a document to the search index
app.MapPost("/index", (TextRequest req) =>
{
    var embedding = embedder.Encode(req.Text);
    documents.Add((req.Text, embedding));

    return Results.Json(new
    {
        indexed = true,
        total_documents = documents.Count,
    });
});

// GET /health — Check that all models are loaded
app.MapGet("/health", () => Results.Json(new
{
    status = "healthy",
    models = new
    {
        sentiment = "distilbert-sentiment",
        toxicity = "toxic-bert",
        embedder = "minilm-l6-v2",
    },
    documents_indexed = documents.Count,
}));

app.Run("http://localhost:5000");

// ─── Request types (must be after top-level statements) ───

record TextRequest(string Text);
record BatchTextRequest(string[] Texts);
record SearchRequest(string Query, int? TopK);