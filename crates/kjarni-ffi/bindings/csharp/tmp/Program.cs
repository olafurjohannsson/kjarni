using Kjarni;

// === 1. Intro similarity ===
Console.WriteLine("=== intro ===");
using var embedder = new Embedder("minilm-l6-v2", quiet: true);
Console.WriteLine($"doctor/physician: {embedder.Similarity("doctor", "physician")}");
Console.WriteLine($"doctor/banana: {embedder.Similarity("doctor", "banana")}");
Console.WriteLine();

// === 2. Encode first 5 ===
Console.WriteLine("=== encode ===");
var hw = embedder.Encode("Hello world");
Console.WriteLine($"Length: {hw.Length}");
Console.WriteLine($"First 5: {string.Join(", ", hw[..5])}");
Console.WriteLine();

// === 3. Similarity pairs ===
Console.WriteLine("=== similarity pairs ===");
var pairs = new[] {
    ("doctor", "physician"),
    ("doctor", "hospital"),
    ("doctor", "banana"),
    ("cat", "dog"),
    ("cat", "car"),
    ("machine learning", "artificial intelligence"),
    ("machine learning", "potato soup"),
};

foreach (var (a, b) in pairs)
    Console.WriteLine($"  {embedder.Similarity(a, b):F4}  \"{a}\" / \"{b}\"");

Console.WriteLine();

// === 4. FAQ search ===
Console.WriteLine("=== faq search ===");
var docs = new[] {
    "How do I reset my password?",
    "What is your refund policy?",
    "Do you ship internationally?",
    "How do I update my billing address?",
    "Where can I track my order?",
};

var vectors = embedder.EncodeBatch(docs);
var query = embedder.Encode("I need to change my login credentials");

var results = docs
    .Select((doc, i) => (doc, score: Embedder.CosineSimilarity(query, vectors[i])))
    .OrderByDescending(x => x.score);

foreach (var (doc, score) in results)
    Console.WriteLine($"  {score:F4}: {doc}");

Console.WriteLine();

// === 5. FAQ matching ===
Console.WriteLine("=== faq matching ===");
var faqs = new[] {
    "How do I cancel my subscription?",
    "How do I get a refund?",
    "How do I change my email address?",
    "What payment methods do you accept?",
    "How do I contact support?",
};
var faqVectors = embedder.EncodeBatch(faqs);

string MatchFaq(string userQuestion)
{
    var queryVec = embedder.Encode(userQuestion);
    var best = faqs
        .Select((faq, i) => (faq, score: Embedder.CosineSimilarity(queryVec, faqVectors[i])))
        .OrderByDescending(x => x.score)
        .First();

    return $"{best.faq}  (score: {best.score:F4})";
}

Console.WriteLine($"\"I want to stop paying\" -> {MatchFaq("I want to stop paying")}");
Console.WriteLine($"\"Can I pay with crypto?\" -> {MatchFaq("Can I pay with crypto?")}");