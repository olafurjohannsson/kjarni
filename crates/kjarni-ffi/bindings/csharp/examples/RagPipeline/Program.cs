// RAG Pipeline in C# — Index documents, search by meaning, rerank results
// No Python. No vector database. No cloud APIs.
using Kjarni;

// ─── Step 1: Create some sample documents ───

var docsDir = Path.Combine(Path.GetTempPath(), "kjarni_rag_demo");
Directory.CreateDirectory(docsDir);

var files = new Dictionary<string, string>
{
    ["returns-policy.txt"] = """
    Returns and Refunds Policy

    We accept returns within 30 days of purchase. Items must be in original
    packaging and unused condition. To initiate a return, contact our support
    team with your order number.

    Refunds are processed within 5-7 business days to your original payment
    method. Shipping costs are non-refundable unless the item was defective.

    For defective items, we offer free return shipping and a full refund
    including original shipping costs.
    """,

    ["shipping-guide.txt"] = """
    Shipping Information

    Standard shipping: 3-5 business days ($4.99)
    Express shipping: 1-2 business days ($12.99)
    Free shipping on orders over $50.

    Premium members receive free express shipping on all orders.
    International shipping is available to select countries.
    Tracking numbers are sent via email once your order ships.
    """,

    ["account-help.txt"] = """
    Account Management

    To cancel your subscription, navigate to Settings > Account > Subscription
    and click "Cancel Plan". Your access continues until the end of the billing
    period.

    To enable two-factor authentication, go to Settings > Security > 2FA.
    We support authenticator apps and SMS verification.

    Password resets can be done from the login page by clicking "Forgot Password".
    A reset link will be sent to your registered email address.
    """,

    ["pricing.txt"] = """
    Pricing and Discounts

    Individual plan: $9.99/month or $99/year (save 17%)
    Team plan: $7.99/user/month (minimum 5 users)
    Enterprise: Contact sales for custom pricing.

    Bulk orders of 100+ units qualify for a 15% wholesale discount.
    Non-profit organizations receive a 25% discount with valid documentation.
    Student discounts of 50% are available with a valid .edu email.
    """,
};

foreach (var (name, content) in files)
    File.WriteAllText(Path.Combine(docsDir, name), content.Trim());

Console.WriteLine($"Created {files.Count} documents in {docsDir}\n");

// ─── Step 2: Index the documents ───

Console.WriteLine("Indexing documents...");
using var indexer = new Indexer(
    model: "minilm-l6-v2",
    chunkSize: 256,
    chunkOverlap: 25,
    quiet: true
);

var indexPath = Path.Combine(Path.GetTempPath(), "kjarni_rag_index");
var stats = indexer.Create(indexPath, new[] { docsDir }, force: true);
Console.WriteLine($"Indexed {stats.DocumentsIndexed} documents, {stats.ChunksCreated} chunks\n");

// ─── Step 3: Search with reranking ───

using var searcher = new Searcher(
    model: "minilm-l6-v2",
    rerankerModel: "minilm-l6-v2-cross-encoder",
    quiet: true
);

string[] questions = [
    "How do I get a refund?",
    "Do you have student pricing?",
    "How do I cancel my account?",
    "How long does shipping take?",
];

foreach (var question in questions)
{
    Console.WriteLine($"🔍 \"{question}\"");
    Console.WriteLine(new string('─', 60));

    var results = searcher.Search(indexPath, question,
        mode: SearchMode.Hybrid,
        topK: 3
    );

    foreach (var r in results)
    {
        var source = r.Metadata.TryGetValue("source", out var s) ? Path.GetFileName(s.ToString()!) : "?";
        Console.WriteLine($"  {r.Score:F3}  [{source}]  {Truncate(r.Text, 80)}");
    }
    
    Console.WriteLine();
}

// ─── Cleanup ───

Directory.Delete(docsDir, true);
Directory.Delete(indexPath, true);

static string Truncate(string s, int max) =>
    s.Length <= max ? s.Replace("\n", " ").Trim() : s[..max].Replace("\n", " ").Trim() + "...";