using System;
using Kjarni;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine($"Kjarni version: {Native.GetVersion()}");

        // Test Embedder
        Console.WriteLine("\n--- Embedder ---");
        using var embedder = new Embedder(quiet: true);
        
        var embedding = embedder.Encode("Hello, world!");
        Console.WriteLine($"Embedding dim: {embedding.Length}");
        Console.WriteLine($"First 5 values: [{string.Join(", ", embedding[..5])}]");

        var sim = embedder.Similarity("cat", "dog");
        Console.WriteLine($"Similarity (cat, dog): {sim:F3}");

        var batch = embedder.EncodeBatch(new[] { "hello", "world", "test" });
        Console.WriteLine($"Batch: {batch.Length} embeddings of dim {batch[0].Length}");

        // Test Classifier
        Console.WriteLine("\n--- Classifier ---");
        using var classifier = new Classifier("sentiment", quiet: true);
        
        var result = classifier.Classify("I love this product!");
        Console.WriteLine("Classification:");
        foreach (var (label, score) in result)
        {
            Console.WriteLine($"  {label}: {score:F3}");
        }

        Console.WriteLine("\nAll tests passed!");
    }
}