using System;
using EdgeGpt;
using System.Linq;
class Program
{
    static void Main()
    {
        Console.WriteLine("EdgeGPT C# Example: Sentence Encoding\n" + new string('=', 60));

        try
        {
            using var edgeGpt = new EdgeGpt.EdgeGpt();

            string text = "The quick brown fox jumps over the lazy dog";
            Console.WriteLine($"\nEncoding text: \"{text}\"");

            var embedding = edgeGpt.Encode(text);

            Console.WriteLine($"\n✓ Successfully encoded!");
            Console.WriteLine($"  Embedding dimension: {embedding.Length}");
            Console.WriteLine($"  First 10 values: {string.Join(" ", embedding[..Math.Min(10, embedding.Length)].Select(v => v.ToString("F4")))}");

            float norm = 0;
            foreach (var v in embedding)
                norm += v * v;
            norm = MathF.Sqrt(norm);

            Console.WriteLine($"  L2 norm: {norm:F4} (should be ~1.0)");
        }
        catch (EdgeGptException ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
        }
    }
}
