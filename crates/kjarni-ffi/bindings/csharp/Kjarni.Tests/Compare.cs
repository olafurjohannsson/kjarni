using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Kjarni;

namespace Kjarni.Tests
{
    public static class EmbedderBenchmark
    {
        // -------------------------------------------------
        // TEST DATA (100+ non-trivial sentences)
        // -------------------------------------------------
        
        static readonly string[] Topics = {
            "machine learning", "air traffic control", "distributed systems",
            "natural language processing", "computer graphics",
            "financial regulation", "software architecture",
            "real-time systems", "data pipelines", "neural networks"
        };

        static readonly string[] Verbs = {
            "optimizes", "evaluates", "analyzes", "processes",
            "transforms", "validates", "predicts", "controls"
        };

        static readonly string[] Objects = {
            "large-scale datasets", "complex state transitions",
            "high-dimensional embeddings", "streaming telemetry",
            "regulatory constraints", "fault-tolerant pipelines",
            "user-generated content", "multimodal signals"
        };

        static string[] GenerateSentences(int count, int seed = 42)
        {
            var random = new Random(seed);
            var sentences = new string[count];
            
            for (int i = 0; i < count; i++)
            {
                var topic = Topics[random.Next(Topics.Length)];
                var verb = Verbs[random.Next(Verbs.Length)];
                var obj = Objects[random.Next(Objects.Length)];
                sentences[i] = $"The {topic} system {verb} {obj} under real-world conditions.";
            }
            
            return sentences;
        }

        // -------------------------------------------------
        // BENCHMARK SETTINGS
        // -------------------------------------------------
        
        const int Runs = 10;
        const int SentenceCount = 120;

        static (double Mean, double StdDev) Benchmark(Action action)
        {
            var times = new List<double>();
            
            for (int i = 0; i < Runs; i++)
            {
                var sw = Stopwatch.StartNew();
                action();
                sw.Stop();
                times.Add(sw.Elapsed.TotalSeconds);
            }
            
            var mean = times.Average();
            var stdDev = Math.Sqrt(times.Select(t => Math.Pow(t - mean, 2)).Average());
            
            return (mean, stdDev);
        }

        public static void Run()
        {
            Console.WriteLine();
            Console.WriteLine("============================================================");
            Console.WriteLine("KJARNI C# EMBEDDER BENCHMARK");
            Console.WriteLine("============================================================");
            Console.WriteLine();

            var sentences = GenerateSentences(SentenceCount);
            Console.WriteLine($"Generated {sentences.Length} test sentences");
            Console.WriteLine($"Sample: \"{sentences[0]}\"");
            Console.WriteLine();

            // -------------------------------------------------
            // KJARNI EMBEDDER
            // -------------------------------------------------

            Console.WriteLine("--- Kjarni Embedder (batch) ---");
            
            using var embedder = new Embedder(
                model: "minilm-l6-v2",
                device: "cpu",
                quiet: true
            );

            // Warm-up
            Console.WriteLine("Warming up...");
            _ = embedder.EncodeBatch(sentences);

            // Benchmark batch encoding
            Console.WriteLine($"Running {Runs} iterations...");
            var (mean, stdDev) = Benchmark(() => embedder.EncodeBatch(sentences));

            // Single embedding test
            var embedding = embedder.Encode("Hello, world!");
            Console.WriteLine($"Kjarni dim: {embedding.Length}");
            Console.WriteLine($"Kjarni first 5: [{string.Join(", ", embedding.Take(5).Select(v => v.ToString("F6")))}]");

            // Similarity test
            var similarity = embedder.Similarity("cat", "dog");
            Console.WriteLine($"Kjarni similarity(cat, dog): {similarity:F6}");

            // -------------------------------------------------
            // RESULTS
            // -------------------------------------------------

            Console.WriteLine();
            Console.WriteLine($"--- Benchmark Results ({Runs} runs, batch size = {SentenceCount}) ---");
            Console.WriteLine($"Kjarni batch encode: {mean * 1000:F2} ms ± {stdDev * 1000:F2} ms");
            Console.WriteLine($"Throughput: {SentenceCount / mean:F0} sentences/sec");
            
            // Memory info
            var process = Process.GetCurrentProcess();
            Console.WriteLine($"Working set: {process.WorkingSet64 / (1024 * 1024):F1} MB");

            Console.WriteLine();
            Console.WriteLine("============================================================");
            Console.WriteLine("BENCHMARK COMPLETE");
            Console.WriteLine("============================================================");
        }
    }
}