using System;
using System.IO;
using System.Runtime.InteropServices;
using Xunit;

namespace Kjarni.Tests
{
    /// <summary>
    /// A model loaded from a local folder with <c>modelPath</c> gives what the
    /// same model gives by name. The folder is the one the named model
    /// downloads to.
    /// </summary>
    public class LocalFolderTests
    {
        /// <summary>Where kjarni keeps models, as the Rust side decides it.</summary>
        private static string CacheDir()
        {
            var env = Environment.GetEnvironmentVariable("KJARNI_CACHE_DIR");
            if (!string.IsNullOrEmpty(env)) return env;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "kjarni");
            var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return Path.Combine(home, "Library", "Caches", "kjarni");
            var xdg = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
            return Path.Combine(string.IsNullOrEmpty(xdg) ? Path.Combine(home, ".cache") : xdg, "kjarni");
        }

        [Fact]
        public void Embedder_FromFolder_MatchesByName()
        {
            float[] want;
            using (var byName = new Embedder("minilm-l6-v2", quiet: true))
                want = byName.Encode("hello world");

            var folder = Path.Combine(CacheDir(), "sentence-transformers_all-MiniLM-L6-v2");
            using var byPath = new Embedder(modelPath: folder, quiet: true);
            var got = byPath.Encode("hello world");

            Assert.Equal(want.Length, got.Length);
            for (int i = 0; i < want.Length; i++)
                Assert.Equal(want[i], got[i], 6);
        }

        [Fact]
        public void Classifier_FromFolder_MatchesByName()
        {
            const string text = "The box arrived crushed.";
            ClassificationResult want;
            using (var byName = new Classifier("distilbert-sentiment", quiet: true))
                want = byName.Classify(text);

            var folder = Path.Combine(CacheDir(), "distilbert_distilbert-base-uncased-finetuned-sst-2-english");
            using var byPath = new Classifier(modelPath: folder, quiet: true);
            var got = byPath.Classify(text);

            Assert.Equal(want.Label, got.Label);
            Assert.Equal(want.Score, got.Score, 6);
        }

        [Fact]
        public void Reranker_FromFolder_MatchesByName()
        {
            const string q = "how do I reset my password";
            const string d = "Click Forgot password on the sign-in page.";
            float want;
            using (var byName = new Reranker(quiet: true))
                want = byName.Score(q, d);

            var folder = Path.Combine(CacheDir(), "cross-encoder_ms-marco-MiniLM-L-6-v2");
            using var byPath = new Reranker(modelPath: folder, quiet: true);

            Assert.Equal(want, byPath.Score(q, d), 5);
        }

        [Fact]
        public void Embedder_FromFolderWithoutModel_SaysWhatIsMissing()
        {
            var empty = Directory.CreateTempSubdirectory("kjarni_cs_").FullName;
            try
            {
                var ex = Assert.ThrowsAny<Exception>(() => new Embedder(modelPath: empty, quiet: true));
                Assert.Contains("config.json", ex.Message);
            }
            finally
            {
                Directory.Delete(empty, true);
            }
        }
    }
}
