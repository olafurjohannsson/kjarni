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

        private const string Question = "What is the capital of Iceland?";

        private static string SendGreedy(Chat chat)
        {
            var config = GenerationConfig.Greedy();
            config.MaxNewTokens = 24;
            return chat.Send(Question, config);
        }

        [SkippableFact]
        public void Chat_FromFolder_MatchesByName()
        {
            Heavy.SkipIfSet();
            string want;
            using (var byName = new Chat("qwen2.5-0.5b-instruct", quiet: true))
                want = SendGreedy(byName);
            Assert.Contains("Reykjav", want);

            var folder = Path.Combine(CacheDir(), "Qwen_Qwen2.5-0.5B-Instruct");
            using (var byPath = new Chat(null, modelPath: folder, quiet: true))
                Assert.Equal(want, SendGreedy(byPath));
        }

        [Fact]
        public void Chat_NeedsANameOrAFolder()
        {
            Assert.Throws<ArgumentException>(() => new Chat(null));
        }

        [Fact]
        public void Chat_FromAnEncoderFolder_SaysItIsNotAChatModel()
        {
            var folder = Path.Combine(CacheDir(), "distilbert_distilbert-base-uncased-finetuned-sst-2-english");
            var ex = Assert.ThrowsAny<Exception>(() => new Chat(null, modelPath: folder, quiet: true));
            Assert.Contains("not suitable for chat", ex.Message);
        }

        [Theory]
        [InlineData("summarizer", "marian")]
        [InlineData("translator", "bart")]
        [InlineData("transcriber", "bart")]
        public void Seq2Seq_FromAFolderOfTheWrongFamily_IsInvalidConfig(string kind, string modelType)
        {
            using var dir = new TempDir($"kjarni_cs_{kind}_{modelType}");
            File.WriteAllText(Path.Combine(dir.Path, "config.json"), $"{{\"model_type\": \"{modelType}\"}}");
            var ex = Assert.Throws<KjarniException>(() =>
            {
                IDisposable model = kind switch
                {
                    "summarizer" => new Summarizer(modelPath: dir.Path, quiet: true),
                    "translator" => new Translator(modelPath: dir.Path, quiet: true),
                    _ => new Transcriber(modelPath: dir.Path, quiet: true),
                };
                model.Dispose();
            });
            Assert.Equal(KjarniErrorCode.InvalidConfig, ex.ErrorCode);
        }

        [SkippableFact]
        public void Translator_FromFolder_MatchesByName()
        {
            Heavy.SkipIfSet();
            const string text = "The committee postponed its decision until next spring.";
            string want;
            using (var byName = new Translator("opus-mt-en-de", quiet: true))
                want = byName.Translate(text, "en", "de");

            var folder = Path.Combine(CacheDir(), "kjarni-ai_opus-mt-en-de");
            using var byPath = new Translator(modelPath: folder, quiet: true);
            Assert.Equal(want, byPath.Translate(text, "en", "de"));
        }

        [SkippableFact]
        public void Summarizer_FromFolder_MatchesByName()
        {
            Heavy.SkipIfSet();
            const string article = "The city council voted on Tuesday to extend the bus network to the new harbour district, which is expected to house 4,000 residents by 2030. The plan adds two lines and raises evening frequency to every ten minutes. Opponents said the cost should be covered by the state rather than the city.";
            string want;
            using (var byName = new Summarizer(quiet: true))
                want = byName.Summarize(article);

            var folder = Path.Combine(CacheDir(), "olafuraron_distilbart-cnn-12-6");
            using var byPath = new Summarizer(modelPath: folder, quiet: true);
            Assert.Equal(want, byPath.Summarize(article));
        }

        [SkippableFact]
        public void Transcriber_FromFolder_MatchesByName()
        {
            Heavy.SkipIfSet();
            string want;
            using (var byName = new Transcriber(language: "en", quiet: true))
                want = byName.TranscribeFile(Fixtures.SpeechWav).Text;

            var folder = Path.Combine(CacheDir(), "openai_whisper-small");
            using var byPath = new Transcriber(modelPath: folder, language: "en", quiet: true);
            Assert.Equal(want, byPath.TranscribeFile(Fixtures.SpeechWav).Text);
        }
    }
}
