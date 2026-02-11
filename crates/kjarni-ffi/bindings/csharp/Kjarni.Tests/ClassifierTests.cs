using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Kjarni.Tests
{
    public class DistilBertSentimentTests : IDisposable
    {
        private readonly Classifier _classifier;
        private readonly ITestOutputHelper _output;

        public DistilBertSentimentTests(ITestOutputHelper output)
        {
            _output = output;
            _classifier = new Classifier("distilbert-sentiment", quiet: true);
        }

        public void Dispose() => _classifier.Dispose();

        [Fact]
        public void NumLabels_Is2()
        {
            Assert.Equal(2, _classifier.NumLabels);
        }

        [Theory]
        [InlineData("I love this product, it's absolutely amazing!", "POSITIVE")]
        [InlineData("This is the best movie I have ever seen.", "POSITIVE")]
        [InlineData("Wonderful experience, highly recommended.", "POSITIVE")]
        public void Classify_PositiveText(string text, string expectedLabel)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");

            Assert.Equal(expectedLabel, result.Label);
            Assert.True(result.Score > 0.9f, $"Expected high confidence, got {result.Score:F4}");
        }

        [Theory]
        [InlineData("This is terrible, I want my money back.", "NEGATIVE")]
        [InlineData("Worst experience of my life, absolutely awful.", "NEGATIVE")]
        [InlineData("The food was disgusting and the service was horrible.", "NEGATIVE")]
        public void Classify_NegativeText(string text, string expectedLabel)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");

            Assert.Equal(expectedLabel, result.Label);
            Assert.True(result.Score > 0.9f, $"Expected high confidence, got {result.Score:F4}");
        }

        [Fact]
        public void Classify_AllScoresSumToOne()
        {
            var result = _classifier.Classify("This is a test sentence.");
            var sum = result.AllScores.Sum(s => s.Score);

            _output.WriteLine($"Score sum: {sum:F8}");
            Assert.InRange(sum, 0.99f, 1.01f);
        }

        [Fact]
        public void Classify_IsDeterministic()
        {
            var a = _classifier.Classify("Deterministic test");
            var b = _classifier.Classify("Deterministic test");

            Assert.Equal(a.Label, b.Label);
            Assert.Equal(a.Score, b.Score);
        }

        [Fact]
        public void Classify_TopKReturnsCorrectCount()
        {
            var result = _classifier.Classify("Test sentence");
            var top1 = result.TopK(1).ToArray();

            Assert.Single(top1);
            Assert.Equal(result.Label, top1[0].Label);
        }

        [Fact]
        public void Classify_ToString_FormatsCorrectly()
        {
            var result = _classifier.Classify("Great product!");
            var str = result.ToString();

            _output.WriteLine(str);
            Assert.Contains("POSITIVE", str);
            Assert.Contains("%", str);
        }
    }
    public class RobertaSentimentTests : IDisposable
    {
        private readonly Classifier _classifier;
        private readonly ITestOutputHelper _output;

        public RobertaSentimentTests(ITestOutputHelper output)
        {
            _output = output;
            _classifier = new Classifier("roberta-sentiment", quiet: true);
        }

        public void Dispose() => _classifier.Dispose();

        [Fact]
        public void NumLabels_Is3()
        {
            Assert.Equal(3, _classifier.NumLabels);
        }

        [Theory]
        [InlineData("I absolutely love this! Best day ever! 🎉", "positive")]
        [InlineData("This made my day, so happy right now", "positive")]
        public void Classify_PositiveText(string text, string expectedLabel)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
        }

        [Theory]
        [InlineData("This is the worst thing ever, so angry", "negative")]
        [InlineData("Terrible service, never coming back", "negative")]
        public void Classify_NegativeText(string text, string expectedLabel)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
        }

        [Fact]
        public void Classify_NeutralText()
        {
            var result = _classifier.Classify("The meeting is scheduled for 3pm tomorrow.");
            _output.WriteLine($"Neutral test => {result}");
            LogAllScores(result);

            // Neutral or at least not strongly positive/negative
            Assert.Contains(result.Label, new[] { "neutral", "positive" });
        }

        [Fact]
        public void Classify_AllScoresSumToOne()
        {
            var result = _classifier.Classify("Testing score distribution");
            var sum = result.AllScores.Sum(s => s.Score);

            _output.WriteLine($"Score sum: {sum:F8}");
            Assert.InRange(sum, 0.99f, 1.01f);
        }

        private void LogAllScores(ClassificationResult result)
        {
            foreach (var (label, score) in result.AllScores)
                _output.WriteLine($"  {label}: {score:F6}");
        }
    }
    public class BertMultilingualSentimentTests : IDisposable
    {
        private readonly Classifier _classifier;
        private readonly ITestOutputHelper _output;

        public BertMultilingualSentimentTests(ITestOutputHelper output)
        {
            _output = output;
            _classifier = new Classifier("bert-sentiment-multilingual", quiet: true);
        }

        public void Dispose() => _classifier.Dispose();

        [Fact]
        public void NumLabels_Is5()
        {
            Assert.Equal(5, _classifier.NumLabels);
        }

        [Theory]
        [InlineData("This product is absolutely fantastic, I love everything about it!", "5 stars")]
        [InlineData("Amazing quality, exceeded all my expectations", "5 stars")]
        public void Classify_VeryPositive_5Stars(string text, string expectedLabel)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
        }

        [Theory]
        [InlineData("Absolutely terrible, worst purchase ever, broken on arrival", "1 star")]
        public void Classify_VeryNegative_1Star(string text, string expectedLabel)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
        }

        [Fact]
        public void Classify_Multilingual_German()
        {
            var result = _classifier.Classify("Dieses Produkt ist wunderbar, ich bin sehr zufrieden!");
            _output.WriteLine($"German positive => {result}");
            LogAllScores(result);

            // Should be 4 or 5 stars
            Assert.Contains(result.Label, new[] { "4 stars", "5 stars" });
        }

        [Fact]
        public void Classify_Multilingual_French()
        {
            var result = _classifier.Classify("C'est terrible, je suis très déçu de cet achat.");
            _output.WriteLine($"French negative => {result}");
            LogAllScores(result);

            // Should be 1 or 2 stars
            Assert.Contains(result.Label, new[] { "1 star", "2 stars" });
        }

        [Fact]
        public void Classify_Multilingual_Spanish()
        {
            var result = _classifier.Classify("¡Me encanta este producto! Es perfecto.");
            _output.WriteLine($"Spanish positive => {result}");
            LogAllScores(result);

            Assert.Contains(result.Label, new[] { "4 stars", "5 stars" });
        }

        [Fact]
        public void Classify_AllScoresSumToOne()
        {
            var result = _classifier.Classify("Average product, nothing special");
            var sum = result.AllScores.Sum(s => s.Score);

            _output.WriteLine($"Score sum: {sum:F8}");
            Assert.InRange(sum, 0.99f, 1.01f);
        }

        private void LogAllScores(ClassificationResult result)
        {
            foreach (var (label, score) in result.AllScores)
                _output.WriteLine($"  {label}: {score:F6}");
        }
    }

    public class DistilRobertaEmotionTests : IDisposable
    {
        private readonly Classifier _classifier;
        private readonly ITestOutputHelper _output;

        public DistilRobertaEmotionTests(ITestOutputHelper output)
        {
            _output = output;
            _classifier = new Classifier("distilroberta-emotion", quiet: true);
        }

        public void Dispose() => _classifier.Dispose();

        [Fact]
        public void NumLabels_Is7()
        {
            Assert.Equal(7, _classifier.NumLabels);
        }

        [Theory]
        [InlineData("I just got promoted and I'm so happy!", "joy")]
        [InlineData("Today is the best day of my life!", "joy")]
        public void Classify_Joy(string text, string expectedLabel)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
        }

        [Theory]
        [InlineData("I'm so angry about what happened, this is unfair!", "anger")]
        public void Classify_Anger(string text, string expectedLabel)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
        }

        [Theory]
        [InlineData("I lost my best friend today, I'm devastated.", "sadness")]
        public void Classify_Sadness(string text, string expectedLabel)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
        }

        [Theory]
        [InlineData("There's a strange noise outside and I'm home alone.", "fear")]
        public void Classify_Fear(string text, string expectedLabel)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
        }

        [Fact]
        public void Classify_HasAllExpectedLabels()
        {
            var result = _classifier.Classify("Test text for label inspection");
            var labels = result.AllScores.Select(s => s.Label).OrderBy(l => l).ToArray();

            _output.WriteLine($"Labels: {string.Join(", ", labels)}");

            Assert.Contains("joy", labels);
            Assert.Contains("anger", labels);
            Assert.Contains("sadness", labels);
            Assert.Contains("fear", labels);
            Assert.Contains("surprise", labels);
            Assert.Contains("disgust", labels);
            Assert.Contains("neutral", labels);
        }

        [Fact]
        public void Classify_AllScoresSumToOne()
        {
            var result = _classifier.Classify("Just another regular day.");
            var sum = result.AllScores.Sum(s => s.Score);

            _output.WriteLine($"Score sum: {sum:F8}");
            Assert.InRange(sum, 0.99f, 1.01f);
        }

        private void LogAllScores(ClassificationResult result)
        {
            foreach (var (label, score) in result.AllScores)
                _output.WriteLine($"  {label}: {score:F6}");
        }
    }
    public class ToxicBertTests : IDisposable
    {
        private readonly Classifier _classifier;
        private readonly ITestOutputHelper _output;

        public ToxicBertTests(ITestOutputHelper output)
        {
            _output = output;
            _classifier = new Classifier("toxic-bert", quiet: true);
        }

        public void Dispose() => _classifier.Dispose();

        [Fact]
        public void NumLabels_Is6()
        {
            Assert.Equal(6, _classifier.NumLabels);
        }

        [Fact]
        public void Classify_NonToxicText_LowScores()
        {
            var result = _classifier.Classify("Have a wonderful day, I hope everything goes well for you!");
            _output.WriteLine($"Non-toxic => {result}");
            LogAllScores(result);

            // For non-toxic text, the top toxic label score should be low
            // The model outputs per-label probabilities, not softmax
            foreach (var (label, score) in result.AllScores)
            {
                Assert.True(score < 0.5f, $"Non-toxic text scored {score:F4} on '{label}'");
            }
        }

        [Fact]
        public void Classify_ToxicText_HighToxicScore()
        {
            var result = _classifier.Classify("You are an absolute idiot and I hate everything about you.");
            _output.WriteLine($"Toxic => {result}");
            LogAllScores(result);

            // Should flag as toxic with reasonable confidence
            var toxicScore = result.AllScores.FirstOrDefault(s => s.Label == "toxic").Score;
            _output.WriteLine($"Toxic score: {toxicScore:F6}");

            Assert.True(toxicScore > 0.5f, $"Expected toxic score > 0.5, got {toxicScore:F4}");
        }

        [Fact]
        public void Classify_HasAllExpectedLabels()
        {
            var result = _classifier.Classify("Test text");
            var labels = result.AllScores.Select(s => s.Label).OrderBy(l => l).ToArray();

            _output.WriteLine($"Labels: {string.Join(", ", labels)}");

            Assert.Contains("toxic", labels);
            Assert.Contains("insult", labels);
            Assert.Contains("obscene", labels);
        }

        [Fact]
        public void Classify_ContrastToxicVsClean()
        {
            var toxic = _classifier.Classify("Shut up you moron, nobody likes you.");
            var clean = _classifier.Classify("Thank you for your kind help today, I really appreciate it.");

            var toxicScore = toxic.AllScores.First(s => s.Label == "toxic").Score;
            var cleanScore = clean.AllScores.First(s => s.Label == "toxic").Score;

            _output.WriteLine($"Toxic text -> toxic score: {toxicScore:F6}");
            _output.WriteLine($"Clean text -> toxic score: {cleanScore:F6}");

            Assert.True(toxicScore > cleanScore,
                $"Toxic text ({toxicScore:F4}) should score higher than clean text ({cleanScore:F4})");
        }

        private void LogAllScores(ClassificationResult result)
        {
            foreach (var (label, score) in result.AllScores)
                _output.WriteLine($"  {label}: {score:F6}");
        }
    }

    public class ClassifierGeneralTests
    {
        [Fact]
        public void Classify_AfterDispose_Throws()
        {
            var classifier = new Classifier("distilbert-sentiment", quiet: true);
            classifier.Dispose();

            Assert.Throws<ObjectDisposedException>(() => classifier.Classify("test"));
        }

        [Fact]
        public void DoubleDispose_DoesNotThrow()
        {
            var classifier = new Classifier("distilbert-sentiment", quiet: true);
            classifier.Dispose();
            classifier.Dispose();
        }

        [Fact]
        public void Classify_UnicodeText_DoesNotCrash()
        {
            using var classifier = new Classifier("distilbert-sentiment", quiet: true);
            var result = classifier.Classify("日本語テスト 🌍 café résumé");

            Assert.NotNull(result);
            Assert.NotNull(result.Label);
            Assert.True(result.Score > 0);
        }

        [Fact]
        public void Classify_EmptyString_DoesNotCrash()
        {
            using var classifier = new Classifier("distilbert-sentiment", quiet: true);
            var result = classifier.Classify("");

            Assert.NotNull(result);
        }

        [Fact]
        public void Classify_VeryLongText_DoesNotCrash()
        {
            using var classifier = new Classifier("distilbert-sentiment", quiet: true);
            var longText = string.Join(" ", Enumerable.Repeat("This is great!", 200));
            var result = classifier.Classify(longText);

            Assert.NotNull(result);
            Assert.Equal("POSITIVE", result.Label);
        }
    }
}