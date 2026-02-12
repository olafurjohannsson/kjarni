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
        [InlineData("I love this product, it's absolutely amazing!", "POSITIVE", 0.9999f)]
        [InlineData("This is the best movie I have ever seen.", "POSITIVE", 0.9998f)]
        [InlineData("Wonderful experience, highly recommended.", "POSITIVE", 0.9999f)]
        public void Classify_PositiveText(string text, string expectedLabel, float expectedScore)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");

            Assert.Equal(expectedLabel, result.Label);
            Assert.Equal(expectedScore, result.Score, 3);
        }

        [Theory]
        [InlineData("This is terrible, I want my money back.", "NEGATIVE", 0.9997f)]
        [InlineData("Worst experience of my life, absolutely awful.", "NEGATIVE", 0.9998f)]
        [InlineData("The food was disgusting and the service was horrible.", "NEGATIVE", 0.9997f)]
        public void Classify_NegativeText(string text, string expectedLabel, float expectedScore)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");

            Assert.Equal(expectedLabel, result.Label);
            Assert.Equal(expectedScore, result.Score, 3);
        }

        [Fact]
        public void Classify_AmbiguousText()
        {
            var result = _classifier.Classify("This is a test sentence.");
            _output.WriteLine($"Ambiguous => {result}");

            Assert.Equal("NEGATIVE", result.Label);
            Assert.Equal(0.981f, result.Score, 3);
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
            var result = _classifier.Classify("Great product!");
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
        [InlineData("This made my day, so happy right now", "positive", 0.983f)]
        public void Classify_PositiveText(string text, string expectedLabel, float expectedScore)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
            Assert.Equal(expectedScore, result.Score, 3);
        }

        [Theory]
        [InlineData("This is the worst thing ever, so angry", "negative", 0.951f)]
        [InlineData("Terrible service, never coming back", "negative", 0.931f)]
        public void Classify_NegativeText(string text, string expectedLabel, float expectedScore)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
            Assert.Equal(expectedScore, result.Score, 3);
        }

        [Fact]
        public void Classify_NeutralText()
        {
            var result = _classifier.Classify("The meeting is scheduled for 3pm tomorrow.");
            _output.WriteLine($"Neutral test => {result}");
            LogAllScores(result);

            Assert.Equal("neutral", result.Label);
            Assert.Equal(0.951f, result.Score, 3);
        }

        [Fact]
        public void Classify_ScoreOrdering()
        {
            var result = _classifier.Classify("Testing score distribution");
            var labels = result.AllScores.Select(s => s.Label).ToArray();

            Assert.Equal("neutral", labels[0]);
            Assert.Equal("positive", labels[1]);
            Assert.Equal("negative", labels[2]);
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
        [InlineData("This product is absolutely fantastic, I love everything about it!", "5 stars", 0.971f)]
        [InlineData("Amazing quality, exceeded all my expectations", "5 stars", 0.925f)]
        public void Classify_VeryPositive_5Stars(string text, string expectedLabel, float expectedScore)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
            Assert.Equal(expectedScore, result.Score, 3);
        }

        [Theory]
        [InlineData("Absolutely terrible, worst purchase ever, broken on arrival", "1 star", 0.982f)]
        public void Classify_VeryNegative_1Star(string text, string expectedLabel, float expectedScore)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
            Assert.Equal(expectedScore, result.Score, 3);
        }

        [Fact]
        public void Classify_Multilingual_German()
        {
            var result = _classifier.Classify("Dieses Produkt ist wunderbar, ich bin sehr zufrieden!");
            _output.WriteLine($"German positive => {result}");
            LogAllScores(result);

            Assert.Equal("5 stars", result.Label);
            Assert.Equal(0.882f, result.Score, 3);
        }

        [Fact]
        public void Classify_Multilingual_French()
        {
            var result = _classifier.Classify("C'est terrible, je suis tres decu de cet achat.");
            _output.WriteLine($"French negative => {result}");
            LogAllScores(result);

            Assert.Equal("1 star", result.Label);
            Assert.Equal(0.846f, result.Score, 3);
        }

        [Fact]
        public void Classify_Multilingual_Spanish()
        {
            var result = _classifier.Classify("\u00a1Me encanta este producto! Es perfecto.");
            _output.WriteLine($"Spanish positive => {result}");
            LogAllScores(result);

            Assert.Equal("5 stars", result.Label);
            Assert.Equal(0.933f, result.Score, 3);
        }

        [Fact]
        public void Classify_MidRange()
        {
            var result = _classifier.Classify("Average product, nothing special");
            _output.WriteLine($"Mid-range => {result}");
            LogAllScores(result);

            Assert.Equal("3 stars", result.Label);
            Assert.Equal(0.662f, result.Score, 3);

            var labels = result.AllScores.Select(s => s.Label).ToArray();
            Assert.Equal("3 stars", labels[0]);
            Assert.Equal("2 stars", labels[1]);
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
        [InlineData("I just got promoted and I'm so happy!", "joy", 0.964f)]
        [InlineData("Today is the best day of my life!", "joy", 0.981f)]
        public void Classify_Joy(string text, string expectedLabel, float expectedScore)
        {
            var result = _classifier.Classify(text);
            _output.WriteLine($"'{text}' => {result}");
            LogAllScores(result);

            Assert.Equal(expectedLabel, result.Label);
            Assert.Equal(expectedScore, result.Score, 3);
        }

        [Fact]
        public void Classify_Anger()
        {
            var result = _classifier.Classify("I'm so angry about what happened, this is unfair!");
            _output.WriteLine($"Anger => {result}");
            LogAllScores(result);

            Assert.Equal("anger", result.Label);
            Assert.Equal(0.984f, result.Score, 3);
        }

        [Fact]
        public void Classify_Sadness()
        {
            var result = _classifier.Classify("I lost my best friend today, I'm devastated.");
            _output.WriteLine($"Sadness => {result}");
            LogAllScores(result);

            Assert.Equal("sadness", result.Label);
            Assert.Equal(0.984f, result.Score, 3);
        }

        [Fact]
        public void Classify_Fear()
        {
            var result = _classifier.Classify("There's a strange noise outside and I'm home alone.");
            _output.WriteLine($"Fear => {result}");
            LogAllScores(result);

            Assert.Equal("fear", result.Label);
            Assert.Equal(0.873f, result.Score, 3);
        }

        [Fact]
        public void Classify_Neutral()
        {
            var result = _classifier.Classify("Just another regular day.");
            _output.WriteLine($"Neutral => {result}");
            LogAllScores(result);

            Assert.Equal("neutral", result.Label);
            Assert.Equal(0.923f, result.Score, 3);
        }

        [Fact]
        public void Classify_HasAllExpectedLabels()
        {
            var result = _classifier.Classify("Just another regular day.");
            var labels = result.AllScores.Select(s => s.Label).OrderBy(l => l).ToArray();

            _output.WriteLine($"Labels: {string.Join(", ", labels)}");

            Assert.Equal(7, labels.Length);
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
        public void Classify_NonToxicText()
        {
            var result = _classifier.Classify("Have a wonderful day, I hope everything goes well for you!");
            _output.WriteLine($"Non-toxic => {result}");
            LogAllScores(result);

            Assert.Equal("toxic", result.Label);
            Assert.Equal(0.001f, result.Score, 3);

            foreach (var (label, score) in result.AllScores)
                Assert.True(score < 0.01f, $"Non-toxic text scored {score:F4} on '{label}'");
        }

        [Fact]
        public void Classify_ToxicText()
        {
            var result = _classifier.Classify("You are an absolute idiot and I hate everything about you.");
            _output.WriteLine($"Toxic => {result}");
            LogAllScores(result);

            var toxic = result.AllScores.First(s => s.Label == "toxic");
            var insult = result.AllScores.First(s => s.Label == "insult");
            var obscene = result.AllScores.First(s => s.Label == "obscene");

            Assert.Equal(0.991f, toxic.Score, 3);
            Assert.Equal(0.945f, insult.Score, 3);
            Assert.Equal(0.714f, obscene.Score, 3);
        }

        [Fact]
        public void Classify_ScoreOrdering()
        {
            var result = _classifier.Classify("Shut up you moron, nobody likes you.");
            var labels = result.AllScores.Select(s => s.Label).ToArray();

            Assert.Equal("toxic", labels[0]);
            Assert.Equal("insult", labels[1]);
            Assert.Equal("obscene", labels[2]);
            Assert.Equal("severe_toxic", labels[3]);
            Assert.Equal("identity_hate", labels[4]);
            Assert.Equal("threat", labels[5]);
        }

        [Fact]
        public void Classify_ToxicScores()
        {
            var result = _classifier.Classify("Shut up you moron, nobody likes you.");
            _output.WriteLine($"Toxic => {result}");
            LogAllScores(result);

            var toxic = result.AllScores.First(s => s.Label == "toxic");
            var insult = result.AllScores.First(s => s.Label == "insult");
            var obscene = result.AllScores.First(s => s.Label == "obscene");

            Assert.Equal(0.992f, toxic.Score, 3);
            Assert.Equal(0.944f, insult.Score, 3);
            Assert.Equal(0.787f, obscene.Score, 3);
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

            Assert.Equal(0.992f, toxicScore, 3);
            Assert.Equal(0.001f, cleanScore, 3);
        }

        [Fact]
        public void Classify_MultiLabel_ScoresDoNotSumToOne()
        {
            var result = _classifier.Classify("You are an absolute idiot and I hate everything about you.");
            var sum = result.AllScores.Sum(s => s.Score);

            _output.WriteLine($"Score sum: {sum:F4}");
            Assert.True(sum > 1.0f, "Multi-label scores should sum to more than 1.0 for toxic text");
        }

        [Fact]
        public void Classify_HasAllExpectedLabels()
        {
            var result = _classifier.Classify("Have a wonderful day, I hope everything goes well for you!");
            var labels = result.AllScores.Select(s => s.Label).OrderBy(l => l).ToArray();

            _output.WriteLine($"Labels: {string.Join(", ", labels)}");

            Assert.Equal(6, labels.Length);
            Assert.Contains("toxic", labels);
            Assert.Contains("severe_toxic", labels);
            Assert.Contains("obscene", labels);
            Assert.Contains("threat", labels);
            Assert.Contains("insult", labels);
            Assert.Contains("identity_hate", labels);
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
            var result = classifier.Classify("Bonjour le monde, cafe resume");

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