using System;
using Xunit;
using Xunit.Abstractions;

namespace Kjarni.Tests
{
    public class UnknownModelTests
    {
        private readonly ITestOutputHelper _output;

        public UnknownModelTests(ITestOutputHelper output)
        {
            _output = output;
        }

        [Theory]
        [InlineData("minilm", "minilm-l6-v2")]
        [InlineData("minilm-l6", "minilm-l6-v2")]
        [InlineData("minnilm-l6-v2", "minilm-l6-v2")] 
        [InlineData("mpnet", "mpnet-base-v2")]
        [InlineData("distilbert", "distilbert-base")]
        public void Embedder_UnknownModel_SuggestsSimilar(string badName, string expectedSuggestion)
        {
            var ex = Assert.Throws<KjarniException>(() =>
                new Embedder(model: badName, quiet: true));

            _output.WriteLine($"Input: '{badName}'");
            _output.WriteLine($"Error: {ex.Message}");

            Assert.Contains("Did you mean", ex.Message);
            Assert.Contains(expectedSuggestion, ex.Message);
        }

        [Fact]
        public void Embedder_CompletelyWrongName_NoSuggestion()
        {
            var ex = Assert.Throws<KjarniException>(() =>
                new Embedder(model: "xyzzy123notamodel", quiet: true));

            _output.WriteLine($"Error: {ex.Message}");
            Assert.NotNull(ex.Message);
            Assert.True(ex.Message.Length > 0);
        }

        [Theory]
        [InlineData("distilbert-sentment", "distilbert-sentiment")]   // typo
        [InlineData("roberta-emotion", "distilroberta-emotion")]      // close match
        [InlineData("toxic", "toxic-bert")]                           // partial
        [InlineData("bert-sentiment", "bert-sentiment-multilingual")] // partial
        public void Classifier_UnknownModel_SuggestsSimilar(string badName, string expectedSuggestion)
        {
            var ex = Assert.Throws<KjarniException>(() =>
                new Classifier(model: badName, quiet: true));

            _output.WriteLine($"Input: '{badName}'");
            _output.WriteLine($"Error: {ex.Message}");

            Assert.Contains("Did you mean", ex.Message);
            Assert.Contains(expectedSuggestion, ex.Message);
        }

        [Theory]
        [InlineData("minilm-cross-encoder", "minilm-l6-v2-cross-encoder")]
        [InlineData("minilm-l6-cross", "minilm-l6-v2-cross-encoder")]
        public void Reranker_UnknownModel_SuggestsSimilar(string badName, string expectedSuggestion)
        {
            var ex = Assert.Throws<KjarniException>(() =>
                new Reranker(model: badName, quiet: true));

            _output.WriteLine($"Input: '{badName}'");
            _output.WriteLine($"Error: {ex.Message}");

            Assert.Contains("Did you mean", ex.Message);
            Assert.Contains(expectedSuggestion, ex.Message);
        }

        [Fact]
        public void Embedder_UnknownModel_HasCorrectErrorCode()
        {
            var ex = Assert.Throws<KjarniException>(() =>
                new Embedder(model: "not-a-model", quiet: true));

            _output.WriteLine($"ErrorCode: {ex.ErrorCode}");
            _output.WriteLine($"Message: {ex.Message}");
            Assert.Equal(KjarniErrorCode.ModelNotFound, ex.ErrorCode);
        }

        [Fact]
        public void Classifier_UnknownModel_HasCorrectErrorCode()
        {
            var ex = Assert.Throws<KjarniException>(() =>
                new Classifier(model: "not-a-model", quiet: true));

            _output.WriteLine($"ErrorCode: {ex.ErrorCode}");
            _output.WriteLine($"Message: {ex.Message}");

            Assert.Equal(KjarniErrorCode.ModelNotFound, ex.ErrorCode);
        }
    }
}