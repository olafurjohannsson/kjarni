using System;
using Xunit;

namespace Kjarni.Tests
{
    public class SummarizerTests
    {
        // Lazy, so a skipped test never loads the model.
        private static readonly Lazy<Summarizer> Shared = new Lazy<Summarizer>(() => new Summarizer(quiet: true));
        private static Summarizer _summarizer => Shared.Value;

        internal const string Article =
            "The city council voted on Tuesday to approve a new public library in the " +
            "downtown district. The library will include a children's wing, a maker space " +
            "with 3D printers, and a cafe. Construction is expected to begin next spring and " +
            "take about eighteen months. Council members said the project would be funded " +
            "through a combination of municipal bonds and private donations. Several " +
            "residents spoke in favour of the plan at the meeting, citing the lack of quiet " +
            "study spaces in the area. A small group opposed it, arguing the money would be " +
            "better spent on road repairs. The mayor said the library would become a " +
            "cornerstone of the community and help revitalise the downtown area, which has " +
            "seen several businesses close over the past few years.";

        [SkippableFact]
        public void SummarizesAnArticle()
        {
            Heavy.SkipIfSet();
            var summary = _summarizer.Summarize(Article);
            Assert.False(string.IsNullOrWhiteSpace(summary));
            Assert.True(summary.Length < Article.Length);
        }

        [SkippableFact]
        public void UnknownModelFails()
        {
            Heavy.SkipIfSet();
            var e = Assert.Throws<KjarniException>(() => new Summarizer(model: "no-such-model", quiet: true));
            Assert.Equal(KjarniErrorCode.ModelNotFound, e.ErrorCode);
        }
    }

    public class TranslatorTests
    {
        private static readonly Lazy<Translator> Shared = new Lazy<Translator>(() => new Translator(from: "en", to: "de", quiet: true));
        private static Translator _translator => Shared.Value;

        [SkippableFact]
        public void TranslatesWithDefaultLanguages()
        {
            Heavy.SkipIfSet();
            Assert.Contains("haus", _translator.Translate("The house is small."), StringComparison.OrdinalIgnoreCase);
        }

        [SkippableFact]
        public void LanguagesCanBeNamedPerCall()
        {
            Heavy.SkipIfSet();
            var text = _translator.Translate("The house is small.", from: "English", to: "German");
            Assert.Contains("haus", text, StringComparison.OrdinalIgnoreCase);
        }

        [SkippableFact]
        public void NoLanguagesIsInvalidConfig()
        {
            Heavy.SkipIfSet();
            using var bare = new Translator(quiet: true);
            var e = Assert.Throws<KjarniException>(() => bare.Translate("hello"));
            Assert.Equal(KjarniErrorCode.InvalidConfig, e.ErrorCode);
        }
    }
}
