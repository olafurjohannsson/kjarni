using System;
using System.IO;
using System.Linq;
using Xunit;

namespace Kjarni.Tests
{
    /// <summary>
    /// The entities are what transformers' pipeline("ner", aggregation_strategy="first")
    /// gives for dslim/distilbert-NER on the CPU.
    /// </summary>
    public class EntityRecognizerTests
    {
        [SkippableFact]
        public void Recognize_MatchesTransformers()
        {
            Heavy.SkipIfSet();
            using var ner = new EntityRecognizer(quiet: true);
            var got = ner.Recognize("Sarah Connor met Apple's CEO in San Francisco on Tuesday.");

            var want = new[]
            {
                ("PER", "Sarah Connor", 0, 12, 0.99879f),
                ("ORG", "Apple", 17, 22, 0.99710f),
                ("LOC", "San Francisco", 32, 45, 0.99761f),
            };
            Assert.Equal(want.Length, got.Length);
            for (int i = 0; i < want.Length; i++)
            {
                Assert.Equal(want[i].Item1, got[i].Label);
                Assert.Equal(want[i].Item2, got[i].Text);
                Assert.Equal(want[i].Item3, got[i].Start);
                Assert.Equal(want[i].Item4, got[i].End);
                Assert.Equal(want[i].Item5, got[i].Score, 4);
            }
        }

        [SkippableFact]
        public void Spans_IndexTheDotNetString_EvenAfterAnEmoji()
        {
            Heavy.SkipIfSet();
            using var ner = new EntityRecognizer(quiet: true);
            const string text = "🎉 Angela Merkel visited Berlin.";
            var got = ner.Recognize(text);
            Assert.NotEmpty(got);
            foreach (var e in got)
                Assert.Equal(e.Text, text.Substring(e.Start, e.End - e.Start));
        }

        [Fact]
        public void Utf16Index_CountsSurrogatePairsAsOneCodePoint()
        {
            Assert.Equal(0, EntityRecognizer.Utf16Index("🎉a", 0));
            Assert.Equal(2, EntityRecognizer.Utf16Index("🎉a", 1));
            Assert.Equal(3, EntityRecognizer.Utf16Index("🎉a", 2));
            Assert.Equal(4, EntityRecognizer.Utf16Index("Þórð", 4));
        }

        [Fact]
        public void New_FromFolderWithoutModel_Throws()
        {
            var empty = Directory.CreateTempSubdirectory("kjarni_cs_ner_").FullName;
            try
            {
                Assert.ThrowsAny<Exception>(() => new EntityRecognizer(modelPath: empty, quiet: true));
            }
            finally
            {
                Directory.Delete(empty, true);
            }
        }
    }
}
