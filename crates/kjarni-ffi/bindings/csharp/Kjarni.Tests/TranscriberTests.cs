using System;
using Xunit;

namespace Kjarni.Tests
{
    public class TranscriberTests
    {
        // Lazy, so a skipped test never loads the model.
        private static readonly Lazy<Transcriber> Shared = new Lazy<Transcriber>(() => new Transcriber(language: "en", quiet: true));
        private static Transcriber _transcriber => Shared.Value;

        [SkippableFact]
        public void TranscribesAWavFile()
        {
            Heavy.SkipIfSet();
            var t = _transcriber.TranscribeFile(Fixtures.SpeechWav);
            Assert.False(string.IsNullOrWhiteSpace(t.Text));
            Assert.Equal("en", t.Language);
            Assert.True(t.DurationSecs > 0);
            Assert.NotEmpty(t.Segments);
            Assert.All(t.Segments, s => Assert.True(s.End >= s.Start));
        }

        [SkippableFact]
        public void AcceptsPcmAtAnotherSampleRate()
        {
            Heavy.SkipIfSet();
            // A second of silence at 8 kHz exercises resampling.
            var t = _transcriber.Transcribe(new float[8000], 8000);
            Assert.NotNull(t.Text);
        }

        [SkippableFact]
        public void RejectsBadInput()
        {
            Heavy.SkipIfSet();
            Assert.Throws<ArgumentOutOfRangeException>(() => _transcriber.Transcribe(new float[100], 0));
            Assert.Throws<KjarniException>(() => _transcriber.TranscribeFile("/no/such/file.wav"));
        }

        [SkippableFact]
        public void ThrowsAfterDispose()
        {
            Heavy.SkipIfSet();
            var t = new Transcriber(language: "en", quiet: true);
            t.Dispose();
            Assert.Throws<ObjectDisposedException>(() => t.TranscribeFile(Fixtures.SpeechWav));
        }

        [Fact]
        public void EmptySegmentsProduceEmptySrtAndVttHeader()
        {
            Assert.Equal("", Transcript.ToSrt(Array.Empty<TranscriptSegment>()));
            Assert.Equal("WEBVTT\n\n", Transcript.ToVtt(Array.Empty<TranscriptSegment>()));
        }

        [Fact]
        public void ShortAndEmptySegmentsAreSkippedWithoutGaps()
        {
            var segs = new[]
            {
                new TranscriptSegment(0.0f, 1.5f, "  Hello there. "),
                new TranscriptSegment(1.5f, 2.0f, "  \n\t  "),
                new TranscriptSegment(61.25f, 3725.004f, " General\n\tKenobi. "),
            };
            Assert.Equal(
                "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n" +
                "2\n00:01:01,250 --> 01:02:05,004\nGeneral Kenobi.\n\n",
                Transcript.ToSrt(segs));
            Assert.Equal(
                "WEBVTT\n\n" +
                "00:00:00.000 --> 00:00:01.500\nHello there.\n\n" +
                "00:01:01.250 --> 01:02:05.004\nGeneral Kenobi.\n\n",
                Transcript.ToVtt(segs));
        }

        [Fact]
        public void WrapsLongSegmentOver42CharsIntoTwoLines()
        {
            var segs = new[]
            {
                new TranscriptSegment(0.0f, 4.0f, "This is a moderately long subtitle line that should wrap onto two lines."),
            };
            Assert.Equal(
                "1\n00:00:00,000 --> 00:00:04,000\n" +
                "This is a moderately long subtitle line\n" +
                "that should wrap onto two lines.\n\n",
                Transcript.ToSrt(segs));
        }

        [Fact]
        public void SplitsVeryLongSegmentIntoMultipleCuesProportionally()
        {
            var segs = new[]
            {
                new TranscriptSegment(
                    10.0f,
                    20.0f,
                    "Alpha bravo charlie delta echo foxtrot golf " +
                    "hotel india juliet kilo lima mike november " +
                    "oscar papa quebec."),
            };
            Assert.Equal(
                "1\n00:00:10,000 --> 00:00:17,404\n" +
                "Alpha bravo charlie delta echo foxtrot\n" +
                "golf hotel india juliet kilo lima mike\n\n" +
                "2\n00:00:17,404 --> 00:00:20,000\n" +
                "november oscar papa quebec.\n\n",
                Transcript.ToSrt(segs));
        }

        [Fact]
        public void KeepsSingleWordOver42CharsIntact()
        {
            const string longWord = "supercalifragilisticexpialidociousandmorewords";
            var segs = new[] { new TranscriptSegment(0.0f, 6.0f, $"Prefix {longWord} suffix") };
            Assert.Equal(
                $"1\n00:00:00,000 --> 00:00:05,390\nPrefix\n{longWord}\n\n" +
                "2\n00:00:05,390 --> 00:00:06,000\nsuffix\n\n",
                Transcript.ToSrt(segs));
        }

        [Fact]
        public void CountsUnicodeScalarCharactersNotUtf16CodeUnits()
        {
            const string line = "Þórður sagði halló við alla í Reykjavík í";
            var segs = new[] { new TranscriptSegment(0.0f, 3.0f, $"{line} dag.") };
            Assert.Equal(
                $"1\n00:00:00,000 --> 00:00:03,000\n{line}\ndag.\n\n",
                Transcript.ToSrt(segs));

            // Surrogate pair (U+1F600) is 2 UTF-16 chars in C#, but 1 Unicode scalar.
            var emojiLine = new string('\u0061', 41); // 41 * 'a'
            var emojiSeg = new[] { new TranscriptSegment(0.0f, 2.0f, $"{emojiLine} \U0001F600") };
            Assert.Equal(
                $"1\n00:00:00,000 --> 00:00:02,000\n{emojiLine}\n\U0001F600\n\n",
                Transcript.ToSrt(emojiSeg));

            var exact42 = new string('a', 40) + " \U0001F600"; // 40 + 1 + 1 = 42 Unicode chars (43 UTF-16 units)
            Assert.Equal(
                $"1\n00:00:00,000 --> 00:00:02,000\n{exact42}\n\n",
                Transcript.ToSrt(new[] { new TranscriptSegment(0.0f, 2.0f, exact42) }));
        }

        [Fact]
        public void HandlesTimestampRoundingAndBounds()
        {
            var segs = new[]
            {
                new TranscriptSegment(0.0f, 0.9996f, "Rounding up."),
                new TranscriptSegment(float.NaN, float.PositiveInfinity, "Non-finite."),
                new TranscriptSegment(-2.5f, -1.0f, "Negative."),
                new TranscriptSegment(5.0f, 3.0f, "Reversed."),
            };
            Assert.Equal(
                "1\n00:00:00,000 --> 00:00:01,000\nRounding up.\n\n" +
                "2\n00:00:00,000 --> 00:00:00,000\nNon-finite.\n\n" +
                "3\n00:00:00,000 --> 00:00:00,000\nNegative.\n\n" +
                "4\n00:00:05,000 --> 00:00:05,000\nReversed.\n\n",
                Transcript.ToSrt(segs));
        }

        [Fact]
        public void VttEscapesArrowInCueText()
        {
            var segs = new[] { new TranscriptSegment(0.0f, 2.0f, "From A --> B and x-->y") };
            Assert.Equal(
                "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nFrom A -- > B and x-- >y\n\n",
                Transcript.ToVtt(segs));
        }

        [Fact]
        public void TranscriptMethodsFallbackWhenSegmentsEmpty()
        {
            var transcript = new Transcript("  Just full text. ", "en", 4.5f, Array.Empty<TranscriptSegment>());
            Assert.Equal("1\n00:00:00,000 --> 00:00:04,500\nJust full text.\n\n", transcript.ToSrt());
            Assert.Equal("WEBVTT\n\n00:00:00.000 --> 00:00:04.500\nJust full text.\n\n", transcript.ToVtt());

            var empty = new Transcript("   ", "en", 1.0f, Array.Empty<TranscriptSegment>());
            Assert.Equal("", empty.ToSrt());
            Assert.Equal("WEBVTT\n\n", empty.ToVtt());
        }
    }
}
