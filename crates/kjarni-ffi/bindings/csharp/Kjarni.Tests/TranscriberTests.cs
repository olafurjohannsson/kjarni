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
    }
}
