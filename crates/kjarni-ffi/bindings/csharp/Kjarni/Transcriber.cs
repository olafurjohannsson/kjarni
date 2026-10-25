using System;
using System.Runtime.InteropServices;

namespace Kjarni
{
    /// <summary>
    /// One timed piece of a transcript, in seconds from the start.
    /// </summary>
    public record TranscriptSegment(float Start, float End, string Text);

    /// <summary>
    /// The result of transcribing audio.
    /// </summary>
    /// <param name="Text">The whole transcript.</param>
    /// <param name="Language">ISO code of the language transcribed: the one given, or the one detected.</param>
    /// <param name="DurationSecs">Length of the audio.</param>
    /// <param name="Segments">The transcript split into timed pieces.</param>
    public record Transcript(string Text, string Language, float DurationSecs, TranscriptSegment[] Segments);

    /// <summary>
    /// Speech to text with Whisper.
    /// </summary>
    /// <example>
    /// <code>
    /// using var transcriber = new Transcriber(language: "en");
    /// var t = transcriber.TranscribeFile("meeting.wav");
    /// Console.WriteLine(t.Text);
    /// </code>
    /// </example>
    public class Transcriber : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>
        /// Load a Whisper model, downloading it on first use.
        /// </summary>
        /// <param name="model">Model name, e.g. "whisper-small" (the default) or "whisper-large-v3".</param>
        /// <param name="language">ISO code such as "en" or "is"; null detects it from the audio.</param>
        /// <param name="translate">Translate the speech into English instead of transcribing it.</param>
        /// <param name="device">Compute device - "cpu" or "gpu".</param>
        /// <param name="quiet">Suppress progress output.</param>
        public Transcriber(
            string? model = null,
            string? language = null,
            bool translate = false,
            string device = "cpu",
            bool quiet = false)
        {
            var config = Native.kjarni_transcriber_config_default();
            config.Device = device == "gpu" ? KjarniDevice.Gpu : KjarniDevice.Cpu;
            config.Translate = translate ? 1 : 0;
            config.Quiet = quiet ? 1 : 0;

            var modelPtr = Native.Utf8OrNull(model);
            var languagePtr = Native.Utf8OrNull(language);
            try
            {
                if (modelPtr != IntPtr.Zero) config.ModelName = modelPtr;
                config.Language = languagePtr;
                Native.CheckError(Native.kjarni_transcriber_new(ref config, out _handle));
            }
            finally
            {
                Marshal.FreeCoTaskMem(modelPtr);
                Marshal.FreeCoTaskMem(languagePtr);
            }
        }

        /// <summary>
        /// Transcribe a wav, mp3, flac or ogg file.
        /// </summary>
        public Transcript TranscribeFile(string path)
        {
            ThrowIfDisposed();
            Native.CheckError(Native.kjarni_transcriber_transcribe_file(_handle, path, out var raw));
            return Take(raw);
        }

        /// <summary>
        /// Transcribe mono samples in -1..1 at any sample rate; they are resampled to 16 kHz.
        /// </summary>
        public Transcript Transcribe(float[] samples, int sampleRate)
        {
            ThrowIfDisposed();
            if (sampleRate <= 0)
                throw new ArgumentOutOfRangeException(nameof(sampleRate), "sample rate must be positive");
            Native.CheckError(Native.kjarni_transcriber_transcribe_pcm(
                _handle, samples, (UIntPtr)samples.Length, (uint)sampleRate, out var raw));
            return Take(raw);
        }

        private static Transcript Take(Native.KjarniTranscription raw)
        {
            try
            {
                var len = (int)raw.Len;
                var segments = new TranscriptSegment[len];
                var size = Marshal.SizeOf<Native.KjarniTranscriptionSegment>();
                for (int i = 0; i < len; i++)
                {
                    var s = Marshal.PtrToStructure<Native.KjarniTranscriptionSegment>(IntPtr.Add(raw.Segments, i * size));
                    segments[i] = new TranscriptSegment(s.Start, s.End, PtrToString(s.Text));
                }
                return new Transcript(PtrToString(raw.Text), PtrToString(raw.Language), raw.DurationSecs, segments);
            }
            finally
            {
                Native.kjarni_transcription_free(raw);
            }
        }

        private static string PtrToString(IntPtr p) =>
            p == IntPtr.Zero ? "" : Marshal.PtrToStringUTF8(p) ?? "";

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    Native.kjarni_transcriber_free(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~Transcriber()
        {
            Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Transcriber));
        }
    }
}
