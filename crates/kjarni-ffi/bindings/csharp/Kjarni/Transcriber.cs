using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

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
    public record Transcript(string Text, string Language, float DurationSecs, TranscriptSegment[] Segments)
    {
        /// <summary>
        /// Format this transcript as SubRip (<c>.srt</c>) subtitle text.
        /// </summary>
        /// <returns>The formatted SRT subtitles, or an empty string if there is no speech.</returns>
        public string ToSrt()
        {
            if (Segments == null || Segments.Length == 0)
            {
                if (string.IsNullOrEmpty(Text))
                    return ToSrt(Array.Empty<TranscriptSegment>());
                var end = IsFinite(DurationSecs) ? Math.Max(0f, DurationSecs) : 0f;
                return ToSrt(new[] { new TranscriptSegment(0f, end, Text) });
            }
            return ToSrt(Segments);
        }

        /// <summary>
        /// Format this transcript as WebVTT (<c>.vtt</c>) subtitle text.
        /// </summary>
        /// <returns>The formatted WebVTT subtitles, beginning with <c>WEBVTT</c>.</returns>
        public string ToVtt()
        {
            if (Segments == null || Segments.Length == 0)
            {
                if (string.IsNullOrEmpty(Text))
                    return ToVtt(Array.Empty<TranscriptSegment>());
                var end = IsFinite(DurationSecs) ? Math.Max(0f, DurationSecs) : 0f;
                return ToVtt(new[] { new TranscriptSegment(0f, end, Text) });
            }
            return ToVtt(Segments);
        }

        /// <summary>
        /// Format timed segments as SubRip (<c>.srt</c>) subtitle text, wrapping long
        /// segments on word boundaries at 42 characters per line and 2 lines per cue.
        /// </summary>
        /// <param name="segments">Timed transcription segments.</param>
        /// <returns>The formatted SRT subtitles.</returns>
        public static string ToSrt(IReadOnlyList<TranscriptSegment> segments)
        {
            if (segments == null) throw new ArgumentNullException(nameof(segments));
            return FormatSubtitles(segments, isVtt: false);
        }

        /// <summary>
        /// Format timed segments as WebVTT (<c>.vtt</c>) subtitle text, wrapping long
        /// segments on word boundaries at 42 characters per line and 2 lines per cue.
        /// </summary>
        /// <param name="segments">Timed transcription segments.</param>
        /// <returns>The formatted WebVTT subtitles.</returns>
        public static string ToVtt(IReadOnlyList<TranscriptSegment> segments)
        {
            if (segments == null) throw new ArgumentNullException(nameof(segments));
            return FormatSubtitles(segments, isVtt: true);
        }

        private static bool IsFinite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);

        private static string FormatSubtitles(IReadOnlyList<TranscriptSegment> segments, bool isVtt)
        {
            if (segments.Count == 0)
            {
                Native.CheckError(isVtt
                    ? Native.kjarni_transcription_to_vtt(null, UIntPtr.Zero, out var emptyPtr)
                    : Native.kjarni_transcription_to_srt(null, UIntPtr.Zero, out emptyPtr));
                return Native.TakeString(emptyPtr);
            }

            var nativeSegs = new Native.KjarniTranscriptionSegment[segments.Count];
            var textPtrs = new IntPtr[segments.Count];
            try
            {
                for (int i = 0; i < segments.Count; i++)
                {
                    var seg = segments[i];
                    if (!string.IsNullOrEmpty(seg.Text))
                    {
                        var byteCount = Encoding.UTF8.GetByteCount(seg.Text);
                        var bytes = new byte[byteCount + 1];
                        Encoding.UTF8.GetBytes(seg.Text, 0, seg.Text.Length, bytes, 0);
                        bytes[byteCount] = 0;
                        var ptr = Marshal.AllocHGlobal(bytes.Length);
                        textPtrs[i] = ptr;
                        Marshal.Copy(bytes, 0, ptr, bytes.Length);
                    }
                    nativeSegs[i] = new Native.KjarniTranscriptionSegment
                    {
                        Start = seg.Start,
                        End = seg.End,
                        Text = textPtrs[i],
                    };
                }

                Native.CheckError(isVtt
                    ? Native.kjarni_transcription_to_vtt(nativeSegs, (UIntPtr)nativeSegs.Length, out var resultPtr)
                    : Native.kjarni_transcription_to_srt(nativeSegs, (UIntPtr)nativeSegs.Length, out resultPtr));
                return Native.TakeString(resultPtr);
            }
            finally
            {
                for (int i = 0; i < textPtrs.Length; i++)
                {
                    if (textPtrs[i] != IntPtr.Zero)
                        Marshal.FreeHGlobal(textPtrs[i]);
                }
            }
        }
    }

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
        /// <param name="modelPath">A local folder holding a Whisper model in the Hugging Face layout. Used instead of <paramref name="model"/> when given.</param>
        public Transcriber(
            string? model = null,
            string? language = null,
            bool translate = false,
            string device = "cpu",
            bool quiet = false,
            string? modelPath = null)
        {
            var config = Native.kjarni_transcriber_config_default();
            config.Device = device == "gpu" ? KjarniDevice.Gpu : KjarniDevice.Cpu;
            config.Translate = translate ? 1 : 0;
            config.Quiet = quiet ? 1 : 0;

            var modelPtr = Native.Utf8OrNull(model);
            var languagePtr = Native.Utf8OrNull(language);
            var modelPathPtr = Native.Utf8OrNull(modelPath);
            try
            {
                config.ModelPath = modelPathPtr;
                if (modelPtr != IntPtr.Zero) config.ModelName = modelPtr;
                config.Language = languagePtr;
                Native.CheckError(Native.kjarni_transcriber_new(ref config, out _handle));
            }
            finally
            {
                Marshal.FreeCoTaskMem(modelPtr);
                Marshal.FreeCoTaskMem(languagePtr);
                Marshal.FreeCoTaskMem(modelPathPtr);
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

        /// <summary>Releases the native model. The object cannot be used afterwards.</summary>
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

        /// <summary>Releases the native model if <see cref="Dispose"/> was not called.</summary>
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
