using System;
using System.Collections.Generic;
using System.Globalization;
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
                if (string.IsNullOrWhiteSpace(Text))
                    return string.Empty;
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
                if (string.IsNullOrWhiteSpace(Text))
                    return "WEBVTT\n\n";
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
            return SubtitleFormatter.Format(segments, isVtt: false);
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
            return SubtitleFormatter.Format(segments, isVtt: true);
        }

        private static bool IsFinite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);
    }

    internal static class SubtitleFormatter
    {
        private const int MaxLineChars = 42;
        private const int MaxLines = 2;

        public static string Format(IReadOnlyList<TranscriptSegment> segments, bool isVtt)
        {
            var sb = new StringBuilder();
            if (isVtt)
                sb.Append("WEBVTT\n\n");

            var decimalSep = isVtt ? '.' : ',';
            int cueIndex = 1;

            for (int s = 0; s < segments.Count; s++)
            {
                var segment = segments[s];
                var cues = BuildSegmentCues(segment.Text ?? string.Empty, isVtt);
                if (cues.Count == 0)
                    continue;

                ulong startMs = ToMillis(segment.Start);
                ulong endMs = Math.Max(startMs, ToMillis(segment.End));

                foreach (var (cueStart, cueEnd, cueText) in DistributeTimestamps(cues, startMs, endMs))
                {
                    if (!isVtt)
                    {
                        sb.Append(cueIndex.ToString(CultureInfo.InvariantCulture));
                        sb.Append('\n');
                        cueIndex++;
                    }
                    AppendTimestamp(sb, cueStart, decimalSep);
                    sb.Append(" --> ");
                    AppendTimestamp(sb, cueEnd, decimalSep);
                    sb.Append('\n');
                    sb.Append(cueText);
                    sb.Append("\n\n");
                }
            }

            return sb.ToString();
        }

        private static List<string> BuildSegmentCues(string text, bool isVtt)
        {
            var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (words.Length == 0)
                return new List<string>();

            if (isVtt)
            {
                var joined = string.Join(" ", words);
                if (joined.IndexOf("-->", StringComparison.Ordinal) >= 0)
                    words = joined.Replace("-->", "-- >").Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            }

            var lines = WrapWords(words);
            var cues = new List<string>((lines.Count + MaxLines - 1) / MaxLines);
            for (int i = 0; i < lines.Count; i += MaxLines)
            {
                int count = Math.Min(MaxLines, lines.Count - i);
                cues.Add(string.Join("\n", lines.GetRange(i, count)));
            }
            return cues;
        }

        private static List<string> WrapWords(string[] words)
        {
            var lines = new List<string>();
            var current = new StringBuilder();
            int currentChars = 0;

            foreach (var word in words)
            {
                int wordChars = RuneCount(word);
                if (currentChars == 0)
                {
                    current.Append(word);
                    currentChars = wordChars;
                }
                else if (currentChars + 1 + wordChars <= MaxLineChars)
                {
                    current.Append(' ');
                    current.Append(word);
                    currentChars += 1 + wordChars;
                }
                else
                {
                    lines.Add(current.ToString());
                    current.Clear();
                    current.Append(word);
                    currentChars = wordChars;
                }
            }

            if (currentChars > 0)
                lines.Add(current.ToString());

            return lines;
        }

        private static List<(ulong Start, ulong End, string Text)> DistributeTimestamps(
            List<string> cues, ulong startMs, ulong endMs)
        {
            int count = cues.Count;
            if (count == 1)
                return new List<(ulong, ulong, string)> { (startMs, endMs, cues[0]) };

            var weights = new ulong[count];
            ulong totalWeight = 0;
            for (int i = 0; i < count; i++)
            {
                ulong w = Math.Max(1UL, (ulong)RuneCount(cues[i]));
                weights[i] = w;
                totalWeight += w;
            }

            ulong totalDur = endMs - startMs;
            var result = new List<(ulong, ulong, string)>(count);
            ulong prevEnd = startMs;
            ulong cumWeight = 0;

            for (int i = 0; i < count; i++)
            {
                ulong cueStart = prevEnd;
                ulong cueEnd;
                if (i + 1 == count)
                {
                    cueEnd = endMs;
                }
                else
                {
                    cumWeight += weights[i];
                    ulong offset = totalDur == 0 || cumWeight <= (ulong.MaxValue - totalWeight / 2) / totalDur
                        ? (totalDur * cumWeight + totalWeight / 2) / totalWeight
                        : (ulong)(((decimal)totalDur * cumWeight + (decimal)totalWeight / 2m) / totalWeight);
                    cueEnd = Math.Min(endMs, Math.Max(cueStart, startMs + offset));
                }
                result.Add((cueStart, cueEnd, cues[i]));
                prevEnd = cueEnd;
            }

            return result;
        }

        private static int RuneCount(string s)
        {
            int count = 0;
            for (int i = 0; i < s.Length; i++)
            {
                if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                    i++;
                count++;
            }
            return count;
        }

        private static ulong ToMillis(float seconds)
        {
            if (float.IsNaN(seconds) || float.IsInfinity(seconds) || seconds <= 0f)
                return 0;
            double ms = (double)seconds * 1000.0;
            if (ms >= 18446744073709551616.0)
                return ulong.MaxValue;
            return (ulong)Math.Round(ms, MidpointRounding.AwayFromZero);
        }

        private static void AppendTimestamp(StringBuilder sb, ulong millis, char decimalSep)
        {
            ulong ms = millis % 1000;
            ulong totalSecs = millis / 1000;
            ulong s = totalSecs % 60;
            ulong m = (totalSecs / 60) % 60;
            ulong h = totalSecs / 3600;
            sb.Append(h.ToString("D2", CultureInfo.InvariantCulture));
            sb.Append(':');
            sb.Append(m.ToString("D2", CultureInfo.InvariantCulture));
            sb.Append(':');
            sb.Append(s.ToString("D2", CultureInfo.InvariantCulture));
            sb.Append(decimalSep);
            sb.Append(ms.ToString("D3", CultureInfo.InvariantCulture));
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
