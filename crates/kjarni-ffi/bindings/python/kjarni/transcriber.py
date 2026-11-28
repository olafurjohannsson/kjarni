"""High-level Transcriber API."""

import math
import os
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union
from ctypes import (
    c_void_p, c_char_p, c_int, c_int32, c_float, c_size_t, c_uint32, byref, POINTER, Structure,
)

from ._ffi import _lib, check_error, encode_or_none, KjarniDevice

_MAX_LINE_CHARS = 42
_MAX_LINES = 2


class KjarniTranscriberConfig(Structure):
    _fields_ = [
        ("device", c_int),
        ("model_name", c_char_p),
        ("language", c_char_p),
        ("translate", c_int32),
        ("quiet", c_int32),
        ("model_path", c_char_p),
    ]


class KjarniTranscriptionSegment(Structure):
    _fields_ = [
        ("start", c_float),
        ("end", c_float),
        ("text", c_char_p),
    ]


class KjarniTranscription(Structure):
    _fields_ = [
        ("text", c_char_p),
        ("language", c_char_p),
        ("duration_secs", c_float),
        ("segments", POINTER(KjarniTranscriptionSegment)),
        ("len", c_size_t),
    ]


_lib.kjarni_transcriber_config_default.restype = KjarniTranscriberConfig
_lib.kjarni_transcriber_config_default.argtypes = []
_lib.kjarni_transcriber_new.restype = c_int
_lib.kjarni_transcriber_new.argtypes = [POINTER(KjarniTranscriberConfig), POINTER(c_void_p)]
_lib.kjarni_transcriber_free.restype = None
_lib.kjarni_transcriber_free.argtypes = [c_void_p]
_lib.kjarni_transcriber_transcribe_file.restype = c_int
_lib.kjarni_transcriber_transcribe_file.argtypes = [c_void_p, c_char_p, POINTER(KjarniTranscription)]
_lib.kjarni_transcriber_transcribe_pcm.restype = c_int
_lib.kjarni_transcriber_transcribe_pcm.argtypes = [
    c_void_p, POINTER(c_float), c_size_t, c_uint32, POINTER(KjarniTranscription)
]
_lib.kjarni_transcription_free.restype = None
_lib.kjarni_transcription_free.argtypes = [KjarniTranscription]


class Segment(NamedTuple):
    """One timed piece of a transcript, in seconds from the start."""
    start: float
    end: float
    text: str


class Transcript(NamedTuple):
    """The result of transcribing audio."""
    text: str
    #: ISO code of the language transcribed: the one given, or the one detected.
    language: str
    duration_secs: float
    segments: List[Segment]

    def to_srt(self) -> str:
        """Format this transcript as SubRip (.srt) subtitle text."""
        if not self.segments:
            if not self.text.strip():
                return ""
            end = max(0.0, self.duration_secs) if math.isfinite(self.duration_secs) else 0.0
            return segments_to_srt([Segment(0.0, end, self.text)])
        return segments_to_srt(self.segments)

    def to_vtt(self) -> str:
        """Format this transcript as WebVTT (.vtt) subtitle text."""
        if not self.segments:
            if not self.text.strip():
                return "WEBVTT\n\n"
            end = max(0.0, self.duration_secs) if math.isfinite(self.duration_secs) else 0.0
            return segments_to_vtt([Segment(0.0, end, self.text)])
        return segments_to_vtt(self.segments)


def segments_to_srt(segments: Sequence[Segment]) -> str:
    """Format timed segments as SubRip (.srt) subtitle text.

    Long segments are split on word boundaries into lines of at most 42
    characters and 2 lines per cue, with timestamps distributed proportionally.
    """
    return _format_subtitles(segments, is_vtt=False)


def segments_to_vtt(segments: Sequence[Segment]) -> str:
    """Format timed segments as WebVTT (.vtt) subtitle text.

    Always begins with the ``WEBVTT`` header. Long segments are split on word
    boundaries into lines of at most 42 characters and 2 lines per cue, with
    timestamps distributed proportionally.
    """
    return _format_subtitles(segments, is_vtt=True)


def _format_subtitles(segments: Sequence[Segment], is_vtt: bool) -> str:
    parts: List[str] = ["WEBVTT\n\n"] if is_vtt else []
    decimal = "." if is_vtt else ","
    cue_index = 1

    for segment in segments:
        cues = _build_segment_cues(segment.text, is_vtt)
        if not cues:
            continue

        start_ms = _to_millis(segment.start)
        end_ms = max(start_ms, _to_millis(segment.end))
        for cue_start, cue_end, cue_text in _distribute_timestamps(cues, start_ms, end_ms):
            if not is_vtt:
                parts.append(f"{cue_index}\n")
                cue_index += 1
            parts.append(
                f"{_format_timestamp(cue_start, decimal)} --> "
                f"{_format_timestamp(cue_end, decimal)}\n{cue_text}\n\n"
            )

    return "".join(parts)


def _build_segment_cues(text: str, is_vtt: bool) -> List[str]:
    words = text.split()
    if not words:
        return []
    if is_vtt:
        joined = " ".join(words)
        if "-->" in joined:
            words = joined.replace("-->", "-- >").split()

    lines = _wrap_words(words, _MAX_LINE_CHARS)
    return ["\n".join(lines[i:i + _MAX_LINES]) for i in range(0, len(lines), _MAX_LINES)]


def _wrap_words(words: Sequence[str], max_line_chars: int) -> List[str]:
    lines: List[str] = []
    current_parts: List[str] = []
    current_chars = 0

    for word in words:
        word_chars = len(word)
        if current_chars == 0:
            current_parts.append(word)
            current_chars = word_chars
        elif current_chars + 1 + word_chars <= max_line_chars:
            current_parts.append(word)
            current_chars += 1 + word_chars
        else:
            lines.append(" ".join(current_parts))
            current_parts = [word]
            current_chars = word_chars

    if current_chars > 0:
        lines.append(" ".join(current_parts))

    return lines


def _distribute_timestamps(
    cues: Sequence[str], start_ms: int, end_ms: int
) -> List[Tuple[int, int, str]]:
    count = len(cues)
    if count == 1:
        return [(start_ms, end_ms, cues[0])]

    weights = [max(len(c), 1) for c in cues]
    total_weight = sum(weights)
    total_dur = end_ms - start_ms

    out: List[Tuple[int, int, str]] = []
    prev_end = start_ms
    cum_weight = 0
    for i, cue in enumerate(cues):
        cue_start = prev_end
        if i + 1 == count:
            cue_end = end_ms
        else:
            cum_weight += weights[i]
            offset = (total_dur * cum_weight + total_weight // 2) // total_weight
            cue_end = min(end_ms, max(cue_start, start_ms + offset))
        out.append((cue_start, cue_end, cue))
        prev_end = cue_end

    return out


def _to_millis(seconds: float) -> int:
    if not math.isfinite(seconds) or seconds <= 0.0:
        return 0
    ms = float(seconds) * 1000.0
    whole = math.floor(ms)
    return int(whole + 1 if (ms - whole) >= 0.5 else whole)


def _format_timestamp(millis: int, decimal: str) -> str:
    ms = millis % 1000
    total_secs = millis // 1000
    s = total_secs % 60
    m = (total_secs // 60) % 60
    h = total_secs // 3600
    return f"{h:02d}:{m:02d}:{s:02d}{decimal}{ms:03d}"


def _decode(b: Optional[bytes]) -> str:
    return b.decode("utf-8") if b else ""


class Transcriber:
    """Speech to text with Whisper.

    Example:
        >>> transcriber = Transcriber(language="en")
        >>> print(transcriber.transcribe_file("meeting.wav").text)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        language: Optional[str] = None,
        translate: bool = False,
        device: str = "cpu",
        quiet: bool = False,
        model_path: Optional[Union[str, os.PathLike]] = None,
    ):
        """Load a Whisper model, downloading it on first use.

        Args:
            model: e.g. "whisper-small" (the default) or "whisper-large-v3"
            language: ISO code such as "en" or "is"; None detects it from the audio
            translate: Translate the speech into English instead of transcribing it
            device: "cpu" or "gpu"
            quiet: Suppress progress output
            model_path: A local folder holding a Whisper model in the Hugging Face
                layout, used instead of `model`
        """
        config = _lib.kjarni_transcriber_config_default()
        config.device = KjarniDevice.GPU if device == "gpu" else KjarniDevice.CPU
        config.translate = 1 if translate else 0
        config.quiet = 1 if quiet else 0
        if model:
            config.model_name = model.encode("utf-8")
        config.language = encode_or_none(language)
        if model_path:
            config.model_path = os.fspath(model_path).encode("utf-8")

        self._handle = c_void_p()
        check_error(_lib.kjarni_transcriber_new(byref(config), byref(self._handle)))

    def __del__(self):
        if getattr(self, "_handle", None):
            _lib.kjarni_transcriber_free(self._handle)
            self._handle = None

    def transcribe_file(self, path: str) -> Transcript:
        """Transcribe a wav, mp3, flac or ogg file."""
        raw = KjarniTranscription()
        check_error(_lib.kjarni_transcriber_transcribe_file(self._handle, str(path).encode("utf-8"), byref(raw)))
        return self._take(raw)

    def transcribe(self, samples: Sequence[float], sample_rate: int) -> Transcript:
        """Transcribe mono samples in -1..1 at any sample rate.

        Accepts a list or any buffer of float32, such as a NumPy array, which is
        passed without copying.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        buf = _float_buffer(samples)
        raw = KjarniTranscription()
        check_error(_lib.kjarni_transcriber_transcribe_pcm(
            self._handle, buf, len(buf), sample_rate, byref(raw)))
        return self._take(raw)

    @staticmethod
    def _take(raw: KjarniTranscription) -> Transcript:
        try:
            segments = [
                Segment(s.start, s.end, _decode(s.text))
                for s in (raw.segments[i] for i in range(raw.len))
            ]
            return Transcript(_decode(raw.text), _decode(raw.language), raw.duration_secs, segments)
        finally:
            _lib.kjarni_transcription_free(raw)


def _float_buffer(samples):
    """A ctypes float array over `samples`, sharing memory when it is float32 already.

    Building one from a list unpacks every sample through Python, which for an hour
    of 16 kHz audio is 57 million calls, so buffers are used as they are.
    """
    if hasattr(samples, "astype") and hasattr(samples, "__array_interface__"):
        import numpy as np  # only reached when given an array, so numpy is present
        samples = np.ascontiguousarray(samples, dtype=np.float32)
    try:
        view = memoryview(samples)
    except TypeError:
        return (c_float * len(samples))(*samples)
    if view.format != "f" or not view.c_contiguous:
        return (c_float * len(samples))(*samples)
    array_type = c_float * (view.nbytes // 4)
    return array_type.from_buffer_copy(view) if view.readonly else array_type.from_buffer(view)
