"""High-level Transcriber API."""

from typing import List, NamedTuple, Optional, Sequence
from ctypes import (
    c_void_p, c_char_p, c_int, c_int32, c_float, c_size_t, c_uint32, byref, POINTER, Structure,
)

from ._ffi import _lib, check_error, encode_or_none, KjarniDevice


class KjarniTranscriberConfig(Structure):
    _fields_ = [
        ("device", c_int),
        ("model_name", c_char_p),
        ("language", c_char_p),
        ("translate", c_int32),
        ("quiet", c_int32),
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
    ):
        """Load a Whisper model, downloading it on first use.

        Args:
            model: e.g. "whisper-small" (the default) or "whisper-large-v3"
            language: ISO code such as "en" or "is"; None detects it from the audio
            translate: Translate the speech into English instead of transcribing it
            device: "cpu" or "gpu"
            quiet: Suppress progress output
        """
        config = _lib.kjarni_transcriber_config_default()
        config.device = KjarniDevice.GPU if device == "gpu" else KjarniDevice.CPU
        config.translate = 1 if translate else 0
        config.quiet = 1 if quiet else 0
        if model:
            config.model_name = model.encode("utf-8")
        config.language = encode_or_none(language)

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
