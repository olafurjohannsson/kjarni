"""High-level Chat API."""

from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple
from ctypes import (
    c_void_p, c_char_p, c_int, c_int32, c_float, c_size_t, c_bool, byref, POINTER, Structure,
    CFUNCTYPE, create_string_buffer,
)

from ._ffi import _lib, check_error, take_string, encode_or_none, KjarniDevice


class KjarniChatConfig(Structure):
    _fields_ = [
        ("device", c_int),
        ("cache_dir", c_char_p),
        ("model_name", c_char_p),
        ("model_path", c_char_p),
        ("system_prompt", c_char_p),
        ("mode", c_int32),
        ("quiet", c_int32),
        ("quantized", c_int32),
    ]


class KjarniGenerationConfig(Structure):
    _fields_ = [
        ("temperature", c_float),
        ("top_k", c_int32),
        ("top_p", c_float),
        ("min_p", c_float),
        ("repetition_penalty", c_float),
        ("max_new_tokens", c_int32),
        ("do_sample", c_int32),
    ]


# bool callback(const char* text, void* user_data); returning false stops generation.
STREAM_CALLBACK_TYPE = CFUNCTYPE(c_bool, c_char_p, c_void_p)

_lib.kjarni_chat_config_default.restype = KjarniChatConfig
_lib.kjarni_chat_config_default.argtypes = []
_lib.kjarni_generation_config_default.restype = KjarniGenerationConfig
_lib.kjarni_generation_config_default.argtypes = []
_lib.kjarni_chat_new.restype = c_int
_lib.kjarni_chat_new.argtypes = [POINTER(KjarniChatConfig), POINTER(c_void_p)]
_lib.kjarni_chat_free.restype = None
_lib.kjarni_chat_free.argtypes = [c_void_p]
_lib.kjarni_chat_send.restype = c_int
_lib.kjarni_chat_send.argtypes = [c_void_p, c_char_p, POINTER(KjarniGenerationConfig), POINTER(c_void_p)]
_lib.kjarni_chat_stream.restype = c_int
_lib.kjarni_chat_stream.argtypes = [
    c_void_p, c_char_p, POINTER(KjarniGenerationConfig), STREAM_CALLBACK_TYPE, c_void_p, c_void_p
]
_lib.kjarni_chat_send_with_history.restype = c_int
_lib.kjarni_chat_send_with_history.argtypes = [
    c_void_p, POINTER(c_int32), POINTER(c_char_p), c_size_t, c_char_p,
    POINTER(KjarniGenerationConfig), POINTER(c_void_p),
]
_lib.kjarni_chat_stream_with_history.restype = c_int
_lib.kjarni_chat_stream_with_history.argtypes = [
    c_void_p, POINTER(c_int32), POINTER(c_char_p), c_size_t, c_char_p,
    POINTER(KjarniGenerationConfig), STREAM_CALLBACK_TYPE, c_void_p, c_void_p,
]
_lib.kjarni_chat_conversation_new.restype = c_int
_lib.kjarni_chat_conversation_new.argtypes = [c_void_p, POINTER(c_void_p)]
_lib.kjarni_chat_conversation_free.restype = None
_lib.kjarni_chat_conversation_free.argtypes = [c_void_p]
_lib.kjarni_chat_conversation_send.restype = c_int
_lib.kjarni_chat_conversation_send.argtypes = [
    c_void_p, c_char_p, POINTER(KjarniGenerationConfig), POINTER(c_void_p)
]
_lib.kjarni_chat_conversation_stream.restype = c_int
_lib.kjarni_chat_conversation_stream.argtypes = [
    c_void_p, c_char_p, POINTER(KjarniGenerationConfig), STREAM_CALLBACK_TYPE, c_void_p, c_void_p
]
_lib.kjarni_chat_conversation_len.restype = c_size_t
_lib.kjarni_chat_conversation_len.argtypes = [c_void_p]
_lib.kjarni_chat_conversation_clear.restype = None
_lib.kjarni_chat_conversation_clear.argtypes = [c_void_p, c_int32]
_lib.kjarni_chat_model_name.restype = c_size_t
_lib.kjarni_chat_model_name.argtypes = [c_void_p, c_char_p, c_size_t]
_lib.kjarni_chat_context_size.restype = c_size_t
_lib.kjarni_chat_context_size.argtypes = [c_void_p]


_MODES = {"default": 0, "creative": 1, "reasoning": 2}
_ROLES = {"system": 0, "user": 1, "assistant": 2}


class Turn(NamedTuple):
    """One message of a transcript: role is "system", "user" or "assistant"."""
    role: str
    content: str


def _generation_config(
    temperature: Optional[float],
    top_k: Optional[int],
    top_p: Optional[float],
    min_p: Optional[float],
    repetition_penalty: Optional[float],
    max_new_tokens: Optional[int],
    do_sample: Optional[bool],
):
    """A config pointer for the given overrides, or None when there are none."""
    overrides = (temperature, top_k, top_p, min_p, repetition_penalty, max_new_tokens, do_sample)
    if all(v is None for v in overrides):
        return None
    cfg = _lib.kjarni_generation_config_default()
    if temperature is not None:
        cfg.temperature = temperature
    if top_k is not None:
        cfg.top_k = top_k
    if top_p is not None:
        cfg.top_p = top_p
    if min_p is not None:
        cfg.min_p = min_p
    if repetition_penalty is not None:
        cfg.repetition_penalty = repetition_penalty
    if max_new_tokens is not None:
        cfg.max_new_tokens = max_new_tokens
    if do_sample is not None:
        cfg.do_sample = 1 if do_sample else 0
    return byref(cfg)


class _Stream:
    """Adapts a Python callable to the native token callback.

    Keeps the CFUNCTYPE object alive for the whole call, collects the text, and
    carries an exception raised by `on_token` back across the native frames, which
    ctypes would otherwise print and swallow.
    """

    def __init__(self, on_token: Optional[Callable[[str], Optional[bool]]]):
        self.parts: List[str] = []
        self.error: Optional[BaseException] = None
        self._on_token = on_token
        self.callback = STREAM_CALLBACK_TYPE(self._call)

    def _call(self, text, _user_data):
        token = text.decode("utf-8", errors="replace") if text else ""
        self.parts.append(token)
        if self._on_token is None:
            return True
        try:
            return self._on_token(token) is not False
        except BaseException as e:  # noqa: BLE001 - re-raised after the native call
            self.error = e
            return False

    def finish(self, err: int) -> str:
        if self.error is not None:
            raise self.error
        check_error(err)
        return "".join(self.parts)


class _History:
    """Native arrays for a transcript, kept alive for the duration of a call."""

    def __init__(self, history: Iterable[Tuple[str, str]]):
        turns = list(history)
        self.len = len(turns)
        self.roles = (c_int32 * max(self.len, 1))()
        self.contents = (c_char_p * max(self.len, 1))()
        for i, (role, content) in enumerate(turns):
            if role not in _ROLES:
                raise ValueError(f"unknown role {role!r}; expected system, user or assistant")
            self.roles[i] = _ROLES[role]
            self.contents[i] = content.encode("utf-8")


class Chat:
    """Chat with a local language model.

    Example:
        >>> chat = Chat("qwen2.5-0.5b-instruct")
        >>> print(chat.send("What is the capital of France?"))
        >>> chat.stream("Tell me a story", on_token=lambda t: print(t, end="", flush=True))
    """

    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        mode: str = "default",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        quiet: bool = False,
        quantized: bool = False,
    ):
        """Load a chat model, downloading it on first use.

        Args:
            model: e.g. "qwen2.5-0.5b-instruct" or "llama3.2-1b-instruct"
            system_prompt: Instructions for the assistant
            mode: "default", "creative" or "reasoning"
            device: "cpu" or "gpu"
            cache_dir: Directory to cache models
            quiet: Suppress progress output
            quantized: Load the 4-bit GGUF weights instead of full precision: a
                fraction of the memory and several times faster to generate on a CPU
        """
        if mode not in _MODES:
            raise ValueError(f"unknown mode {mode!r}; expected default, creative or reasoning")
        config = _lib.kjarni_chat_config_default()
        config.device = KjarniDevice.GPU if device == "gpu" else KjarniDevice.CPU
        config.model_name = model.encode("utf-8")
        config.system_prompt = encode_or_none(system_prompt)
        config.cache_dir = encode_or_none(cache_dir)
        config.mode = _MODES[mode]
        config.quiet = 1 if quiet else 0
        config.quantized = 1 if quantized else 0

        self._handle = c_void_p()
        check_error(_lib.kjarni_chat_new(byref(config), byref(self._handle)))

    def __del__(self):
        if getattr(self, "_handle", None):
            _lib.kjarni_chat_free(self._handle)
            self._handle = None

    @property
    def model_name(self) -> str:
        size = _lib.kjarni_chat_model_name(self._handle, None, 0)
        buf = create_string_buffer(size + 1)
        _lib.kjarni_chat_model_name(self._handle, buf, size + 1)
        return buf.value.decode("utf-8")

    @property
    def context_size(self) -> int:
        return _lib.kjarni_chat_context_size(self._handle)

    def send(self, message: str, *, temperature: Optional[float] = None, top_k: Optional[int] = None,
             top_p: Optional[float] = None, min_p: Optional[float] = None,
             repetition_penalty: Optional[float] = None, max_new_tokens: Optional[int] = None,
             do_sample: Optional[bool] = None) -> str:
        """Send one message and return the whole reply. Keeps no history."""
        gen = _generation_config(temperature, top_k, top_p, min_p, repetition_penalty, max_new_tokens, do_sample)
        out = c_void_p()
        check_error(_lib.kjarni_chat_send(self._handle, message.encode("utf-8"), gen, byref(out)))
        return take_string(out)

    def stream(self, message: str, on_token: Optional[Callable[[str], Optional[bool]]] = None, *,
               temperature: Optional[float] = None, top_k: Optional[int] = None,
               top_p: Optional[float] = None, min_p: Optional[float] = None,
               repetition_penalty: Optional[float] = None, max_new_tokens: Optional[int] = None,
               do_sample: Optional[bool] = None) -> str:
        """Stream a reply, calling `on_token` with each piece; return False from it to stop.

        Returns the text generated, including the piece that stopped it.
        """
        gen = _generation_config(temperature, top_k, top_p, min_p, repetition_penalty, max_new_tokens, do_sample)
        s = _Stream(on_token)
        err = _lib.kjarni_chat_stream(self._handle, message.encode("utf-8"), gen, s.callback, None, None)
        return s.finish(err)

    def send_with_history(self, history: Iterable[Tuple[str, str]], message: str, *,
                          temperature: Optional[float] = None, top_k: Optional[int] = None,
                          top_p: Optional[float] = None, min_p: Optional[float] = None,
                          repetition_penalty: Optional[float] = None, max_new_tokens: Optional[int] = None,
                          do_sample: Optional[bool] = None) -> str:
        """Reply to `message` given prior `(role, content)` turns, oldest first.

        For callers that own the transcript and replay it every turn. A system turn
        replaces the system prompt given at construction.
        """
        gen = _generation_config(temperature, top_k, top_p, min_p, repetition_penalty, max_new_tokens, do_sample)
        h = _History(history)
        out = c_void_p()
        check_error(_lib.kjarni_chat_send_with_history(
            self._handle, h.roles, h.contents, h.len, message.encode("utf-8"), gen, byref(out)))
        return take_string(out)

    def stream_with_history(self, history: Iterable[Tuple[str, str]], message: str,
                            on_token: Optional[Callable[[str], Optional[bool]]] = None, *,
                            temperature: Optional[float] = None, top_k: Optional[int] = None,
                            top_p: Optional[float] = None, min_p: Optional[float] = None,
                            repetition_penalty: Optional[float] = None, max_new_tokens: Optional[int] = None,
                            do_sample: Optional[bool] = None) -> str:
        """`send_with_history`, streamed through `on_token`."""
        gen = _generation_config(temperature, top_k, top_p, min_p, repetition_penalty, max_new_tokens, do_sample)
        h = _History(history)
        s = _Stream(on_token)
        err = _lib.kjarni_chat_stream_with_history(
            self._handle, h.roles, h.contents, h.len, message.encode("utf-8"), gen, s.callback, None, None)
        return s.finish(err)

    def conversation(self) -> "Conversation":
        """Start a conversation that keeps its own history."""
        return Conversation(self)


class Conversation:
    """A conversation whose history is kept by the engine between messages."""

    def __init__(self, chat: Chat):
        self._chat = chat  # the conversation borrows the model, so keep it alive
        self._handle = c_void_p()
        check_error(_lib.kjarni_chat_conversation_new(chat._handle, byref(self._handle)))

    def __del__(self):
        if getattr(self, "_handle", None):
            _lib.kjarni_chat_conversation_free(self._handle)
            self._handle = None

    def __len__(self) -> int:
        """Messages in the history, including the system prompt."""
        return _lib.kjarni_chat_conversation_len(self._handle)

    def send(self, message: str, *, temperature: Optional[float] = None, top_k: Optional[int] = None,
             top_p: Optional[float] = None, min_p: Optional[float] = None,
             repetition_penalty: Optional[float] = None, max_new_tokens: Optional[int] = None,
             do_sample: Optional[bool] = None) -> str:
        gen = _generation_config(temperature, top_k, top_p, min_p, repetition_penalty, max_new_tokens, do_sample)
        out = c_void_p()
        check_error(_lib.kjarni_chat_conversation_send(self._handle, message.encode("utf-8"), gen, byref(out)))
        return take_string(out)

    def stream(self, message: str, on_token: Optional[Callable[[str], Optional[bool]]] = None, *,
               temperature: Optional[float] = None, top_k: Optional[int] = None,
               top_p: Optional[float] = None, min_p: Optional[float] = None,
               repetition_penalty: Optional[float] = None, max_new_tokens: Optional[int] = None,
               do_sample: Optional[bool] = None) -> str:
        gen = _generation_config(temperature, top_k, top_p, min_p, repetition_penalty, max_new_tokens, do_sample)
        s = _Stream(on_token)
        err = _lib.kjarni_chat_conversation_stream(self._handle, message.encode("utf-8"), gen, s.callback, None, None)
        return s.finish(err)

    def clear(self, keep_system: bool = True) -> None:
        """Forget the history, keeping the system prompt unless told otherwise."""
        _lib.kjarni_chat_conversation_clear(self._handle, 1 if keep_system else 0)
