package kjarni

import (
	"errors"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Segment is one timed piece of a transcript, in seconds from the start.
type Segment struct {
	Start float32
	End   float32
	Text  string
}

// Transcript is the result of transcribing audio.
type Transcript struct {
	Text string
	// Language is the ISO code transcribed: the one given, or the one detected.
	Language     string
	DurationSecs float32
	Segments     []Segment
}

type ffiTranscriberConfig struct {
	Device    int32
	_         int32 // padding
	ModelName uintptr
	Language  uintptr
	Translate int32
	Quiet     int32
}

type ffiTranscriptionSegment struct {
	Start float32
	End   float32
	Text  uintptr
}

type ffiTranscription struct {
	Text         uintptr
	Language     uintptr
	DurationSecs float32
	_            int32 // padding
	Segments     uintptr
	Len          uintptr
}

// Transcriber turns speech into text with Whisper.
type Transcriber struct {
	handle uintptr
	mu     sync.Mutex
	closed bool
}

// NewTranscriber loads a Whisper model, downloading it on first use. Pass "" for
// whisper-small. See WithLanguage and WithTranslate.
func NewTranscriber(model string, opts ...Option) (*Transcriber, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	o := applyOptions(opts)

	modelPtr, keepModel := cStringOrNull(model)
	defer keepModel()
	languagePtr, keepLanguage := cStringOrNull(o.language)
	defer keepLanguage()

	config := ffiTranscriberConfig{
		Device:    deviceCode(o.device),
		ModelName: modelPtr,
		Language:  languagePtr,
		Translate: boolToInt(o.translate),
		Quiet:     boolToInt(o.quiet),
	}
	var handle uintptr
	r1, _, _ := purego.SyscallN(_transcriberNewSym, uintptr(unsafe.Pointer(&config)), uintptr(unsafe.Pointer(&handle)))
	if code := int32(r1); code != 0 {
		return nil, lastError(code)
	}
	return &Transcriber{handle: handle}, nil
}

// TranscribeFile transcribes a wav, mp3, flac or ogg file.
func (t *Transcriber) TranscribeFile(path string) (*Transcript, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil, errors.New("transcriber is closed")
	}

	pathPtr, keepPath := cString(path)
	defer keepPath()
	var raw ffiTranscription
	r1, _, _ := purego.SyscallN(_transcriberTranscribeFileSym, t.handle, pathPtr, uintptr(unsafe.Pointer(&raw)))
	if code := int32(r1); code != 0 {
		return nil, lastError(code)
	}
	return takeTranscription(raw), nil
}

// Transcribe transcribes mono samples in -1..1 at any sample rate.
func (t *Transcriber) Transcribe(samples []float32, sampleRate int) (*Transcript, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil, errors.New("transcriber is closed")
	}
	if sampleRate <= 0 {
		return nil, errors.New("sample rate must be positive")
	}

	var data uintptr
	if len(samples) > 0 {
		data = uintptr(unsafe.Pointer(&samples[0]))
	}
	var raw ffiTranscription
	r1, _, _ := purego.SyscallN(_transcriberTranscribePCMSym, t.handle, data, uintptr(len(samples)),
		uintptr(sampleRate), uintptr(unsafe.Pointer(&raw)))
	runtime.KeepAlive(samples)
	if code := int32(r1); code != 0 {
		return nil, lastError(code)
	}
	return takeTranscription(raw), nil
}

// Close releases the model. Safe to call more than once.
func (t *Transcriber) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil
	}
	t.closed = true
	_transcriberFree(t.handle)
	t.handle = 0
	return nil
}

func takeTranscription(raw ffiTranscription) *Transcript {
	defer callWithStruct(_transcriptionFreeSym, unsafe.Pointer(&raw), unsafe.Sizeof(raw))

	out := &Transcript{
		Text:         goString(raw.Text),
		Language:     goString(raw.Language),
		DurationSecs: raw.DurationSecs,
		Segments:     make([]Segment, raw.Len),
	}
	if raw.Len > 0 {
		segs := unsafe.Slice((*ffiTranscriptionSegment)(unsafe.Pointer(raw.Segments)), raw.Len)
		for i, s := range segs {
			out.Segments[i] = Segment{Start: s.Start, End: s.End, Text: goString(s.Text)}
		}
	}
	return out
}
