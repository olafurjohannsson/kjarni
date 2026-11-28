package kjarni

import (
	"errors"
	"math"
	"runtime"
	"strings"
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

// ToSRT formats the transcript as SubRip (.srt) subtitle text.
func (t *Transcript) ToSRT() (string, error) {
	if t == nil {
		return SegmentsToSRT(nil)
	}
	if len(t.Segments) == 0 {
		if strings.TrimSpace(t.Text) == "" {
			return SegmentsToSRT(nil)
		}
		end := float32(0)
		d := float64(t.DurationSecs)
		if !math.IsNaN(d) && !math.IsInf(d, 0) && t.DurationSecs > 0 {
			end = t.DurationSecs
		}
		return SegmentsToSRT([]Segment{{Start: 0, End: end, Text: t.Text}})
	}
	return SegmentsToSRT(t.Segments)
}

// ToVTT formats the transcript as WebVTT (.vtt) subtitle text.
func (t *Transcript) ToVTT() (string, error) {
	if t == nil {
		return SegmentsToVTT(nil)
	}
	if len(t.Segments) == 0 {
		if strings.TrimSpace(t.Text) == "" {
			return SegmentsToVTT(nil)
		}
		end := float32(0)
		d := float64(t.DurationSecs)
		if !math.IsNaN(d) && !math.IsInf(d, 0) && t.DurationSecs > 0 {
			end = t.DurationSecs
		}
		return SegmentsToVTT([]Segment{{Start: 0, End: end, Text: t.Text}})
	}
	return SegmentsToVTT(t.Segments)
}

// SegmentsToSRT formats timed segments as SubRip (.srt) subtitle text.
// Long segments are wrapped on word boundaries at around 42 characters per
// line and at most 2 lines per cue, with timestamps split proportionally.
func SegmentsToSRT(segments []Segment) (string, error) {
	return formatSubtitles(segments, _transcriptionToSRTSym)
}

// SegmentsToVTT formats timed segments as WebVTT (.vtt) subtitle text.
// Always starts with the WEBVTT header. Long segments are wrapped on word
// boundaries at around 42 characters per line and at most 2 lines per cue,
// with timestamps split proportionally.
func SegmentsToVTT(segments []Segment) (string, error) {
	return formatSubtitles(segments, _transcriptionToVTTSym)
}

func formatSubtitles(segments []Segment, sym uintptr) (string, error) {
	if err := ensureInit(); err != nil {
		return "", err
	}
	if sym == 0 {
		return "", errors.New("this Kjarni library has no subtitle formatting; update it")
	}

	var data uintptr
	var raw []ffiTranscriptionSegment
	var keeps []func()
	if len(segments) > 0 {
		raw = make([]ffiTranscriptionSegment, len(segments))
		keeps = make([]func(), len(segments))
		for i, seg := range segments {
			var textPtr uintptr
			textPtr, keeps[i] = cString(seg.Text)
			raw[i] = ffiTranscriptionSegment{
				Start: seg.Start,
				End:   seg.End,
				Text:  textPtr,
			}
		}
		data = uintptr(unsafe.Pointer(&raw[0]))
	}

	var out uintptr
	r1, _, _ := purego.SyscallN(sym, data, uintptr(len(raw)), uintptr(unsafe.Pointer(&out)))
	for _, keep := range keeps {
		keep()
	}
	runtime.KeepAlive(raw)

	if code := int32(r1); code != 0 {
		return "", lastError(code)
	}
	return takeString(out), nil
}

type ffiTranscriberConfig struct {
	Device    int32
	_         int32 // padding
	ModelName uintptr
	Language  uintptr
	Translate int32
	Quiet     int32
	ModelPath uintptr
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
	modelPath, keepModelPath := cStringOrNull(o.modelPath)
	defer keepModelPath()

	config := ffiTranscriberConfig{
		Device:    deviceCode(o.device),
		ModelName: modelPtr,
		Language:  languagePtr,
		Translate: boolToInt(o.translate),
		Quiet:     boolToInt(o.quiet),
		ModelPath: modelPath,
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
