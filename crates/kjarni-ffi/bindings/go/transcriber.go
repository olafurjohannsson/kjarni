package kjarni

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
	"runtime"
	"strings"
	"sync"
	"unicode/utf8"
	"unsafe"

	"github.com/ebitengine/purego"
)

const (
	subtitleMaxLineChars = 42
	subtitleMaxLines     = 2
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
func (t *Transcript) ToSRT() string {
	if t == nil {
		return ""
	}
	if len(t.Segments) == 0 {
		if strings.TrimSpace(t.Text) == "" {
			return ""
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
func (t *Transcript) ToVTT() string {
	if t == nil {
		return "WEBVTT\n\n"
	}
	if len(t.Segments) == 0 {
		if strings.TrimSpace(t.Text) == "" {
			return "WEBVTT\n\n"
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
func SegmentsToSRT(segments []Segment) string {
	return formatSubtitles(segments, false)
}

// SegmentsToVTT formats timed segments as WebVTT (.vtt) subtitle text.
// Always starts with the WEBVTT header. Long segments are wrapped on word
// boundaries at around 42 characters per line and at most 2 lines per cue,
// with timestamps split proportionally.
func SegmentsToVTT(segments []Segment) string {
	return formatSubtitles(segments, true)
}

func formatSubtitles(segments []Segment, isVTT bool) string {
	var out strings.Builder
	decimal := ','
	if isVTT {
		out.WriteString("WEBVTT\n\n")
		decimal = '.'
	}

	cueIndex := 1
	for _, seg := range segments {
		cues := buildSegmentCues(seg.Text, isVTT)
		if len(cues) == 0 {
			continue
		}

		startMs := subtitleToMillis(seg.Start)
		endMs := subtitleToMillis(seg.End)
		if endMs < startMs {
			endMs = startMs
		}

		for _, cue := range distributeSubtitleTimestamps(cues, startMs, endMs) {
			if !isVTT {
				fmt.Fprintf(&out, "%d\n", cueIndex)
				cueIndex++
			}
			fmt.Fprintf(
				&out,
				"%s --> %s\n%s\n\n",
				formatSubtitleTimestamp(cue.start, decimal),
				formatSubtitleTimestamp(cue.end, decimal),
				cue.text,
			)
		}
	}

	return out.String()
}

func buildSegmentCues(text string, isVTT bool) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}
	if isVTT {
		joined := strings.Join(words, " ")
		if strings.Contains(joined, "-->") {
			words = strings.Fields(strings.ReplaceAll(joined, "-->", "-- >"))
		}
	}

	lines := wrapSubtitleWords(words, subtitleMaxLineChars)
	cues := make([]string, 0, (len(lines)+subtitleMaxLines-1)/subtitleMaxLines)
	for i := 0; i < len(lines); i += subtitleMaxLines {
		end := i + subtitleMaxLines
		if end > len(lines) {
			end = len(lines)
		}
		cues = append(cues, strings.Join(lines[i:end], "\n"))
	}
	return cues
}

func wrapSubtitleWords(words []string, maxLineChars int) []string {
	var lines []string
	var current strings.Builder
	currentChars := 0

	for _, word := range words {
		wordChars := utf8.RuneCountInString(word)
		if currentChars == 0 {
			current.WriteString(word)
			currentChars = wordChars
		} else if currentChars+1+wordChars <= maxLineChars {
			current.WriteByte(' ')
			current.WriteString(word)
			currentChars += 1 + wordChars
		} else {
			lines = append(lines, current.String())
			current.Reset()
			current.WriteString(word)
			currentChars = wordChars
		}
	}

	if currentChars > 0 {
		lines = append(lines, current.String())
	}

	return lines
}

type timedCue struct {
	start uint64
	end   uint64
	text  string
}

func distributeSubtitleTimestamps(cues []string, startMs, endMs uint64) []timedCue {
	count := len(cues)
	if count == 1 {
		return []timedCue{{start: startMs, end: endMs, text: cues[0]}}
	}

	weights := make([]uint64, count)
	var totalWeight uint64
	for i, c := range cues {
		w := uint64(utf8.RuneCountInString(c))
		if w == 0 {
			w = 1
		}
		weights[i] = w
		totalWeight += w
	}

	totalDur := endMs - startMs
	out := make([]timedCue, 0, count)
	prevEnd := startMs
	var cumWeight uint64

	for i, cue := range cues {
		cueStart := prevEnd
		var cueEnd uint64
		if i+1 == count {
			cueEnd = endMs
		} else {
			cumWeight += weights[i]
			offset := mulDivRound(totalDur, cumWeight, totalWeight)
			cueEnd = startMs + offset
			if cueEnd < cueStart {
				cueEnd = cueStart
			}
			if cueEnd > endMs {
				cueEnd = endMs
			}
		}
		out = append(out, timedCue{start: cueStart, end: cueEnd, text: cue})
		prevEnd = cueEnd
	}

	return out
}

func mulDivRound(a, b, denom uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	lo, carry := bits.Add64(lo, denom/2, 0)
	hi += carry
	q, _ := bits.Div64(hi, lo, denom)
	return q
}

func subtitleToMillis(seconds float32) uint64 {
	d := float64(seconds)
	if math.IsNaN(d) || math.IsInf(d, 0) || d <= 0 {
		return 0
	}
	ms := d * 1000.0
	if ms >= float64(math.MaxUint64) {
		return math.MaxUint64
	}
	return uint64(math.Round(ms))
}

func formatSubtitleTimestamp(millis uint64, decimal rune) string {
	ms := millis % 1000
	totalSecs := millis / 1000
	s := totalSecs % 60
	m := (totalSecs / 60) % 60
	h := totalSecs / 3600
	return fmt.Sprintf("%02d:%02d:%02d%c%03d", h, m, s, decimal, ms)
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
