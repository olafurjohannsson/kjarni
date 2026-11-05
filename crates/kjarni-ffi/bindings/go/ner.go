package kjarni

import (
	"errors"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Aggregation decides how tokens become entities, as transformers'
// aggregation_strategy.
type Aggregation int32

const (
	// AggregationFirst gives a word the label of its first token, so it is never split.
	AggregationFirst Aggregation = 0
	// AggregationSimple keeps each token's label; neighbours with the same tag join.
	AggregationSimple Aggregation = 1
	// AggregationMax gives a word the label of its most confident token.
	AggregationMax Aggregation = 2
	// AggregationAverage gives a word the label with the highest mean probability.
	AggregationAverage Aggregation = 3
)

// Entity is one named entity. Start and End are byte offsets into the input,
// so text[e.Start:e.End] == e.Text.
type Entity struct {
	Label string
	Text  string
	Start int
	End   int
	Score float32
}

type ffiNerConfig struct {
	Device      int32
	_           int32 // padding
	CacheDir    uintptr
	ModelName   uintptr
	ModelPath   uintptr
	Aggregation int32
	Quiet       int32
}

type ffiEntity struct {
	Label uintptr
	Text  uintptr
	Start uintptr
	End   uintptr
	Score float32
	_     [4]byte // padding
}

type ffiEntities struct {
	Entities uintptr
	Len      uintptr
}

// EntityRecognizer finds names of people, places, organizations and other
// entities with a token classification model. The entities are those
// transformers' pipeline("ner") gives for the same aggregation.
type EntityRecognizer struct {
	mu     sync.Mutex
	handle uintptr
	closed bool
}

// NewEntityRecognizer loads an entity recognizer. An empty model means the
// default, distilbert-ner, for English. WithModelPath loads any
// *ForTokenClassification model from a folder instead.
func NewEntityRecognizer(model string, aggregation Aggregation, opts ...Option) (*EntityRecognizer, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if _nerNewSym == 0 {
		return nil, errors.New("this Kjarni library has no entity recognition; update it")
	}

	o := applyOptions(opts)
	modelPtr, keepModel := cStringOrNull(model)
	defer keepModel()
	cacheDir, keepCacheDir := cStringOrNull(o.cacheDir)
	defer keepCacheDir()
	modelPath, keepModelPath := cStringOrNull(o.modelPath)
	defer keepModelPath()

	config := ffiNerConfig{
		Device:      deviceCode(o.device),
		CacheDir:    cacheDir,
		ModelName:   modelPtr,
		ModelPath:   modelPath,
		Aggregation: int32(aggregation),
		Quiet:       boolToInt(o.quiet),
	}

	var handle uintptr
	r1, _, _ := purego.SyscallN(_nerNewSym, uintptr(unsafe.Pointer(&config)), uintptr(unsafe.Pointer(&handle)))
	if code := int32(r1); code != 0 {
		return nil, lastError(code)
	}
	return &EntityRecognizer{handle: handle}, nil
}

// Recognize returns the entities in text, in order.
func (r *EntityRecognizer) Recognize(text string) ([]Entity, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("entity recognizer is closed")
	}

	textPtr, keepText := cString(text)
	defer keepText()

	var list ffiEntities
	r1, _, _ := purego.SyscallN(_nerRecognizeSym, r.handle, textPtr, uintptr(unsafe.Pointer(&list)))
	if code := int32(r1); code != 0 {
		return nil, lastError(code)
	}
	defer callWithStruct(_entitiesFreeSym, unsafe.Pointer(&list), unsafe.Sizeof(list))

	if list.Len == 0 || list.Entities == 0 {
		return []Entity{}, nil
	}
	raw := unsafe.Slice((*ffiEntity)(unsafe.Pointer(list.Entities)), int(list.Len))
	offsets := runeToByteOffsets(text)
	out := make([]Entity, len(raw))
	for i, e := range raw {
		out[i] = Entity{
			Label: goString(e.Label),
			Text:  goString(e.Text),
			Start: offsets[min(int(e.Start), len(offsets)-1)],
			End:   offsets[min(int(e.End), len(offsets)-1)],
			Score: e.Score,
		}
	}
	return out, nil
}

// runeToByteOffsets maps a code point offset, which the engine reports, to a
// byte offset into s: element n is where the n-th rune starts, and the last
// element is len(s).
func runeToByteOffsets(s string) []int {
	offsets := make([]int, 0, len(s)+1)
	for i := range s {
		offsets = append(offsets, i)
	}
	return append(offsets, len(s))
}

// Close frees the native model. Calling it twice is safe.
func (r *EntityRecognizer) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.closed && r.handle != 0 {
		purego.SyscallN(_nerFreeSym, r.handle)
		r.handle = 0
	}
	r.closed = true
	return nil
}
