package kjarni

import (
	"errors"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

type ffiSummarizerConfig struct {
	Device    int32
	_         int32 // padding
	CacheDir  uintptr
	ModelName uintptr
	MinLength uintptr
	MaxLength uintptr
	Quiet     int32
	_2        int32 // padding
}

// Summarizer summarizes article-length text.
type Summarizer struct {
	handle uintptr
	mu     sync.Mutex
	closed bool
}

// NewSummarizer loads a summarization model, downloading it on first use. Pass ""
// for distilbart-cnn. See WithSummaryLength.
func NewSummarizer(model string, opts ...Option) (*Summarizer, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	o := applyOptions(opts)
	if o.minLength < 0 || o.maxLength < 0 {
		return nil, errors.New("summary lengths must not be negative")
	}

	modelPtr, keepModel := cStringOrNull(model)
	defer keepModel()
	cacheDir, keepCacheDir := cStringOrNull(o.cacheDir)
	defer keepCacheDir()

	config := ffiSummarizerConfig{
		Device:    deviceCode(o.device),
		CacheDir:  cacheDir,
		ModelName: modelPtr,
		MinLength: uintptr(o.minLength),
		MaxLength: uintptr(o.maxLength),
		Quiet:     boolToInt(o.quiet),
	}
	var handle uintptr
	r1, _, _ := purego.SyscallN(_summarizerNewSym, uintptr(unsafe.Pointer(&config)), uintptr(unsafe.Pointer(&handle)))
	if code := int32(r1); code != 0 {
		return nil, lastError(code)
	}
	return &Summarizer{handle: handle}, nil
}

// Summarize summarizes text.
func (s *Summarizer) Summarize(text string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return "", errors.New("summarizer is closed")
	}

	textPtr, keepText := cString(text)
	defer keepText()
	var out uintptr
	r1, _, _ := purego.SyscallN(_summarizerSummarizeSym, s.handle, textPtr, uintptr(unsafe.Pointer(&out)))
	if code := int32(r1); code != 0 {
		return "", lastError(code)
	}
	return takeString(out), nil
}

// Close releases the model. Safe to call more than once.
func (s *Summarizer) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	_summarizerFree(s.handle)
	s.handle = 0
	return nil
}
