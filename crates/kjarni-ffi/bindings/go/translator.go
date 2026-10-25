package kjarni

import (
	"errors"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

type ffiTranslatorConfig struct {
	Device    int32
	_         int32 // padding
	CacheDir  uintptr
	ModelName uintptr
	From      uintptr
	To        uintptr
	Quiet     int32
	_2        int32 // padding
}

// Translator translates text between languages.
type Translator struct {
	handle uintptr
	mu     sync.Mutex
	closed bool
}

// NewTranslator loads a translation model, downloading it on first use. Pass ""
// for flan-t5-base. WithLanguages sets the languages Translate uses by default.
func NewTranslator(model string, opts ...Option) (*Translator, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	o := applyOptions(opts)

	modelPtr, keepModel := cStringOrNull(model)
	defer keepModel()
	cacheDir, keepCacheDir := cStringOrNull(o.cacheDir)
	defer keepCacheDir()
	fromPtr, keepFrom := cStringOrNull(o.from)
	defer keepFrom()
	toPtr, keepTo := cStringOrNull(o.to)
	defer keepTo()

	config := ffiTranslatorConfig{
		Device:    deviceCode(o.device),
		CacheDir:  cacheDir,
		ModelName: modelPtr,
		From:      fromPtr,
		To:        toPtr,
		Quiet:     boolToInt(o.quiet),
	}
	var handle uintptr
	r1, _, _ := purego.SyscallN(_translatorNewSym, uintptr(unsafe.Pointer(&config)), uintptr(unsafe.Pointer(&handle)))
	if code := int32(r1); code != 0 {
		return nil, lastError(code)
	}
	return &Translator{handle: handle}, nil
}

// Translate translates text between the default languages given to NewTranslator.
func (t *Translator) Translate(text string) (string, error) {
	return t.TranslateBetween(text, "", "")
}

// TranslateBetween translates text from one language to another, e.g. "en" to
// "de" or "english" to "german". An empty language falls back to the default;
// with neither, the error has code ErrInvalidConfig.
func (t *Translator) TranslateBetween(text, from, to string) (string, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return "", errors.New("translator is closed")
	}

	textPtr, keepText := cString(text)
	defer keepText()
	fromPtr, keepFrom := cStringOrNull(from)
	defer keepFrom()
	toPtr, keepTo := cStringOrNull(to)
	defer keepTo()
	var out uintptr
	r1, _, _ := purego.SyscallN(_translatorTranslateSym, t.handle, textPtr, fromPtr, toPtr, uintptr(unsafe.Pointer(&out)))
	if code := int32(r1); code != 0 {
		return "", lastError(code)
	}
	return takeString(out), nil
}

// Close releases the model. Safe to call more than once.
func (t *Translator) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil
	}
	t.closed = true
	_translatorFree(t.handle)
	t.handle = 0
	return nil
}
