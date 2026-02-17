package kjarni

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ClassifyResult holds the output of a classification. Label and Score
// contain the top prediction. AllScores contains scores for every label.
type ClassifyResult struct {
	Label     string
	Score     float32
	AllScores []LabelScore
}

// LabelScore is a single label with its confidence score.
type LabelScore struct {
	Label string
	Score float32
}

// String returns the result as "label (score%)".
func (r *ClassifyResult) String() string {
	return fmt.Sprintf("%s (%.1f%%)", r.Label, r.Score*100)
}

// ToJSON returns the result as a JSON string.
func (r *ClassifyResult) ToJSON() string {
	var sb strings.Builder
	sb.WriteString("{\n")
	sb.WriteString(fmt.Sprintf("  \"label\": \"%s\",\n", r.Label))
	sb.WriteString(fmt.Sprintf("  \"score\": %.4f,\n", r.Score))
	sb.WriteString("  \"predictions\": [\n")
	for i, s := range r.AllScores {
		comma := ","
		if i == len(r.AllScores)-1 {
			comma = ""
		}
		sb.WriteString(fmt.Sprintf("    {\"label\": \"%s\", \"score\": %.4f}%s\n", s.Label, s.Score, comma))
	}
	sb.WriteString("  ]\n}")
	return sb.String()
}

// Classifier runs text classification using a pre-trained model.
type Classifier struct {
	handle uintptr
	mu     sync.Mutex
	closed bool
}

// NewClassifier creates a classifier for the given model.
// Available models: distilbert-sentiment, roberta-sentiment,
// bert-sentiment-multilingual, distilroberta-emotion, roberta-emotions, toxic-bert.
// Models download automatically on first use and are cached locally.
func NewClassifier(model string, opts ...Option) (*Classifier, error) {
	var initErr error
	ffiOnce.Do(func() { initErr = initFFI() })
	if initErr != nil {
		return nil, fmt.Errorf("initializing kjarni: %w", initErr)
	}

	o := applyOptions(opts)

	modelStr, keepModel := cString(model)
	defer keepModel()

	var config ffiClassifierConfig
	config.Device = deviceCode(o.device)
	config.ModelName = modelStr
	config.Quiet = boolToInt(o.quiet)

	var handle uintptr
	r1, _, _ := purego.SyscallN(
		_classifierNewSym,
		uintptr(unsafe.Pointer(&config)),
		uintptr(unsafe.Pointer(&handle)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	return &Classifier{handle: handle}, nil
}

// Classify runs the model on the given text and returns scored labels.
func (c *Classifier) Classify(text string) (*ClassifyResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, errors.New("classifier is closed")
	}

	textPtr, keepText := cString(text)
	defer keepText()

	var results ffiClassResults
	r1, _, _ := purego.SyscallN(
		_classifierClassifySym,
		c.handle,
		textPtr,
		uintptr(unsafe.Pointer(&results)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	defer freeClassResults(results)
	return parseClassResults(results), nil
}

// NumLabels returns the number of labels the model supports.
func (c *Classifier) NumLabels() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return int(_classifierNumLabels(c.handle))
}

// Close releases the classifier resources. Safe to call multiple times.
func (c *Classifier) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}
	c.closed = true
	_classifierFree(c.handle)
	c.handle = 0
	return nil
}

func parseClassResults(results ffiClassResults) *ClassifyResult {
	count := int(results.Len)
	if count == 0 {
		return &ClassifyResult{}
	}

	structSize := unsafe.Sizeof(ffiClassResult{})
	allScores := make([]LabelScore, count)

	for i := 0; i < count; i++ {
		ptr := results.Results + uintptr(i)*structSize
		item := (*ffiClassResult)(unsafe.Pointer(ptr))
		label := goString(item.Label)
		allScores[i] = LabelScore{
			Label: label,
			Score: item.Score,
		}
	}

	// top score
	bestIdx := 0
	bestScore := float32(-math.MaxFloat32)
	for i, s := range allScores {
		if s.Score > bestScore {
			bestScore = s.Score
			bestIdx = i
		}
	}

	return &ClassifyResult{
		Label:     allScores[bestIdx].Label,
		Score:     allScores[bestIdx].Score,
		AllScores: allScores,
	}
}

func freeClassResults(results ffiClassResults) {
	if _classResultsFreeSymGlobal != 0 {
		purego.SyscallN(_classResultsFreeSymGlobal, results.Results, results.Len)
	}
}