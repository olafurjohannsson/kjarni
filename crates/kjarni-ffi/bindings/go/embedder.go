package kjarni

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Embedder encodes text into vector embeddings for similarity and search.
type Embedder struct {
	handle uintptr
	mu     sync.Mutex
	closed bool
}

// NewEmbedder creates an embedder for the given model.
// Available models: minilm-l6-v2 (384d), mpnet-base-v2 (768d), distilbert-base (768d).
// Models download automatically on first use and are cached locally.
func NewEmbedder(model string, opts ...Option) (*Embedder, error) {
	var initErr error
	ffiOnce.Do(func() { initErr = initFFI() })
	if initErr != nil {
		return nil, fmt.Errorf("initializing kjarni: %w", initErr)
	}

	o := applyOptions(opts)

	modelStr, keepModel := cString(model)
	defer keepModel()

	var config ffiEmbedderConfig
	config.Device = deviceCode(o.device)
	config.ModelName = modelStr
	config.Normalize = 1
	config.Quiet = boolToInt(o.quiet)

	var handle uintptr
	r1, _, _ := purego.SyscallN(
		_embedderNewSym,
		uintptr(unsafe.Pointer(&config)),
		uintptr(unsafe.Pointer(&handle)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	return &Embedder{handle: handle}, nil
}

// Encode returns the embedding vector for the given text.
func (e *Embedder) Encode(text string) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil, errors.New("embedder is closed")
	}

	textPtr, keepText := cString(text)
	defer keepText()

	var result ffiFloatArray
	r1, _, _ := purego.SyscallN(
		_embedderEncodeSym,
		e.handle,
		textPtr,
		uintptr(unsafe.Pointer(&result)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	vec := floatArrayToSlice(result)
	freeFloatArray(result)
	return vec, nil
}

// EncodeBatch encodes multiple texts and returns their embedding vectors.
func (e *Embedder) EncodeBatch(texts []string) ([][]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil, errors.New("embedder is closed")
	}

	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	// array of C string
	cStrs := make([]uintptr, len(texts))
	keeps := make([]func(), len(texts))
	for i, t := range texts {
		cStrs[i], keeps[i] = cString(t)
	}
	defer func() {
		for _, k := range keeps {
			k()
		}
	}()

	var result ffiFloat2DArray
	r1, _, _ := purego.SyscallN(
		_embedderEncodeBatchSym,
		e.handle,
		uintptr(unsafe.Pointer(&cStrs[0])),
		uintptr(len(texts)),
		uintptr(unsafe.Pointer(&result)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	vecs := float2DArrayToSlice(result)
	freeFloat2DArray(result)
	return vecs, nil
}

// Similarity returns the cosine similarity between two texts, computed by the engine.
func (e *Embedder) Similarity(a, b string) (float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return 0, errors.New("embedder is closed")
	}

	aPtr, keepA := cString(a)
	defer keepA()
	bPtr, keepB := cString(b)
	defer keepB()

	var result float32
	r1, _, _ := purego.SyscallN(
		_embedderSimilaritySym,
		e.handle,
		aPtr,
		bPtr,
		uintptr(unsafe.Pointer(&result)),
	)

	code := int32(r1)
	if code != 0 {
		return 0, lastError(code)
	}

	return result, nil
}

// Dim returns the dimensionality of the embedding model.
func (e *Embedder) Dim() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return int(_embedderDim(e.handle))
}

// Close releases the embedder resources. Safe to call multiple times.
func (e *Embedder) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil
	}
	e.closed = true
	_embedderFree(e.handle)
	e.handle = 0
	return nil
}

// CosineSimilarity computes cosine similarity between two vectors in Go.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))
	if denom == 0 {
		return 0
	}
	return dot / denom
}

func floatArrayToSlice(arr ffiFloatArray) []float32 {
	count := int(arr.Len)
	if count == 0 || arr.Data == 0 {
		return []float32{}
	}
	result := make([]float32, count)
	src := unsafe.Slice((*float32)(unsafe.Pointer(arr.Data)), count)
	copy(result, src)
	return result
}

func float2DArrayToSlice(arr ffiFloat2DArray) [][]float32 {
	rows := int(arr.Rows)
	cols := int(arr.Cols)
	if rows == 0 || cols == 0 || arr.Data == 0 {
		return [][]float32{}
	}
	flat := unsafe.Slice((*float32)(unsafe.Pointer(arr.Data)), rows*cols)
	result := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float32, cols)
		copy(result[i], flat[i*cols:(i+1)*cols])
	}
	return result
}

func freeFloatArray(arr ffiFloatArray) {
	if _floatArrayFreeSym != 0 {
		purego.SyscallN(_floatArrayFreeSym, arr.Data, arr.Len)
	}
}

func freeFloat2DArray(arr ffiFloat2DArray) {
	if _float2DArrayFreeSym != 0 {
		purego.SyscallN(_float2DArrayFreeSym, arr.Data, arr.Rows, arr.Cols)
	}
}