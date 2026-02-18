package kjarni

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// RerankResult holds a single reranked document with its relevance score.
type RerankResult struct {
	Index    int
	Score    float32
	Document string
}

// Reranker scores query-document relevance using a cross-encoder model.
type Reranker struct {
	handle uintptr
	mu     sync.Mutex
	closed bool
}

// NewReranker creates a reranker using the default cross-encoder model.
// The model downloads automatically on first use and is cached locally.
func NewReranker(opts ...Option) (*Reranker, error) {
	var initErr error
	ffiOnce.Do(func() { initErr = initFFI() })
	if initErr != nil {
		return nil, fmt.Errorf("initializing kjarni: %w", initErr)
	}

	o := applyOptions(opts)

	var config ffiRerankerConfig
	config.Device = deviceCode(o.device)
	config.Quiet = boolToInt(o.quiet)

	var handle uintptr
	r1, _, _ := purego.SyscallN(
		_rerankerNewSym,
		uintptr(unsafe.Pointer(&config)),
		uintptr(unsafe.Pointer(&handle)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	return &Reranker{handle: handle}, nil
}

// Score returns the relevance score for a single query-document pair.
func (r *Reranker) Score(query, document string) (float32, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return 0, errors.New("reranker is closed")
	}

	qPtr, keepQ := cString(query)
	defer keepQ()
	dPtr, keepD := cString(document)
	defer keepD()

	var result float32
	r1, _, _ := purego.SyscallN(
		_rerankerScoreSym,
		r.handle,
		qPtr,
		dPtr,
		uintptr(unsafe.Pointer(&result)),
	)

	code := int32(r1)
	if code != 0 {
		return 0, lastError(code)
	}

	return result, nil
}

// Rerank scores all documents and returns them sorted by relevance to the query.
func (r *Reranker) Rerank(query string, documents []string) ([]RerankResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return nil, errors.New("reranker is closed")
	}

	if len(documents) == 0 {
		return []RerankResult{}, nil
	}

	qPtr, keepQ := cString(query)
	defer keepQ()

	cStrs := make([]uintptr, len(documents))
	keeps := make([]func(), len(documents))
	for i, d := range documents {
		cStrs[i], keeps[i] = cString(d)
	}
	defer func() {
		for _, k := range keeps {
			k()
		}
	}()

	var results ffiRerankResults
	r1, _, _ := purego.SyscallN(
		_rerankerRerankSym,
		r.handle,
		qPtr,
		uintptr(unsafe.Pointer(&cStrs[0])),
		uintptr(len(documents)),
		uintptr(unsafe.Pointer(&results)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	defer freeRerankResults(results)
	return parseRerankResults(results, documents), nil
}

// RerankTopK scores all documents and returns the top k sorted by relevance.
func (r *Reranker) RerankTopK(query string, documents []string, k int) ([]RerankResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return nil, errors.New("reranker is closed")
	}

	if len(documents) == 0 {
		return []RerankResult{}, nil
	}

	qPtr, keepQ := cString(query)
	defer keepQ()

	cStrs := make([]uintptr, len(documents))
	keeps := make([]func(), len(documents))
	for i, d := range documents {
		cStrs[i], keeps[i] = cString(d)
	}
	defer func() {
		for _, k := range keeps {
			k()
		}
	}()

	var results ffiRerankResults
	r1, _, _ := purego.SyscallN(
		_rerankerRerankTopKSym,
		r.handle,
		qPtr,
		uintptr(unsafe.Pointer(&cStrs[0])),
		uintptr(len(documents)),
		uintptr(k),
		uintptr(unsafe.Pointer(&results)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	defer freeRerankResults(results)
	return parseRerankResults(results, documents), nil
}

// Close releases the reranker resources. Safe to call multiple times.
func (r *Reranker) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return nil
	}
	r.closed = true
	_rerankerFree(r.handle)
	r.handle = 0
	return nil
}

func parseRerankResults(results ffiRerankResults, documents []string) []RerankResult {
	count := int(results.Len)
	if count == 0 {
		return []RerankResult{}
	}

	structSize := unsafe.Sizeof(ffiRerankResult{})
	out := make([]RerankResult, count)

	for i := 0; i < count; i++ {
		ptr := results.Results + uintptr(i)*structSize
		item := (*ffiRerankResult)(unsafe.Pointer(ptr))
		idx := int(item.Index)
		doc := ""
		if idx >= 0 && idx < len(documents) {
			doc = documents[idx]
		}
		out[i] = RerankResult{
			Index:    idx,
			Score:    item.Score,
			Document: doc,
		}
	}

	return out
}

func freeRerankResults(results ffiRerankResults) {
    if _rerankResultsFreeSym != 0 {
        purego.SyscallN(_rerankResultsFreeSym, uintptr(unsafe.Pointer(&results)))
    }
}