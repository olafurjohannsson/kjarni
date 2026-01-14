package kjarni

/*
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// RerankerConfig configures the Reranker.
type RerankerConfig struct {
	Model    string
	Device   string // "cpu" or "gpu"
	CacheDir string
	Quiet    bool
}

// RerankResult represents a single rerank result.
type RerankResult struct {
	Index    int
	Score    float32
	Document string
}

// Reranker reranks documents by relevance to a query.
type Reranker struct {
	handle *C.KjarniReranker
}

// NewReranker creates a new Reranker.
func NewReranker(cfg *RerankerConfig) (*Reranker, error) {
	config := C.kjarni_reranker_config_default()

	var modelName *C.char
	var cacheDir *C.char

	if cfg != nil {
		if cfg.Device == "gpu" {
			config.device = C.KJARNI_DEVICE_GPU
		}
		config.quiet = boolToInt(cfg.Quiet)

		if cfg.Model != "" {
			modelName = C.CString(cfg.Model)
			defer C.free(unsafe.Pointer(modelName))
			config.model_name = modelName
		}
		if cfg.CacheDir != "" {
			cacheDir = C.CString(cfg.CacheDir)
			defer C.free(unsafe.Pointer(cacheDir))
			config.cache_dir = cacheDir
		}
	}

	var handle *C.KjarniReranker
	err := C.kjarni_reranker_new(&config, &handle)
	if e := checkError(err); e != nil {
		return nil, e
	}

	reranker := &Reranker{handle: handle}
	runtime.SetFinalizer(reranker, (*Reranker).Close)
	return reranker, nil
}

// Close frees the Reranker resources.
func (r *Reranker) Close() {
	if r.handle != nil {
		C.kjarni_reranker_free(r.handle)
		r.handle = nil
	}
}

// Score scores a single query-document pair.
func (r *Reranker) Score(query, document string) (float32, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	cDoc := C.CString(document)
	defer C.free(unsafe.Pointer(cDoc))

	var result C.float
	err := C.kjarni_reranker_score(r.handle, cQuery, cDoc, &result)
	if e := checkError(err); e != nil {
		return 0, e
	}

	return float32(result), nil
}

// Rerank reranks documents by relevance to query.
func (r *Reranker) Rerank(query string, documents []string) ([]RerankResult, error) {
	if len(documents) == 0 {
		return []RerankResult{}, nil
	}

	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	cDocs := make([]*C.char, len(documents))
	for i, d := range documents {
		cDocs[i] = C.CString(d)
		defer C.free(unsafe.Pointer(cDocs[i]))
	}

	var results C.KjarniRerankResults
	err := C.kjarni_reranker_rerank(
		r.handle,
		cQuery,
		(**C.char)(unsafe.Pointer(&cDocs[0])),
		C.size_t(len(documents)),
		&results,
	)
	if e := checkError(err); e != nil {
		return nil, e
	}
	defer C.kjarni_rerank_results_free(results)

	return rerankResultsToSlice(results, documents), nil
}

// RerankTopK reranks and returns top-k results.
func (r *Reranker) RerankTopK(query string, documents []string, k int) ([]RerankResult, error) {
	if len(documents) == 0 {
		return []RerankResult{}, nil
	}

	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	cDocs := make([]*C.char, len(documents))
	for i, d := range documents {
		cDocs[i] = C.CString(d)
		defer C.free(unsafe.Pointer(cDocs[i]))
	}

	var results C.KjarniRerankResults
	err := C.kjarni_reranker_rerank_top_k(
		r.handle,
		cQuery,
		(**C.char)(unsafe.Pointer(&cDocs[0])),
		C.size_t(len(documents)),
		C.size_t(k),
		&results,
	)
	if e := checkError(err); e != nil {
		return nil, e
	}
	defer C.kjarni_rerank_results_free(results)

	return rerankResultsToSlice(results, documents), nil
}

func rerankResultsToSlice(results C.KjarniRerankResults, documents []string) []RerankResult {
	if results.results == nil || results.len == 0 {
		return []RerankResult{}
	}
	length := int(results.len)
	cSlice := (*[1 << 30]C.KjarniRerankResult)(unsafe.Pointer(results.results))[:length:length]

	out := make([]RerankResult, length)
	for i, r := range cSlice {
		idx := int(r.index)
		out[i] = RerankResult{
			Index:    idx,
			Score:    float32(r.score),
			Document: documents[idx],
		}
	}
	return out
}