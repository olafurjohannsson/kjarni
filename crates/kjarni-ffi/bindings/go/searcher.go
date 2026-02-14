package kjarni

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// search strategy hybrid, semantic, keyword(bm25)
type SearchMode int

const (
	Keyword  SearchMode = 0
	Semantic SearchMode = 1
	Hybrid   SearchMode = 2
)

// single search result
type SearchResult struct {
	Score float32
	Text  string
}

type ffiSearcherConfig struct {
	Device      int32
	_           int32 // padding
	CacheDir    uintptr
	ModelName   uintptr
	RerankModel uintptr
	DefaultMode int32
	_2          int32 // padding
	DefaultTopK uintptr
	Quiet       int32
	_3          int32 // padding
}

type ffiSearchOptions struct {
	Mode         int32
	_            int32 // padding
	TopK         uintptr
	UseReranker  int32
	Threshold    float32
	SourcePattern uintptr
	FilterKey    uintptr
	FilterValue  uintptr
}

type ffiSearchResult struct {
	Score        float32
	_            int32 // padding
	DocumentId   uintptr
	Text         uintptr
	MetadataJson uintptr
}

type ffiSearchResults struct {
	Results uintptr
	Len     uintptr
}

// Searcher queries indexes created by Indexer
type Searcher struct {
	handle uintptr
	mu     sync.Mutex
	closed bool
}

// new searcher
func NewSearcher(model string, rerankerModel string, opts ...Option) (*Searcher, error) {
	var initErr error
	ffiOnce.Do(func() { initErr = initFFI() })
	if initErr != nil {
		return nil, fmt.Errorf("initializing kjarni: %w", initErr)
	}

	o := applyOptions(opts)

	modelStr, keepModel := cString(model)
	defer keepModel()

	var rerankPtr uintptr
	var keepRerank func()
	if rerankerModel != "" {
		rerankPtr, keepRerank = cString(rerankerModel)
		defer keepRerank()
	}

	var config ffiSearcherConfig
	config.Device = deviceCode(o.device)
	config.ModelName = modelStr
	config.RerankModel = rerankPtr
	config.DefaultMode = int32(Hybrid)
	config.Quiet = boolToInt(o.quiet)

	var handle uintptr
	r1, _, _ := purego.SyscallN(
		_searcherNewSym,
		uintptr(unsafe.Pointer(&config)),
		uintptr(unsafe.Pointer(&handle)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	return &Searcher{handle: handle}, nil
}

//queries the index with the given mode
func (s *Searcher) Search(indexPath string, query string, mode SearchMode) ([]SearchResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, errors.New("searcher is closed")
	}

	pathPtr, keepPath := cString(indexPath)
	defer keepPath()
	queryPtr, keepQuery := cString(query)
	defer keepQuery()

	var searchOpts ffiSearchOptions
	searchOpts.Mode = int32(mode)
	searchOpts.TopK = 10
	searchOpts.UseReranker = 1

	var results ffiSearchResults
	r1, _, _ := purego.SyscallN(
		_searcherSearchWithOptionsSym,
		s.handle,
		pathPtr,
		queryPtr,
		uintptr(unsafe.Pointer(&searchOpts)),
		uintptr(unsafe.Pointer(&results)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	defer freeSearchResults(results)
	return parseSearchResults(results), nil
}

// releases  resources
func (s *Searcher) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}
	s.closed = true
	_searcherFree(s.handle)
	s.handle = 0
	return nil
}

func parseSearchResults(results ffiSearchResults) []SearchResult {
	count := int(results.Len)
	if count == 0 {
		return []SearchResult{}
	}

	structSize := unsafe.Sizeof(ffiSearchResult{})
	out := make([]SearchResult, count)

	for i := 0; i < count; i++ {
		ptr := results.Results + uintptr(i)*structSize
		item := (*ffiSearchResult)(unsafe.Pointer(ptr))
		out[i] = SearchResult{
			Score: item.Score,
			Text:  goString(item.Text),
		}
	}

	return out
}

func freeSearchResults(results ffiSearchResults) {
	if _searchResultsFreeSym != 0 {
		purego.SyscallN(_searchResultsFreeSym, results.Results, results.Len)
	}
}