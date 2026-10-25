package kjarni

import (
	"encoding/json"
	"errors"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// SearchMode determines the search strategy.
type SearchMode int

const (
	// Keyword uses BM25 term matching.
	Keyword SearchMode = 0
	// Semantic uses vector similarity.
	Semantic SearchMode = 1
	// Hybrid combines BM25 and vector similarity.
	Hybrid SearchMode = 2
)

// SearchResult holds a single search result with its relevance score.
type SearchResult struct {
	Score      float32
	Text       string
	DocumentID int
	// Metadata stored with the chunk, such as "source", the file it came from.
	Metadata map[string]any
}

// SearchOptions overrides a Searcher's defaults for one query. The zero value
// changes nothing.
type SearchOptions struct {
	// TopK is the most results to return; 0 means 10.
	TopK int
	// NoRerank skips the reranker for this query when the Searcher has one.
	NoRerank bool
	// Threshold drops results scoring below it.
	Threshold float32
	// SourcePattern keeps only chunks from files matching this `*` pattern, e.g. "*.md".
	SourcePattern string
	// FilterKey and FilterValue keep only chunks whose metadata has that pair.
	FilterKey   string
	FilterValue string
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
	Mode          int32
	_             int32 // padding
	TopK          uintptr
	UseReranker   int32
	Threshold     float32
	SourcePattern uintptr
	FilterKey     uintptr
	FilterValue   uintptr
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

// Searcher queries indexes created by an Indexer.
type Searcher struct {
	handle uintptr
	mu     sync.Mutex
	closed bool
}

// NewSearcher creates a searcher using the given embedding model.
// Pass a non-empty rerankerModel to enable cross-encoder reranking of results.
// Pass an empty string to disable reranking.
func NewSearcher(model string, rerankerModel string, opts ...Option) (*Searcher, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}

	o := applyOptions(opts)

	modelStr, keepModel := cString(model)
	defer keepModel()
	cacheDir, keepCacheDir := cStringOrNull(o.cacheDir)
	defer keepCacheDir()

	var rerankPtr uintptr
	var keepRerank func()
	if rerankerModel != "" {
		rerankPtr, keepRerank = cString(rerankerModel)
		defer keepRerank()
	}

	var config ffiSearcherConfig
	config.Device = deviceCode(o.device)
	config.ModelName = modelStr
	config.CacheDir = cacheDir
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

// Search queries the index at indexPath and returns up to 10 results using the
// given mode, reranked when the Searcher has a reranker.
func (s *Searcher) Search(indexPath string, query string, mode SearchMode) ([]SearchResult, error) {
	return s.SearchWithOptions(indexPath, query, mode, SearchOptions{})
}

// SearchWithOptions is Search with per-query overrides.
func (s *Searcher) SearchWithOptions(indexPath string, query string, mode SearchMode, opts SearchOptions) ([]SearchResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, errors.New("searcher is closed")
	}

	pathPtr, keepPath := cString(indexPath)
	defer keepPath()
	queryPtr, keepQuery := cString(query)
	defer keepQuery()
	sourcePtr, keepSource := cStringOrNull(opts.SourcePattern)
	defer keepSource()
	keyPtr, keepKey := cStringOrNull(opts.FilterKey)
	defer keepKey()
	valuePtr, keepValue := cStringOrNull(opts.FilterValue)
	defer keepValue()

	var searchOpts ffiSearchOptions
	searchOpts.Mode = int32(mode)
	searchOpts.TopK = 10
	if opts.TopK > 0 {
		searchOpts.TopK = uintptr(opts.TopK)
	}
	// -1 reranks when a reranker was configured. This used to send 1, asking for
	// a reranker on every query whether or not one existed.
	searchOpts.UseReranker = -1
	if opts.NoRerank {
		searchOpts.UseReranker = 0
	}
	searchOpts.Threshold = opts.Threshold
	searchOpts.SourcePattern = sourcePtr
	searchOpts.FilterKey = keyPtr
	searchOpts.FilterValue = valuePtr

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

// Close releases the searcher resources. Safe to call multiple times.
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
			Score:      item.Score,
			Text:       goString(item.Text),
			DocumentID: int(item.DocumentId),
		}
		if meta := goString(item.MetadataJson); meta != "" {
			_ = json.Unmarshal([]byte(meta), &out[i].Metadata)
		}
	}

	return out
}

func freeSearchResults(results ffiSearchResults) {
    if _searchResultsFreeSym != 0 {
        callWithStruct(_searchResultsFreeSym, unsafe.Pointer(&results), unsafe.Sizeof(results))
    }
}