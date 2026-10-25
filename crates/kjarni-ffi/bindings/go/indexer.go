package kjarni

import (
	"errors"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// IndexStats holds statistics from an indexing operation.
type IndexStats struct {
	DocumentsIndexed int
	ChunksCreated    int
	Dimension        int
	SizeBytes        uint64
	FilesProcessed   int
	FilesSkipped     int
	ElapsedMs        uint64
}

type ffiIndexerConfig struct {
	Device          int32
	_               int32 // padding
	CacheDir        uintptr
	ModelName       uintptr
	ChunkSize       uintptr
	ChunkOverlap    uintptr
	BatchSize       uintptr
	Extensions      uintptr
	ExcludePatterns uintptr
	Recursive       int32
	IncludeHidden   int32
	MaxFileSize     uintptr
	Quiet           int32
	_2              int32 // padding
}

type ffiIndexStats struct {
	DocumentsIndexed uintptr
	ChunksCreated    uintptr
	Dimension        uintptr
	SizeBytes        uint64
	FilesProcessed   uintptr
	FilesSkipped     uintptr
	ElapsedMs        uint64
}

// Indexer creates search indexes from files in a directory.
type Indexer struct {
	handle uintptr
	mu     sync.Mutex
	closed bool
}

// NewIndexer creates an indexer using the given embedding model.
// The model is used to generate vectors for each text chunk during indexing.
func NewIndexer(model string, opts ...Option) (*Indexer, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}

	o := applyOptions(opts)

	modelStr, keepModel := cString(model)
	defer keepModel()
	cacheDir, keepCacheDir := cStringOrNull(o.cacheDir)
	defer keepCacheDir()
	extensions, keepExtensions := cStringOrNull(strings.Join(o.extensions, ","))
	defer keepExtensions()
	exclude, keepExclude := cStringOrNull(strings.Join(o.exclude, ","))
	defer keepExclude()

	var config ffiIndexerConfig
	config.Device = deviceCode(o.device)
	config.CacheDir = cacheDir
	config.ModelName = modelStr
	config.ChunkSize = 512
	config.ChunkOverlap = 100
	if o.chunkSize > 0 {
		config.ChunkSize = uintptr(o.chunkSize)
		config.ChunkOverlap = uintptr(o.chunkOverlap)
	}
	config.BatchSize = 32
	config.Extensions = extensions
	config.ExcludePatterns = exclude
	config.Recursive = boolToInt(!o.notRecursive)
	config.IncludeHidden = boolToInt(o.includeHidden)
	config.MaxFileSize = 10 * 1024 * 1024
	config.Quiet = boolToInt(o.quiet)

	var handle uintptr
	r1, _, _ := purego.SyscallN(
		_indexerNewSym,
		uintptr(unsafe.Pointer(&config)),
		uintptr(unsafe.Pointer(&handle)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	return &Indexer{handle: handle}, nil
}

// Create builds a new search index at indexPath from the given files and
// directories. Files are chunked, embedded, and stored for later retrieval with a
// Searcher. It fails if an index already exists there; see Recreate.
func (idx *Indexer) Create(indexPath string, inputs []string) (*IndexStats, error) {
	return idx.create(indexPath, inputs, false)
}

// Recreate is Create, replacing any index already at indexPath.
func (idx *Indexer) Recreate(indexPath string, inputs []string) (*IndexStats, error) {
	return idx.create(indexPath, inputs, true)
}

// Add indexes more files and directories into the existing index at indexPath,
// returning how many chunks were added.
func (idx *Indexer) Add(indexPath string, inputs []string) (int, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if idx.closed {
		return 0, errors.New("indexer is closed")
	}
	if len(inputs) == 0 {
		return 0, errors.New("no input paths given")
	}

	pathPtr, keepPath := cString(indexPath)
	defer keepPath()
	inputsPtr, keepInputs := cStringArray(inputs)
	defer keepInputs()

	var added uintptr
	r1, _, _ := purego.SyscallN(_indexerAddSym, idx.handle, pathPtr, inputsPtr, uintptr(len(inputs)),
		uintptr(unsafe.Pointer(&added)))
	if code := int32(r1); code != 0 {
		return 0, lastError(code)
	}
	return int(added), nil
}

func (idx *Indexer) create(indexPath string, inputs []string, force bool) (*IndexStats, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return nil, errors.New("indexer is closed")
	}
	// The C call reads the first input unconditionally through &cStrs[0].
	if len(inputs) == 0 {
		return nil, errors.New("no input paths given")
	}

	pathPtr, keepPath := cString(indexPath)
	defer keepPath()

	cStrs := make([]uintptr, len(inputs))
	keeps := make([]func(), len(inputs))
	for i, s := range inputs {
		cStrs[i], keeps[i] = cString(s)
	}
	defer func() {
		for _, k := range keeps {
			k()
		}
	}()

	var stats ffiIndexStats
	r1, _, _ := purego.SyscallN(
		_indexerCreateSym,
		idx.handle,
		pathPtr,
		uintptr(unsafe.Pointer(&cStrs[0])),
		uintptr(len(inputs)),
		uintptr(boolToInt(force)),
		uintptr(unsafe.Pointer(&stats)),
	)

	code := int32(r1)
	if code != 0 {
		return nil, lastError(code)
	}

	return &IndexStats{
		DocumentsIndexed: int(stats.DocumentsIndexed),
		ChunksCreated:    int(stats.ChunksCreated),
		Dimension:        int(stats.Dimension),
		SizeBytes:        stats.SizeBytes,
		FilesProcessed:   int(stats.FilesProcessed),
		FilesSkipped:     int(stats.FilesSkipped),
		ElapsedMs:        stats.ElapsedMs,
	}, nil
}

// Close releases the indexer resources. Safe to call multiple times.
func (idx *Indexer) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return nil
	}
	idx.closed = true
	_indexerFree(idx.handle)
	idx.handle = 0
	return nil
}