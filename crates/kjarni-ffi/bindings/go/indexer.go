package kjarni

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// statistics from an indexing operation.
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
	Device         int32
	_              int32 // padding
	CacheDir       uintptr
	ModelName      uintptr
	ChunkSize      uintptr
	ChunkOverlap   uintptr
	BatchSize      uintptr
	Extensions     uintptr
	ExcludePatterns uintptr
	Recursive      int32
	IncludeHidden  int32
	MaxFileSize    uintptr
	Quiet          int32
	_2             int32 // padding
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

// create search indexes from files
type Indexer struct {
	handle uintptr
	mu     sync.Mutex
	closed bool
}

//  new indexer
func NewIndexer(model string, opts ...Option) (*Indexer, error) {
	var initErr error
	ffiOnce.Do(func() { initErr = initFFI() })
	if initErr != nil {
		return nil, fmt.Errorf("initializing kjarni: %w", initErr)
	}

	o := applyOptions(opts)

	modelStr, keepModel := cString(model)
	defer keepModel()

	var config ffiIndexerConfig
	config.Device = deviceCode(o.device)
	config.ModelName = modelStr
	config.ChunkSize = 512
	config.ChunkOverlap = 50
	config.BatchSize = 32
	config.Recursive = 1
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

// build a new index from the given dirs
func (idx *Indexer) Create(indexPath string, inputs []string) (*IndexStats, error) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if idx.closed {
		return nil, errors.New("indexer is closed")
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
		0, // force = false
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

// releases resources
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