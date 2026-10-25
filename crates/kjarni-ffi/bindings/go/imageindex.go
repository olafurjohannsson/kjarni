package kjarni

import (
	"errors"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// ImageHit is an image matching a search.
type ImageHit struct {
	Path string
	// Score is a cosine similarity, small even for good matches, so only the
	// order is meaningful.
	Score float32
}

// ScanReport says what scanning a directory did.
type ScanReport struct {
	Added int
	// Skipped counts files that could not be decoded.
	Skipped int
}

type ffiImageHit struct {
	Path  uintptr
	Score float32
	_     int32 // padding
}

type ffiImageResults struct {
	Hits uintptr
	Len  uintptr
}

type ffiScanReport struct {
	Added   uintptr
	Skipped uintptr
}

// ImageIndex searches photos by description, with CLIP.
type ImageIndex struct {
	handle uintptr
	mu     sync.Mutex
	closed bool
}

// NewImageIndex loads a CLIP model from modelDir, or clip-vit-base-32 for "",
// downloaded on first use.
func NewImageIndex(modelDir string) (*ImageIndex, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if _imageIndexNewSym == 0 {
		return nil, errors.New("this Kjarni library was built without image support")
	}

	dirPtr, keepDir := cStringOrNull(modelDir)
	defer keepDir()
	var handle uintptr
	r1, _, _ := purego.SyscallN(_imageIndexNewSym, dirPtr, uintptr(unsafe.Pointer(&handle)))
	if code := int32(r1); code != 0 {
		return nil, lastError(code)
	}
	return &ImageIndex{handle: handle}, nil
}

func (x *ImageIndex) check() error {
	if x.closed {
		return errors.New("image index is closed")
	}
	return nil
}

// Add indexes one JPEG or PNG.
func (x *ImageIndex) Add(path string) error {
	x.mu.Lock()
	defer x.mu.Unlock()
	if err := x.check(); err != nil {
		return err
	}
	pathPtr, keep := cString(path)
	defer keep()
	r1, _, _ := purego.SyscallN(_imageIndexAddSym, x.handle, pathPtr)
	if code := int32(r1); code != 0 {
		return lastError(code)
	}
	return nil
}

// AddDirectory indexes every JPEG and PNG under dir.
func (x *ImageIndex) AddDirectory(dir string) (ScanReport, error) {
	x.mu.Lock()
	defer x.mu.Unlock()
	if err := x.check(); err != nil {
		return ScanReport{}, err
	}
	dirPtr, keep := cString(dir)
	defer keep()
	var report ffiScanReport
	r1, _, _ := purego.SyscallN(_imageIndexAddDirectorySym, x.handle, dirPtr, uintptr(unsafe.Pointer(&report)))
	if code := int32(r1); code != 0 {
		return ScanReport{}, lastError(code)
	}
	return ScanReport{Added: int(report.Added), Skipped: int(report.Skipped)}, nil
}

// Search returns the topK images best matching query.
func (x *ImageIndex) Search(query string, topK int) ([]ImageHit, error) {
	x.mu.Lock()
	defer x.mu.Unlock()
	if err := x.check(); err != nil {
		return nil, err
	}
	if topK < 0 {
		return nil, errors.New("topK must not be negative")
	}
	queryPtr, keep := cString(query)
	defer keep()
	var results ffiImageResults
	r1, _, _ := purego.SyscallN(_imageIndexSearchSym, x.handle, queryPtr, uintptr(topK), uintptr(unsafe.Pointer(&results)))
	if code := int32(r1); code != 0 {
		return nil, lastError(code)
	}
	defer callWithStruct(_imageResultsFreeSym, unsafe.Pointer(&results), unsafe.Sizeof(results))

	hits := make([]ImageHit, results.Len)
	if results.Len > 0 {
		for i, h := range unsafe.Slice((*ffiImageHit)(unsafe.Pointer(results.Hits)), results.Len) {
			hits[i] = ImageHit{Path: goString(h.Path), Score: h.Score}
		}
	}
	return hits, nil
}

// Save writes the vectors (not the images) to path.
func (x *ImageIndex) Save(path string) error {
	x.mu.Lock()
	defer x.mu.Unlock()
	if err := x.check(); err != nil {
		return err
	}
	pathPtr, keep := cString(path)
	defer keep()
	r1, _, _ := purego.SyscallN(_imageIndexSaveSym, x.handle, pathPtr)
	if code := int32(r1); code != 0 {
		return lastError(code)
	}
	return nil
}

// Load restores vectors written by Save and returns how many.
func (x *ImageIndex) Load(path string) (int, error) {
	x.mu.Lock()
	defer x.mu.Unlock()
	if err := x.check(); err != nil {
		return 0, err
	}
	pathPtr, keep := cString(path)
	defer keep()
	var count uintptr
	r1, _, _ := purego.SyscallN(_imageIndexLoadSym, x.handle, pathPtr, uintptr(unsafe.Pointer(&count)))
	if code := int32(r1); code != 0 {
		return 0, lastError(code)
	}
	return int(count), nil
}

// Len reports how many images are indexed.
func (x *ImageIndex) Len() int {
	x.mu.Lock()
	defer x.mu.Unlock()
	if x.closed {
		return 0
	}
	r1, _, _ := purego.SyscallN(_imageIndexLenSym, x.handle)
	return int(r1)
}

// Close releases the model. Safe to call more than once.
func (x *ImageIndex) Close() error {
	x.mu.Lock()
	defer x.mu.Unlock()
	if x.closed {
		return nil
	}
	x.closed = true
	purego.SyscallN(_imageIndexFreeSym, x.handle)
	x.handle = 0
	return nil
}
