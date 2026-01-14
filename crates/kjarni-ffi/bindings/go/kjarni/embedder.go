package kjarni

/*
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// EmbedderConfig configures the Embedder.
type EmbedderConfig struct {
	Model     string
	Device    string // "cpu" or "gpu"
	CacheDir  string
	Normalize bool
	Quiet     bool
}

// Embedder encodes text to embedding vectors.
type Embedder struct {
	handle *C.KjarniEmbedder
}

// NewEmbedder creates a new Embedder.
func NewEmbedder(cfg *EmbedderConfig) (*Embedder, error) {
	config := C.kjarni_embedder_config_default()

	var modelName *C.char
	var cacheDir *C.char

	if cfg != nil {
		if cfg.Device == "gpu" {
			config.device = C.KJARNI_DEVICE_GPU
		}
		config.normalize = boolToInt(cfg.Normalize)
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
	} else {
		config.normalize = 1
	}

	var handle *C.KjarniEmbedder
	err := C.kjarni_embedder_new(&config, &handle)
	if e := checkError(err); e != nil {
		return nil, e
	}

	embedder := &Embedder{handle: handle}
	runtime.SetFinalizer(embedder, (*Embedder).Close)
	return embedder, nil
}

// Close frees the Embedder resources.
func (e *Embedder) Close() {
	if e.handle != nil {
		C.kjarni_embedder_free(e.handle)
		e.handle = nil
	}
}

// Encode encodes a single text to an embedding vector.
func (e *Embedder) Encode(text string) ([]float32, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var result C.KjarniFloatArray
	err := C.kjarni_embedder_encode(e.handle, cText, &result)
	if e := checkError(err); e != nil {
		return nil, e
	}
	defer C.kjarni_float_array_free(result)

	return floatArrayToSlice(result), nil
}

// EncodeBatch encodes multiple texts.
func (e *Embedder) EncodeBatch(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}

	cTexts := make([]*C.char, len(texts))
	for i, t := range texts {
		cTexts[i] = C.CString(t)
		defer C.free(unsafe.Pointer(cTexts[i]))
	}

	var result C.KjarniFloat2DArray
	err := C.kjarni_embedder_encode_batch(
		e.handle,
		(**C.char)(unsafe.Pointer(&cTexts[0])),
		C.size_t(len(texts)),
		&result,
	)
	if e := checkError(err); e != nil {
		return nil, e
	}
	defer C.kjarni_float_2d_array_free(result)

	return float2DArrayToSlice(result), nil
}

// Similarity computes cosine similarity between two texts.
func (e *Embedder) Similarity(text1, text2 string) (float32, error) {
	cText1 := C.CString(text1)
	defer C.free(unsafe.Pointer(cText1))
	cText2 := C.CString(text2)
	defer C.free(unsafe.Pointer(cText2))

	var result C.float
	err := C.kjarni_embedder_similarity(e.handle, cText1, cText2, &result)
	if e := checkError(err); e != nil {
		return 0, e
	}

	return float32(result), nil
}

// Dim returns the embedding dimension.
func (e *Embedder) Dim() int {
	return int(C.kjarni_embedder_dim(e.handle))
}

// Helper functions
func boolToInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

func floatArrayToSlice(arr C.KjarniFloatArray) []float32 {
	if arr.data == nil || arr.len == 0 {
		return []float32{}
	}
	length := int(arr.len)
	result := make([]float32, length)
	cSlice := (*[1 << 30]C.float)(unsafe.Pointer(arr.data))[:length:length]
	for i, v := range cSlice {
		result[i] = float32(v)
	}
	return result
}

func float2DArrayToSlice(arr C.KjarniFloat2DArray) [][]float32 {
	if arr.data == nil || arr.rows == 0 || arr.cols == 0 {
		return [][]float32{}
	}
	rows := int(arr.rows)
	cols := int(arr.cols)
	total := rows * cols
	cSlice := (*[1 << 30]C.float)(unsafe.Pointer(arr.data))[:total:total]

	result := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float32, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = float32(cSlice[i*cols+j])
		}
	}
	return result
}