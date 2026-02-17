package kjarni

import (
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

// C struct layouts

type ffiClassifierConfig struct {
	Device    int32
	_         int32 // padding
	CacheDir  uintptr
	ModelName uintptr
	ModelPath uintptr
	Labels    uintptr
	NumLabels uintptr
	MultiLabel int32
	Quiet     int32
}

type ffiClassResult struct {
	Label uintptr
	Score float32
	_     [4]byte // padding to 16 bytes
}

type ffiClassResults struct {
	Results uintptr
	Len     uintptr
}

type ffiEmbedderConfig struct {
	Device    int32
	_         int32 // padding
	CacheDir  uintptr
	ModelName uintptr
	ModelPath uintptr
	Normalize int32
	Quiet     int32
}

type ffiFloatArray struct {
	Data uintptr
	Len  uintptr
}

type ffiFloat2DArray struct {
	Data uintptr
	Rows uintptr
	Cols uintptr
}

type ffiRerankerConfig struct {
	Device    int32
	_         int32 // padding
	CacheDir  uintptr
	ModelName uintptr
	ModelPath uintptr
	Quiet     int32
	_2        int32 // padding
}

type ffiRerankResult struct {
	Index uintptr
	Score float32
	_     [4]byte // padding
}

type ffiRerankResults struct {
	Results uintptr
	Len     uintptr
}

var (
	ffiOnce sync.Once

	// Error handling
	_lastErrorMessage func() uintptr
	_clearError       func()

	// Classifier
	_classifierNewSym  uintptr
	_classifierFree    func(handle uintptr)
	_classifierClassifySym uintptr
	_classifierNumLabels func(handle uintptr) uintptr
	_classResultsFree  func(results uintptr, len uintptr)

	// Embedder
	_embedderNewSym    uintptr
	_embedderFree      func(handle uintptr)
	_embedderEncodeSym uintptr
	_embedderEncodeBatchSym uintptr
	_embedderSimilaritySym uintptr
	_embedderDim       func(handle uintptr) uintptr
	_floatArrayFree    func(data uintptr, len uintptr)
	_float2DArrayFree  func(data uintptr, rows uintptr, cols uintptr)

	// Reranker
	_rerankerNewSym    uintptr
	_rerankerFree      func(handle uintptr)
	_rerankerScoreSym  uintptr
	_rerankerRerankSym uintptr
	_rerankerRerankTopKSym uintptr
	_rerankResultsFree func(results uintptr, len uintptr)

	// Indexer
	_indexerNewSym    uintptr
	_indexerFree      func(handle uintptr)
	_indexerCreateSym uintptr

	// Searcher
	_searcherNewSym              uintptr
	_searcherFree                func(handle uintptr)
	_searcherSearchWithOptionsSym uintptr

	// Cosine similarity
	_cosineSimilarity func(a unsafe.Pointer, b unsafe.Pointer, len uintptr) float32

	// Chat
	_chatNewSym              uintptr
	_chatFree                func(handle uintptr)
	_chatSendSym             uintptr
	_chatStreamSym           uintptr
	_chatModelNameSym        uintptr
	_chatContextSizeSym      uintptr
	_chatConversationNewSym  uintptr
	_chatConversationFree    func(handle uintptr)
	_chatConversationSendSym   uintptr
	_chatConversationStreamSym uintptr
	_chatConversationLenSym    uintptr
	_chatConversationClearSym  uintptr

	// String free (needed for chat results)
	_stringFreeSym uintptr
)

func initFFI() error {
	handle, err := loadLibrary()
	if err != nil {
		return err
	}

	// Error handling
	purego.RegisterLibFunc(&_lastErrorMessage, handle, "kjarni_last_error_message")
	purego.RegisterLibFunc(&_clearError, handle, "kjarni_clear_error")

	// Classifier
	_classifierNewSym, err = purego.Dlsym(handle, "kjarni_classifier_new")
	if err != nil {
		return err
	}
	purego.RegisterLibFunc(&_classifierFree, handle, "kjarni_classifier_free")
	_classifierClassifySym, err = purego.Dlsym(handle, "kjarni_classifier_classify")
	if err != nil {
		return err
	}
	purego.RegisterLibFunc(&_classifierNumLabels, handle, "kjarni_classifier_num_labels")

	// class_results_free takes the struct by value (2 fields: ptr + len)
	_classResultsFreeSym, err := purego.Dlsym(handle, "kjarni_class_results_free")
	if err != nil {
		return err
	}
	_ = _classResultsFreeSym // used via SyscallN
	// Store it 
	_classResultsFreeSymGlobal = _classResultsFreeSym

	// Embedder
	_embedderNewSym, err = purego.Dlsym(handle, "kjarni_embedder_new")
	if err != nil {
		return err
	}
	purego.RegisterLibFunc(&_embedderFree, handle, "kjarni_embedder_free")
	_embedderEncodeSym, err = purego.Dlsym(handle, "kjarni_embedder_encode")
	if err != nil {
		return err
	}
	_embedderEncodeBatchSym, err = purego.Dlsym(handle, "kjarni_embedder_encode_batch")
	if err != nil {
		return err
	}
	_embedderSimilaritySym, err = purego.Dlsym(handle, "kjarni_embedder_similarity")
	if err != nil {
		return err
	}
	purego.RegisterLibFunc(&_embedderDim, handle, "kjarni_embedder_dim")

	floatArrayFreeSym, err := purego.Dlsym(handle, "kjarni_float_array_free")
	if err != nil {
		return err
	}
	_floatArrayFreeSym = floatArrayFreeSym

	float2DArrayFreeSym, err := purego.Dlsym(handle, "kjarni_float_2d_array_free")
	if err != nil {
		return err
	}
	_float2DArrayFreeSym = float2DArrayFreeSym

	// Reranker
	_rerankerNewSym, err = purego.Dlsym(handle, "kjarni_reranker_new")
	if err != nil {
		return err
	}
	purego.RegisterLibFunc(&_rerankerFree, handle, "kjarni_reranker_free")
	_rerankerScoreSym, err = purego.Dlsym(handle, "kjarni_reranker_score")
	if err != nil {
		return err
	}
	_rerankerRerankSym, err = purego.Dlsym(handle, "kjarni_reranker_rerank")
	if err != nil {
		return err
	}
	_rerankerRerankTopKSym, err = purego.Dlsym(handle, "kjarni_reranker_rerank_top_k")
	if err != nil {
		return err
	}

	rerankResultsFreeSym, err := purego.Dlsym(handle, "kjarni_rerank_results_free")
	if err != nil {
		return err
	}
	_rerankResultsFreeSym = rerankResultsFreeSym

	// Indexer
	_indexerNewSym, err = purego.Dlsym(handle, "kjarni_indexer_new")
	if err != nil {
		return err
	}
	purego.RegisterLibFunc(&_indexerFree, handle, "kjarni_indexer_free")
	_indexerCreateSym, err = purego.Dlsym(handle, "kjarni_indexer_create")
	if err != nil {
		return err
	}

	// Searcher
	_searcherNewSym, err = purego.Dlsym(handle, "kjarni_searcher_new")
	if err != nil {
		return err
	}
	purego.RegisterLibFunc(&_searcherFree, handle, "kjarni_searcher_free")
	_searcherSearchWithOptionsSym, err = purego.Dlsym(handle, "kjarni_searcher_search_with_options")
	if err != nil {
		return err
	}

	searchResultsFreeSym, err := purego.Dlsym(handle, "kjarni_search_results_free")
	if err != nil {
		return err
	}
	_searchResultsFreeSym = searchResultsFreeSym

	// Chat
	_chatNewSym, err = purego.Dlsym(handle, "kjarni_chat_new")
	if err != nil {
		return err
	}
	purego.RegisterLibFunc(&_chatFree, handle, "kjarni_chat_free")
	_chatSendSym, err = purego.Dlsym(handle, "kjarni_chat_send")
	if err != nil {
		return err
	}
	_chatStreamSym, err = purego.Dlsym(handle, "kjarni_chat_stream")
	if err != nil {
		return err
	}
	_chatModelNameSym, err = purego.Dlsym(handle, "kjarni_chat_model_name")
	if err != nil {
		return err
	}
	_chatContextSizeSym, err = purego.Dlsym(handle, "kjarni_chat_context_size")
	if err != nil {
		return err
	}
	_chatConversationNewSym, err = purego.Dlsym(handle, "kjarni_chat_conversation_new")
	if err != nil {
		return err
	}
	purego.RegisterLibFunc(&_chatConversationFree, handle, "kjarni_chat_conversation_free")
	_chatConversationSendSym, err = purego.Dlsym(handle, "kjarni_chat_conversation_send")
	if err != nil {
		return err
	}
	_chatConversationStreamSym, err = purego.Dlsym(handle, "kjarni_chat_conversation_stream")
	if err != nil {
		return err
	}
	_chatConversationLenSym, err = purego.Dlsym(handle, "kjarni_chat_conversation_len")
	if err != nil {
		return err
	}
	_chatConversationClearSym, err = purego.Dlsym(handle, "kjarni_chat_conversation_clear")
	if err != nil {
		return err
	}

	// String free
	_stringFreeSym, err = purego.Dlsym(handle, "kjarni_string_free")
	if err != nil {
		return err
	}

	return nil
}

// Symbols stored for SyscallN usage
var (
	_classResultsFreeSymGlobal uintptr
	_floatArrayFreeSym         uintptr
	_float2DArrayFreeSym       uintptr
	_rerankResultsFreeSym      uintptr
	_searchResultsFreeSym      uintptr
)

// convert Go string to null-terminated C string, returns pointer and cleanup func
func cString(s string) (uintptr, func()) {
	b := append([]byte(s), 0)
	ptr := unsafe.Pointer(&b[0])
	// prevent GC from collecting the slice while we use the pointer
	return uintptr(ptr), func() {
		_ = b
	}
}

// get last error message from FFI
func lastError(code int32) error {
	ptr := _lastErrorMessage()
	if ptr == 0 {
		return &KjarniError{
			Code:    ErrorCode(code),
			Message: "unknown error",
		}
	}
	msg := goString(ptr)
	return &KjarniError{
		Code:    ErrorCode(code),
		Message: msg,
	}
}

// read null-terminated C string from pointer
func goString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	var length int
	for {
		b := *(*byte)(unsafe.Pointer(ptr + uintptr(length)))
		if b == 0 {
			break
		}
		length++
	}
	bytes := make([]byte, length)
	for i := 0; i < length; i++ {
		bytes[i] = *(*byte)(unsafe.Pointer(ptr + uintptr(i)))
	}
	return string(bytes)
}