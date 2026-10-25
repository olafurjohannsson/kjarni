package kjarni

import (
	"fmt"
	"runtime"
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

	// Chat with caller-held history
	_chatSendWithHistorySym   uintptr
	_chatStreamWithHistorySym uintptr

	// Indexer additions
	_indexerAddSym uintptr

	// Transcriber
	_transcriberNewSym            uintptr
	_transcriberFree              func(handle uintptr)
	_transcriberTranscribeFileSym uintptr
	_transcriberTranscribePCMSym  uintptr
	_transcriptionFreeSym         uintptr

	// Summarizer
	_summarizerNewSym       uintptr
	_summarizerFree         func(handle uintptr)
	_summarizerSummarizeSym uintptr

	// Translator
	_translatorNewSym       uintptr
	_translatorFree         func(handle uintptr)
	_translatorTranslateSym uintptr

	// Image search. Absent from a library built without the image-io feature,
	// so a failed lookup leaves these zero and NewImageIndex reports it.
	_imageIndexNewSym          uintptr
	_imageIndexFreeSym         uintptr
	_imageIndexAddSym          uintptr
	_imageIndexAddDirectorySym uintptr
	_imageIndexSearchSym       uintptr
	_imageResultsFreeSym       uintptr
	_imageIndexSaveSym         uintptr
	_imageIndexLoadSym         uintptr
	_imageIndexLenSym          uintptr
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

	if _chatSendWithHistorySym, err = purego.Dlsym(handle, "kjarni_chat_send_with_history"); err != nil {
		return err
	}
	if _chatStreamWithHistorySym, err = purego.Dlsym(handle, "kjarni_chat_stream_with_history"); err != nil {
		return err
	}
	if _indexerAddSym, err = purego.Dlsym(handle, "kjarni_indexer_add"); err != nil {
		return err
	}

	// Transcriber
	if _transcriberNewSym, err = purego.Dlsym(handle, "kjarni_transcriber_new"); err != nil {
		return err
	}
	purego.RegisterLibFunc(&_transcriberFree, handle, "kjarni_transcriber_free")
	if _transcriberTranscribeFileSym, err = purego.Dlsym(handle, "kjarni_transcriber_transcribe_file"); err != nil {
		return err
	}
	if _transcriberTranscribePCMSym, err = purego.Dlsym(handle, "kjarni_transcriber_transcribe_pcm"); err != nil {
		return err
	}
	if _transcriptionFreeSym, err = purego.Dlsym(handle, "kjarni_transcription_free"); err != nil {
		return err
	}

	// Summarizer
	if _summarizerNewSym, err = purego.Dlsym(handle, "kjarni_summarizer_new"); err != nil {
		return err
	}
	purego.RegisterLibFunc(&_summarizerFree, handle, "kjarni_summarizer_free")
	if _summarizerSummarizeSym, err = purego.Dlsym(handle, "kjarni_summarizer_summarize"); err != nil {
		return err
	}

	// Translator
	if _translatorNewSym, err = purego.Dlsym(handle, "kjarni_translator_new"); err != nil {
		return err
	}
	purego.RegisterLibFunc(&_translatorFree, handle, "kjarni_translator_free")
	if _translatorTranslateSym, err = purego.Dlsym(handle, "kjarni_translator_translate"); err != nil {
		return err
	}

	// Image search: optional, so lookup failures are not errors.
	_imageIndexNewSym, _ = purego.Dlsym(handle, "kjarni_image_index_new")
	_imageIndexFreeSym, _ = purego.Dlsym(handle, "kjarni_image_index_free")
	_imageIndexAddSym, _ = purego.Dlsym(handle, "kjarni_image_index_add")
	_imageIndexAddDirectorySym, _ = purego.Dlsym(handle, "kjarni_image_index_add_directory")
	_imageIndexSearchSym, _ = purego.Dlsym(handle, "kjarni_image_index_search")
	_imageResultsFreeSym, _ = purego.Dlsym(handle, "kjarni_image_results_free")
	_imageIndexSaveSym, _ = purego.Dlsym(handle, "kjarni_image_index_save")
	_imageIndexLoadSym, _ = purego.Dlsym(handle, "kjarni_image_index_load")
	_imageIndexLenSym, _ = purego.Dlsym(handle, "kjarni_image_index_len")

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
	p := unsafe.Pointer(ptr)
	n := 0
	for *(*byte)(unsafe.Add(p, n)) != 0 {
		n++
	}
	// One copy, rather than a bounds-checked read per byte into a second buffer.
	return string(unsafe.Slice((*byte)(p), n))
}

// callWithStruct calls a C function whose only parameter is a struct passed by
// value, such as kjarni_float_array_free(KjarniFloatArray).
//
// purego.SyscallN takes machine words, so the struct has to be laid out the way
// the platform ABI would pass it:
//
//   - Windows x64 passes any struct larger than 8 bytes as a pointer to a copy.
//   - System V x86-64 passes a struct of up to 16 bytes in two registers, and a
//     larger one on the stack, after the six integer argument registers.
//
// Passing &s everywhere, as these bindings used to, is only right on Windows. On
// Linux the callee read the pointer as the struct's data field and an unrelated
// register as its length, so every result it was asked to free was leaked.
func callWithStruct(sym uintptr, s unsafe.Pointer, size uintptr) {
	if runtime.GOOS == "windows" {
		purego.SyscallN(sym, uintptr(s))
		return
	}
	words := unsafe.Slice((*uintptr)(s), (size+7)/8)
	if size <= 16 {
		purego.SyscallN(sym, words...)
		return
	}
	args := make([]uintptr, 6, 6+len(words))
	purego.SyscallN(sym, append(args, words...)...)
}

// ensureInit loads the library once, for constructors.
func ensureInit() error {
	var initErr error
	ffiOnce.Do(func() { initErr = initFFI() })
	if initErr != nil {
		return fmt.Errorf("initializing kjarni: %w", initErr)
	}
	return nil
}

// cStringOrNull is cString, but NULL for "", which the C API reads as "use the
// default".
func cStringOrNull(s string) (uintptr, func()) {
	if s == "" {
		return 0, func() {}
	}
	return cString(s)
}

// takeString copies a string the engine allocated and frees it.
func takeString(ptr uintptr) string {
	if ptr == 0 {
		return ""
	}
	s := goString(ptr)
	purego.SyscallN(_stringFreeSym, ptr)
	return s
}

// cStringArray converts strings to a C `const char* const*`. The returned func
// keeps the backing memory alive and must be deferred.
func cStringArray(items []string) (uintptr, func()) {
	if len(items) == 0 {
		return 0, func() {}
	}
	ptrs := make([]uintptr, len(items))
	keeps := make([]func(), len(items))
	for i, s := range items {
		ptrs[i], keeps[i] = cString(s)
	}
	return uintptr(unsafe.Pointer(&ptrs[0])), func() {
		for _, k := range keeps {
			k()
		}
		runtime.KeepAlive(ptrs)
	}
}

// Streaming.
//
// purego never releases a callback and allows about two thousand per process, so
// creating one per Stream call, as these bindings used to, panicked a long-running
// program after that many streamed replies. One trampoline is created instead,
// and each call is told apart by the user_data the C API hands back.

var (
	streamOnce sync.Once
	streamCB   uintptr
	streamMu   sync.Mutex
	streamFns  = map[uintptr]func(string) bool{}
	streamNext uintptr
)

// registerStream returns the shared callback and the user_data identifying
// onToken; release must be called once the native call has returned.
func registerStream(onToken func(string) bool) (cb uintptr, userData uintptr, release func()) {
	streamOnce.Do(func() {
		streamCB = purego.NewCallback(func(textPtr uintptr, userData uintptr) uintptr {
			streamMu.Lock()
			fn := streamFns[userData]
			streamMu.Unlock()
			if fn == nil || !fn(goString(textPtr)) {
				return 0 // stop
			}
			return 1 // continue
		})
	})
	streamMu.Lock()
	streamNext++
	id := streamNext
	streamFns[id] = onToken
	streamMu.Unlock()
	return streamCB, id, func() {
		streamMu.Lock()
		delete(streamFns, id)
		streamMu.Unlock()
	}
}
