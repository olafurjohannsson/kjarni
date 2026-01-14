package kjarni

/*
#cgo LDFLAGS: -L${SRCDIR}/../../../target/release -lkjarni_ffi
#cgo linux LDFLAGS: -lm -ldl -lpthread
#include <stdlib.h>

// Error codes
typedef enum {
    KJARNI_OK = 0,
    KJARNI_ERROR_NULL_POINTER = 1,
    KJARNI_ERROR_INVALID_UTF8 = 2,
    KJARNI_ERROR_MODEL_NOT_FOUND = 3,
    KJARNI_ERROR_LOAD_FAILED = 4,
    KJARNI_ERROR_INFERENCE_FAILED = 5,
    KJARNI_ERROR_GPU_UNAVAILABLE = 6,
    KJARNI_ERROR_INVALID_CONFIG = 7,
    KJARNI_ERROR_CANCELLED = 8,
    KJARNI_ERROR_TIMEOUT = 9,
    KJARNI_ERROR_STREAM_ENDED = 10,
    KJARNI_ERROR_UNKNOWN = 255,
} KjarniError;

typedef enum {
    KJARNI_DEVICE_CPU = 0,
    KJARNI_DEVICE_GPU = 1,
} KjarniDevice;

typedef struct {
    float* data;
    size_t len;
} KjarniFloatArray;

typedef struct {
    float* data;
    size_t rows;
    size_t cols;
} KjarniFloat2DArray;

typedef struct {
    KjarniDevice device;
    const char* cache_dir;
    const char* model_name;
    const char* model_path;
    int normalize;
    int quiet;
} KjarniEmbedderConfig;

typedef struct {
    char* label;
    float score;
} KjarniClassResult;

typedef struct {
    KjarniClassResult* results;
    size_t len;
} KjarniClassResults;

typedef struct {
    KjarniDevice device;
    const char* cache_dir;
    const char* model_name;
    const char* model_path;
    const char** labels;
    size_t num_labels;
    int multi_label;
    int quiet;
} KjarniClassifierConfig;

// Opaque handles
typedef struct KjarniEmbedder KjarniEmbedder;
typedef struct KjarniClassifier KjarniClassifier;

// Functions
extern const char* kjarni_last_error_message();
extern void kjarni_clear_error();
extern const char* kjarni_version();
extern void kjarni_float_array_free(KjarniFloatArray arr);
extern void kjarni_float_2d_array_free(KjarniFloat2DArray arr);
extern void kjarni_class_results_free(KjarniClassResults results);

extern KjarniEmbedderConfig kjarni_embedder_config_default();
extern KjarniError kjarni_embedder_new(const KjarniEmbedderConfig* config, KjarniEmbedder** out);
extern void kjarni_embedder_free(KjarniEmbedder* embedder);
extern KjarniError kjarni_embedder_encode(KjarniEmbedder* embedder, const char* text, KjarniFloatArray* out);
extern KjarniError kjarni_embedder_encode_batch(KjarniEmbedder* embedder, const char** texts, size_t num_texts, KjarniFloat2DArray* out);
extern KjarniError kjarni_embedder_similarity(KjarniEmbedder* embedder, const char* text1, const char* text2, float* out);
extern size_t kjarni_embedder_dim(const KjarniEmbedder* embedder);

extern KjarniClassifierConfig kjarni_classifier_config_default();
extern KjarniError kjarni_classifier_new(const KjarniClassifierConfig* config, KjarniClassifier** out);
extern void kjarni_classifier_free(KjarniClassifier* classifier);
extern KjarniError kjarni_classifier_classify(KjarniClassifier* classifier, const char* text, KjarniClassResults* out);
extern size_t kjarni_classifier_num_labels(const KjarniClassifier* classifier);
*/
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

// KjarniError represents an error from the Kjarni library.
type KjarniError struct {
	Code    int
	Message string
}

func (e *KjarniError) Error() string {
	return e.Message
}

func checkError(code C.KjarniError) error {
	if code == C.KJARNI_OK {
		return nil
	}
	msg := C.kjarni_last_error_message()
	var message string
	if msg != nil {
		message = C.GoString(msg)
	} else {
		message = "Unknown error"
	}
	return &KjarniError{Code: int(code), Message: message}
}

// Version returns the Kjarni library version.
func Version() string {
	return C.GoString(C.kjarni_version())
}