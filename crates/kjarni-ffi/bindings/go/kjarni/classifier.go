package kjarni

/*
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// ClassifierConfig configures the Classifier.
type ClassifierConfig struct {
	Model      string
	Device     string // "cpu" or "gpu"
	CacheDir   string
	Labels     []string
	MultiLabel bool
	Quiet      bool
}

// ClassResult represents a single classification result.
type ClassResult struct {
	Label string
	Score float32
}

// Classifier classifies text.
type Classifier struct {
	handle *C.KjarniClassifier
}

// NewClassifier creates a new Classifier.
func NewClassifier(cfg *ClassifierConfig) (*Classifier, error) {
	config := C.kjarni_classifier_config_default()

	var modelName *C.char
	var cacheDir *C.char
	var cLabels []*C.char

	if cfg != nil {
		if cfg.Device == "gpu" {
			config.device = C.KJARNI_DEVICE_GPU
		}
		config.multi_label = boolToInt(cfg.MultiLabel)
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
		if len(cfg.Labels) > 0 {
			cLabels = make([]*C.char, len(cfg.Labels))
			for i, l := range cfg.Labels {
				cLabels[i] = C.CString(l)
				defer C.free(unsafe.Pointer(cLabels[i]))
			}
			config.labels = (**C.char)(unsafe.Pointer(&cLabels[0]))
			config.num_labels = C.size_t(len(cfg.Labels))
		}
	}

	var handle *C.KjarniClassifier
	err := C.kjarni_classifier_new(&config, &handle)
	if e := checkError(err); e != nil {
		return nil, e
	}

	classifier := &Classifier{handle: handle}
	runtime.SetFinalizer(classifier, (*Classifier).Close)
	return classifier, nil
}

// Close frees the Classifier resources.
func (c *Classifier) Close() {
	if c.handle != nil {
		C.kjarni_classifier_free(c.handle)
		c.handle = nil
	}
}

// Classify classifies a single text.
func (c *Classifier) Classify(text string) ([]ClassResult, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var result C.KjarniClassResults
	err := C.kjarni_classifier_classify(c.handle, cText, &result)
	if e := checkError(err); e != nil {
		return nil, e
	}
	defer C.kjarni_class_results_free(result)

	return classResultsToSlice(result), nil
}

// NumLabels returns the number of classification labels.
func (c *Classifier) NumLabels() int {
	return int(C.kjarni_classifier_num_labels(c.handle))
}

func classResultsToSlice(results C.KjarniClassResults) []ClassResult {
	if results.results == nil || results.len == 0 {
		return []ClassResult{}
	}
	length := int(results.len)
	cSlice := (*[1 << 30]C.KjarniClassResult)(unsafe.Pointer(results.results))[:length:length]

	out := make([]ClassResult, length)
	for i, r := range cSlice {
		label := ""
		if r.label != nil {
			label = C.GoString(r.label)
		}
		out[i] = ClassResult{Label: label, Score: float32(r.score)}
	}
	return out
}