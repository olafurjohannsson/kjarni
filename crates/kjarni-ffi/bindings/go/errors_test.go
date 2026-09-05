package kjarni

import (
	"os"
	"regexp"
	"strconv"
	"testing"
)

// Error codes cross the ABI as bare integers. If the Rust enum gains a variant
// in the middle, every code after it shifts and Go silently reports the wrong
// error for the rest of the module's life. Pin each constant to the generated
// header instead of to a copy of the numbers.
func TestErrorCodesMatchHeader(t *testing.T) {
	// Go name for each C enumerator. The C side owns the values; this map only
	// records which Go constant is meant to mirror which enumerator.
	want := map[string]ErrorCode{
		"KJARNI_ERROR_CODE_OK":               ErrOk,
		"KJARNI_ERROR_CODE_NULL_POINTER":     ErrNullPointer,
		"KJARNI_ERROR_CODE_INVALID_UTF8":     ErrInvalidUtf8,
		"KJARNI_ERROR_CODE_MODEL_NOT_FOUND":  ErrModelNotFound,
		"KJARNI_ERROR_CODE_LOAD_FAILED":      ErrLoadFailed,
		"KJARNI_ERROR_CODE_INFERENCE_FAILED": ErrInferenceFailed,
		"KJARNI_ERROR_CODE_GPU_UNAVAILABLE":  ErrGpuUnavailable,
		"KJARNI_ERROR_CODE_INVALID_CONFIG":   ErrInvalidConfig,
		"KJARNI_ERROR_CODE_CANCELLED":        ErrCancelled,
		"KJARNI_ERROR_CODE_TIMEOUT":          ErrTimeout,
		"KJARNI_ERROR_CODE_STREAM_ENDED":     ErrStreamEnded,
		"KJARNI_ERROR_CODE_PANIC":            ErrPanic,
		"KJARNI_ERROR_CODE_UNKNOWN":          ErrUnknown,
	}

	header, err := os.ReadFile(findHeader(t))
	if err != nil {
		t.Fatalf("reading header: %v", err)
	}

	pattern := regexp.MustCompile(`(KJARNI_ERROR_CODE_[A-Z0-9_]+) = (\d+)`)
	matches := pattern.FindAllStringSubmatch(string(header), -1)
	if len(matches) == 0 {
		t.Fatal("found no error enumerators in the header")
	}

	found := map[string]bool{}
	for _, m := range matches {
		name, raw := m[1], m[2]
		found[name] = true

		value, err := strconv.Atoi(raw)
		if err != nil {
			t.Fatalf("parsing %s = %s: %v", name, raw, err)
		}

		got, ok := want[name]
		if !ok {
			t.Errorf("the header declares %s = %d, which the Go bindings do not expose", name, value)
			continue
		}
		if int(got) != value {
			t.Errorf("%s is %d in the header, %d in Go", name, value, int(got))
		}
	}

	for name := range want {
		if !found[name] {
			t.Errorf("Go declares a constant for %s, which the header no longer has", name)
		}
	}
}

func TestKjarniErrorMessage(t *testing.T) {
	err := &KjarniError{Code: ErrModelNotFound, Message: "no such model"}
	want := "kjarni: no such model (code 3)"
	if got := err.Error(); got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

// lastError has to produce a usable error even when the native side left no
// message behind, which is the path taken when a call fails before the engine
// records anything.
func TestLastErrorWithoutNativeMessage(t *testing.T) {
	if err := initFFI(); err != nil {
		t.Skipf("native library unavailable: %v", err)
	}
	_clearError()

	err := lastError(int32(ErrInvalidConfig))
	if err == nil {
		t.Fatal("lastError returned nil")
	}
	ke, ok := err.(*KjarniError)
	if !ok {
		t.Fatalf("lastError returned %T, want *KjarniError", err)
	}
	if ke.Code != ErrInvalidConfig {
		t.Errorf("Code = %d, want %d", ke.Code, ErrInvalidConfig)
	}
	if ke.Message == "" {
		t.Error("Message is empty; the error would tell a caller nothing")
	}
}
