//go:build integration

// A model loaded from a local folder with WithModelPath gives what the same
// model gives by name. The folder is the one the named model downloads to.
package kjarni

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// kjarniCacheDir is where kjarni keeps models, as the Rust side decides it.
func kjarniCacheDir(t *testing.T) string {
	if dir := os.Getenv("KJARNI_CACHE_DIR"); dir != "" {
		return dir
	}
	if runtime.GOOS == "linux" && os.Getenv("XDG_CACHE_HOME") == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			t.Fatal(err)
		}
		return filepath.Join(home, ".cache", "kjarni")
	}
	dir, err := os.UserCacheDir()
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(dir, "kjarni")
}

func TestEmbedderFromFolderMatchesByName(t *testing.T) {
	byName, err := NewEmbedder("minilm-l6-v2", WithQuiet(true))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	defer byName.Close()
	want, err := byName.Encode("hello world")
	if err != nil {
		t.Fatal(err)
	}

	folder := filepath.Join(kjarniCacheDir(t), "sentence-transformers_all-MiniLM-L6-v2")
	byPath, err := NewEmbedder("", WithModelPath(folder), WithQuiet(true))
	if err != nil {
		t.Fatalf("NewEmbedder with WithModelPath(%s): %v", folder, err)
	}
	defer byPath.Close()
	got, err := byPath.Encode("hello world")
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != len(want) {
		t.Fatalf("dimension %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("component %d: %f, want %f", i, got[i], want[i])
		}
	}
}

func TestClassifierFromFolderMatchesByName(t *testing.T) {
	const text = "The box arrived crushed."
	byName, err := NewClassifier("distilbert-sentiment", WithQuiet(true))
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	defer byName.Close()
	want, err := byName.Classify(text)
	if err != nil {
		t.Fatal(err)
	}

	folder := filepath.Join(kjarniCacheDir(t), "distilbert_distilbert-base-uncased-finetuned-sst-2-english")
	byPath, err := NewClassifier("", WithModelPath(folder), WithQuiet(true))
	if err != nil {
		t.Fatalf("NewClassifier with WithModelPath(%s): %v", folder, err)
	}
	defer byPath.Close()
	got, err := byPath.Classify(text)
	if err != nil {
		t.Fatal(err)
	}

	if got.Label != want.Label || math.Abs(float64(got.Score-want.Score)) > 1e-6 {
		t.Errorf("by path %s %f, by name %s %f", got.Label, got.Score, want.Label, want.Score)
	}
}

func TestRerankerFromFolderMatchesByName(t *testing.T) {
	const q, d = "how do I reset my password", "Click Forgot password on the sign-in page."
	byName, err := NewReranker(WithQuiet(true))
	if err != nil {
		t.Fatalf("NewReranker: %v", err)
	}
	defer byName.Close()
	want, err := byName.Score(q, d)
	if err != nil {
		t.Fatal(err)
	}

	folder := filepath.Join(kjarniCacheDir(t), "cross-encoder_ms-marco-MiniLM-L-6-v2")
	byPath, err := NewReranker(WithModelPath(folder), WithQuiet(true))
	if err != nil {
		t.Fatalf("NewReranker with WithModelPath(%s): %v", folder, err)
	}
	defer byPath.Close()
	got, err := byPath.Score(q, d)
	if err != nil {
		t.Fatal(err)
	}

	if math.Abs(float64(got-want)) > 1e-5 {
		t.Errorf("by path %f, by name %f", got, want)
	}
}

func TestEmbedderDownloadsIntoCacheDir(t *testing.T) {
	// WithCacheDir is where the model goes; the default cache is left alone.
	dir := t.TempDir()
	e, err := NewEmbedder("minilm-l6-v2", WithCacheDir(dir), WithQuiet(true))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	defer e.Close()
	weights := filepath.Join(dir, "sentence-transformers_all-MiniLM-L6-v2", "model.safetensors")
	if _, err := os.Stat(weights); err != nil {
		t.Errorf("model not in the cache dir given: %v", err)
	}
}
