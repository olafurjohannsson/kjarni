//go:build integration

// A model loaded from a local folder with WithModelPath gives what the same
// model gives by name. The folder is the one the named model downloads to.
package kjarni

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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

// hfSnapshot is the folder of repo in the Hugging Face cache, or "" when it
// is not there.
func hfSnapshot(repo string) string {
	home := os.Getenv("HF_HOME")
	if home == "" {
		h, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		home = filepath.Join(h, ".cache", "huggingface")
	}
	found, _ := filepath.Glob(filepath.Join(home, "hub", "models--"+strings.ReplaceAll(repo, "/", "--"), "snapshots", "*", "config.json"))
	if len(found) == 0 {
		return ""
	}
	return filepath.Dir(found[0])
}

func TestChatFromFolderMatchesByName(t *testing.T) {
	skipHeavy(t)
	const question = "What is the capital of Iceland?"
	greedy := GreedyGenerationConfig()
	greedy.MaxNewTokens = 24
	send := func(c *Chat, err error) string {
		t.Helper()
		if err != nil {
			t.Fatalf("NewChat: %v", err)
		}
		defer c.Close()
		reply, err := c.SendWithConfig(question, &greedy)
		if err != nil {
			t.Fatalf("Send: %v", err)
		}
		return reply
	}

	want := send(NewChat("qwen2.5-0.5b-instruct", WithQuiet(true)))
	if !strings.Contains(want, "Reykjav") {
		t.Fatalf("by name: %q", want)
	}
	folder := filepath.Join(kjarniCacheDir(t), "Qwen_Qwen2.5-0.5B-Instruct")
	if got := send(NewChat("", WithModelPath(folder), WithQuiet(true))); got != want {
		t.Errorf("from %s: %q, want %q", folder, got, want)
	}
	// A Hub folder brings its own Jinja chat template.
	if hub := hfSnapshot("Qwen/Qwen2.5-0.5B-Instruct"); hub != "" {
		if got := send(NewChat("", WithModelPath(hub), WithQuiet(true))); got != want {
			t.Errorf("from %s: %q, want %q", hub, got, want)
		}
	}
}

func TestChatNeedsANameOrAFolder(t *testing.T) {
	if _, err := NewChat(""); err == nil {
		t.Fatal("NewChat with neither a name nor a folder succeeded")
	}
	folder := filepath.Join(kjarniCacheDir(t), "distilbert_distilbert-base-uncased-finetuned-sst-2-english")
	_, err := NewChat("", WithModelPath(folder), WithQuiet(true))
	if err == nil || !strings.Contains(err.Error(), "not suitable for chat") {
		t.Fatalf("NewChat on an encoder folder: %v", err)
	}
}
