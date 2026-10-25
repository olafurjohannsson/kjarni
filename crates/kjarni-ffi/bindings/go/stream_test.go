package kjarni

import "testing"

// purego caps callbacks at about two thousand per process and never frees them,
// so creating one per stream panics a long-running program. Every stream must
// share one, and finished streams must not stay registered.
func TestStreamsShareOneCallback(t *testing.T) {
	first, _, release := registerStream(func(string) bool { return true })
	release()
	for i := 0; i < 2500; i++ {
		cb, id, release := registerStream(func(string) bool { return true })
		if cb != first {
			t.Fatalf("stream %d got a new callback", i)
		}
		if id == 0 {
			t.Fatal("user_data 0 is reserved for no stream")
		}
		release()
	}
	streamMu.Lock()
	left := len(streamFns)
	streamMu.Unlock()
	if left != 0 {
		t.Errorf("%d streams still registered after release", left)
	}
}

func TestOptionsForNewCapabilities(t *testing.T) {
	o := applyOptions([]Option{
		WithLanguage("is"), WithTranslate(true), WithLanguages("en", "de"),
		WithSummaryLength(10, 80), WithExtensions("md", "txt"), WithExclude("vendor"),
		WithChunking(256, 32), WithRecursive(false), WithHidden(true), WithCacheDir("/tmp/m"),
	})
	if o.language != "is" || !o.translate || o.from != "en" || o.to != "de" ||
		o.minLength != 10 || o.maxLength != 80 || len(o.extensions) != 2 || o.exclude[0] != "vendor" ||
		o.chunkSize != 256 || o.chunkOverlap != 32 || !o.notRecursive || !o.includeHidden || o.cacheDir != "/tmp/m" {
		t.Errorf("options = %+v", o)
	}
	if d := applyOptions(nil); d.notRecursive || d.includeHidden || d.translate {
		t.Errorf("defaults = %+v; recursion on, hidden and translation off", d)
	}
}
