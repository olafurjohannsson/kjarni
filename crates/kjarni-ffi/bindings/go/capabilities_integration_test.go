//go:build integration

package kjarni

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"testing"
)

// skipHeavy skips tests that need a model of several hundred megabytes when
// KJARNI_SKIP_HEAVY is set, so CI can run the rest without the downloads.
func skipHeavy(t *testing.T) {
	t.Helper()
	if os.Getenv("KJARNI_SKIP_HEAVY") != "" {
		t.Skip("KJARNI_SKIP_HEAVY is set")
	}
}

// repoFixture resolves a path under the repo root, four levels above this
// package. In the published kjarni-go module the fixtures are absent, so the
// tests that need them skip.
func repoFixture(t *testing.T, rel string) string {
	t.Helper()
	p := filepath.Join("..", "..", "..", "..", rel)
	if _, err := os.Stat(p); err != nil {
		t.Skipf("fixture %s not found", rel)
	}
	abs, _ := filepath.Abs(p)
	return abs
}

func wantCode(t *testing.T, err error, code ErrorCode) {
	t.Helper()
	var ke *KjarniError
	if !errors.As(err, &ke) {
		t.Fatalf("got %v (%T), want a *KjarniError with code %d", err, err, code)
	}
	if ke.Code != code {
		t.Fatalf("error code = %d (%s), want %d", ke.Code, ke.Message, code)
	}
}

func rssKB() int {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return -1
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			n, _ := strconv.Atoi(strings.Fields(line)[1])
			return n
		}
	}
	return -1
}

// Results are structs passed to their free functions by value, which purego can
// only do by laying out the words itself. Getting that wrong does not crash: the
// free reads a zero length and does nothing, so every result leaks. 300 batches
// of 64 vectors is 28MB; a correct binding stops growing once the allocator has
// settled.
func TestResultsAreFreed(t *testing.T) {
	if rssKB() < 0 {
		t.Skip("needs /proc/self/status")
	}
	// One OS thread for every call. Go moves goroutines between threads, and
	// glibc gives each thread its own malloc arena; memory freed in one is not
	// reused by another, so resident memory creeps up until the arenas fill.
	// That is the allocator, not a leak, and it would hide one.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	e, err := NewEmbedder(testModel, WithQuiet(true))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	defer e.Close()

	texts := make([]string, 64)
	for i := range texts {
		texts[i] = "some text to embed " + strconv.Itoa(i)
	}
	for i := 0; i < 20; i++ { // reach a steady state first
		if _, err := e.EncodeBatch(texts); err != nil {
			t.Fatal(err)
		}
	}
	// Go's collector frees the returned slices on its own schedule and hands
	// memory back to the OS lazily, which made this test measure GC timing.
	// Collecting and releasing first leaves only native memory in the reading.
	settledRSS := func() int {
		runtime.GC()
		debug.FreeOSMemory()
		return rssKB()
	}
	round := func() int {
		before := settledRSS()
		for i := 0; i < 300; i++ {
			if _, err := e.EncodeBatch(texts); err != nil {
				t.Fatal(err)
			}
			if _, err := e.Encode(texts[i%len(texts)]); err != nil {
				t.Fatal(err)
			}
		}
		return settledRSS() - before
	}
	// The allocator settles in the first round, growing a few MB as it
	// arranges its arenas for this mix of sizes. A leak keeps growing in the
	// second round too, by about the 28MB the results come to.
	first := round()
	second := round()
	t.Logf("RSS grew %d KB, then %d KB, over two rounds of 300 batches", first, second)
	if second > 8*1024 {
		t.Errorf("RSS grew %d KB in the second round; results are not being freed", second)
	}
}

func TestIndexAndSearch(t *testing.T) {
	docs := t.TempDir()
	write := func(name, text string) {
		if err := os.WriteFile(filepath.Join(docs, name), []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("cats.txt", "Cats are small carnivorous mammals that purr and chase mice.")
	write("rust.md", "Rust is a systems programming language focused on memory safety.")
	write("bread.txt", "Sourdough bread is leavened with a starter of wild yeast.")
	write("skip.log", "This file has an extension that is not indexed.")
	index := filepath.Join(docs, "index")

	idx, err := NewIndexer(testModel, WithQuiet(true), WithExtensions("txt", "md"))
	if err != nil {
		t.Fatalf("NewIndexer: %v", err)
	}
	defer idx.Close()

	stats, err := idx.Create(index, []string{docs})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if stats.DocumentsIndexed != 3 || stats.Dimension != 384 {
		t.Errorf("stats = %+v, want 3 documents of dimension 384", stats)
	}
	if _, err := idx.Create(index, []string{docs}); err == nil {
		t.Error("Create over an existing index succeeded")
	}
	if _, err := idx.Recreate(index, []string{docs}); err != nil {
		t.Errorf("Recreate: %v", err)
	}
	if _, err := idx.Create(index, nil); err == nil {
		t.Error("Create with no inputs succeeded")
	}

	write("tea.txt", "Green tea is brewed from unoxidised leaves of the tea plant.")
	if added, err := idx.Add(index, []string{filepath.Join(docs, "tea.txt")}); err != nil || added < 1 {
		t.Errorf("Add = %d, %v", added, err)
	}

	s, err := NewSearcher(testModel, "", WithQuiet(true))
	if err != nil {
		t.Fatalf("NewSearcher: %v", err)
	}
	defer s.Close()

	hits, err := s.Search(index, "feline pets", Hybrid)
	if err != nil || len(hits) == 0 {
		t.Fatalf("Search = %v, %v", hits, err)
	}
	if !strings.Contains(strings.ToLower(hits[0].Text), "cats") {
		t.Errorf("top hit %q is not the cat document", hits[0].Text)
	}
	if src, _ := hits[0].Metadata["source"].(string); !strings.HasSuffix(src, "cats.txt") {
		t.Errorf("top hit metadata = %v, want source cats.txt", hits[0].Metadata)
	}

	semantic, err := s.SearchWithOptions(index, "a drink made from leaves", Semantic, SearchOptions{TopK: 1})
	if err != nil || len(semantic) != 1 || !strings.Contains(strings.ToLower(semantic[0].Text), "tea") {
		t.Errorf("semantic top 1 = %v, %v; want the added tea document", semantic, err)
	}

	keyword, err := s.Search(index, "sourdough", Keyword)
	if err != nil || len(keyword) == 0 || !strings.Contains(strings.ToLower(keyword[0].Text), "sourdough") {
		t.Errorf("keyword search = %v, %v", keyword, err)
	}

	filtered, err := s.SearchWithOptions(index, "programming", Hybrid, SearchOptions{SourcePattern: "*.md"})
	if err != nil || len(filtered) == 0 {
		t.Fatalf("filtered search = %v, %v", filtered, err)
	}
	for _, h := range filtered {
		if src, _ := h.Metadata["source"].(string); !strings.HasSuffix(src, "rust.md") {
			t.Errorf("source pattern *.md let through %v", h.Metadata)
		}
	}

	if _, err := s.Search(filepath.Join(docs, "nope"), "x", Hybrid); err == nil {
		t.Error("searching a missing index succeeded")
	}
}

func TestTranscriber(t *testing.T) {
	skipHeavy(t)
	wav := repoFixture(t, "crates/kjarni/tests/fixtures/speech.wav")

	tr, err := NewTranscriber("", WithLanguage("en"), WithQuiet(true))
	if err != nil {
		t.Fatalf("NewTranscriber: %v", err)
	}
	defer tr.Close()

	got, err := tr.TranscribeFile(wav)
	if err != nil {
		t.Fatalf("TranscribeFile: %v", err)
	}
	if strings.TrimSpace(got.Text) == "" || got.Language != "en" || got.DurationSecs <= 0 || len(got.Segments) == 0 {
		t.Errorf("transcript = %+v", got)
	}
	for _, s := range got.Segments {
		if s.End < s.Start {
			t.Errorf("segment ends before it starts: %+v", s)
		}
	}

	// A second of silence at 8 kHz exercises resampling.
	if _, err := tr.Transcribe(make([]float32, 8000), 8000); err != nil {
		t.Errorf("Transcribe at 8 kHz: %v", err)
	}
	if _, err := tr.Transcribe(make([]float32, 100), 0); err == nil {
		t.Error("a zero sample rate was accepted")
	}
	if _, err := tr.TranscribeFile("/no/such/file.wav"); err == nil {
		t.Error("a missing file was accepted")
	}
}

const article = "The city council voted on Tuesday to approve a new public library in the " +
	"downtown district. The library will include a children's wing, a maker space " +
	"with 3D printers, and a cafe. Construction is expected to begin next spring and " +
	"take about eighteen months. Council members said the project would be funded " +
	"through a combination of municipal bonds and private donations. Several " +
	"residents spoke in favour of the plan at the meeting, citing the lack of quiet " +
	"study spaces in the area. A small group opposed it, arguing the money would be " +
	"better spent on road repairs. The mayor said the library would become a " +
	"cornerstone of the community and help revitalise the downtown area, which has " +
	"seen several businesses close over the past few years."

func TestSummarizer(t *testing.T) {
	skipHeavy(t)
	s, err := NewSummarizer("", WithQuiet(true))
	if err != nil {
		t.Fatalf("NewSummarizer: %v", err)
	}
	defer s.Close()

	summary, err := s.Summarize(article)
	if err != nil || strings.TrimSpace(summary) == "" || len(summary) >= len(article) {
		t.Errorf("Summarize = %q, %v", summary, err)
	}

	_, err = NewSummarizer("no-such-model", WithQuiet(true))
	wantCode(t, err, ErrModelNotFound)
}

func TestTranslator(t *testing.T) {
	skipHeavy(t)
	tr, err := NewTranslator("", WithLanguages("en", "de"), WithQuiet(true))
	if err != nil {
		t.Fatalf("NewTranslator: %v", err)
	}
	defer tr.Close()

	if out, err := tr.Translate("The house is small."); err != nil || !strings.Contains(strings.ToLower(out), "haus") {
		t.Errorf("Translate = %q, %v", out, err)
	}
	if out, err := tr.TranslateBetween("The house is small.", "English", "German"); err != nil ||
		!strings.Contains(strings.ToLower(out), "haus") {
		t.Errorf("TranslateBetween = %q, %v", out, err)
	}

	bare, err := NewTranslator("", WithQuiet(true))
	if err != nil {
		t.Fatalf("NewTranslator: %v", err)
	}
	defer bare.Close()
	_, err = bare.Translate("hello")
	wantCode(t, err, ErrInvalidConfig)
}

func TestImageIndex(t *testing.T) {
	skipHeavy(t)
	photos := repoFixture(t, "tests/fixtures/clip/photos")

	x, err := NewImageIndex("")
	if err != nil {
		t.Fatalf("NewImageIndex: %v", err)
	}
	defer x.Close()

	report, err := x.AddDirectory(photos)
	if err != nil || report.Added < 4 || report.Skipped != 0 {
		t.Fatalf("AddDirectory = %+v, %v", report, err)
	}
	if x.Len() != report.Added {
		t.Errorf("Len = %d, want %d", x.Len(), report.Added)
	}

	for _, c := range []struct{ query, want string }{
		{"a sandy beach under a blue sky", "beach_holiday"},
		{"a dense green forest with trees", "deep_forest"},
		{"a red car", "red_car"},
		{"a snowy mountain peak", "snowy_mountain"},
	} {
		hits, err := x.Search(c.query, 2)
		if err != nil || len(hits) != 2 || !strings.Contains(hits[0].Path, c.want) {
			t.Errorf("Search(%q) = %v, %v; want %s first", c.query, hits, err, c.want)
		}
	}

	saved := filepath.Join(t.TempDir(), "photos.idx")
	if err := x.Save(saved); err != nil {
		t.Fatalf("Save: %v", err)
	}
	restored, err := NewImageIndex("")
	if err != nil {
		t.Fatalf("NewImageIndex: %v", err)
	}
	defer restored.Close()
	if n, err := restored.Load(saved); err != nil || n != x.Len() {
		t.Errorf("Load = %d, %v; want %d", n, err, x.Len())
	}
	if hits, err := restored.Search("a red car", 1); err != nil || len(hits) != 1 || !strings.Contains(hits[0].Path, "red_car") {
		t.Errorf("restored Search = %v, %v", hits, err)
	}

	if err := x.Add("/no/such/photo.jpg"); err == nil {
		t.Error("a missing photo was accepted")
	}
}

func TestChat(t *testing.T) {
	skipHeavy(t)
	c, err := NewChat("qwen2.5-0.5b-instruct", WithQuiet(true))
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	defer c.Close()
	greedy := GreedyGenerationConfig()
	greedy.MaxNewTokens = 24

	reply, err := c.SendWithConfig("What is the capital of France? Answer in one word.", &greedy)
	if err != nil || !strings.Contains(strings.ToLower(reply), "paris") {
		t.Errorf("Send = %q, %v", reply, err)
	}

	var streamed strings.Builder
	if err := c.StreamWithConfig("What is the capital of France? Answer in one word.", &greedy, func(tok string) bool {
		streamed.WriteString(tok)
		return true
	}); err != nil || strings.TrimSpace(streamed.String()) != strings.TrimSpace(reply) {
		t.Errorf("Stream = %q, %v; want the same text as Send, %q", streamed.String(), err, reply)
	}

	tokens := 0
	if err := c.StreamWithConfig("Count from one to fifty.", &greedy, func(string) bool {
		tokens++
		return tokens < 3
	}); err != nil || tokens != 3 {
		t.Errorf("stopping the stream: %d tokens, %v; want 3", tokens, err)
	}

	history := []ChatTurn{
		{RoleUser, "My name is Anna."},
		{RoleAssistant, "Nice to meet you, Anna."},
	}
	if out, err := c.SendWithHistory(history, "What is my name?", &greedy); err != nil ||
		!strings.Contains(strings.ToLower(out), "anna") {
		t.Errorf("SendWithHistory = %q, %v", out, err)
	}
	var withHistory strings.Builder
	if err := c.StreamWithHistory(history, "What is my name?", &greedy, func(tok string) bool {
		withHistory.WriteString(tok)
		return true
	}); err != nil || !strings.Contains(strings.ToLower(withHistory.String()), "anna") {
		t.Errorf("StreamWithHistory = %q, %v", withHistory.String(), err)
	}

	q, err := NewChat("qwen2.5-0.5b-instruct", WithQuiet(true), WithQuantized(true))
	if err != nil {
		t.Fatalf("NewChat quantized: %v", err)
	}
	defer q.Close()
	if out, err := q.SendWithConfig("What is the capital of France? Answer in one word.", &greedy); err != nil ||
		!strings.Contains(strings.ToLower(out), "paris") {
		t.Errorf("quantized Send = %q, %v", out, err)
	}

	convo, err := c.Conversation()
	if err != nil {
		t.Fatalf("Conversation: %v", err)
	}
	defer convo.Close()
	if _, err := convo.SendWithConfig("Remember this: the secret word is banana.", &greedy); err != nil {
		t.Fatal(err)
	}
	if out, err := convo.SendWithConfig("What is the secret word?", &greedy); err != nil ||
		!strings.Contains(strings.ToLower(out), "banana") {
		t.Errorf("conversation recall = %q, %v", out, err)
	}
}
