//go:build integration

package kjarni

import (
	"math"
	"testing"
)

// The entities are what transformers' pipeline("ner", aggregation_strategy="first")
// gives for dslim/distilbert-NER on the CPU.
func TestEntityRecognizerMatchesTransformers(t *testing.T) {
	ner, err := NewEntityRecognizer("", AggregationFirst, WithQuiet(true))
	if err != nil {
		t.Fatalf("NewEntityRecognizer: %v", err)
	}
	defer ner.Close()

	got, err := ner.Recognize("Sarah Connor met Apple's CEO in San Francisco on Tuesday.")
	if err != nil {
		t.Fatal(err)
	}
	want := []Entity{
		{"PER", "Sarah Connor", 0, 12, 0.99879},
		{"ORG", "Apple", 17, 22, 0.99710},
		{"LOC", "San Francisco", 32, 45, 0.99761},
	}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		g, w := got[i], want[i]
		if g.Label != w.Label || g.Text != w.Text || g.Start != w.Start || g.End != w.End ||
			math.Abs(float64(g.Score-w.Score)) > 1e-4 {
			t.Errorf("entity %d: got %+v, want %+v", i, g, w)
		}
	}
}

func TestEntitySpansAreByteOffsets(t *testing.T) {
	ner, err := NewEntityRecognizer("", AggregationFirst, WithQuiet(true))
	if err != nil {
		t.Fatalf("NewEntityRecognizer: %v", err)
	}
	defer ner.Close()

	text := "Þórður hitti Angela Merkel í Berlín."
	got, err := ner.Recognize(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 {
		t.Fatal("no entities found")
	}
	for _, e := range got {
		if text[e.Start:e.End] != e.Text {
			t.Errorf("text[%d:%d] = %q, want %q", e.Start, e.End, text[e.Start:e.End], e.Text)
		}
	}
}
