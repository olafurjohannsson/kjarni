//go:build integration

package kjarni

import (
	"math"
	"testing"
)

// A batch must give each text exactly what classifying it alone gives.
func TestClassifyBatchMatchesSingle(t *testing.T) {
	c, err := NewClassifier("distilbert-sentiment", WithQuiet(true))
	if err != nil {
		t.Fatalf("NewClassifier: %v", err)
	}
	defer c.Close()

	texts := []string{
		"I love this product, it's absolutely amazing!",
		"Worst purchase of my life.",
		"It is fine I guess.",
		"Quick delivery, but the colour is not what the photos showed and setup took ages.",
	}
	batch, err := c.ClassifyBatch(texts)
	if err != nil {
		t.Fatalf("ClassifyBatch: %v", err)
	}
	if len(batch) != len(texts) {
		t.Fatalf("got %d results for %d texts", len(batch), len(texts))
	}
	for i, text := range texts {
		single, err := c.Classify(text)
		if err != nil {
			t.Fatalf("Classify(%q): %v", text, err)
		}
		if single.Label != batch[i].Label || math.Abs(float64(single.Score-batch[i].Score)) > 1e-5 {
			t.Errorf("%q: batch %s %.6f, alone %s %.6f", text, batch[i].Label, batch[i].Score, single.Label, single.Score)
		}
	}

	empty, err := c.ClassifyBatch(nil)
	if err != nil || len(empty) != 0 {
		t.Errorf("ClassifyBatch(nil) = %v, %v; want empty", empty, err)
	}
}
