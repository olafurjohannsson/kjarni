//go:build integration

// These tests download models (roughly 90MB for minilm) and run real inference,
// so they are behind a build tag and stay out of the default `go test ./...`.
// Run them with: go test -tags integration -v ./...
package kjarni

import (
	"math"
	"testing"
)

const testModel = "minilm-l6-v2"

func TestEmbedderEncode(t *testing.T) {
	e, err := NewEmbedder(testModel, WithQuiet(true))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	defer e.Close()

	if got := e.Dim(); got != 384 {
		t.Errorf("Dim() = %d, want 384", got)
	}

	vec, err := e.Encode("Hello world")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(vec) != 384 {
		t.Fatalf("Encode returned %d dimensions, want 384", len(vec))
	}

	// Vectors are normalised by default, so the norm is the cheapest assertion
	// that the values are real output rather than a zeroed or partial buffer.
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	if norm := math.Sqrt(sum); math.Abs(norm-1.0) > 1e-3 {
		t.Errorf("embedding norm = %f, want 1.0", norm)
	}
}

func TestEmbedderSimilarityOrdering(t *testing.T) {
	e, err := NewEmbedder(testModel, WithQuiet(true))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	defer e.Close()

	related, err := e.Similarity("doctor", "physician")
	if err != nil {
		t.Fatalf("Similarity: %v", err)
	}
	unrelated, err := e.Similarity("doctor", "banana")
	if err != nil {
		t.Fatalf("Similarity: %v", err)
	}

	// Asserting the ordering rather than exact scores: this catches a broken
	// pipeline without pinning the test to a model revision.
	if related <= unrelated {
		t.Errorf("similarity(doctor, physician) = %f is not above "+
			"similarity(doctor, banana) = %f", related, unrelated)
	}
	if related < 0.7 {
		t.Errorf("similarity(doctor, physician) = %f, expected a near-synonym score", related)
	}
}

func TestEmbedderEncodeBatchMatchesEncode(t *testing.T) {
	e, err := NewEmbedder(testModel, WithQuiet(true))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	defer e.Close()

	texts := []string{"first document", "second document", "third document"}

	batch, err := e.EncodeBatch(texts)
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}
	if len(batch) != len(texts) {
		t.Fatalf("EncodeBatch returned %d vectors for %d texts", len(batch), len(texts))
	}

	// The batched path is a separate FFI entry point, so it can drift from the
	// single-text one. Rows must line up with their inputs and agree in value.
	for i, text := range texts {
		single, err := e.Encode(text)
		if err != nil {
			t.Fatalf("Encode(%q): %v", text, err)
		}
		if len(batch[i]) != len(single) {
			t.Fatalf("row %d has %d dimensions, Encode gave %d", i, len(batch[i]), len(single))
		}
		for j := range single {
			if math.Abs(float64(batch[i][j]-single[j])) > 1e-4 {
				t.Fatalf("row %d differs from Encode(%q) at dimension %d: %f vs %f",
					i, text, j, batch[i][j], single[j])
			}
		}
	}
}

func TestEmbedderClosedTwice(t *testing.T) {
	e, err := NewEmbedder(testModel, WithQuiet(true))
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Errorf("first Close: %v", err)
	}
	// A double Close must not free the same handle twice.
	if err := e.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

func TestNewEmbedderWithUnknownModel(t *testing.T) {
	e, err := NewEmbedder("no-such-model-exists", WithQuiet(true))
	if err == nil {
		e.Close()
		t.Fatal("NewEmbedder accepted a model that does not exist")
	}
	if _, ok := err.(*KjarniError); !ok {
		t.Errorf("got %T, want *KjarniError", err)
	}
}
