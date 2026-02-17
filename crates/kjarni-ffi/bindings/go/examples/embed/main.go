package main

import (
	"fmt"
	"os"

	kjarni "github.com/olafurjohannsson/kjarni-go"
)

// generate embeddings and compute semantic similarity.
// available models: minilm-l6-v2 (384d), mpnet-base-v2 (768d), distilbert-base (768d)
func main() {
	e, err := kjarni.NewEmbedder("minilm-l6-v2", kjarni.WithQuiet(true))
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer e.Close()

	// single embedding
	vec, _ := e.Encode("hello world")
	fmt.Printf("dimensions: %d\n", len(vec))
	// dimensions: 384

	// similarity via the engine
	sim, _ := e.Similarity("doctor", "physician")
	fmt.Printf("doctor/physician: %.4f\n", sim)
	// doctor/physician: 0.8598

	sim2, _ := e.Similarity("doctor", "banana")
	fmt.Printf("doctor/banana:    %.4f\n", sim2)
	// doctor/banana:    0.3379

	// batch encoding
	vecs, _ := e.EncodeBatch([]string{"cat", "dog", "airplane"})
	fmt.Printf("batch: %d embeddings\n", len(vecs))

	// you can also compute cosine similarity directly in go
	v1, _ := e.Encode("cat")
	v2, _ := e.Encode("dog")
	fmt.Printf("cat/dog (go cosine): %.4f\n", kjarni.CosineSimilarity(v1, v2))
}