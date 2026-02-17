package main

import (
	"fmt"
	"os"

	kjarni "github.com/olafurjohannsson/kjarni-go"
)

// rerank documents by relevance to a query using a cross-encoder model.
// useful for re-ordering search results, rag pipelines, or any
// scenario where you need precise query-document relevance scoring.
func main() {
	r, err := kjarni.NewReranker(kjarni.WithQuiet(true))
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer r.Close()

	query := "What is machine learning?"
	docs := []string{
		"The weather is nice today.",
		"Machine learning is a branch of artificial intelligence.",
		"Deep learning uses neural networks to learn from data.",
		"I like to eat pizza on fridays.",
	}

	// rerank all documents
	ranked, _ := r.Rerank(query, docs)
	fmt.Printf("query: %q\n\n", query)
	for _, result := range ranked {
		fmt.Printf("  [%d] %7.2f  %s\n", result.Index, result.Score, result.Document)
	}

	// or get only the top k results
	top, _ := r.RerankTopK(query, docs, 2)
	fmt.Printf("\ntop 2:\n")
	for _, result := range top {
		fmt.Printf("  [%d] %7.2f  %s\n", result.Index, result.Score, result.Document)
	}
}