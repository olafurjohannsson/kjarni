package main

import (
	"fmt"
	"os"
	"path/filepath"

	kjarni "github.com/olafurjohannsson/kjarni-go"
)

// build a search index from text files and query it three ways.
// the index is a self-contained directory: commit it, ship it in a container
// image, or drop it on a share. there is no server to run.
func main() {
	dir, err := os.MkdirTemp("", "kjarni-search")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer os.RemoveAll(dir)

	docs := map[string]string{
		"neural.txt":   "Neural networks consist of interconnected layers of artificial neurons.",
		"roman.txt":    "The Roman Empire collapsed in 476 AD after centuries of instability.",
		"pizza.txt":    "Neapolitan pizza is baked in a wood fired oven for ninety seconds.",
		"gradient.txt": "Gradient descent iteratively adjusts weights to minimise a loss function.",
	}
	for name, body := range docs {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	}

	indexPath := filepath.Join(dir, "docs.idx")

	idx, err := kjarni.NewIndexer("minilm-l6-v2", kjarni.WithQuiet(true))
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer idx.Close()

	stats, err := idx.Create(indexPath, []string{dir})
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("indexed %d documents\n\n", stats.DocumentsIndexed)

	// the second argument is the reranker; empty means no reranking.
	s, err := kjarni.NewSearcher("minilm-l6-v2", "", kjarni.WithQuiet(true))
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer s.Close()

	query := "how do machines learn"

	// Keyword is BM25, so it only matches words that actually appear.
	// Semantic matches meaning. Hybrid combines the two, which is usually
	// what you want: it still finds an exact term, but is not defeated by
	// a document that never uses the query's words.
	for _, mode := range []struct {
		name string
		mode kjarni.SearchMode
	}{
		{"keyword", kjarni.Keyword},
		{"semantic", kjarni.Semantic},
		{"hybrid", kjarni.Hybrid},
	} {
		results, err := s.Search(indexPath, query, mode.mode)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("%s search for %q:\n", mode.name, query)
		for _, r := range results {
			fmt.Printf("  %6.3f  %s\n", r.Score, r.Text)
		}
		fmt.Println()
	}
}
