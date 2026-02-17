package main

import (
	"fmt"
	"os"

	kjarni "github.com/olafurjohannsson/kjarni-go"
)

// classify text using a pre-trained sentiment model.
// models download automatically on first use.
func main() {
	c, err := kjarni.NewClassifier("roberta-sentiment", kjarni.WithQuiet(true))
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer c.Close()

	result, err := c.Classify("I love this product!")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(result)
	// positive (98.5%)

	// multi-label models like toxic-bert return scores for all labels
	t, _ := kjarni.NewClassifier("toxic-bert", kjarni.WithQuiet(true))
	defer t.Close()

	toxicity, _ := t.Classify("You are an idiot")
	for _, s := range toxicity.AllScores {
		fmt.Printf("  %s: %.1f%%\n", s.Label, s.Score*100)
	}
}