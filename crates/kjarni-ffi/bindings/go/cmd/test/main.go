package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	kjarni "github.com/olafurjohannsson/kjarni-go"
)

func main() {
	fmt.Println("=== Kjarni Go Bindings Test ===")
	fmt.Println()

	fmt.Println("--- Classifiers ---")

	models := []struct {
		name string
		text string
	}{
		{"distilbert-sentiment", "I love this product!"},
		{"roberta-sentiment", "I love this product!"},
		{"bert-sentiment-multilingual", "Absolutely amazing!"},
		{"distilroberta-emotion", "I'm so nervous about the interview tomorrow."},
		{"roberta-emotions", "Thank you so much for your help!"},
		{"toxic-bert", "You are an idiot"},
	}

	for _, m := range models {
		c, err := kjarni.NewClassifier(m.name, kjarni.WithQuiet(true))
		if err != nil {
			fmt.Printf("  ERROR %s: %v\n", m.name, err)
			os.Exit(1)
		}
		result, _ := c.Classify(m.text)
		fmt.Printf("  %s: %s\n", m.name, result)
		c.Close()
	}

	// toxic-bert multi-label check
	c, _ := kjarni.NewClassifier("toxic-bert", kjarni.WithQuiet(true))
	result, _ := c.Classify("You are an idiot")
	var sum float32
	for _, s := range result.AllScores {
		sum += s.Score
	}
	fmt.Printf("  toxic-bert scores sum: %.4f (should be > 1.0 for multi-label)\n", sum)
	c.Close()

	fmt.Println()
	fmt.Println("--- Embedders ---")

	embModels := []string{"minilm-l6-v2", "mpnet-base-v2", "distilbert-base"}
	for _, m := range embModels {
		e, err := kjarni.NewEmbedder(m, kjarni.WithQuiet(true))
		if err != nil {
			fmt.Printf("  ERROR %s: %v\n", m, err)
			os.Exit(1)
		}
		vec, _ := e.Encode("hello world")
		fmt.Printf("  %s: dim=%d, vec.len=%d\n", m, e.Dim(), len(vec))
		e.Close()
	}

	e, _ := kjarni.NewEmbedder("minilm-l6-v2", kjarni.WithQuiet(true))
	sim, _ := e.Similarity("cat", "dog")
	fmt.Printf("  cat/dog similarity: %.4f\n", sim)
	sim2, _ := e.Similarity("cat", "airplane")
	fmt.Printf("  cat/airplane similarity: %.4f\n", sim2)

	vecs, _ := e.EncodeBatch([]string{"hello", "world", "test"})
	fmt.Printf("  batch: %d embeddings, each %dd\n", len(vecs), len(vecs[0]))

	v1, _ := e.Encode("doctor")
	v2, _ := e.Encode("physician")
	goSim := kjarni.CosineSimilarity(v1, v2)
	fmt.Printf("  doctor/physician (Go cosine): %.4f\n", goSim)
	e.Close()

	// --- Reranker ---
	fmt.Println()
	fmt.Println("--- Reranker ---")

	r, err := kjarni.NewReranker(kjarni.WithQuiet(true))
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		os.Exit(1)
	}

	score, _ := r.Score("What is machine learning?", "Machine learning is a branch of AI.")
	fmt.Printf("  Score: %.4f\n", score)

	docs := []string{
		"Machine learning is AI.",
		"The weather is nice.",
		"Deep learning uses neural networks.",
	}
	ranked, _ := r.Rerank("What is machine learning?", docs)
	fmt.Println("  Reranked:")
	for _, rr := range ranked {
		fmt.Printf("    [%d] %.4f: %s\n", rr.Index, rr.Score, rr.Document)
	}

	topk, _ := r.RerankTopK("machine learning overview", docs, 1)
	fmt.Printf("  TopK=1: [%d] %.4f: %s\n", topk[0].Index, topk[0].Score, topk[0].Document)
	r.Close()

	fmt.Println()
	fmt.Println("--- Indexer + Searcher ---")

	testDir := filepath.Join(os.TempDir(), "kjarni_go_test_docs")
	os.MkdirAll(testDir, 0755)
	os.WriteFile(filepath.Join(testDir, "a.txt"), []byte("Machine learning is a branch of artificial intelligence."), 0644)
	os.WriteFile(filepath.Join(testDir, "b.txt"), []byte("Neural networks learn hierarchical representations of data."), 0644)
	os.WriteFile(filepath.Join(testDir, "c.txt"), []byte("The weather forecast calls for rain tomorrow."), 0644)

	indexPath := filepath.Join(os.TempDir(), "kjarni_go_test_index")
	os.RemoveAll(indexPath)

	idx, err := kjarni.NewIndexer("minilm-l6-v2", kjarni.WithQuiet(true))
	if err != nil {
		fmt.Printf("  ERROR creating indexer: %v\n", err)
		os.Exit(1)
	}

	stats, err := idx.Create(indexPath, []string{testDir})
	if err != nil {
		fmt.Printf("  ERROR indexing: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  Indexed: %d docs, %d chunks\n", stats.DocumentsIndexed, stats.ChunksCreated)
	idx.Close()

	searcher, err := kjarni.NewSearcher("minilm-l6-v2", "", kjarni.WithQuiet(true))
	if err != nil {
		fmt.Printf("  ERROR creating searcher: %v\n", err)
		os.Exit(1)
	}

	for _, mode := range []struct {
		name string
		mode kjarni.SearchMode
	}{
		{"Keyword", kjarni.Keyword},
		{"Semantic", kjarni.Semantic},
		{"Hybrid", kjarni.Hybrid},
	} {
		results, err := searcher.Search(indexPath, "machine learning", mode.mode)
		if err != nil {
			fmt.Printf("  ERROR %s search: %v\n", mode.name, err)
			os.Exit(1)
		}
		fmt.Printf("  %s:\n", mode.name)
		for _, sr := range results {
			fmt.Printf("    %.4f: %s\n", sr.Score, sr.Text)
		}
	}
	searcher.Close()

	os.RemoveAll(testDir)
	os.RemoveAll(indexPath)

	fmt.Println()
	fmt.Println("--- Chat: Blocking send ---")

	chat, err := kjarni.NewChat("llama3.2-1b-instruct", kjarni.WithQuiet(true))
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		os.Exit(1)
	}

	response, err := chat.Send("What is 2 + 2? Answer in one sentence.")
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("  Response: %s\n", response)
	fmt.Printf("  Model: %s\n", chat.ModelName())
	fmt.Printf("  Context: %d\n", chat.ContextSize())

	fmt.Println()
	fmt.Println("--- Chat: Streaming ---")

	var sb strings.Builder
	fmt.Print("  ")
	err = chat.Stream("Name three colors. One word each, comma separated.", func(token string) bool {
		fmt.Print(token)
		sb.WriteString(token)
		return true
	})
	if err != nil {
		fmt.Printf("\n  ERROR: %v\n", err)
		os.Exit(1)
	}
	fmt.Println()
	fmt.Printf("  Full: %s\n", sb.String())

	fmt.Println()
	fmt.Println("--- Chat: Greedy generation ---")

	greedy := kjarni.GreedyGenerationConfig()
	greedy.MaxNewTokens = 50

	fmt.Print("  ")
	err = chat.StreamWithConfig("What is the capital of France? One sentence.", &greedy, func(token string) bool {
		fmt.Print(token)
		return true
	})
	fmt.Println()

	fmt.Println()
	fmt.Println("--- Chat: Conversation ---")

	convo, err := chat.Conversation()
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		os.Exit(1)
	}

	r1, _ := convo.Send("My name is Bob. Remember that.")
	fmt.Printf("  Turn 1: %s\n", r1)
	fmt.Printf("  History: %d\n", convo.Length())

	r2, _ := convo.Send("What is my name?")
	fmt.Printf("  Turn 2: %s\n", r2)
	fmt.Printf("  History: %d\n", convo.Length())

	fmt.Println()
	fmt.Println("--- Chat: Conversation streaming ---")

	fmt.Print("  Turn 3: ")
	convo.Stream("Tell me a one-sentence joke.", func(token string) bool {
		fmt.Print(token)
		return true
	})
	fmt.Println()
	fmt.Printf("  History: %d\n", convo.Length())
	convo.Close()

	fmt.Println()
	fmt.Println("--- Chat: System prompt ---")

	pirate, err := kjarni.NewChat("llama3.2-1b-instruct",
		kjarni.WithQuiet(true),
		kjarni.WithSystemPrompt("You are a pirate. Speak like a pirate. Keep responses under 20 words."),
	)
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		os.Exit(1)
	}
	r3, _ := pirate.Send("What is machine learning?")
	fmt.Printf("  Pirate: %s\n", r3)
	pirate.Close()

	fmt.Println()
	fmt.Println("--- Chat: Early stop ---")

	tokenCount := 0
	fmt.Print("  ")
	chat.Stream("Write a long essay about computing.", func(token string) bool {
		fmt.Print(token)
		tokenCount++
		return tokenCount < 10
	})
	fmt.Printf("\n  (stopped after %d tokens)\n", tokenCount)

	fmt.Println()
	fmt.Println("--- Chat: Error handling ---")

	_, err = kjarni.NewChat("not-a-model", kjarni.WithQuiet(true))
	if err != nil {
		fmt.Printf("  Bad model: %v\n", err)
	}

	fmt.Println()
	fmt.Println("--- Chat: Dispose safety ---")

	tmp, _ := kjarni.NewChat("llama3.2-1b-instruct", kjarni.WithQuiet(true))
	tmp.Close()
	tmp.Close()
	_, err = tmp.Send("test")
	if err != nil {
		fmt.Printf("  Closed chat: %v\n", err)
	}

	chat.Close()

	fmt.Println()
	fmt.Println("--- Error handling ---")
	_, err = kjarni.NewClassifier("not-a-model", kjarni.WithQuiet(true))
	if err != nil {
		fmt.Printf("  Bad classifier: %v\n", err)
	}
	_, err = kjarni.NewEmbedder("not-a-model", kjarni.WithQuiet(true))
	if err != nil {
		fmt.Printf("  Bad embedder: %v\n", err)
	}

	fmt.Println()
	fmt.Println("--- Dispose safety ---")
	c2, _ := kjarni.NewClassifier("distilbert-sentiment", kjarni.WithQuiet(true))
	c2.Close()
	c2.Close()
	_, err = c2.Classify("test")
	if err != nil {
		fmt.Printf("  Closed classifier: %v\n", err)
	}

	e2, _ := kjarni.NewEmbedder("minilm-l6-v2", kjarni.WithQuiet(true))
	e2.Close()
	e2.Close()
	_, err = e2.Encode("test")
	if err != nil {
		fmt.Printf("  Closed embedder: %v\n", err)
	}

	fmt.Println()
	fmt.Println("=== ALL PASSED ===")
}