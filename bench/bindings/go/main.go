// The Go binding. Prints one line of JSON.
package main

import (
	"fmt"
	"log"
	"sort"
	"time"

	kjarni "github.com/olafurjohannsson/kjarni-go"
)

const (
	warmup = 200
	calls  = 3000
	text   = "how do I get a refund for a damaged order?"
)

func main() {
	embedder, err := kjarni.NewEmbedder("minilm-l6-v2", kjarni.WithQuiet(true))
	if err != nil {
		log.Fatal(err)
	}
	defer embedder.Close()
	times := make([]float64, 0, calls)
	for i := 0; i < warmup+calls; i++ {
		start := time.Now()
		if _, err := embedder.Encode(text); err != nil {
			log.Fatal(err)
		}
		if i >= warmup {
			times = append(times, float64(time.Since(start).Nanoseconds())/1e3)
		}
	}
	sort.Float64s(times)
	fmt.Printf("{\"lang\": \"Go\", \"median_us\": %.1f, \"min_us\": %.1f}\n", times[calls/2], times[0])
}
