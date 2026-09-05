// Package kjarni provides text classification, embeddings, semantic search,
// and reranking using pre-trained transformer models, running inside your
// process against a native library. Models download automatically on first use
// and are cached on disk, so everything after that works offline.
//
// The same engine ships for other runtimes: Kjarni on NuGet for C#, kjarni-wasm
// on npm for the browser, and a command line binary. Documentation and guides
// live at https://kjarni.ai and the source is at
// https://github.com/olafurjohannsson/kjarni
//
// # Getting started
//
//	import "github.com/olafurjohannsson/kjarni-go"
//
//	emb, err := kjarni.NewEmbedder("minilm-l6-v2")
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer emb.Close()
//
//	vec, err := emb.Encode("hello world")
package kjarni