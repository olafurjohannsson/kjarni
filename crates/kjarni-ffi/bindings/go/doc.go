// Package kjarni provides text classification, embeddings, semantic search,
// reranking, chat, speech to text, summarization, translation and image search
// using pre-trained models, running inside your process against a native
// library. Models download automatically on first use
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
//
// # Beyond text embeddings
//
//	tr, _ := kjarni.NewTranscriber("", kjarni.WithLanguage("en"))
//	transcript, _ := tr.TranscribeFile("meeting.wav")
//
//	sum, _ := kjarni.NewSummarizer("")
//	summary, _ := sum.Summarize(article)
//
//	tl, _ := kjarni.NewTranslator("", kjarni.WithLanguages("en", "de"))
//	german, _ := tl.Translate("The house is small.")
//
//	photos, _ := kjarni.NewImageIndex("")
//	photos.AddDirectory("Pictures")
//	hits, _ := photos.Search("a red car", 5)
package kjarni