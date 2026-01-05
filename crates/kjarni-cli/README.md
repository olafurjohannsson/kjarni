# Kjarni CLI Commands

## Model Management

### List available models
```bash
# List all models
kjarni model list

# Filter by architecture
kjarni model list --arch encoder
kjarni model list --arch decoder
kjarni model list --arch encoder-decoder
kjarni model list --arch cross-encoder
```

### Download a model
```bash
kjarni model download llama-3.2-1b
kjarni model download minilm-l6-v2
```

### Show model info
```bash
kjarni model info llama-3.2-1b
```

### Search for models
```bash
kjarni model search llama
kjarni model search summarize
```

---

## Text Generation

### Basic generation
```bash
kjarni generate "Once upon a time"
```

### With options
```bash
kjarni generate "The meaning of life is" \
  --model llama-3.2-3b \
  --max-tokens 200 \
  --temperature 0.8
```

### Greedy decoding (deterministic)
```bash
kjarni generate "The capital of France is" --greedy
```

### Creative writing
```bash
kjarni generate "Write a poem about" \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-tokens 200
```

### From file
```bash
kjarni generate prompt.txt --max-tokens 500
```

### GPU acceleration
```bash
kjarni generate "Hello" --gpu
```

### Quiet mode (for piping)
```bash
echo "Explain quantum computing:" | kjarni generate -q
```

---

## Chat

### Interactive chat
```bash
kjarni chat
```

### With custom model
```bash
kjarni chat --model llama-3-8b-instruct
```

### With system prompt
```bash
kjarni chat --system "You are a pirate. Respond in pirate speak."
```

### Chat commands
```
> /help           Show available commands
> /quit           Exit chat
> /clear          Clear conversation history
> /system         Show current system prompt
> /system <text>  Set new system prompt
> /history        Show conversation history
```

---

## Summarization

### Basic summarization
```bash
kjarni summarize article.txt
```

### From stdin
```bash
cat long_document.txt | kjarni summarize
```

### With options
```bash
kjarni summarize article.txt \
  --max-length 100 \
  --min-length 30 \
  --num-beams 6
```

### GPU acceleration
```bash
kjarni summarize large_doc.txt --model bart-large-cnn --gpu
```

### Quiet mode for scripting
```bash
cat article.txt | kjarni summarize -q > summary.txt
```

---

## Text Encoding (Embeddings)

### Encode text
```bash
kjarni encode "Hello world"
```

### Encode from file
```bash
kjarni encode document.txt
```

### Batch encoding from stdin
```bash
echo -e "First text\nSecond text\nThird text" | kjarni encode
```

### Output formats
```bash
kjarni encode "text" --format json    # Full JSON with metadata
kjarni encode "text" --format jsonl   # JSON lines
kjarni encode "text" --format raw     # Space-separated floats
```

### Custom pooling and normalization
```bash
kjarni encode "text" --pooling cls --normalize false
kjarni encode "text" --pooling mean --normalize true
```

### Different model
```bash
kjarni encode "text" --model mpnet-base-v2
```

---

## Semantic Similarity

### Compare two texts
```bash
kjarni similarity "The cat sat on the mat" "A feline was sitting on a rug"
```

### Compare files
```bash
kjarni similarity doc1.txt doc2.txt
```

### Quiet mode (output score only)
```bash
kjarni similarity "text1" "text2" -q
# Output: 0.8234
```

---

## Reranking

### Rerank documents
```bash
kjarni rerank "machine learning" doc1.txt doc2.txt doc3.txt
```

### From stdin (one document per line)
```bash
echo -e "Python is great\nRust is fast\nJava is verbose" | \
  kjarni rerank "systems programming"
```

### Top-K results
```bash
kjarni rerank "query" doc1.txt doc2.txt doc3.txt --top-k 2
```

### Output formats
```bash
kjarni rerank "query" docs/*.txt --format json
kjarni rerank "query" docs/*.txt --format text
kjarni rerank "query" docs/*.txt --format docs  # Just documents, for piping
```

---

## Indexing

### Create an index
```bash
kjarni index create ./docs.idx ./documents/
```

### Index multiple paths
```bash
kjarni index create ./code.idx src/ lib/ tests/
```

### Custom chunking
```bash
kjarni index create ./large.idx ./books/ \
  --chunk-size 2000 \
  --chunk-overlap 400
```

### Add to existing index
```bash
kjarni index add ./docs.idx ./new-documents/
```

### Show index info
```bash
kjarni index info ./docs.idx
```

---

## Search

### Hybrid search (default)
```bash
kjarni search ./docs.idx "how to deploy kubernetes"
```

### Keyword-only (BM25)
```bash
kjarni search ./docs.idx "deployment error" --mode keyword
```

### Semantic search
```bash
kjarni search ./docs.idx "container orchestration" --mode semantic
```

### Limit results
```bash
kjarni search ./docs.idx "query" --top-k 5
```

### Output formats
```bash
kjarni search ./docs.idx "query" --format json
kjarni search ./docs.idx "query" --format docs  # Just text, for piping
```

---

## Advanced Pipelines

### Semantic grep (find relevant lines)
```bash
cat README.md | kjarni rerank "installation" -k 3 --format docs
```

### RAG pipeline
```bash
# Search, rerank, then generate
kjarni search ./knowledge.idx "deployment" -k 10 --format docs | \
  kjarni rerank "kubernetes deployment" -k 3 -q --format docs | \
  kjarni generate "Based on this context, explain:" -q
```

### Multi-document summarization
```bash
for f in articles/*.txt; do
  kjarni summarize "$f" -q
done | kjarni rerank "key findings" -k 1 --format docs
```

### Build embeddings for external use
```bash
cat texts.txt | kjarni encode --format raw > embeddings.txt
```

### Zero-shot classification via reranking
```bash
echo -e "positive sentiment\nnegative sentiment\nneutral" | \
  kjarni rerank "This product is amazing!" -k 1 --format docs
# Output: positive sentiment
```

### Find similar content
```bash
kjarni search ./docs.idx "$(cat target.txt)" --mode semantic -k 5
```

### Batch processing
```bash
# Summarize all articles
find articles/ -name "*.txt" -exec sh -c \
  'kjarni summarize "$1" -q > summaries/$(basename "$1")' _ {} \;
```

### Full RAG with chat
```bash
CONTEXT=$(kjarni search ./brain.idx "$QUERY" -k 3 -q --format docs)
kjarni chat --system "Answer using only this context: $CONTEXT"
```

---

## Global Options

| Option | Description |
|--------|-------------|
| `--gpu` | Use GPU acceleration (WGPU) |
| `-q, --quiet` | Suppress status messages |
| `-m, --model` | Specify model name |
| `--model-path` | Use local model directory |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `KJARNI_CACHE_DIR` | Model cache directory (default: `~/.cache/kjarni/`) |
| `RUST_LOG` | Logging level (e.g., `info`, `debug`) |