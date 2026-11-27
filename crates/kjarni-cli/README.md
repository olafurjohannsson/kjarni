<p align="center">
  <a href="https://kjarni.ai"><img src="https://raw.githubusercontent.com/kjarni-ai/kjarni/main/assets/kjarni-banner.png" alt="kjarni: local AI inference in a single native library" width="600"></a>
</p>

# Kjarni CLI

**Run transformer models from your shell. Same output as PyTorch.**

`kjarni` is the command-line build of Kjarni, a transformer inference library:
classification, embeddings, similarity, reranking, indexing and search, chat,
summarization, translation and speech to text, as ordinary commands that read stdin
and write text or JSON. `kjarni mcp` serves the same tools to AI agents over MCP.

```bash
curl -fsSL https://kjarni.ai/install.sh | sh     # Linux and macOS
irm https://kjarni.ai/install.ps1 | iex          # Windows
cargo install kjarni-cli                         # from source
```

# Commands

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

### With a model folder
```bash
kjarni chat --model-path ./Qwen2.5-0.5B-Instruct
```
Any Llama, Qwen2, Qwen 3.5, Mistral, Phi-3 or GPT-2 folder from the Hugging Face Hub, formatted
with its own chat template. `kjarni generate --model-path` takes the same folders.

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

### With a model folder
```bash
kjarni summarize -i article.txt --model-path ./distilbart-cnn-12-6
kjarni translate -i "Where is the library?" --model-path ./opus-mt-en-ROMANCE --src en --dst es
kjarni transcribe meeting.wav --model-path ./whisper-base.en
```
A BART or T5 folder summarizes, a Marian (`Helsinki-NLP/opus-mt-*`) or T5 folder
translates, and a Whisper folder transcribes. Weights may be safetensors or
`pytorch_model.bin`. A Marian folder translates the languages its
`tokenizer_config.json` names.

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

## HTTP server

`kjarni serve` serves every task over HTTP: chat, completions, embeddings and
speech-to-text in OpenAI's format, reranking in Cohere's and Jina's, and
classification, translation, summarization and entity recognition in Kjarni's own.

```bash
kjarni serve                                    # http://127.0.0.1:8080/v1
kjarni serve --chat qwen2.5-1.5b-instruct --preload
kjarni serve --host 0.0.0.0 --api-key secret    # accept other machines
```

```bash
curl http://127.0.0.1:8080/v1/classify -H "Content-Type: application/json" \
  -d '{"input": "The shoes run a full size small.", "labels": ["sizing", "shipping", "price"]}'

curl http://127.0.0.1:8080/v1/rerank -H "Content-Type: application/json" \
  -d '{"query": "capital of France", "documents": ["Paris is the capital of France.", "Bananas are yellow."]}'

curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@speech.wav -F response_format=srt
```

Each request names a model with `"model"`, or gets the endpoint's default:
`--chat`, `--embed`, `--classify`, `--zero-shot`, `--rerank`, `--transcribe`,
`--translate`, `--summarize` and `--ner` set them. `GET /v1/models` lists every
model and the endpoint it serves.

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

### Zero-shot classification with labels you choose
```bash
kjarni classify -m minilm-zeroshot --labels "sizing,shipping,price" \
  "The shoes run a full size small."
#   ✓         sizing  ██████████████████░░   91.8%
#              price  █░░░░░░░░░░░░░░░░░░░    6.6%
#           shipping  ░░░░░░░░░░░░░░░░░░░░    1.6%

# One text per line; --multi-label judges each label on its own, and
# --hypothesis changes the sentence each label is put into
kjarni classify -q -m minilm-zeroshot --multi-label \
  --hypothesis "This review is about {}." \
  --labels "sizing,shipping,price,quality,customer service" < reviews.txt
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