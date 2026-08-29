# Kjarni for agents

Kjarni is a single native binary that runs transformer models locally. No Python,
no ONNX Runtime, no CUDA, no model server, no API key. The only runtime dependency
is glibc.

If you are an AI agent with shell access, this gives you transcription,
classification, embeddings, reranking, semantic search, summarization, translation
and local LLM chat as ordinary commands you can pipe and chain.

## Install

```bash
curl -fsSL https://kjarni.ai/install.sh | sh     # Linux / macOS
irm https://kjarni.ai/install.ps1 | iex          # Windows
```

## Download models before you need them

Models download on first use. **A first run can therefore pull hundreds of
megabytes and may exceed a tool-call timeout.** Pre-warm the ones you intend to
use:

```bash
kjarni model download minilm-l6-v2        #  88 MB  embeddings
kjarni model download distilbert-sentiment # 257 MB  sentiment
kjarni model download whisper-small        # 925 MB  speech-to-text
kjarni model list                          # everything available, and what is cached
```

## Commands

Every command reads from stdin where it makes sense and writes to stdout. Add
`-q` to suppress progress output so only the result is emitted.

### Transcribe audio

```bash
kjarni transcribe -q recording.mp3
kjarni transcribe -q --timestamps interview.wav
kjarni transcribe -q --translate icelandic.wav      # translate to English
```

Accepts wav, mp3, flac, ogg. Roughly realtime on CPU with `whisper-small`.

### Classify text

```bash
kjarni classify -q --format json "The build keeps failing and I am losing patience"
echo "great work" | kjarni classify -q --format json
kjarni classify -q -m toxic-bert --format json "$COMMENT"
kjarni classify -q -m distilroberta-emotion --format json "I just got promoted"
```

Emits `{"label": "...", "label_index": N, "score": 0.98, ...}`. Use
`--threshold` to drop low-confidence predictions, `--top-k` to limit output.

### Embeddings and similarity

```bash
kjarni embed -q "text to embed"                    # space-separated floats
kjarni similarity -q "doctor" "physician"          # single float, e.g. 0.859813
```

`similarity` is the one to reach for when you want a yes/no semantic comparison
without handling vectors yourself.

### Rerank candidates by relevance

```bash
kjarni rerank -q --format json "what is machine learning?" \
  "Machine learning is a subset of AI." \
  "The weather today is sunny."
```

Higher score means more relevant. Useful for picking the best of several
candidate answers, files, or search hits.

### Index and search a directory

```bash
kjarni index create ./notes --name mynotes
kjarni search -q --format json mynotes "how do refunds work?" --mode hybrid
```

Modes: `hybrid` (default), `semantic`, `keyword`. Add `--rerank-model` for
cross-encoder reranking of the results.

### Summarize and translate

```bash
cat long-report.txt | kjarni summarize -q
kjarni translate -q --src en --dst de -i "Where is the station?"
```

### Text generation and chat

```bash
kjarni generate -q -m llama3.2-3b-instruct -n 200 "Explain RAG in one paragraph"
kjarni chat --model llama3.2-3b-instruct        # interactive
```

## Output formats

`classify`, `rerank` and `search` accept `--format json` (also `jsonl`, and
`docs` for rerank). `embed` writes raw floats. `similarity` writes a single
float. `transcribe` writes plain text.

Pair `-q` with `--format json` and pipe into `jq`:

```bash
kjarni classify -q --format json "$TEXT" | jq -r '.label'
```

## Notes

- Everything runs locally. No data leaves the machine, and no network is used
  after the model is cached.
- Set `KJARNI_CACHE_DIR` to relocate the model cache; `HF_TOKEN` for gated models.
- Add `--gpu` to use a GPU where supported. GPU is verified for the encoder
  models (embeddings, classification, reranking). Decoder and encoder-decoder
  models on GPU are less well tested — prefer CPU for `chat`, `generate` and
  `transcribe` unless you have checked the output.
- Model names come from `kjarni model list`; that is the authoritative registry.

MIT licensed. Source: https://github.com/olafurjohannsson/kjarni
