# Changelog

## 0.8.1

- **Faster prompts on the GPU for quantised models.** Q4_K, Q5_K, Q6_K and
  Q8_0 models read the prompt on the GPU's tensor cores in f16, as llama.cpp
  does: on an RTX A2000, a 512-token prompt runs 1.8–1.9x faster (Qwen2.5-0.5B
  2576 tokens/s, Llama 3.2 1B 1313, Qwen3.5-4B 436) and a 4096-token one
  1.1–1.4x. Quantised models also keep their KV cache in f16 on the GPU.
  `KJARNI_GPU_EXACT=1` turns both off, for output that matches the CPU.
  Models with f32, f16 or bf16 weights are unchanged.
- **GPU memory is freed with the model.** A GPU context and the buffers it
  reused held on to each other, so dropping a model on the GPU kept its memory
  until the process exited.
- **Q5_K and Q6_K linear layers move to the GPU.** A Q5_K layer was refused
  and a Q6_K layer gave NaN when moved on its own.
- **wgpu 30.**
- **Long prompts on the GPU.** A prompt that added thousands of tokens at once,
  such as a large tool result in the agent, could stop the daemon with "Buffer
  binding exceeds max_storage_buffer_binding_size": the attention scores for
  the whole prompt did not fit in one buffer. They are now computed a slice of
  the prompt at a time, with the same result. A long prompt also goes to the
  GPU 512 tokens at a time, so it no longer stops with "timed out after
  30 s": 7,000 tokens on Qwen3.5-4B take about two minutes on an RTX A2000.
- **A failing model no longer leaves a conversation stuck.** If the model
  panics during a turn, the turn ends with an error saying why, and the
  conversation takes the next message.

## 0.8.0

- **Whisper no longer drops speech after a window that ends early.** When
  Whisper finished a 30-second window before its end, the speech after the
  last segment was skipped. That part is now transcribed again on its own,
  in files and in streaming, unless it is silence.
- **Send a message while Kjarni works.** It waits above the message box as
  "next step", and the agent reads it before its next step, after the results
  of the calls it made, so "only the first item" or "answer in Icelandic"
  steers a turn already running. A message that arrives as the answer is
  written gets an answer of its own in the same turn. Stopping the turn
  puts the unread messages back in the message box. In the API, a message
  sent mid-turn returns `{"queued": true}` instead of 409, with a `queued`
  event, then a `user` event when the agent reads it. In the library,
  `AgentEvents::interjections`.
- **Sign another device in with a code.** Settings has an "Other devices"
  section with the addresses the daemon answers to (such as its Tailscale
  name) and a button that shows an eight-digit code. Typed on the other
  device's sign-in page, it signs that browser in once, within two minutes;
  ten wrong codes cancel the pending ones. The token file still works. Also
  `POST /api/device-code`, and `POST /login` with `{"code"}`.
- **A context meter.** A small bar above the message box shows how full the
  conversation's context is; opened, it says how many tokens of how many,
  with a button to summarize the earlier messages now instead of waiting.
  Also `POST /api/conversations/{id}/compact`, a `context` event after each
  turn, and `context` in `GET /api/conversations/{id}`. In the library,
  `Agent::context_usage` and `Agent::compact`.
- **Decline a call and say what to do instead.** Declining in the web UI now
  asks "What should Kjarni do instead?"; the answer goes to the model with the
  decline, so "put it in notes/ instead" steers it rather than only stopping
  it. Leave it empty for a plain decline. `kjarni agent` asks the same after a
  "no". In the API, `POST .../approvals/{id}` takes an optional `reason` with
  `deny`, and the `approved` event carries it. In the library, a new
  `Approval::DenyWith(String)`; `Approval` is no longer `Copy`.
- **See every file a conversation changed.** A "files changed" button above
  the message box opens a panel with each file the conversation wrote or
  edited and its diff from before the conversation first touched it. The list
  is saved beside the conversation and deleted with it. Files changed by
  commands the agent ran are not listed. Also at
  `GET /api/conversations/{id}/changes`, with a `file_changed` event as each
  write lands. In the library, tools can say which file a call writes
  (`Tool::writes`), and `AgentEvents::file_changed` reports the content before
  and after.
- **The agent can transcribe recordings and find photos.** Two new tools,
  `transcribe` (Whisper, timestamped) and `search_images` (CLIP: "a receipt",
  "a dog on a beach"), the same as the MCP server's. Each loads its model on
  its first call and runs on the CPU.

## 0.7.0

The daemon, its web UI and schedules are a preview: they work and are tested,
but the API and the settings files may still change.

- **Kjarni declares the Rust it needs.** Every crate now says
  `rust-version = "1.90"`, the oldest Rust the GPU library builds with, so an
  older compiler gets a plain message naming the version instead of an error
  from a dependency.
- **Schedules in the daemon.** A schedule is a message the daemon sends
  itself at set times: every day at 07:30, on chosen weekdays, or every few
  minutes or hours. Each run is a new conversation in the web UI's list.
  Nobody is there to approve its calls, so a schedule says up front which
  actions it may take (open web pages, change files, run commands, update
  memory); anything else it tries is declined, and reading files and notes
  always works. A run missed while the computer was off happens once when
  the daemon is back. Make and edit them on the new Schedules page, or
  through `/api/schedules`.
- **Switch the daemon's model from the web UI.** The new Settings page lists
  the models the agent can run (the Qwen 3.5 sizes, from 0.8B to 35B-A3B),
  which are downloaded, and which one is running. Picking another downloads
  it if needed, waits for replies in progress to finish, unloads the old
  model and loads the new one, holding new messages and schedules until it
  is ready; if it will not load, the old one is loaded again. The choice is
  saved in the daemon's folder and loaded at the next start, unless
  `daemon run` is given `--model`. `daemon install` saves its model there
  too instead of putting `--model` in the startup entry. Also at
  `GET /api/models` and `PUT /api/models/current`.
- **A Memory page.** Everything Kjarni remembers, newest first, with a search
  once the list grows. Add a memory by hand, or forget one that is wrong.
- **Phone notifications through ntfy.** When a schedule finishes, the
  daemon can publish its reply to an ntfy topic you choose, on ntfy.sh or
  your own server, with an access token if it needs one. Tapping the
  notification opens the run. Set it up on the Schedules page, which can
  send a test.
- **The daemon's web UI.** `kjarni daemon open` now opens a chat page served
  by the daemon itself, built into the binary and working with no internet
  connection: conversations listed on the side, replies streamed in, each
  tool call shown as a step with its result folded under it, and calls that
  change files, run commands or go online waiting for Allow once, Always
  allow or Decline in the page. Edits show as the lines they take out and
  put in. It works on a phone as on a laptop. A browser on another device
  signs in with the daemon's token. The page sanitizes model output and
  sends a Content-Security-Policy that lets it load and talk to nothing but
  the daemon, and refuses to be framed by other sites.
- **`kjarni daemon run --allow-host <name>`** (and `install`). The daemon
  also answers to that name, over HTTP or HTTPS, so `tailscale serve` can
  put it on your tailnet while it still listens on 127.0.0.1 only. A request
  naming any other host is refused with the `--allow-host` that would allow
  it.
- **`kjarni daemon run`.** The agent as a background service with a local
  HTTP API: conversations, their events streamed as server-sent events (reply
  text, tool calls, results), approvals answered over HTTP, stopping a turn,
  and the agent's memory. The OpenAI-compatible endpoints of `kjarni serve`
  sit beside it on the same loaded model, taking turns with the agent. It
  listens on 127.0.0.1 only unless given `--listen`, and every request needs
  the token in its folder: as a bearer token, or as the cookie a browser gets
  from the one-time link `kjarni daemon open` opens. Requests naming another
  host or sent from another web origin are refused. `kjarni daemon status`
  shows what it runs. The agent works in `~/kjarni` unless given `--dir`.
- **`kjarni daemon install`.** Starts the daemon at every login, as the user,
  with no admin rights: a systemd user unit on Linux (kept running after
  logout with `loginctl enable-linger` unless `--no-linger`), a LaunchAgent
  on macOS, a logon task on Windows. Each restarts the daemon after a crash.
  Without `--model` it picks a Qwen 3.5 model for the machine from free GPU
  memory and system memory, and downloads it before starting. `start`,
  `stop`, `uninstall` (`--purge` also deletes the token, logs, memory and
  conversations) and `logs` (`-f` to follow) work the same on all three; the
  log is `logs/daemon.log` in the daemon's folder, rotated at 10 MB.
- **The agent answers a new message without re-reading the conversation.**
  Qwen 3.5's template drops the think blocks of earlier turns, which changed
  the prompt from the last reply on, so each new message re-read the previous
  exchange, tool output included. The agent now keeps those blocks
  (`Conversation::set_preserve_thinking`), and the second message of a
  conversation reuses the whole cached prompt. Chat renders as Qwen's own
  template does.
- **Qwen 3.5 on the GPU.** The dense models run on the GPU with `--gpu`, in
  `kjarni generate`, `chat` and now `agent`: gated attention, the Gated DeltaNet
  convolution and delta rule, and their recurrent state, with snapshots so a
  prompt that shares its start with the last one reuses the cache as it does on
  the CPU. `qwen3.5-4b` (Q4_K_M) on an RTX A2000 decodes about 29 tokens/s
  against 7.2 on four CPU cores, and prefills a 1,500-token prompt in 8.4 s against
  66 s; an agent session of two messages and a tool call took 22 s against
  75 s. The 0.8B matches transformers within 1e-3 and continues both reference
  prompts token for token. The mixture-of-experts models stay on the CPU.
- **Q5_K stays packed on the GPU.** The linear layer reads Q5_K blocks as
  stored, in decode and prefill, where it expanded them to BF16 before: Q4_K_M
  files use Q5_K for some tensors (Qwen 3.5's `attn_qkv` and `ssm_out`). For
  `qwen3.5-4b` that is about 1 GB less VRAM, decode 11% faster and loading
  9.6 s down to 6.8 s, measured back to back with the build before.
- **Project notes.** The agent reads `KJARNI.md`, or else `AGENTS.md`, from the
  folder it starts in and from each folder above it up to the repository root,
  into its instructions (up to 16 KB): how the project builds, tests and is
  laid out. `kjarni::agent::project_notes` finds them for other front ends.

## 0.6.0

- **Models straight from Hugging Face.** Anywhere a model name goes, a Hub
  repository works too: `kjarni chat --model Qwen/Qwen3.5-9B`, a
  `https://huggingface.co/...` link, a GGUF repository such as
  `unsloth/Qwen3.5-9B-GGUF` (Q4_K_M unless `:Q8_0` or a link to one file says
  otherwise), in every command, in `kjarni model download`, and in the library
  builders the bindings use. `kjarni serve` starts with one (`--chat org/name`),
  and a request may name any the server has downloaded, but never makes it
  download a new one. The files go into the
  model cache once; after that the folder loads with no network. A GGUF repository
  takes its tokenizer and config from the model its card names as `base_model`. The
  small files come first, so a repository that cannot work, such as an encoder
  given to `chat`, is turned down before gigabytes of weights download. `HF_TOKEN`
  is sent for gated and private repositories, and `HF_ENDPOINT` points at a mirror.
- `kjarni embed` and `kjarni classify` load `--model-path` whatever `--model`
  says; they turned away a `--model` missing from the registry first.
- **`kjarni agent`, an assistant in the terminal that uses tools.** It reads,
  searches, writes and edits files in the folder it is started in, runs commands,
  reads web pages, and remembers facts between conversations (`remember`,
  `recall`, `forget`; recalled by meaning with `minilm-l6-v2`, by words without
  it). Reading is never asked about; writing, running and going online ask first
  (yes, always, no), or never with `--yes`. Conversations are saved, and continue
  with `-c` or `--resume <id>`; `-p` answers one prompt and exits. When a
  conversation outgrows 16k tokens, its older part is folded into a summary the
  model writes. Defaults to `qwen3.5-4b` from GGUF. The same agent is a library
  module, `kjarni::agent`, with the `Tool` trait for adding tools of your own.
- **The agent searches documents by meaning.** `search_documents` finds the
  passages in the working folder, or a folder inside it, that answer a question,
  by meaning and keywords together, reranked by a cross-encoder
  (`minilm-l6-v2-cross-encoder`). A folder is indexed on its first search and only
  changed files are embedded after that; a folder inside an indexed one uses that
  index. A folder with more than 10,000 documents, such as a whole home folder, is
  refused with the subfolders that hold the most, so the model searches one of
  those. Text and code files are read, not PDF or Word.
- **The agent reuses its prompt cache after `remember`.** The system prompt lists
  the memory's facts as they were when the conversation started, so saving a fact
  no longer makes the model read the whole conversation again; a forgotten fact
  still leaves at once. A list, remember and answer run went from 271 s to 112 s
  on four cores with `qwen3.5-4b`.
- **`kjarni::folder_search`.** The folder search behind `search_documents` and the
  MCP server's `search_folder`, as a library module. The two share their indexes,
  now kept in `folder-indexes` in the model cache; the old `mcp-indexes` folder
  there is no longer read and can be deleted.
- **Tool calling.** Qwen 3.5's chat template renders tools, calls and tool results
  as transformers renders them, and `parse_tool_calls` reads calls back in both
  Qwen 3.5's XML form and the JSON form of Qwen 2.5 and 3. Jinja templates that
  read `tools` get them.
- **Tool calling in `kjarni serve`.** `/v1/chat/completions` takes `tools`,
  assistant messages with `tool_calls`, and `tool` messages, and answers with
  `tool_calls` and `finish_reason: "tool_calls"`, streamed or not. Streamed, the
  call markup never reaches the client's text. `tool_choice` takes `"auto"` and
  `"none"`; forcing a call is refused, since it can't be guaranteed yet.
- **Faster K-quant kernels.** AVX2 dot products for Q5_K and Q6_K weights, and
  integer accumulation for Q4_K, each bit for bit the same as the scalar code.
  With a faster Gated DeltaNet recurrence, a 2,300-token prompt to the 4B is
  read in 21.7s instead of 24.
- **Qwen 3.5.** Qwen's hybrid decoder, with a Gated DeltaNet layer (linear
  attention with a fixed-size recurrent state) in three layers of every four and
  gated attention in the fourth. The registry has `qwen3.5-0.8b`, `qwen3.5-2b` and
  `qwen3.5-4b`; any dense Qwen 3.5 folder from the Hugging Face Hub loads with
  `--model-path`, the multimodal checkpoints' text decoder included. Logits match
  transformers to within 3e-5 on Qwen3.5-0.8B, and greedy output matches token for
  token. Chat uses the model's non-thinking mode, as its own template does by
  default. `--prefix-cache` works: the recurrent state is kept at the end of each
  prompt and just before it, which is where the next turn's prompt diverges, so a
  second turn after a 2,200-token system prompt takes 1.9s instead of 7.8s on the
  0.8B. CPU only for now; the GPU, images and speculative decoding are not
  supported yet, and speculative decoding says so.
- **Qwen 3.5 from GGUF.** `--gguf` downloads unsloth's Q4_K_M for `qwen3.5-0.8b`,
  `qwen3.5-2b` and `qwen3.5-4b`, and any Qwen 3.5 GGUF written by llama.cpp's
  converter loads with `--model-path`, including the newer files that merge
  `attn_qkv` with `attn_gate` and `ssm_beta` with `ssm_alpha`, and those that keep
  the multi-token-prediction block. The 4B Q4_K_M continues a 272-token prompt
  token for token as the f32 transformers model does, and peaks at 4.0 GB of
  memory instead of 9.7: the token table stays in its K-quant format and each
  token's row is dequantized as it is looked up. On the 0.8B, loading and generating 100 tokens takes
  6.7s instead of 11.8.
- **Qwen 3.5 mixture of experts.** `qwen3.5-35b-a3b`, and any Qwen 3.5 MoE
  folder or GGUF with `--model-path`. In every layer a router picks 8 of 256
  small experts per token, and a gated shared expert always runs. Experts load
  from transformers' per-expert or fused tensors and from llama.cpp's stacked
  `ffn_*_exps` tensors; a tiny Qwen 3.5 MoE checkpoint matches transformers'
  logits to within 1e-4 and its greedy output token for token, from
  safetensors and from GGUF. The unsloth Q4_K_M (22 GB) decodes at 5.9 tok/s on
  an i7-13700 whose memory reads about 17 GB/s: each token reads about 2.6 GB
  of weights, so that is within 10% of what the memory allows. For one token,
  all the chosen experts' rows go to the threads in one dispatch. CPU only.
- `qwen3.5-9b` and `qwen3.5-27b` in the registry, from safetensors or unsloth's
  Q4_K_M with `--gguf`.
- **Thinking mode for Qwen 3.5.** Off by default, as before. `kjarni chat
  --think`, `ChatBuilder::thinking(true)` or `Conversation::set_thinking` turn it
  on; the chat prints the reasoning dimmed before the answer. `kjarni serve`
  takes vLLM's `chat_template_kwargs: {"enable_thinking": true}` or OpenAI's
  `reasoning_effort` (anything but `"none"` turns it on), and returns the
  reasoning apart from the answer in `reasoning_content`, streamed or not.
- Qwen 3.5 decodes faster: the DeltaNet's short convolution runs one token
  without a task per channel. Qwen3.5-0.8B from GGUF goes from 28.8 to 31.8
  tok/s on an i7-13700.
- `ModelWeights` reads `name[start:end]`, rows `start..end` of a stacked tensor
  (every dimension but the last counting as rows), borrowed from the file where
  it is mapped, so a layer can be built on one expert of a stacked tensor.
- **Q5_K weights.** Q4_K_M files store some tensors as Q5_K, so until now no Q4_K_M
  Qwen 3.5 file could load. CPU only; a Q5_K tensor on the GPU is an error that
  says so.
- `kjarni generate --gguf` no longer downloads the model's safetensors when they
  are not cached; it reads only the GGUF.
- GGUF files whose tensor sizes are not all multiples of 32 bytes load. The start
  of the tensor data was computed from the sum of the tensor sizes, which falls
  short by the padding between tensors.
- `Cache::reusable_prefix`, for a cache that can only be cut back to some
  lengths. `PrefixCache` uses it, so a model with a recurrent state reuses what it
  can rather than reading state from the wrong position.

- Per-model settings in `kjarni.toml` apply when the section names the model as
  it is written, `[models."qwen3.5-4b"]`; only the dashed form
  `[models.qwen3-5-4b]` matched before. A model's `quantize_embeddings` setting
  is applied; it was read and ignored.
- **Phi-3.5 past 4,096 tokens.** LongRoPE now switches to its long factors once a
  forward pass reaches past the context the model was trained on, as
  transformers does; Kjarni used the short factors at every length. The trained
  length sits at the top level of Phi-3's config and was not passed on. A tiny
  Phi-3 checkpoint now matches transformers logit for logit on both sides of the
  switch. The GPU path still uses the short factors.
- `/v1/completions` names `prompt`, not `messages`, when a prompt is longer than
  the model's context.

## 0.5.1

- **`kjarni serve`, an OpenAI-compatible HTTP server.** The OpenAI client libraries
  and the tools built on them work against Kjarni by changing the base URL.
  `/v1/chat/completions` (streamed or whole, with `stop`, `max_tokens`,
  `temperature`, `top_p` and token usage), `/v1/completions`, `/v1/embeddings`
  (floats or base64), `/v1/audio/transcriptions` and `/v1/audio/translations`
  (JSON, text, SRT and WebVTT) and `/v1/models` follow OpenAI's format;
  `/v1/rerank` follows Cohere's and Jina's; `/v1/classify` (including zero-shot
  with `"labels"`), `/v1/translate`, `/v1/summarize` and `/v1/ner` are Kjarni's own.
  Requests name a registry model or get the endpoint's default, set with flags such
  as `--chat`. Models load on their first request, or at start with `--preload`.
  `--api-key` (or `KJARNI_API_KEY`) guards the API, `--cors` lets web pages call it,
  and `--prefix-cache` reuses earlier turns of a conversation. Tool calling, JSON
  mode and log probabilities are refused with an error rather than ignored. The
  responses match Kjarni's library exactly, and through it transformers.
- New in the Rust API, for the server and anyone building one:
  `Chat::render_messages` and `Chat::reply_prefix`, `Generator::count_tokens` and
  `Generator::max_new_tokens`, `Embedder::count_tokens`, and
  `Transcriber::with_settings` for one job's language, task and timestamps without
  reloading the model.
- **Zero-shot classification.** Classify text into labels you choose at run time,
  with no training: `kjarni classify -m minilm-zeroshot --labels "sizing,shipping"`
  and `Classifier::zero_shot`, `zero_shot_with` and `zero_shot_batch` in Rust, with
  `ZeroShotOptions` for the hypothesis template (default `"This example is {}."`)
  and for judging each label on its own (`--multi-label`, `--hypothesis` in the
  CLI). Any natural-language-inference model works, from the registry or a folder
  with an entailment label. The new registry model `minilm-zeroshot` is
  `cross-encoder/nli-MiniLM2-L6-H768` (82M parameters). Scores match transformers
  4.57 `pipeline("zero-shot-classification")`.

## 0.5.0

- **Summarize, translate and transcribe with any model folder.** `--model-path` on
  `kjarni summarize`, `translate` and `transcribe`, `model_path(..)` on the Rust
  `Summarizer`, `Translator`, `Transcriber` and `Seq2SeqGenerator` builders,
  `model_path` in Python and in the C configs, `modelPath` in C# and `WithModelPath`
  in Go load a Hugging Face folder instead of a registry model. The family comes
  from `model_type` in `config.json`: BART and T5 summarize, Marian (the
  Helsinki-NLP `opus-mt` models) and T5 translate, and Whisper transcribes,
  including the English-only `.en` models. A folder of another family is refused by
  name. A Marian folder translates the languages its `tokenizer_config.json` names;
  one with several targets, such as `opus-mt-en-ROMANCE`, is given the target as its
  `>>fr<<` token, and a code the language table does not know, such as `fr_BE`, can
  be passed as is. Output matches transformers 4.57 for `opus-mt` en-de, de-en,
  en-ROMANCE, is-en and en-is, `distilbart-cnn-12-6`, `flan-t5-base` and Whisper
  tiny, base, small, tiny.en and base.en. `kjarni rerank --model-path` works too.

- **The Go module builds again.** `kjarni-go` v0.4.3 and v0.4.4 failed to compile on
  every platform with `openLibrary redeclared`: publishing copied the new
  `dl_unix.go` and `dl_windows.go` over the repository but left the old
  `lib_unix.go` and `lib_windows.go`. Publishing now mirrors the Go files, the
  examples and the libraries, and builds and tests the tree it pushes.

- **Weights load from `pytorch_model.bin`.** Many Hub models, all Helsinki-NLP
  translation models among them, ship only PyTorch's pickle format. Kjarni reads it
  directly, both the zip format of PyTorch 1.6 and later and the older legacy
  format, and sharded checkpoints with `pytorch_model.bin.index.json`. The pickle is
  read by a restricted unpickler that accepts only the few names a state dict uses,
  such as `collections.OrderedDict` and `torch._utils._rebuild_tensor_v2`, so a file
  cannot run code. Safetensors is still preferred when a folder has both.

- **Marian tokenizers load from `source.spm` and `vocab.json`.** Ids match
  `MarianTokenizer`, including the `>>xx<<` language tokens.

- **Attention with a bias on only some projections.** The fused query/key/value
  projection, used on the CPU for hidden sizes of 512 or less, dropped all three
  biases when one projection had none, and the decoder's fused key/value projection
  did the same. Whisper tiny and base, whose key projection has no bias, gave wrong
  text through it; their encoder output now matches PyTorch.

- **Icelandic is a known language for translation** (`is`, `icelandic`,
  `íslenska`).

- **Breaking, C API: `KjarniSummarizerConfig`, `KjarniTranslatorConfig` and
  `KjarniTranscriberConfig` end with a new `model_path` field.** Code that fills
  these from `kjarni_*_config_default()` only needs recompiling; bindings that
  declare the structs themselves must add the field.

- **Breaking: `model_type()` on `Summarizer`, `Translator` and `Seq2SeqGenerator`
  returns `Option<ModelType>`**, `None` for a model loaded from a folder.
  `Seq2SeqGeneratorBuilder::from_path` is now `model_path`. The `Summarizer` and
  `Translator` builders gain `with_load_config`.

- **Named-entity recognition.** `kjarni ner`, `kjarni::ner::EntityRecognizer` and
  `EntityRecognizer` in Python, C# and Go find people, places, organizations and other
  entities. The default model is `distilbert-ner`, for English; `--model-path` (or
  `model_path`) loads any `*ForTokenClassification` model, such as the Icelandic
  `vesteinn/IceBERT-ner` or the multilingual
  `Davlan/distilbert-base-multilingual-cased-ner-hrl`. Token labels match PyTorch, and
  entities match transformers' `pipeline("ner")` for each `aggregation_strategy`:
  `first` (the default, which keeps words whole), `simple`, `max` and `average`.
  Spans count characters in Rust, Python and C, UTF-16 units in C# and bytes in Go, so
  each language can slice its own string with them.

- **Python, C# and Go load models from a folder.** `Embedder`, `Classifier` and
  `Reranker` take `model_path` in Python, `modelPath` in C# and `WithModelPath` in Go,
  as the CLI's `--model-path` already did. In Python, `Classifier` no longer needs a
  model name when given a folder.

- **`WithCacheDir` works for the Go `Embedder`, `Classifier` and `Reranker`.** They
  did not pass it to the library, so models went to the default cache instead.

- **Models without `tokenizer.json` load.** Many models on the Hugging Face Hub ship
  only the files their Python tokenizer reads: `vocab.json` and `merges.txt` (GPT-2,
  RoBERTa, BART, Longformer) or `vocab.txt` (BERT, DistilBERT, ELECTRA, MPNet).
  Kjarni now builds the tokenizer transformers would build from them, with the same
  token ids, so a folder such as IceBERT, `unitary/toxic-bert` or
  `distilbert-base-uncased-finetuned-sst-2-english` works with `--model-path` and
  `from_pretrained` as downloaded. A folder that has `tokenizer.json` uses it
  exactly as before. Other tokenizers without one fail with a message saying how to
  save it, rather than guessing.

- **SentencePiece tokenizers load from their `.model` file.** XLM-RoBERTa, CamemBERT,
  T5 and mT5 (`sentencepiece.bpe.model`, `spiece.model`) and Llama 2 style models
  such as TinyLlama, Mistral and Phi-3 (`tokenizer.model`) work without
  `tokenizer.json`. The tokenizer is built as transformers 4.57 converts these files,
  following `legacy`, `add_prefix_space`, `add_bos_token` and `add_eos_token` in
  `tokenizer_config.json`, and gives the same ids and offsets. This covers models
  such as `intfloat/multilingual-e5-small`, `BAAI/bge-m3` and `google/flan-t5-base`.

- **XLM-RoBERTa and CamemBERT classifiers and NER models load.** They were read as
  BERT and failed with a missing `bert.embeddings` tensor. Models such as
  `Davlan/xlm-roberta-large-ner-hrl` now match PyTorch, and CamemBERT sentence
  encoders pool like RoBERTa ones.

- **Chat with any decoder model folder.** `kjarni chat --model-path`,
  `Chat::builder(..).model_path(..)`, `model_path` on the Python `Chat`, `modelPath`
  in C#, `WithModelPath` in Go and `model_path` in the C `KjarniChatConfig` load a
  Llama, Qwen2, Mistral, Phi-3 or GPT-2 folder from the Hugging Face Hub. The
  conversation is formatted by the folder's own Jinja `chat_template` (from
  `chat_template.jinja` or `tokenizer_config.json`), rendered as transformers
  renders it; a Kjarni download folder, which keeps no template file, uses the
  template of the registry model it holds. `kjarni generate --model-path` accepts
  such a folder too, as well as a `.gguf` file. A folder without a chat template, or
  of a model that does not generate text, is reported by name. In the C API the
  field was reserved and rejected; `model_name` may now be NULL when it is set.

- **Repetition penalty counts each token once.** A token seen several times was
  penalized once per occurrence; transformers penalizes it once. This changed the
  output of every model whose generation defaults set a penalty, such as Qwen 2.5.

- **Breaking: `Chat::model_type()` and `Generator::model_type()` return
  `Option<ModelType>`.** A model loaded from a folder has none.

- **Debug builds no longer print "Encoder buffers allocated" to stdout.** It is a
  debug log message now, so it no longer corrupts JSON output.

## 0.4.4

- **`kjarni mcp` finds photos by description.** The new `search_images` tool takes a
  folder and a description such as "a dog on a beach" and returns the best matching
  JPEG and PNG photos, with CLIP running locally. The first search of a folder embeds
  every photo, about 20 ms each on a desktop CPU, and later searches reuse those
  vectors until the folder changes. CLIP downloads on first use.

- **Folder search updates its index instead of rebuilding it.** `search_folder` in
  `kjarni mcp` embedded the whole folder again whenever any file changed. It now
  embeds only new and changed files and drops deleted ones. The library exposes this
  as `Indexer::sync`, which records each file's size and modification time per chunk.
  Indexes built by earlier versions have no such record and are re-embedded once.

- **Qwen prompts no longer start with a stray `<|endoftext|>`.** `GenerationConfig`
  defaults to adding a BOS token, and the CLI and the browser build on that default,
  so Qwen got one its tokenizer never adds, which changed its first logits. Whether a
  prompt starts with BOS is now the model's call: Llama and Mistral yes, Qwen, GPT-2
  and Phi no. `add_bos_token: false` still turns it off.

- **GPT-2 on the CPU matches HuggingFace.** The CPU prefill fed the last prompt token
  to the model twice, so CPU output differed from both transformers and Kjarni's own
  GPU path. Greedy gpt2 and distilgpt2 now match transformers token for token.

## 0.4.3

- **Transcription reads MP3, FLAC and Ogg Vorbis.** The documentation, the CLI help and
  the bindings all listed these formats, but releases decoded only WAV and failed on the
  rest with "requires the 'symphonia' feature". `TranscribeFile`, `transcribe_file`,
  `kjarni transcribe` and the MCP tool now decode them, at any sample rate, mono or stereo.

- **The Go module builds on Windows.** kjarni-go loaded its library through functions
  purego provides only on Linux and macOS, so `go build` failed on Windows with
  "undefined: purego.Dlsym". It now uses LoadLibrary there, and CI runs the Go tests
  on Windows against the DLL the module embeds.

- **`kjarni transcribe --timestamps` prints timestamps.** Without `--no-stream` the flag
  was ignored. It now waits for the full transcript and prints each segment with its
  start and end time.

- **`kjarni index create` reports how many documents it indexed.** The progress line
  counted documents before each batch was embedded, so a small index showed "Indexed 0
  documents". It now counts documents written, in the CLI and in progress callbacks
  from every binding.

- The Python `Reranker` docstring named a model, "cross-encoder-minilm", that the
  binding rejects; it now names the default, `minilm-l6-v2-cross-encoder`. The
  `--model-path` help for `kjarni transcribe` says plainly that it is not supported yet.

## 0.4.2

- **Chat replies no longer stop at accented letters and emoji.** A character the model
  writes as two or more byte tokens, such as the "Ó" in "Ólafur", ended the reply with
  "�", and nothing after it was generated. `Chat`, conversations, streaming, the CLI and
  the WebAssembly build were all affected. Speculative decoding printed such characters
  as "��"; it now prints them whole.

- **Image search indexes black-and-white JPEGs.** Greyscale JPEGs failed to decode and were
  skipped by `ImageIndex.AddDirectory` and `kjarni image`.

- **Faster on Apple Silicon and ARM Linux.** Quantized weights (Q8_0, Q4_K, Q6_K), RoPE and
  the norm layers have NEON kernels, where they ran as plain scalar code before. Decoding
  on the macOS CI runner went from 9.8 to 45.2 tokens per second. Embedding speed is
  unchanged.

## 0.4.1

- **`kjarni translate` uses a real translation model by default.** English to German and
  German to English now use Opus-MT, since the old default, `flan-t5-base`, returned
  questions and many other inputs unchanged. Other pairs still use `flan-t5-base` and print a
  warning on stderr. Choosing a model with `-m` works as before.

- **The bindings pick the same model.** A Python, C#, Go or C++ `Translator` with no model
  and default languages of English and German loads Opus-MT. It translates that pair only,
  so another pair on the same instance is now an error rather than a poor translation. With
  no default languages it still loads `flan-t5-base` and warns unless quiet.

- **`kjarni embed` returns unit-length vectors by default.** This matches the library
  bindings and sentence-transformers, so the same text gives the same vector from the CLI
  and from code. **Output changes for existing scripts**: pass `--no-normalize` to get the
  raw pooled vector. `--normalize` is still accepted.

- **`kjarni generate -m gpt2 --gpu` loads.** GPT-2 failed with "GPU device requires context"
  because its loader, unlike the other decoders', did not create one.

- **Each release is installed and run after publishing.** CI runs the website's install
  script on Linux, macOS and Windows, then checks the installed binary's embeddings,
  similarity, reranking, classification and translation against pinned values.

## 0.4.0

- **Long text translates completely.** Translation models end early when given several
  sentences at once, which left out most of a long text with no error. `Translator` and
  `kjarni translate` now translate each sentence on its own and keep the line breaks.
  A `max_length` setting applies to each sentence.

- **`kjarni search --format json` prints `[]` when nothing matches.** It printed nothing,
  which is not valid JSON. An unknown `--format` is an error in this case too.

- **Speculative decoding runs on the GPU, and is exact.** A draft model runs on the device
  it was loaded for, and the target checks a batch of tokens in one pass on the GPU too.
  Sampling now uses speculative sampling and follows the target's distribution, where
  it used to draft greedily and ignore temperature and top-k. Runs stop at `max_new_tokens`,
  and repetition penalties account for the proposed tokens, so greedy output equals plain
  decoding. The draft model continues from its cache instead of re-reading the whole text
  each round. `SpeculationParams::probabilistic` is removed, since the decoding strategy
  decides, and `DecoderGenerationBackend` gains a required `forward_batch` method. A draft
  model on the GPU can be slower than plain decoding, so measure before relying on one.

- **Faster GPU batches of 2 to 10 rows.** Linear layers read each weight once for all the
  rows. A batch of 5 through a 28-layer decoder takes about half the time it did.

- **Faster quantized decode on the GPU for most GGUF models.** Measured against the previous
  kernels on an RTX A2000, Llama 3.2 1B is about 14% faster and 3B up to 22%. Qwen 2.5 0.5B, whose
  file is mostly Q5_0 weights, is about 9% slower. On a release build, Llama 3.2 3B generates 46
  tokens a second and 1B 105, from a short prompt. Quantized batches of 2 to 10 rows use the same kernels, which makes speculation on a
  GGUF model up to 3.4x faster, though it remains slower than plain decoding on the GPU.

- **`--gguf --gpu` no longer refuses a model that fits.** With both formats cached, the VRAM
  check added them together, so `deepseek-r1-8b` asked for 23.4 GB when its GGUF needs 5.5 GB.

- **DeepSeek-R1 replies open with `<think>`.** The chat prompt supplies the tag, so replies
  showed only the closing `</think>`. `Chat` now returns the whole reply, streamed or not.

- **Prompt lookup.** `kjarni generate --prompt-lookup 3` and `.prompt_lookup(n)` propose tokens
  from earlier in the text, with no second model. This helps where the output repeats the
  input.

## 0.3.0

- **Index format 2, with correct updates.** Indexes from earlier versions must be
  rebuilt. Commits are atomic, a writer holds a lock, and small segments merge.
  `index add` and `Indexer::add` replace a file already in the index, and
  `index remove` and `Indexer::remove` take files and folders out.

- **Search fixes.** Source code and plain text are stored as written, and markdown
  cleanup keeps code and plain brackets. `chunk_overlap` takes effect. Keyword scores
  are comparable across segments, results are deterministic, and filters search the
  whole ranking. `force` and `Indexer::delete` refuse folders that are not indexes.
  Searching with a different model from the one that built the index is an error, and
  so is a threshold on hybrid search without a reranker.

- **`kjarni::Conversation` in `kjarni.hpp`.** `Chat::conversation()` starts a chat that
  remembers its turns, with `send`, `stream`, `clear` and `size`. C++ callers previously
  had to use the C API for this, because `Chat::send` and `Chat::stream` keep no history.

- **Opus-MT translation models.** `opus-mt-en-de` and `opus-mt-de-en` add the Marian
  architecture. Each is about 300 MB and translates one direction. On a set of 16
  sentences, the output matches transformers exactly on CPU and GPU. The GPU path now
  applies `final_logits_bias`, as the CPU path does, and the `swish` activation maps to
  SiLU instead of falling back to GELU.

## 0.2.9

### whisper-large-v3 works, and `--translate` asks for a translation

- **whisper-large-v3 failed to load**, with an unsupported dtype on the CPU and a
  panic on the GPU. Its weights are half precision only; they now load as f32, as
  they do in transformers. Other half-precision safetensors files load the same way.
- **Whisper's task and timestamp tokens were fixed numbers**, from whisper-small, and
  the translate one was wrong even there. `--translate` sent `<|startoflm|>`, which
  still came out in English but repeated itself and strayed from transformers. On
  large-v3, transcribing sent `<|translate|>` and streaming sent `<|nospeech|>`, which
  gave an empty result. They now come from each model's tokenizer, and translations
  of German speech match transformers on both models.
- **A panic while streaming a transcription looked like success**, with no text and
  exit code 0. It now reaches the caller as an error.
- **`kjarni index create` indexed hidden folders and build output**, unlike
  `search_folder`. It now skips the same folders. A file or folder named directly is
  always read.
- **`kjarni.hpp` failed to compile after `kjarni.h`**, the order the Qt search example
  used. It now compiles either way; image search is left out if `kjarni.h` came first
  without `KJARNI_IMAGE_IO`.

## 0.2.8

### Faster GPU generation, and Whisper on the GPU

- **Whisper runs on the GPU.** `Transcriber` with `device="gpu"` was an error. It
  now runs the encoder and decoder there, at 15.9x real time on an RTX A2000
  against 8.6x on an i7-13700's CPU, with the same transcript on the test clips.
- **Generation is about twice as fast on the GPU.** Llama 3.2 1B went from 32 to
  72 tokens a second after a 512-token prompt. Attention writes keys and values
  straight into the KV cache and reads them in place, which cuts each layer from
  over twenty dispatches to eight; readback blocks in the driver instead of polling
  on a sleep; a model without an attention bias no longer adds zeros.
- **Prompts in Qwen's Q4_K_M file are about 8x faster on the GPU**, from 186 to
  1,537 tokens a second. Q4_0 and Q5_0 tensors reach the GPU as Q8_0, which had no
  batched kernel: a long prompt dispatched one workgroup per output element.
- **Streaming with beam search returned garbage.** `kjarni translate` streams by
  default, and "The house is small." came back as "Das Haus ist klein.. aber Zimmer
  sind ...". Streaming now gives the same text as not streaming: with more than one
  beam, a token is sent once every hypothesis still able to win agrees on it, so
  text arrives in bursts and is never taken back.
- **`Indexer.exclude()` did nothing for most patterns, and hidden folders were
  indexed.** Patterns were compared with the full path, so `target` or `*.log` never
  matched. `search_folder` over MCP therefore indexed build output and could return
  passages from folders like `.aws` to the agent. Excluded and hidden folders are
  now skipped.
- **Short texts on the GPU are a little faster**: bge-m3 from 74 to 58 ms. Layer
  norm gives a few rows a workgroup each instead of one thread each. Batches are
  unchanged.

## 0.2.7

### GPU results match PyTorch

Every embedding model, classifier and the reranker now run against PyTorch on
both CPU and GPU in one test, logit by logit, and agree to within 6e-6 on each.
Several GPU paths were wrong before, some silently:

- **GPU classification from the bindings ran on the CPU.** `kjarni_classifier_new`
  started a fresh builder from the model name after setting the device and cache
  directory, dropping both. Python, C#, Go and C++ classifiers asked for the GPU
  and got the CPU, and ignored `cache_dir`.
- **RoBERTa, DistilRoBERTa and XLM-R (bge-m3) were wrong on the GPU.** The GPU
  encoder built its own embedding setup without the position offset, so position
  embeddings were read from row 0 instead of row 2. bge-m3 embeddings were at a
  cosine of 0.97 from the CPU's, and RoBERTa sentiment scores off by 0.05.
- **MPNet was wrong on the GPU.** Its relative position bias was never loaded on
  the GPU path (cosine 0.87).
- **nomic-embed-text panicked on the GPU**, with "FFN type mismatch": the GPU
  encoder always used the standard feed-forward kernel, and had no rotary
  positions either.
- **Large GPU batches panicked.** Kernels launched one workgroup per row or per
  256 elements in a single dimension, and WebGPU allows 65535: MiniLM failed at
  256 texts, bge-m3 at 128. Launches past the limit now spread over a 2D grid.
- **GPU decode ran out of memory as it generated.** The tensor pool kept a buffer
  for every size it had handed out, and attention's sizes change every token: a
  3B model gained about 70 MB of VRAM per token until the driver started paging.
- **`return_logits` did nothing.** The classifier accepted it and returned
  probabilities.
- **T5 and Whisper on the GPU abort no longer.** They have no GPU path yet and
  reached a `todo!()`, which took a calling process down with it. They now return
  an error saying they run on the CPU only.

### The GPU is faster than the CPU

Embedding, classification and prompt processing on the GPU were between 2 and 15
times slower than the CPU. On an RTX A2000, with the model loaded:

| | CPU | GPU in 0.2.6 | GPU now |
|---|---|---|---|
| MiniLM, 256 texts | 0.7 s | 2.4 s | 0.3 s |
| bge-m3, 256 texts | 13.8 s | 44 s (and wrong) | 5.2 s |
| Llama 3.2 1B, 1800-token prompt, first token with load, Q4_K_M | 7.7 s | 21.0 s | 6.0 s |
| Llama 3.2 3B decode, Q4_K_M / BF16 | | 14.0 / 11.7 tok/s | 24.1 / 18.5 tok/s |

- **Matrix multiplies are tiled.** Batched multiplies gave each thread one output
  and read both operands from global memory for every product, at about 150
  GFLOP/s. They now stage tiles in workgroup memory, 64x64 or 128x128 by row
  count, and reach 2 to 2.7 TFLOP/s. F32, BF16, Q4_K and Q6_K weights all use them;
  packed weights are dequantised as each tile loads.
- **The encoder feed-forward and batched SwiGLU use them.** Both had their own
  per-output kernels for batches.
- **Each layer reuses the previous layer's scratch buffers.** A forward pass was
  one allocation scope, so every layer's intermediates stayed allocated to the end:
  8.3 GB for bge-m3 at a few thousand tokens, enough to push the weights out of
  VRAM. It now holds 3.7 GB, and time grows linearly with the batch.
- **GGUF files load faster.** The embedding table was dequantised and uploaded
  twice, once by the decoder and once by the pipeline; it is now shared, and the
  dequantisation runs in parallel. Llama 3.2 1B loads in 3.2 s instead of 5.6.

### Documentation

- **Every public API has a doc comment** in C#, Go, Python and TypeScript, 101
  that had none.
- **The NuGet package ships its XML documentation**, so IDEs show it. Only
  `Kjarni.Extensions.AI` did.
- **`scripts/gen-reference.sh` generates the Python, .NET and JavaScript API
  references** as Markdown, and fails when a public item has no doc comment. CI
  runs it on every push and attaches the result to each release as
  `kjarni-reference.tar.gz`. Rust and Go are on docs.rs and pkg.go.dev.
- **One license everywhere.** NuGet, npm and the `kjarni-wasm` crate said MIT;
  every manifest now says MIT OR Apache-2.0, and `check-versions.sh` fails if one
  disagrees.
- Package descriptions on NuGet and PyPI list every task, the npm README's format
  guide link works, and the `Capability` type is exported from `kjarni-wasm`.

## 0.2.6

### Outputs match PyTorch

Every model was run against transformers on the same weight files. Embeddings,
classifiers, the reranker and CLIP now agree to within 1e-6; summaries,
translations and greedy chat replies are word for word the same. These were
wrong, most of them silently:

- **nomic-embed-text applied its activation to the wrong projection.** The gate
  and up weights of its gated MLP were swapped in every layer, so its embeddings
  were at a cosine of 0.14 from the model's own.
- **Llama 3 positions drifted with length.** The Llama 3 RoPE scaling blended
  the two divisors instead of the two frequencies across the middle band, so
  the error grew with every token. Llama 3.1 and 3.2 and the DeepSeek-R1 distill
  were affected; their logits were 100 times further from a float64 reference
  than PyTorch's.
- **Phi-3.5 attention was too flat.** LongRoPE scales cos and sin by
  `sqrt(1 + ln(32) / ln(4096))`, about 1.19, which the checkpoint leaves to be
  derived. Without it every attention logit was 1.42 times too small and replies
  differed from the first token.
- **bge-m3 is CLS-pooled**, as its sentence-transformers config says, not
  mean-pooled (cosine 0.65 before). `Embedder::embed_batch_flat` in Rust
  mean-pooled every model whatever its pooling.
- **Multi-label classifiers use a sigmoid per label.** RoBERTa configs dropped
  `problem_type`, so GoEmotions labels were softmaxed against each other and
  scores were off by up to 0.42.
- **RoBERTa, XLM-R and MPNet stop at 512 tokens**, not 514. Their position table
  starts at offset 2, and the last two tokens read past its end.
- **Trailing whitespace is no longer a token** for SentencePiece Unigram
  tokenizers (XLM-R, T5), matching transformers.
- **CLIP resizes with PIL's fixed-point arithmetic**, byte for byte, instead of
  floats that left a few pixels per image one level off.
- **Beam search scores the way transformers does**: log-softmax before the
  logit processors, and lengths counted from the decoder start token. It changed
  which distilbart summary won.
- **Chat templates match each model's own.** Llama 3.1 and 3.2 get their date
  header, DeepSeek-R1 its own format instead of Llama 3's, Mistral v0.3 no space
  before `[/INST]` and its system prompt in the last user turn, and no template
  invents a system prompt the model's does not.

### Benchmarks

- **The translation comparison now runs the same search on both sides.** PyTorch
  was given greedy decoding while Kjarni ran flan-t5's own four-beam settings, so
  the published "1.4 times PyTorch's time" compared different work. With the same
  settings the two produce the same German and take 989 and 947 ms.

### Faster batches for reranking, classification and embeddings

Measured on a batch of 64 texts against PyTorch and ONNX Runtime; the full method
and results are in [bench/BENCHMARKS.md](bench/BENCHMARKS.md).

- **Reranking and classification group inputs by length.** Both padded a whole batch
  to its longest member and used the encoder's allocating forward pass. They now
  use the same length-bucketed, buffered path as embeddings: reranking 64 documents
  went from 561 ms to 246 ms, faster than PyTorch's 273 ms.
- **Batch classification in the C API.** `kjarni_classifier_classify_batch` runs many
  texts through the model together, and every binding uses it: `classify_batch` in
  Python, `ClassifyBatch` in C# and Go, `classify(span)` in C++. 64 texts went from
  681 ms, one at a time, to 157 ms, against PyTorch's 223 ms.
- **Encoder buffers are reused across calls.** Each forward pass allocated and zeroed
  a couple of hundred megabytes of scratch space, 40 ms of a 260 ms MiniLM batch.
  Whisper and the BART and T5 encoders reuse them too.
- **faer 0.24.** Matrix multiplies on encoder shapes are 1.1 to 1.7 times faster.
- The embedding layer norm and the position and token-type additions run in
  parallel instead of on one thread.

### Prompts are processed more than ten times faster

A 500-token prompt took Qwen2.5 0.5B ten seconds before the first token; it now
takes about one. Prompt processing on the same GGUF files went from 41 to 615
tokens a second (llama.cpp: 611) on Qwen2.5 0.5B and from 20 to 282 on Llama 3.2
1B (llama.cpp: 424).

- **Prompts use the streaming attention path.** It was reserved for an all-ones
  mask, but Llama-family models pass a causal mask for a prompt, which asks for
  exactly what that path does. Every prompt took the path that materialises the
  full score tensor instead: 330 ms per layer on a 500-token prompt, against 14.
- **Quantized and BF16 weights are multiplied as tiles for prompts.** The
  per-row kernels, right for decoding one token, read and dequantized every
  weight once per prompt token. From 16 rows up, tiles are dequantized once and
  multiplied with faer.
- **Only the last position's logits are computed for a prompt**, the only ones
  used: on a 152k vocabulary, 70 GFLOP saved per 500-token prompt.

### Models hold one copy of their weights

Whisper-small went from 2.8 GB to 1.1 GB once loaded, distilbart-cnn from 3.5 GB
to 1.3 GB and CLIP from 1.3 GB to 0.65 GB: about the size of their weight files,
and less than the same models take in PyTorch. Three things held extra copies:

- **The mmap cache never let go.** It held every weight file mapped for the life
  of the process, after its weights had been copied out and even after the model
  was dropped. It holds weak references now.
- **`StdFeedForward` stored its matrices twice**, in public fields beside the
  layers that use them. In Whisper and BART the feed-forward is most of the model.
- **The token embedding table was loaded three times** for encoder-decoder
  models, by the encoder, the decoder and the pipeline. `ModelWeights` now hands
  out one shared copy (`get_shared_array2`).
- **Loading peaked at twice the model.** The mapped file stayed resident while
  its weights were copied out of it. Each tensor's pages are now released as soon
  as it is copied, so Whisper-small peaks at 1.28 GB instead of 2.97 GB.

A test loads Whisper-small and distilbart-cnn and checks each holds under 1.3
times its weight file once loaded, and peaks under 1.5 times while loading; they
are at 1.03 and 1.05.

### Whisper keeps every sentence on long audio

Audio was cut into fixed 30-second windows, wherever the 30 seconds fell. Whisper
drops an unfinished phrase at the end of a window and guesses at the fragment that
opens the next, so a sentence spanning a cut went missing. A cut could also send a
window into a loop to the 448-token limit: a 54-second recording took 30 seconds.

- **Windows are cut at pauses**: the latest clear pause in a window's last five
  seconds.
- **Whisper's timestamp rules** from OpenAI's decoder are applied, so segments carry
  real start and end times instead of one segment per 30-second window.
- **Text invented in a window's padding is dropped.** Past the end of the speech,
  Whisper carries on into the silence, continuing a phrase it expects. A segment
  that starts there, and `[BLANK_AUDIO]`, are discarded, and no segment opens in the
  window's last half-second.

Transcribing long audio takes about a third longer, from the timestamp tokens; it
is still faster than openai-whisper and than faster-whisper at full precision.

### Quantized chat in every language, and Qwen from GGUF

- **`quantized`** (C: `KjarniChatConfig.quantized`; `quantized=True` in Python and
  C#, `WithQuantized` in Go, `.quantized` in C++) loads a model's 4-bit GGUF weights:
  a fraction of the memory, and several times faster to generate on a CPU. Only the
  CLI could reach them before.
- **Qwen GGUF files load and generate correctly.** Three bugs stood in the way, each
  enough on its own: Q5_0 tensors, which Q4_K_M files use wherever a row is not a
  multiple of 256, were rejected; the Q/K row permutation llama.cpp applies only to
  Llama files was undone for every architecture; and the attention biases were never
  mapped to their GGUF names, so Qwen ran without them. Q5_0 and Q4_0 are now
  converted to Q8_0 at load, which is exact.
- `KjarniChatConfig.model_path` was ignored. It is now rejected with a message, until
  loading from a path is supported.

### Fixes

- `kjarni_embedder_encode_batch` always mean-pooled and normalised, ignoring the
  embedder's `normalize` setting, so a batch disagreed with encoding the same texts
  one at a time. It now uses the same path.
- `Classifier::classify_batch` applied sigmoid to probabilities instead of logits in
  multi-label mode, scored custom labels as zero, and dropped any text with nothing
  above the threshold, shifting every later result onto the wrong text. It now does
  exactly what classifying each text alone does.
- Four C API results freed their arrays with the wrong size: class results and string
  arrays as a single element, and collections whose capacity could exceed their
  length. All are boxed slices of exactly their length now.
- Releases publish `SHA256SUMS` and a signed build-provenance attestation for every
  shipped library (`gh attestation verify`).

## 0.2.5

### Every capability in every language

Speech to text, summarization, translation and image search were in the engine and
on the command line but not in the C ABI, so no binding could reach them. They are
now `kjarni_transcriber_*`, `kjarni_summarizer_*`, `kjarni_translator_*` and
`kjarni_image_index_*`, and every binding wraps all of them:

| Added | C++ | C# | Go | Python |
| ----- | :-: | :-: | :-: | :----: |
| Transcriber, Summarizer, Translator, ImageIndex | ✓ | ✓ | ✓ | ✓ |
| Indexer, Searcher | ✓ | | | |
| Chat (with streaming, history, conversations) | | | | ✓ |
| Chat with caller-held history | | | ✓ | |
| Indexer `Add`, `Recreate`, extensions, excludes, chunking | | | ✓ | |
| Search options and result metadata | | | ✓ | |

The same input now gives byte-identical output from C++, C#, Go and Python for
transcription, summaries, translation, image and hybrid search scores, and greedy
chat.

`image-io` is now a default feature of `kjarni-ffi`, so the published libraries
include image search; `--no-default-features` leaves it out, and each binding
reports that rather than failing to load. `kjarni_image_index_new` with a null model
directory now uses clip-vit-base-32, downloaded on first use.

### The Go binding leaked every result

C functions that take a struct by value, such as
`kjarni_float_array_free(KjarniFloatArray)`, were called through purego with a
pointer to the struct. That is the Windows x64 convention; on Linux the struct
travels in registers, so the free read the Go pointer as its data field and an
unrelated register as its length, found zero, and did nothing. Every embedding,
classification, rerank and search result leaked: about 28MB per 300 batches of 64
embeddings. A new test measures RSS across that loop.

### The Go binding panicked after about two thousand streamed replies

Each `Stream` call created a purego callback. purego never frees one and allows about
two thousand per process, so a long-running chat server panicked. All streams now
share one callback and are told apart by `user_data`.

### Smaller fixes

- Go: `Indexer.Create` with no inputs indexed out of range and panicked; it now
  returns an error. `Searcher.Search` requested the reranker on every query whether
  or not one was configured; it now uses the searcher's default.
- Go: `goString` copied byte by byte through two loops; it is one copy now.
- Python: the in-repo library path pointed at `crates/target/release`, one directory
  short, so a development checkout silently loaded whatever `libkjarni_ffi` was on
  the system path instead.
- Python: embeddings were copied out of the engine one float at a time through the
  interpreter; they are sliced in C now.
- Python now has a test suite, run in CI. The C++ job downloads the larger models
  and runs every capability; the C#, Go and Python jobs set `KJARNI_SKIP_HEAVY` and
  skip those, rather than each caching its own 3.5GB.

The browser build is unchanged: Whisper, the seq2seq models and CLIP are native only.

## 0.2.4

### `kjarni mcp`: Kjarni as tools for AI agents

`kjarni mcp` serves Kjarni to any MCP client (Claude Code, Cursor, Claude Desktop,
VS Code) as tools that run locally: `search_folder` (semantic and hybrid search over
a folder of documents or code, indexed on first use and kept up to date),
`transcribe`, `classify`, `rerank` and `similarity`. Models load once per session,
and indexes live in Kjarni's cache, never in the folders searched.

```bash
claude mcp add kjarni -- kjarni mcp
```

### Whisper is four times faster and detects the language

Three minutes of English went from 64 s to 16 s on CPU.

- **The mel spectrogram** computed a naive DFT per bin. It is now two matrix
  multiplies against cos/sin tables built once: 1.5 s to 15 ms per chunk, and 100
  times closer to HuggingFace's features (worst difference 1.7e-3 to 1.7e-5).
- **The convolutional front end** is im2col and a matmul instead of scalar loops,
  with Whisper's exact erf GELU in place of the tanh approximation. Encoder output
  against HuggingFace: worst difference 0.27 to 0.0025.
- **The encoder** runs the buffered path, about three times faster on 1,500
  positions.
- **Single-query attention** gets one task per head and reads the cache by rows,
  with bit-identical results: cross-attention 10.9 s to 4.9 s on three minutes.
- **Up to four chunks decode together**, sharing each read of the weights. The
  output is identical and decoding takes half the time.
- **The language is detected** from the first chunk. Auto-detect was a stub that
  always answered English.
- A greedy decode stuck repeating a phrase ends its chunk after four repeats.

### Faster CPU encoders, and multilingual-e5-small

- Embedding batches are grouped by token length, so each text is padded to its
  neighbours rather than to the longest in the call.
- Residual add and layer norm are fused and run in parallel across rows. GELU, the
  head transposes and the fused QKV split run in parallel too.
- Per 200-word chunk on a 24-thread i7-13700: multilingual-e5-small 72 to 41 ms,
  mpnet 272 to 148 ms, bge-m3 539 to 352 ms. Vectors match sentence-transformers to
  2e-7.
- **multilingual-e5-small** (intfloat, MIT, about 100 languages) joins the registry.
  Texts take a `query: ` or `passage: ` prefix.
- CLIP: `embed_image_tensors` embeds a batch of images about four times faster per
  image than one at a time, and the patch projection is one matmul (60 to 2 ms per
  image).

### Batched decoding masked the wrong tokens

The decoder multiplied the causal mask into the padding mask and passed the result
as `[batch, key]`. When the batch size equalled the prompt length, each sequence was
masked by a causal row, and the first saw only its first token. Self-attention
already applies the causal mask, so only a real `[batch, total_len]` padding mask
is passed now.

### Smaller fixes

- The indexer honours exclusions: a pattern such as `node_modules`,
  `node_modules/**` or `*.min.js` matches any component of a path, and an excluded
  or hidden directory is never entered, rather than walked and filtered file by
  file.
- `LinearLayer::matmul` with a few rows ran on one thread; it uses the same kernels
  as `matmul_noalloc` now. `matmul_noalloc` asserts its output shape, since its
  kernels write through raw pointers and a wrong-sized buffer was silent memory
  corruption.
- A Qt example, `crates/kjarni-ffi/examples/qt`, searches help articles from QML.

### Windows no longer needs the Visual C++ Redistributable

`kjarni_ffi.dll` and `kjarni.exe` imported `VCRUNTIME140.dll`, the Visual C++
Redistributable. Most machines have it; a clean install or a minimal VM does not, and
there the failure did not point at the cause: .NET reported
`DllNotFoundException: Unable to load DLL 'kjarni_ffi'` with the file sitting next to
the executable, because Windows reports a missing dependency of a DLL as the DLL itself
failing to load.

The C runtime is now linked statically for `x86_64-pc-windows-msvc`
(`.cargo/config.toml`). The binaries import only libraries that ship with Windows,
including the Universal CRT, which is part of Windows 10 and later. Release CI checks
the import table and fails if `VCRUNTIME` comes back.

### Indexing progress callbacks work under IL2CPP

`Indexer.Create` and `Indexer.Add` passed `onProgress` to native code as a lambda.
IL2CPP, Unity's ahead-of-time backend, can only hand native code a delegate to a static
method, so passing `onProgress` there threw. Progress now goes through one static
callback marked `[MonoPInvokeCallback]`, with the handler carried in `user_data`, the
same way chat streaming already worked. Nothing changes for callers.

### The netstandard2.1 package has no dependencies

`Kjarni` for netstandard2.1 depended on `System.Text.Json` 8.0.5, used in exactly one
place: reading the metadata attached to each search result. Unity cannot restore NuGet
dependencies, so that one call meant hand-installing `System.Text.Json` and its chain
(`System.Memory`, `System.Buffers`, `Microsoft.Bcl.AsyncInterfaces` and the rest)
before the package would load.

The metadata is always a flat object of string keys and string values, since the
native side serialises a `HashMap<String, String>`, so a small internal parser now
reads it instead. Both targets have no package dependencies, and using Kjarni from
Unity is two files: `Kjarni.dll` and the native library. `SearchResult.Metadata` is
unchanged.

## 0.2.3

### A panic in the engine no longer kills the host process

The workspace release profile set `panic = "abort"`. Cargo has no per-package panic
strategy, so that applied to `kjarni-ffi` as well — the cdylib that NuGet, the Go
module and the C header all load *into someone else's process*. Any panic reachable
through any of the 80 `extern "C"` entry points terminated that process: a .NET
service, a Go binary, a Qt application. No error code, no message, nothing the caller
could have handled. `kjarni-ffi/src` had 38 `unwrap`/`expect`/`panic!` sites of its
own, and everything the engine can panic on sits behind them.

`panic = "abort"` is gone from the profile, and the 66 fallible entry points now run
their bodies inside a `catch_unwind` barrier (`kjarni-ffi/src/panic.rs`). A panic
becomes the new `KJARNI_ERROR_CODE_PANIC` (11) with the entry point's name and the
panic message on `kjarni_last_error_message()`, which is what every other failure in
the C ABI already looks like. The remaining 14 entry points return compile-time
constants and are left unguarded, which the module documents; guarding
`kjarni_last_error_message` in particular would be circular.

This is a backstop, not a licence to panic — a panic reaching it is still a bug. It
only decides whether the caller gets to hear about it.

Building `kjarni-ffi` under `panic = "abort"` is now a compile error rather than a
silently defeated barrier, since `catch_unwind` cannot intercept an abort. That check
is what would have caught the original setting.

`KJARNI_ERROR_CODE_PANIC = 11` is additive — no existing code is renumbered — and is
mirrored in the C header, the C# `KjarniErrorCode`, Go's `ErrPanic`, and Python's
`PANIC`. The cost of the profile change is a larger binary and landing pads the
optimiser must keep.

### An error message with an interior nul is no longer discarded

`set_last_error` built its `CString` with `CString::new(msg).ok()`, which yields
`None` on an interior nul and cleared the stored message rather than replacing it, so
`kjarni_last_error_message()` returned NULL for an error that had just been reported.
It now truncates at the first nul and keeps the leading, most specific part. For an ordinary error that lost a diagnostic; for a caught panic it
would have lost the only evidence the panic happened.

## 0.1.4

### Encoders now truncate where sentence-transformers does

Encoder models read `max_seq_length` from `sentence_bert_config.json` when the model
ships one, falling back to `max_position_embeddings` as before. For
`all-MiniLM-L6-v2` that changes the default from 512 to **256**, and for
`all-mpnet-base-v2` from 512 to **384**.

This is a behaviour change for long inputs. Anything under the limit is unaffected,
but text above it now produces the same embedding the reference Python
implementation produces, where previously it did not. The default RAG chunk of 1000
characters measures at a median of 286 tokens on ordinary prose, so most indexed
chunks were in the affected range. **Rebuild indexes built with an earlier version**
if you index documents longer than 256 tokens.

Pass `ModelLoadConfig::max_sequence_length` to override.

### Asking for more positions than a model has is now an error

Requesting a sequence length above the model's position-embedding table used to
succeed. Tokens past the table received no position embedding at all and the encoder
returned a normal-looking vector that was quietly wrong. This is now refused at load
with a message naming the request and the ceiling. A length derived from a config
file is clamped with a warning instead, so a malformed config cannot make a model
unloadable.

### `.kjq` carries the tuned sequence length

`scripts/quantize_model.py` folds `max_seq_length` into the config it packs, and the
byte loader reads it back. Without this the browser had no way to learn the number,
since reading `sentence_bert_config.json` needs a filesystem, and WASM would have
truncated at 512 while every native binding truncated at 256. **Regenerate `.kjq`
files** built with an earlier version.

### Vector search is faster and its ordering is now defined

`VectorStore::search` keeps a bounded top-k rather than sorting every candidate and
truncating, hoists the query norm out of the inner loop, and runs in parallel above
2048 vectors. On 384-dimensional vectors, best of five on a 24-core machine:

| Vectors | Before | After |
| ------- | ------ | ----- |
| 10,000 | 2.6 ms | 0.2 ms |
| 100,000 | 24.9 ms | 4.9 ms |
| 1,000,000 | 255 ms | 56 ms |

Results are unchanged. Ties now break on ascending index, which makes the ordering
total: without that the parallel scan could return a different permutation of
equally-scoring documents between runs.

WASM keeps the serial path, since rayon there needs a threaded build.

### A TypeScript package for the browser, published to npm

`npm i kjarni-wasm` gets embeddings, classification, reranking and chat running
in a Web Worker, so loading a model or generating a reply never freezes the page.
Every method returns a promise, and chat is an async iterator.

```ts
const kjarni = await Kjarni.load({ encoder: "/models/minilm-l6-v2-q8.kjq" });
for await (const piece of kjarni.chat("Explain RAG")) out.textContent += piece;
```

The raw wasm-bindgen bundle is still in `kjarni-wasm.tar.gz` for anyone who
would rather drive it directly.

### Browser generation actually streams

`WasmChat` ran the whole decode loop to completion and only then drained the
channel the engine had been streaming into, so callers could show nothing until
the reply was finished. Polling the loop together with its own drain fixes that
on wasm's single thread, and the channel goes back to being a small buffer that
applies real back-pressure rather than one sized to hold an entire response.

`generate_stream` exposes it to JavaScript.

### The WASM bundle is now tested in a browser

Everything testing `kjarni-wasm` ran the Rust natively, which covers the engine
and misses the artifact entirely: a broken import, a bundle compiled from stale
sources, a binding that only fails through wasm-bindgen's glue. A new CI job
loads the built bundle in Chromium and checks it encodes text correctly.

This was not hypothetical. The live demo served a bundle containing neither
classification nor chat for days while every test stayed green.

### C++ is built and tested in CI

The C ABI header had never been compiled by CI, and shipped in every release archive
unverified. A new job regenerates `kjarni.h` with cbindgen and diffs it against the
committed copy, compiles it as C11 under `-Wall -Wextra -Werror`, and runs the C++
wrapper's 61 checks under AddressSanitizer and UndefinedBehaviorSanitizer.

### Other

- `kjarni-wasm`'s 18 binding tests now run in CI; they were excluded when the crate
  could not build natively.
- The `.kjq` test fixtures are exported in CI from cached weights. They previously
  looked for a sibling checkout that CI does not have, so they skipped and reported
  green while testing nothing.
- `quantize_model.py` tolerates an unreadable `sentence_bert_config.json` rather than
  aborting, which is what a truncated or redirected download leaves behind.
