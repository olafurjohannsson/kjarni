"""Writes bench/BENCHMARKS.md from results/*.json, so the published numbers are
exactly the measured ones."""
import json, os, platform, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "BENCHMARKS.md")


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def mem(r):
    """Resident memory after the work, which older result files lack."""
    return f'{r["rss_mb"]:,.0f} MB' if "rss_mb" in r else "–"


def picture(name, alt):
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="charts/{name}-dark.svg">\n'
            f'  <img alt="{alt}" src="charts/{name}-light.svg">\n</picture>\n')


def versions():
    out = {}
    try:
        import torch, transformers, sentence_transformers, onnxruntime
        out.update(torch=torch.__version__, transformers=transformers.__version__,
                   **{"sentence-transformers": sentence_transformers.__version__, "onnxruntime": onnxruntime.__version__})
        import faster_whisper, ctranslate2, open_clip
        out.update(**{"faster-whisper": faster_whisper.__version__, "ctranslate2": ctranslate2.__version__,
                      "open_clip": open_clip.__version__})
    except ImportError:
        pass
    return out


L = []
w = L.append
w("# Benchmarks\n")
w("Kjarni against the usual ways of running the same models: PyTorch (through "
  "`transformers` and `sentence-transformers`), ONNX Runtime, faster-whisper and "
  "llama.cpp. Same weights, same inputs, same machine. The sections up to Text generation are CPU only; CPU and GPU covers both.\n")
w("Reproduce with `bench/compare/run_encoders.py` and `bench/compare/run_rest.py`; "
  "`make_charts.py` and `make_report.py` turn the results into this page.\n")
w("## Method\n")
w("- **Machine:** Intel Core i7-13700 (8 performance and 8 efficiency cores, 24 threads), 64 GB, Linux. Idle: each run waits for a load average under 1 over the last minute and under 1.5 over the last five.")
w("- **Isolation:** every engine runs in its own fresh process, so load time and memory are its own.")
w("- **Memory:** *memory after* is what the process holds once the work is done; *peak* is the most it held at any point, usually while loading.")
w("- **Timing:** minimum of repeated calls after warm-up, the least noisy estimate of what each engine can do. Medians are in the raw JSON.")
w("- **Threads:** each engine's default. Kjarni uses rayon's pool; PyTorch and ONNX Runtime use the physical core count. For the LLMs, both engines get the better of 8 and 16 threads for each of prompt processing and generation.")
w("- **Agreement:** encoder outputs are compared with PyTorch's; the largest difference is listed with each row. Generated text is recorded so the engines can be checked to have done the same work.")
v = versions()
if v:
    w("- **Versions:** " + ", ".join(f"{k} {val}" for k, val in v.items()) + ", llama.cpp b11901.\n")

enc = load("encoders")
if enc:
    w("## Embeddings, reranking and classification\n")
    for c, alt in [("encoders-throughput", "Batch throughput"), ("encoders-load", "Load time"), ("encoders-memory", "Memory after")]:
        w(picture(c, alt))
    name = {"kjarni": "**Kjarni**", "torch": "PyTorch", "onnx": "ONNX Runtime"}
    task = {"embed-minilm": "Embed · all-MiniLM-L6-v2", "embed-mpnet": "Embed · all-mpnet-base-v2",
            "rerank": "Rerank · ms-marco-MiniLM-L6", "classify": "Classify · DistilBERT SST-2"}
    w("| Task | Engine | Batch of 64 | One text | Load | Memory after | Peak memory | Max diff vs PyTorch |")
    w("|---|---|--:|--:|--:|--:|--:|--:|")
    for r in enc["rows"]:
        diff = "reference" if r["engine"] == "torch" else f'{r.get("max_abs_diff_vs_torch", float("nan")):.0e}'
        w(f'| {task[r["task"]]} | {name[r["engine"]]} | {r["batch_ms"]["min"]:.0f} ms ({r["batch_per_s"]:.0f}/s) | '
          f'{r["single_ms"]["min"]:.1f} ms | {r["load_ms"]:,.0f} ms | {mem(r)} | {r["peak_rss_mb"]:,.0f} MB | {diff} |')
    w("")

wh = load("whisper")
if wh:
    w("## Speech to text\n")
    w(picture("whisper", "Whisper speed"))
    w("whisper-small, English, greedy decoding, CPU.\n")
    w("| Audio | Engine | Time | Faster than real time | Load | Memory after | Peak memory |")
    w("|---|---|--:|--:|--:|--:|--:|")
    for r in wh:
        e = "**Kjarni**" if r["engine"] == "kjarni" else r["engine"]
        w(f'| {r["audio_secs"]:.0f} s | {e} | {r["secs"]:.2f} s | {r["realtime_x"]:.1f}× | {r["load_ms"]:,.0f} ms | {mem(r)} | {r["peak_rss_mb"]:,.0f} MB |')
    w("")

s2 = load("seq2seq")
if s2:
    w("## Summarization and translation\n")
    w("distilbart-cnn-12-6 and flan-t5-base, each with its checkpoint's own generation settings.\n")
    w("| Task | Engine | Time | Load | Memory after | Peak memory | Output |")
    w("|---|---|--:|--:|--:|--:|---|")
    for r in s2:
        e = "**Kjarni**" if r["engine"] == "kjarni" else "PyTorch"
        out = r["output"].replace("|", "\\|")
        out = out if len(out) < 90 else out[:87] + "…"
        w(f'| {r["task"]} | {e} | {r["ms"]:,.0f} ms | {r["load_ms"]:,.0f} ms | {mem(r)} | {r["peak_rss_mb"]:,.0f} MB | {out} |')
    w("")

cl = load("clip")
if cl:
    w("## Image search (CLIP)\n")
    w("ViT-B/32 with the LAION-2B weights, indexing 64 photos and running one text query.\n")
    w("| Engine | Images per second | Query | Load | Memory after | Peak memory |")
    w("|---|--:|--:|--:|--:|--:|")
    for r in cl:
        e = "**Kjarni**" if r["engine"] == "kjarni" else r["engine"]
        w(f'| {e} | {r["images_per_s"]:.1f} | {r["query_ms"]:.1f} ms | {r["load_ms"]:,.0f} ms | {mem(r)} | {r["peak_rss_mb"]:,.0f} MB |')
    w("")

dec = load("decoders")
if dec:
    w("## Text generation\n")
    w(picture("decode-generation", "Generation speed"))
    w(picture("decode-prompt", "Prompt processing"))
    w("The same Q4_K_M GGUF file in both engines. llama.cpp is timed with its own `llama-bench` "
      "(pp512, tg128); Kjarni through its Python binding, from a 512-token prompt and 128 greedy tokens.\n")
    w("| Model | Engine | Prompt processing | Generation |")
    w("|---|---|--:|--:|")
    for r in dec:
        e = "**Kjarni**" if r["engine"] == "kjarni" else "llama.cpp"
        w(f'| {r["model"]} | {e} | {r["prompt_tok_s"]:,.0f} tok/s | {r["gen_tok_s"]:,.1f} tok/s |')
    w("")

matrix = load("matrix")
if matrix:
    def rows(**want):
        return [r for r in matrix if "error" not in r and all(r.get(k) == v for k, v in want.items())]

    w("## CPU and GPU\n")
    w("Reproduce with `bench/compare/run_matrix.py`. CPU as above; GPU an NVIDIA RTX A2000 12 GB, "
      "with Kjarni on WebGPU (Vulkan here) and the others on CUDA, in fresh processes on a quiet machine.\n")
    for chart, alt in [("matrix-embeddings-cpu", "Embeddings on the CPU"), ("matrix-embeddings-gpu", "Embeddings on the GPU"),
                       ("matrix-whisper", "Speech to text"), ("matrix-generation-cpu", "Text generation on the CPU"),
                       ("matrix-generation-gpu", "Text generation on the GPU"), ("matrix-prompt-gpu", "Prompt processing on the GPU")]:
        w(picture(chart, alt))
    engine = {"kjarni": "**Kjarni**", "torch": "PyTorch", "onnx": "ONNX Runtime", "llama.cpp": "llama.cpp"}
    w("| Model | Device | Engine | Batch of 64 | One text | Load | Max diff vs PyTorch on CPU |")
    w("|---|---|---|--:|--:|--:|--:|")
    for r in rows(kind="encoder"):
        diff = r.get("max_abs_diff_vs_torch_cpu")
        shown = "reference" if r["engine"] == "torch" and r["device"] == "cpu" else (f"{diff:.0e}" if diff is not None else "")
        w(f'| {r["task"].replace("embed-", "")} | {r["device"].upper()} | {engine[r["engine"]]} | {r["batch_per_s"]:,.1f}/s | '
          f'{r["single_ms"]["min"]:.1f} ms | {r["load_ms"]:,.0f} ms | {shown} |')
    w("")
    w("| Device | Engine | 55 s of speech | Faster than real time |")
    w("|---|---|--:|--:|")
    for r in rows(kind="whisper"):
        w(f'| {r["device"].upper()} | {"**Kjarni**" if r["engine"] == "kjarni" else r["engine"]} | {r["secs"]:.2f} s | {r["realtime_x"]:.1f}× |')
    w("")
    w("Generation is from the same Q4_K_M GGUF, except PyTorch and the rows marked safetensors. *After a prompt* follows "
      "512 tokens; *short* starts from a nearly empty context, as llama-bench's tg128 does, so llama.cpp's one figure is in that column.\n")
    w("| Model | Device | Engine | Prompt | After a prompt | Short |")
    w("|---|---|---|--:|--:|--:|")
    for r in rows(kind="decoder"):
        name = engine[r["engine"]]
        if r["engine"] == "llama.cpp" and r["device"] == "gpu":
            name += " " + {"cuda": "CUDA", "vulkan": "Vulkan"}[r["backend"]]
        if r["engine"] == "kjarni" and r["weights"] != "Q4_K_M":
            name += " safetensors"
        short = r.get("gen_short_tok_s", r["gen_tok_s"] if r["engine"] == "llama.cpp" else None)
        after = "" if r["engine"] == "llama.cpp" else f'{r["gen_tok_s"]:,.1f}'
        w(f'| {r["model"]} | {r["device"].upper()} | {name} | {r["prompt_tok_s"]:,.0f} | {after} | {short:,.1f} |')
    w("")

bindings = load("bindings")
if bindings:
    w("## Binding overhead\n")
    w("One MiniLM embedding of a short text, on the CPU, through each binding against the same library: "
      "the worst case for a binding, since the model does the least work. Median of 3,000 calls, best of three "
      "runs. Reproduce with `bench/bindings/run.py`.\n")
    w("| Language | Median per call | Against C |")
    w("|---|--:|--:|")
    for r in bindings:
        w(f'| {r["lang"]} | {r["median_us"] / 1000:.2f} ms | {"baseline" if r["lang"] == "C" else format(r["overhead_us"], "+.0f") + " µs"} |')
    w("\nThe differences are within run-to-run noise: Go and Rust come out ahead of C, which cannot be overhead.\n")

open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", os.path.abspath(OUT))
