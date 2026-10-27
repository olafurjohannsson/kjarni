"""Writes bench/BENCHMARKS.md from results/*.json, so the published numbers are
exactly the measured ones."""
import json, os, platform, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "BENCHMARKS.md")


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


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
  "llama.cpp. Same weights, same inputs, same machine, CPU only.\n")
w("Reproduce with `bench/compare/run_encoders.py` and `bench/compare/run_rest.py`; "
  "`make_charts.py` and `make_report.py` turn the results into this page.\n")
w("## Method\n")
w("- **Machine:** Intel Core i7-13700 (8 performance and 8 efficiency cores, 24 threads), 64 GB, Linux. Idle: each run waits for a load average under 1.")
w("- **Isolation:** every engine runs in its own fresh process, so load time and memory are its own.")
w("- **Timing:** minimum of repeated calls after warm-up, the least noisy estimate of what each engine can do. Medians are in the raw JSON.")
w("- **Threads:** each engine's default. Kjarni uses rayon's pool; PyTorch and ONNX Runtime use the physical core count. For the LLMs, both engines get the better of 8 and 16 threads.")
w("- **Agreement:** encoder outputs are compared with PyTorch's; the largest difference is listed with each row. Generated text is recorded so the engines can be checked to have done the same work.")
v = versions()
if v:
    w("- **Versions:** " + ", ".join(f"{k} {val}" for k, val in v.items()) + ", llama.cpp b11901.\n")

enc = load("encoders")
if enc:
    w("## Embeddings, reranking and classification\n")
    for c, alt in [("encoders-throughput", "Batch throughput"), ("encoders-load", "Load time"), ("encoders-memory", "Peak memory")]:
        w(picture(c, alt))
    name = {"kjarni": "**Kjarni**", "torch": "PyTorch", "onnx": "ONNX Runtime"}
    task = {"embed-minilm": "Embed · all-MiniLM-L6-v2", "embed-mpnet": "Embed · all-mpnet-base-v2",
            "rerank": "Rerank · ms-marco-MiniLM-L6", "classify": "Classify · DistilBERT SST-2"}
    w("| Task | Engine | Batch of 64 | One text | Load | Peak memory | Max diff vs PyTorch |")
    w("|---|---|--:|--:|--:|--:|--:|")
    for r in enc["rows"]:
        diff = "reference" if r["engine"] == "torch" else f'{r.get("max_abs_diff_vs_torch", float("nan")):.0e}'
        w(f'| {task[r["task"]]} | {name[r["engine"]]} | {r["batch_ms"]["min"]:.0f} ms ({r["batch_per_s"]:.0f}/s) | '
          f'{r["single_ms"]["min"]:.1f} ms | {r["load_ms"]:,.0f} ms | {r["peak_rss_mb"]:,.0f} MB | {diff} |')
    w("")

wh = load("whisper")
if wh:
    w("## Speech to text\n")
    w(picture("whisper", "Whisper speed"))
    w("whisper-small, English, greedy decoding, CPU.\n")
    w("| Audio | Engine | Time | Faster than real time | Load | Peak memory |")
    w("|---|---|--:|--:|--:|--:|")
    for r in wh:
        e = "**Kjarni**" if r["engine"] == "kjarni" else r["engine"]
        w(f'| {r["audio_secs"]:.0f} s | {e} | {r["secs"]:.2f} s | {r["realtime_x"]:.1f}× | {r["load_ms"]:,.0f} ms | {r["peak_rss_mb"]:,.0f} MB |')
    w("")

s2 = load("seq2seq")
if s2:
    w("## Summarization and translation\n")
    w("distilbart-cnn-12-6 and flan-t5-base, each with its checkpoint's own generation settings.\n")
    w("| Task | Engine | Time | Load | Peak memory | Output |")
    w("|---|---|--:|--:|--:|---|")
    for r in s2:
        e = "**Kjarni**" if r["engine"] == "kjarni" else "PyTorch"
        out = r["output"].replace("|", "\\|")
        out = out if len(out) < 90 else out[:87] + "…"
        w(f'| {r["task"]} | {e} | {r["ms"]:,.0f} ms | {r["load_ms"]:,.0f} ms | {r["peak_rss_mb"]:,.0f} MB | {out} |')
    w("")

cl = load("clip")
if cl:
    w("## Image search (CLIP)\n")
    w("ViT-B/32 with the LAION-2B weights, indexing 64 photos and running one text query.\n")
    w("| Engine | Images per second | Query | Load | Peak memory |")
    w("|---|--:|--:|--:|--:|")
    for r in cl:
        e = "**Kjarni**" if r["engine"] == "kjarni" else r["engine"]
        w(f'| {e} | {r["images_per_s"]:.1f} | {r["query_ms"]:.1f} ms | {r["load_ms"]:,.0f} ms | {r["peak_rss_mb"]:,.0f} MB |')
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

open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", os.path.abspath(OUT))
