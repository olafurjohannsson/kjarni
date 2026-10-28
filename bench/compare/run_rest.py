"""Runs the Whisper, seq2seq, CLIP and decoder comparisons, each engine in its
own process, and writes results/*.json. Encoders are run_encoders.py."""
import json, os, subprocess, sys
from idle import wait_idle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
PY = sys.executable
BENCH = os.path.expanduser("~/.cache/kjarni-bench")
CACHE = os.path.expanduser("~/.cache/kjarni")
LLAMA_BENCH = os.path.join(BENCH, "llama/llama-b11901/llama-bench")


def run(args, env=None):
    wait_idle()
    p = subprocess.run(args, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if p.returncode != 0:
        print("FAILED", " ".join(args[:4]), p.stderr[-1500:], file=sys.stderr, flush=True)
        return None
    return json.loads(p.stdout.strip().splitlines()[-1])


def save(name, rows):
    json.dump(rows, open(os.path.join(OUT, f"{name}.json"), "w"), indent=2)


ONLY = os.environ.get("ONLY", "whisper,seq2seq,clip,decode")

# Whisper
rows = []
if "whisper" not in ONLY:
    rows = json.load(open(os.path.join(OUT, "whisper.json")))
for audio in ([] if "whisper" not in ONLY else ["/home/olafurj/dev/edgebert/crates/kjarni/tests/fixtures/speech.wav", f"{BENCH}/speech_55s.wav"]):
    for engine in ["kjarni", "faster-whisper", "faster-whisper-int8", "openai-whisper"]:
        r = run([PY, f"{HERE}/whisper_worker.py", engine, audio])
        if r:
            rows.append(r)
            print(f"whisper {r['audio']:<16}{engine:<22}{r['secs']:6.2f}s  {r['realtime_x']:5.1f}x realtime  load {r['load_ms']:6.0f}ms  rss {r['peak_rss_mb']:5.0f}MB", flush=True)
save("whisper", rows)

# Summarization and translation, the faster of two runs per engine.
if "seq2seq" in ONLY:
    rows = []
    for task in ["summarize", "translate"]:
        for engine in ["kjarni", "torch"]:
            runs = [r for _ in range(2) if (r := run([PY, f"{HERE}/seq2seq_worker.py", engine, task]))]
            if runs:
                r = min(runs, key=lambda r: r["ms"])
                rows.append(r)
                print(f"{task:<10}{engine:<8}{r['ms']:8.0f}ms  load {r['load_ms']:6.0f}ms  rss {r['peak_rss_mb']:5.0f}MB  | {r['output'][:70]}", flush=True)
    save("seq2seq", rows)

# CLIP
if "clip" in ONLY:
    rows = []
    for engine in ["kjarni", "open_clip"]:
        r = run([PY, f"{HERE}/clip_worker.py", engine, f"{BENCH}/photos64"])
        if r:
            rows.append(r)
            print(f"clip {engine:<10}{r['images_per_s']:7.1f} img/s  query {r['query_ms']:5.1f}ms  top={r['top_hit']}  load {r['load_ms']:6.0f}ms  rss {r['peak_rss_mb']:5.0f}MB", flush=True)
    save("clip", rows)

# Decoders: the same GGUF file in both engines; best of 8 and 16 threads each.
if os.environ.get("ONLY") and "decode" not in os.environ["ONLY"]:
    raise SystemExit
rows = []
# The files Kjarni's registry downloads for `quantized`, so llama.cpp is given
# exactly the bytes Kjarni loads.
models = [("qwen2.5-0.5b-instruct", f"{CACHE}/Qwen_Qwen2.5-0.5B-Instruct/model.gguf", f"{CACHE}/Qwen_Qwen2.5-0.5B-Instruct/tokenizer.json"),
          ("llama3.2-1b-instruct", f"{CACHE}/meta-llama_Llama-3.2-1B-Instruct/model.gguf", f"{CACHE}/meta-llama_Llama-3.2-1B-Instruct/tokenizer.json")]
for name, gguf, tokjson in models:
    # Each metric at its own best thread count, as llama-bench reports llama.cpp:
    # prompts usually want 16 threads and generation 8.
    runs = [r for t in ["8", "16"]
            if (r := run([PY, f"{HERE}/decoder_worker.py", name, gguf, tokjson], env={"RAYON_NUM_THREADS": t}))]
    if runs:
        pp = max(runs, key=lambda r: r["prompt_tok_s"])
        tg = max(runs, key=lambda r: r["gen_tok_s"])
        best = {**tg, "model": name, "prompt_tok_s": pp["prompt_tok_s"], "ttft_ms": pp["ttft_ms"],
                "threads": [pp["threads"], tg["threads"]]}
        rows.append(best)
        print(f"decode {name:<40} kjarni     pp {pp['prompt_tok_s']:7.1f} tok/s  tg {tg['gen_tok_s']:6.1f} tok/s  threads {pp['threads']}/{tg['threads']}", flush=True)
    wait_idle()
    p = subprocess.run([LLAMA_BENCH, "-m", gguf, "-p", "512", "-n", "128", "-t", "8,16", "-r", "3", "-o", "json"],
                       capture_output=True, text=True)
    if p.returncode == 0:
        res = json.loads(p.stdout)
        pp = max((x for x in res if x["n_prompt"] > 0), key=lambda x: x["avg_ts"])
        tg = max((x for x in res if x["n_gen"] > 0), key=lambda x: x["avg_ts"])
        rows.append(dict(engine="llama.cpp", model=name, prompt_tok_s=pp["avg_ts"], gen_tok_s=tg["avg_ts"],
                         threads=[pp["n_threads"], tg["n_threads"]], build=pp.get("build_commit")))
        print(f"decode {name:<40} llama.cpp  pp {pp['avg_ts']:7.1f} tok/s  tg {tg['avg_ts']:6.1f} tok/s  threads {pp['n_threads']}/{tg['n_threads']}", flush=True)
    else:
        print("llama-bench FAILED", p.stderr[-800:], file=sys.stderr)
save("decoders", rows)
