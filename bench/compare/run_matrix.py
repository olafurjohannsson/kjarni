"""The full CPU and GPU matrix: Kjarni against PyTorch, ONNX Runtime, llama.cpp,
openai-whisper and faster-whisper on the models people ask about. Every run is a
fresh process on a quiet machine; a run that fails is recorded, not skipped, so
the matrix shows what could not be measured. Writes results/matrix.json.

    python run_matrix.py          (any Python; the workers run in the bench venvs)
    ONLY=encoders,whisper,decoders python run_matrix.py
"""
import json, os, subprocess, sys
import numpy as np
from idle import wait_quiet

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
BENCH = os.path.expanduser("~/.cache/kjarni-bench")
CACHE = os.path.expanduser("~/.cache/kjarni")
CPU_PY, GPU_PY = f"{BENCH}/venv/bin/python", f"{BENCH}/venv-gpu/bin/python"
AUDIO = f"{BENCH}/speech_55s.wav"
LLAMA = {"cpu": f"{BENCH}/llama/llama-b11901", "cuda": f"{BENCH}/llama-gpu/cuda", "vulkan": f"{BENCH}/llama-gpu/vulkan"}
# ctranslate2 is built against CUDA 12, the rest of the GPU venv against 13.
CT2_LIBS = ":".join(f"{BENCH}/venv-gpu/lib/python3.13/site-packages/nvidia/{d}/lib" for d in ["cublas", "cudnn"])
ONLY = os.environ.get("ONLY", "encoders,whisper,decoders")
PATH = os.path.join(OUT, "matrix.json")
rows = json.load(open(PATH)) if os.path.exists(PATH) and os.environ.get("ONLY") else []
# DEVICES=cpu measures the CPU only and keeps the GPU rows already recorded.
DEVICES = os.environ.get("DEVICES", "cpu,gpu").split(",")
# A partial run replaces the rows of the sections and devices it measures again.
KINDS = {"encoders": "encoder", "whisper": "whisper", "decoders": "decoder"}
rerun = {KINDS[s] for s in ONLY.split(",") if s in KINDS}
# TASKS=embed-bge-m3 limits the encoder section to those tasks and keeps the others.
TASKS = [t for t in os.environ.get("TASKS", "").split(",") if t]
# ENGINES=torch limits the decoder section to those engines and keeps the others' rows.
ENGINES = [e for e in os.environ.get("ENGINES", "").split(",") if e]
rows = [r for r in rows if not (r.get("kind") in rerun and r.get("device") in DEVICES
                                and (not TASKS or r.get("kind") != "encoder" or r.get("task") in TASKS)
                                and (not ENGINES or r.get("kind") != "decoder" or r.get("engine") in ENGINES))]


def save():
    json.dump(rows, open(PATH, "w"), indent=2)


def add(row, label):
    rows.append(row)
    save()
    print(f"{label:<58}", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()
                           if k in ("error", "batch_per_s", "load_ms", "realtime_x", "prompt_tok_s", "gen_tok_s", "gen_short_tok_s", "vram_mb", "peak_rss_mb")}, flush=True)


def run(py, args, env=None):
    wait_quiet()
    p = subprocess.run([py, *args], capture_output=True, text=True, env={**os.environ, **(env or {})})
    if p.returncode != 0:
        return dict(error=(p.stderr or p.stdout).strip().splitlines()[-1][:300])
    return json.loads(p.stdout.strip().splitlines()[-1])


def llama_bench(model, gguf, backend):
    wait_quiet()
    d = LLAMA[backend]
    args = [f"{d}/llama-bench", "-m", gguf, "-p", "512", "-n", "128", "-r", "3", "-o", "json"]
    args += ["-t", "8,16"] if backend == "cpu" else ["-ngl", "99"]
    p = subprocess.run(args, capture_output=True, text=True, env={**os.environ, "LD_LIBRARY_PATH": d})
    base = dict(kind="decoder", model=model, engine="llama.cpp", backend=backend, weights="Q4_K_M",
                device="cpu" if backend == "cpu" else "gpu")
    if p.returncode != 0:
        return {**base, "error": p.stderr.strip().splitlines()[-1][:300]}
    res = json.loads(p.stdout)
    pp = max((x for x in res if x["n_prompt"] > 0), key=lambda x: x["avg_ts"])
    tg = max((x for x in res if x["n_gen"] > 0), key=lambda x: x["avg_ts"])
    return {**base, "prompt_tok_s": pp["avg_ts"], "gen_tok_s": tg["avg_ts"], "threads": [pp["n_threads"], tg["n_threads"]]}


if "encoders" in ONLY:
    fresh = []
    for task in [t for t in ["embed-minilm", "embed-bge-m3"] if not TASKS or t in TASKS]:
        for device in [d for d in ["cpu", "gpu"] if d in DEVICES]:
            for engine in ["kjarni", "torch", "onnx"]:
                npy = os.path.join(OUT, f"matrix.{task}.{engine}.{device}.npy")
                py = GPU_PY if device == "gpu" and engine != "kjarni" else CPU_PY
                # On the CPU each engine gets its default thread count and 8 (one per
                # P-core on a hybrid CPU), and keeps its faster one.
                var = {"kjarni": "RAYON_NUM_THREADS", "torch": "TORCH_THREADS", "onnx": "ORT_THREADS"}[engine]
                counts = [None, "8"] if device == "cpu" else [None]
                runs = [run(py, [f"{HERE}/encoder_worker.py", engine, task, npy], {"BENCH_DEVICE": device, **({var: n} if n else {})})
                        for n in counts for _ in range(2)]
                good = [r for r in runs if "error" not in r]
                best = min(good, key=lambda r: r["batch_ms"]["min"]) if good else runs[0]
                row = {**best, "kind": "encoder", "task": task, "engine": engine, "device": device}
                # The first run can include reading the model from disk; report the
                # fastest load, as for the batch.
                if good:
                    row["load_ms"] = min(r["load_ms"] for r in good)
                ref = os.path.join(OUT, f"matrix.{task}.torch.cpu.npy")
                if good and os.path.exists(ref):
                    a, b = np.load(npy), np.load(ref)
                    row["max_abs_diff_vs_torch_cpu"] = float(np.max(np.abs(a - b)))
                add(row, f"{task} {engine} {device}")

if "whisper" in ONLY:
    for device in [d for d in ["cpu", "gpu"] if d in DEVICES]:
        for engine in ["kjarni", "openai-whisper", "faster-whisper", "faster-whisper-int8"]:
            py = GPU_PY if device == "gpu" and engine != "kjarni" else CPU_PY
            env = {"BENCH_DEVICE": device}
            if device == "gpu" and engine.startswith("faster"):
                env["LD_LIBRARY_PATH"] = CT2_LIBS
            # On the CPU each engine also runs at 8 threads, one per P-core, and keeps the faster run.
            var = "RAYON_NUM_THREADS" if engine == "kjarni" else "TORCH_THREADS" if engine == "openai-whisper" else "CT2_THREADS"
            runs = [{**run(py, [f"{HERE}/whisper_worker.py", engine, AUDIO], {**env, **({var: n} if n else {})}), "threads": n or "default"}
                    for n in ([None, "8"] if device == "cpu" else [None])]
            good = [r for r in runs if "error" not in r]
            best = min(good, key=lambda r: r["secs"]) if good else runs[0]
            add({**best, "kind": "whisper", "engine": engine, "device": device}, f"whisper {engine} {device}")

if "decoders" in ONLY:
    MODELS = [("qwen2.5-0.5b-instruct", "Qwen_Qwen2.5-0.5B-Instruct"), ("llama3.2-1b-instruct", "meta-llama_Llama-3.2-1B-Instruct")]
    for model, local in MODELS:
        d = f"{CACHE}/{local}"
        worker = [f"{HERE}/decoder_worker.py", model, f"{d}/model.gguf", f"{d}/tokenizer.json"]
        for quantized, weights in [("1", "Q4_K_M"), ("0", "safetensors")] if not ENGINES or "kjarni" in ENGINES else []:
            if "cpu" not in DEVICES:
                cpu = None
            else:
                cpu = [r for t in ["8", "16"] if "error" not in (r := run(CPU_PY, worker, {"RAYON_NUM_THREADS": t, "KJARNI_QUANTIZED": quantized}))]
            base = dict(kind="decoder", model=model, engine="kjarni", weights=weights)
            if cpu is None:
                pass
            elif cpu:
                pp, tg = max(cpu, key=lambda r: r["prompt_tok_s"]), max(cpu, key=lambda r: r["gen_tok_s"])
                short = max(r["gen_short_tok_s"] for r in cpu)
                add({**tg, **base, "device": "cpu", "prompt_tok_s": pp["prompt_tok_s"], "gen_short_tok_s": short, "threads": [pp["threads"], tg["threads"]]}, f"{model} kjarni {weights} cpu")
            else:
                add({**base, "device": "cpu", "error": "failed at 8 and 16 threads"}, f"{model} kjarni {weights} cpu")
            if "gpu" in DEVICES:
                add({**run(CPU_PY, worker, {"BENCH_DEVICE": "gpu", "KJARNI_QUANTIZED": quantized}), **base, "device": "gpu"}, f"{model} kjarni {weights} gpu")
        # PyTorch runs the same safetensors. On the CPU it gets the better of bfloat16,
        # as stored, and float32, and of 8 and 16 threads, for each figure.
        for device, py, dtypes, threads in [("cpu", CPU_PY, ["bfloat16", "float32"], ["8", "16"]), ("gpu", GPU_PY, ["bfloat16"], [None])]:
            if device not in DEVICES or (ENGINES and "torch" not in ENGINES):
                continue
            tries = [run(py, [f"{HERE}/torch_decoder_worker.py", model, d],
                         {"BENCH_DEVICE": device, "TORCH_DTYPE": t, **({"TORCH_THREADS": n} if n else {})})
                     for t in dtypes for n in threads]
            good = [r for r in tries if "error" not in r]
            if good:
                pp, tg = max(good, key=lambda r: r["prompt_tok_s"]), max(good, key=lambda r: r["gen_tok_s"])
                short = max(good, key=lambda r: r["gen_short_tok_s"])
                best = {**tg, "prompt_tok_s": pp["prompt_tok_s"], "gen_short_tok_s": short["gen_short_tok_s"],
                        "dtype": [pp["dtype"], tg["dtype"]], "threads": [pp["threads"], tg["threads"]],
                        "tries": [{k: r[k] for k in ("dtype", "threads", "prompt_tok_s", "gen_tok_s", "gen_short_tok_s")} for r in good]}
            else:
                best = tries[0]
            add({**best, "kind": "decoder", "model": model, "engine": "torch", "device": device, "weights": "safetensors"}, f"{model} torch {device}")
        for backend in [b for b in LLAMA if ("cpu" if b == "cpu" else "gpu") in DEVICES and (not ENGINES or "llama.cpp" in ENGINES)]:
            add(llama_bench(model, f"{d}/model.gguf", backend), f"{model} llama.cpp {backend}")

print("done", flush=True)
