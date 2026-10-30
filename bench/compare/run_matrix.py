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
    for task in ["embed-minilm", "embed-bge-m3"]:
        for device in ["cpu", "gpu"]:
            for engine in ["kjarni", "torch", "onnx"]:
                npy = os.path.join(OUT, f"matrix.{task}.{engine}.{device}.npy")
                py = GPU_PY if device == "gpu" and engine != "kjarni" else CPU_PY
                runs = [run(py, [f"{HERE}/encoder_worker.py", engine, task, npy], {"BENCH_DEVICE": device}) for _ in range(2)]
                good = [r for r in runs if "error" not in r]
                best = min(good, key=lambda r: r["batch_ms"]["min"]) if good else runs[0]
                row = {**best, "kind": "encoder", "task": task, "engine": engine, "device": device}
                ref = os.path.join(OUT, f"matrix.{task}.torch.cpu.npy")
                if good and os.path.exists(ref):
                    a, b = np.load(npy), np.load(ref)
                    row["max_abs_diff_vs_torch_cpu"] = float(np.max(np.abs(a - b)))
                add(row, f"{task} {engine} {device}")

if "whisper" in ONLY:
    for device in ["cpu", "gpu"]:
        for engine in ["kjarni", "openai-whisper", "faster-whisper", "faster-whisper-int8"]:
            py = GPU_PY if device == "gpu" and engine != "kjarni" else CPU_PY
            env = {"BENCH_DEVICE": device}
            if device == "gpu" and engine.startswith("faster"):
                env["LD_LIBRARY_PATH"] = CT2_LIBS
            add({**run(py, [f"{HERE}/whisper_worker.py", engine, AUDIO], env), "kind": "whisper", "engine": engine, "device": device}, f"whisper {engine} {device}")

if "decoders" in ONLY:
    MODELS = [("qwen2.5-0.5b-instruct", "Qwen_Qwen2.5-0.5B-Instruct"), ("llama3.2-1b-instruct", "meta-llama_Llama-3.2-1B-Instruct")]
    for model, local in MODELS:
        d = f"{CACHE}/{local}"
        worker = [f"{HERE}/decoder_worker.py", model, f"{d}/model.gguf", f"{d}/tokenizer.json"]
        for quantized, weights in [("1", "Q4_K_M"), ("0", "safetensors")]:
            cpu = [r for t in ["8", "16"] if "error" not in (r := run(CPU_PY, worker, {"RAYON_NUM_THREADS": t, "KJARNI_QUANTIZED": quantized}))]
            base = dict(kind="decoder", model=model, engine="kjarni", weights=weights)
            if cpu:
                pp, tg = max(cpu, key=lambda r: r["prompt_tok_s"]), max(cpu, key=lambda r: r["gen_tok_s"])
                short = max(r["gen_short_tok_s"] for r in cpu)
                add({**tg, **base, "device": "cpu", "prompt_tok_s": pp["prompt_tok_s"], "gen_short_tok_s": short, "threads": [pp["threads"], tg["threads"]]}, f"{model} kjarni {weights} cpu")
            else:
                add({**base, "device": "cpu", "error": "failed at 8 and 16 threads"}, f"{model} kjarni {weights} cpu")
            add({**run(CPU_PY, worker, {"BENCH_DEVICE": "gpu", "KJARNI_QUANTIZED": quantized}), **base, "device": "gpu"}, f"{model} kjarni {weights} gpu")
        for device, py in [("cpu", CPU_PY), ("gpu", GPU_PY)]:
            add({**run(py, [f"{HERE}/torch_decoder_worker.py", model, d], {"BENCH_DEVICE": device}), "kind": "decoder", "model": model, "engine": "torch", "device": device, "weights": "safetensors"}, f"{model} torch {device}")
        for backend in LLAMA:
            add(llama_bench(model, f"{d}/model.gguf", backend), f"{model} llama.cpp {backend}")

print("done", flush=True)
