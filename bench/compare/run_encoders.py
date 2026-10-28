"""Runs every engine on every encoder task, each in its own process, twice, and
keeps the faster run. Writes results/encoders.json, including how far each
engine's output is from PyTorch's.
"""
import json, os, subprocess, sys
import numpy as np
from idle import wait_idle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
TASKS = ["embed-minilm", "embed-mpnet", "rerank", "classify"]
ENGINES = ["kjarni", "torch", "onnx"]
REPEATS = 2


def load_avg():
    return os.getloadavg()[0]


rows = []
for task in TASKS:
    for engine in ENGINES:
        best = None
        for r in range(REPEATS):
            npy = os.path.join(OUT, f"{task}.{engine}.npy")
            wait_idle()
            p = subprocess.run([sys.executable, os.path.join(HERE, "encoder_worker.py"), engine, task, npy],
                               capture_output=True, text=True)
            if p.returncode != 0:
                print(f"{task} {engine} FAILED\n{p.stderr[-2000:]}", file=sys.stderr); break
            d = json.loads(p.stdout.strip().splitlines()[-1])
            d["load_avg_before"] = load_avg()
            if best is None or d["batch_ms"]["min"] < best["batch_ms"]["min"]:
                best = d
        if best:
            rows.append(best)
            print(f"{task:<13}{engine:<8} load {best['load_ms']:7.0f} ms  single {best['single_ms']['min']:7.2f} ms  "
                  f"batch {best['batch_ms']['min']:8.1f} ms  ({best['batch_per_s']:6.0f}/s)  rss {best['peak_rss_mb']:5.0f} MB", flush=True)

# Agreement with PyTorch, the reference implementation.
for row in rows:
    ref = os.path.join(OUT, f"{row['task']}.torch.npy")
    mine = os.path.join(OUT, f"{row['task']}.{row['engine']}.npy")
    if not (os.path.exists(ref) and os.path.exists(mine)):
        continue
    a, b = np.load(mine), np.load(ref)
    row["max_abs_diff_vs_torch"] = float(np.max(np.abs(a - b)))
    if row["task"].startswith("embed"):
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        row["min_cosine_vs_torch"] = float(cos.min())

json.dump(dict(machine=dict(cpu="Intel Core i7-13700 (8P+8E, 24 threads)", ram_gb=64), rows=rows),
          open(os.path.join(OUT, "encoders.json"), "w"), indent=2)
for row in rows:
    if "max_abs_diff_vs_torch" in row and row["engine"] != "torch":
        print(f"{row['task']:<13}{row['engine']:<8} max |diff| vs torch {row['max_abs_diff_vs_torch']:.2e}"
              + (f"  min cosine {row['min_cosine_vs_torch']:.6f}" if "min_cosine_vs_torch" in row else ""))
