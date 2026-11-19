"""Install size and cold start, measured on clean installs of the published
packages. Writes results/footprint.json.

Each stack goes into its own new venv under ~/.cache/kjarni-bench/footprint:
  kjarni            pip install kjarni==VERSION (everything: CPU and GPU)
  torch-cpu         torch from the CPU-only index, plus sentence-transformers
  torch             pip install torch sentence-transformers (CUDA build on Linux)
  onnxruntime       onnxruntime plus tokenizers, the runtime alone

Size is the venv's site-packages on disk, minus pip itself. Cold start runs
coldstart_worker.py in the kjarni and torch-cpu venvs: the whole process
(interpreter included) and its import / load / first-vector split, with the OS
file cache warm and dropped, REPEATS times each, keeping the median.
SKIP_INSTALL=1 reuses venvs already there.
"""
import json, os, shutil, statistics, subprocess, sys, time
from idle import wait_quiet

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
ROOT = os.path.expanduser("~/.cache/kjarni-bench/footprint")
VERSION = os.environ.get("KJARNI_VERSION", "0.7.0")
REPEATS = 5
STACKS = {
    "kjarni": [[f"kjarni=={VERSION}"]],
    "torch-cpu": [["torch", "--index-url", "https://download.pytorch.org/whl/cpu"], ["sentence-transformers"]],
    "torch": [["torch", "sentence-transformers"]],
    "onnxruntime": [["onnxruntime", "tokenizers"]],
}


def du(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            p = os.path.join(dirpath, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


def site_packages(venv):
    lib = os.path.join(venv, "lib")
    return os.path.join(lib, os.listdir(lib)[0], "site-packages")


def versions(py):
    out = subprocess.run([py, "-m", "pip", "list", "--format=json"], capture_output=True, text=True, check=True).stdout
    return {p["name"]: p["version"] for p in json.loads(out)}


rows = []
for name, installs in STACKS.items():
    venv = os.path.join(ROOT, name)
    py = os.path.join(venv, "bin", "python")
    install_s = None
    if not (os.environ.get("SKIP_INSTALL") and os.path.exists(py)):
        shutil.rmtree(venv, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "venv", venv], check=True)
        t = time.perf_counter()
        for args in installs:
            subprocess.run([py, "-m", "pip", "install", "-q", "--no-cache-dir", *args], check=True)
        install_s = time.perf_counter() - t
    sp = site_packages(venv)
    pip_bytes = sum(du(os.path.join(sp, d)) for d in os.listdir(sp) if d == "pip" or d.startswith("pip-"))
    v = versions(py)
    row = dict(stack=name, installed_mb=(du(sp) - pip_bytes) / 1e6, install_s=install_s,
               packages=len(v) - 1, versions={k: v[k] for k in ["kjarni", "torch", "sentence-transformers", "onnxruntime"] if k in v})
    rows.append(row)
    print(f"{name:<12} {row['installed_mb']:8,.0f} MB  {row['packages']:3} packages  install {install_s or 0:5.1f} s  {row['versions']}", flush=True)

cold = []
for name, engine in [("kjarni", "kjarni"), ("torch-cpu", "torch")]:
    py = os.path.join(ROOT, name, "bin", "python")
    for mode in ["warm", "cold"]:
        runs = []
        for _ in range(REPEATS):
            wait_quiet()
            t = time.perf_counter()
            p = subprocess.run([py, "-I", os.path.join(HERE, "coldstart_worker.py"), engine, mode],
                               capture_output=True, text=True, check=True)
            process_ms = (time.perf_counter() - t) * 1000
            runs.append({**json.loads(p.stdout.strip().splitlines()[-1]), "process_ms": process_ms})
        med = {k: statistics.median(r[k] for r in runs) for k in ["import_ms", "load_ms", "first_ms", "total_ms", "process_ms"]}
        cold.append(dict(stack=name, engine=engine, mode=mode, **med, runs=runs))
        print(f"cold start {name:<10} {mode}  process {med['process_ms']:6.0f} ms  import {med['import_ms']:6.0f}  "
              f"load {med['load_ms']:5.0f}  first {med['first_ms']:4.0f}", flush=True)

json.dump(dict(machine="Intel Core i7-13700, Linux, Python " + sys.version.split()[0], install=rows, coldstart=cold),
          open(os.path.join(OUT, "footprint.json"), "w"), indent=2)
