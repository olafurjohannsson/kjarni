"""Builds every binding's benchmark, then runs each in turn, three rounds, each run in
a fresh process on a quiet machine. Keeps each language's lowest median, so one busy
moment cannot pass for overhead. Writes results/bindings.json."""
import json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../.."))
sys.path.insert(0, os.path.join(REPO, "bench/compare"))
from idle import wait_quiet

LIB_DIR = os.path.join(REPO, "target/release")
BUILD = os.path.join(REPO, "target/binding-overhead")
os.makedirs(BUILD, exist_ok=True)
ENV = {**os.environ, "LD_LIBRARY_PATH": LIB_DIR}


def sh(args, cwd=REPO):
    subprocess.run(args, cwd=cwd, check=True, env=ENV, stdout=subprocess.DEVNULL)


sh(["cargo", "build", "--release", "-p", "kjarni-ffi"])
sh(["cargo", "build", "--release", "-p", "kjarni", "--example", "binding_overhead"])
c_bin = os.path.join(BUILD, "overhead_c")
sh(["gcc", "-O2", os.path.join(HERE, "overhead.c"), "-I", "crates/kjarni-ffi/include",
    "-L", LIB_DIR, "-lkjarni_ffi", "-o", c_bin])
go_bin = os.path.join(BUILD, "overhead_go")
sh(["go", "mod", "tidy"], cwd=os.path.join(HERE, "go"))
sh(["go", "build", "-o", go_bin, "."], cwd=os.path.join(HERE, "go"))
dotnet_out = os.path.join(BUILD, "csharp")
sh(["dotnet", "publish", "-c", "Release", "-o", dotnet_out], cwd=os.path.join(HERE, "csharp"))
shutil.copy(os.path.join(LIB_DIR, "libkjarni_ffi.so"), dotnet_out)
shutil.copy(os.path.join(LIB_DIR, "libkjarni_ffi.so"), BUILD)

RUNS = {
    "C": [c_bin],
    "Rust": [os.path.join(LIB_DIR, "examples/binding_overhead")],
    "Python": [os.path.expanduser("~/.cache/kjarni-bench/venv/bin/python"), os.path.join(HERE, "overhead.py")],
    "Go": [go_bin],
    "C#": [os.path.join(dotnet_out, "Overhead")],
}
best = {}
for _ in range(3):
    for lang, cmd in RUNS.items():
        wait_quiet()
        p = subprocess.run(cmd, capture_output=True, text=True, env=ENV, cwd=dotnet_out if lang == "C#" else BUILD)
        if p.returncode != 0:
            print(lang, "FAILED", p.stderr[-800:], flush=True)
            continue
        r = json.loads(p.stdout.strip().splitlines()[-1])
        print(r, flush=True)
        if lang not in best or r["median_us"] < best[lang]["median_us"]:
            best[lang] = r
base = best["C"]["median_us"]
for r in best.values():
    r["overhead_us"] = r["median_us"] - base
json.dump(list(best.values()), open(os.path.join(REPO, "bench/compare/results/bindings.json"), "w"), indent=2)
for r in best.values():
    print(f"{r['lang']:<8}{r['median_us']:9.1f} us  {r['overhead_us']:+8.1f} us")
