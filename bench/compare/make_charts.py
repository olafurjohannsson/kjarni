"""Turns results/*.json into light and dark SVG bar charts in bench/charts/.

Hand-written SVG rather than matplotlib: the charts are small, and this keeps
them crisp, theme-aware (the README picks light or dark with <picture>), and
free of a plotting dependency. Colours are the validated reference palette:
one fixed hue per engine, the same in every chart, and every bar carries its
value as a label because two of the light hues are under 3:1 contrast.
"""
import json, os, shutil, subprocess
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "charts")
os.makedirs(OUT, exist_ok=True)

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", base="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", base="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}
# Engine -> palette slot, fixed across every chart: colour follows the engine.
SLOT = {"Kjarni": 0, "PyTorch": 1, "ONNX Runtime": 2, "llama.cpp": 2, "faster-whisper": 2,
        "faster-whisper int8": 3, "openai-whisper": 1, "open_clip": 1,
        "PyTorch CUDA": 1, "ONNX Runtime CUDA": 2, "Kjarni WebGPU": 0, "llama.cpp CUDA": 2, "llama.cpp Vulkan": 3}
FONT = "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\""


def bar_chart(name, title, subtitle, groups, engines, values, fmt, theme, per_group=False):
    """groups: row labels; values[g][e] (None to skip); one horizontal bar per engine.

    per_group scales each group to its own largest value, as small multiples:
    for models an order of magnitude apart, where one shared scale would leave
    the slower model's bars too short to compare. Groups are then separated by a
    rule so they read as separate panels, and every bar is labelled.
    """
    t = THEMES[theme]
    W, label_w, right_pad = 760, 190, 90
    bar_h, bar_gap, group_gap = 16, 2, 18
    top = 92
    plot_w = W - label_w - right_pad
    vmax = max(v for row in values for v in row if v is not None)
    rows_h = sum(len([v for v in row if v is not None]) * (bar_h + bar_gap) - bar_gap for row in values)
    H = top + rows_h + group_gap * (len(groups) - 1) + 28
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" {FONT} role="img" aria-label="{escape(title)}">',
         f'<rect width="{W}" height="{H}" rx="8" fill="{t["surface"]}"/>',
         f'<text x="20" y="32" font-size="17" font-weight="600" fill="{t["ink"]}">{escape(title)}</text>',
         f'<text x="20" y="54" font-size="13" fill="{t["ink2"]}">{escape(subtitle)}</text>']
    # Legend, one row under the subtitle.
    x = 20
    for e in engines:
        c = t["series"][SLOT[e]]
        o.append(f'<rect x="{x}" y="66" width="11" height="11" rx="3" fill="{c}"/>')
        o.append(f'<text x="{x + 16}" y="76" font-size="12.5" fill="{t["ink2"]}">{escape(e)}</text>')
        x += 16 + 7.4 * len(e) + 22
    y = top
    for gi, (g, row) in enumerate(zip(groups, values)):
        if per_group:
            vmax = max(v for v in row if v is not None)
            if gi:
                o.append(f'<line x1="20" y1="{y - group_gap / 2 - 1}" x2="{W - 20}" y2="{y - group_gap / 2 - 1}" stroke="{t["base"]}" stroke-width="0.5" stroke-dasharray="2 3"/>')
        present = [(e, v) for e, v in zip(engines, row) if v is not None]
        block = len(present) * (bar_h + bar_gap) - bar_gap
        o.append(f'<text x="{label_w - 14}" y="{y + block / 2 + 4.5}" text-anchor="end" font-size="13" fill="{t["ink"]}">{escape(g)}</text>')
        o.append(f'<line x1="{label_w}" y1="{y - 3}" x2="{label_w}" y2="{y + block + 3}" stroke="{t["base"]}" stroke-width="1"/>')
        for e, v in present:
            w = max(2.0, plot_w * v / vmax)
            r = min(4, w / 2)
            c = t["series"][SLOT[e]]
            # Square at the baseline, 4px rounded at the data end.
            o.append(f'<path d="M{label_w + 1},{y} h{w - r} a{r},{r} 0 0 1 {r},{r} v{bar_h - 2 * r} a{r},{r} 0 0 1 -{r},{r} h-{w - r} z" fill="{c}"><title>{escape(e)}: {escape(fmt(v))}</title></path>')
            weight = ' font-weight="600"' if e.startswith("Kjarni") else ""
            o.append(f'<text x="{label_w + 1 + w + 6}" y="{y + bar_h - 3.5}" font-size="12" fill="{t["ink2"]}"{weight}>{escape(fmt(v))}</text>')
            y += bar_h + bar_gap
        y += group_gap - bar_gap
    o.append("</svg>")
    path = os.path.join(OUT, f"{name}-{theme}.svg")
    open(path, "w").write("\n".join(o))
    return path


def both(*args, **kw):
    for theme in THEMES:
        bar_chart(*args, theme, **kw)


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


ENGINE = {"kjarni": "Kjarni", "torch": "PyTorch", "onnx": "ONNX Runtime"}


def value_ms(row):
    """The whole process's time, interpreter start included, as a user waits for it."""
    return row["process_ms"] if row else None

enc = load("encoders")
if enc:
    rows = enc["rows"]
    tasks = [("embed-minilm", "Embed · MiniLM-L6"), ("embed-mpnet", "Embed · MPNet-base"),
             ("rerank", "Rerank · MiniLM CE"), ("classify", "Classify · DistilBERT")]
    engines = ["Kjarni", "PyTorch", "ONNX Runtime"]
    def grid(key):
        out = []
        for t, _ in tasks:
            by = {ENGINE[r["engine"]]: r for r in rows if r["task"] == t}
            out.append([key(by[e]) if e in by else None for e in engines])
        return out
    labels = [l for _, l in tasks]
    both("encoders-throughput", "Batch throughput (higher is better)",
         "64 texts of ~80-140 tokens per call, CPU, fp32 · items per second, each model on its own scale",
         labels, engines, grid(lambda r: r["batch_per_s"]), lambda v: f"{v:,.0f}/s", per_group=True)
    both("encoders-latency", "Single-text latency (lower is better)",
         "One short query per call, CPU, fp32 · milliseconds, each model on its own scale", labels, engines,
         grid(lambda r: r["single_ms"]["min"]), lambda v: f"{v:.1f} ms", per_group=True)
    both("encoders-load", "Time to load a model (lower is better)",
         "From constructor call to ready, model already on disk · milliseconds", labels, engines,
         grid(lambda r: r["load_ms"]), lambda v: f"{v:,.0f} ms")
    both("encoders-memory", "Memory in use (lower is better)",
         "Resident memory of the whole process once loaded and after a batch · MB", labels, engines,
         grid(lambda r: r.get("rss_mb", r["peak_rss_mb"])), lambda v: f"{v:,.0f} MB")

wh = load("whisper")
if wh:
    names = {"kjarni": "Kjarni", "openai-whisper": "openai-whisper", "faster-whisper": "faster-whisper",
             "faster-whisper-int8": "faster-whisper int8"}
    engines = ["Kjarni", "faster-whisper", "faster-whisper int8", "openai-whisper"]
    audios = sorted({r["audio"] for r in wh}, key=lambda a: [r["audio_secs"] for r in wh if r["audio"] == a][0])
    vals = [[next((r["realtime_x"] for r in wh if r["audio"] == a and names[r["engine"]] == e), None) for e in engines] for a in audios]
    labels = [f"{[r['audio_secs'] for r in wh if r['audio'] == a][0]:.0f} s of speech" for a in audios]
    both("whisper", "Speech to text speed (higher is better)",
         "whisper-small, English, greedy, CPU · times faster than real time", labels, engines, vals, lambda v: f"{v:.1f}×")

dec = load("decoders")
if dec:
    models = []
    for r in dec:
        if r["model"] not in models:
            models.append(r["model"])
    pretty = lambda m: "Qwen2.5 0.5B Q4_K_M" if "qwen" in m.lower() else "Llama 3.2 1B Q4_K_M"
    engines = ["Kjarni", "llama.cpp"]
    eng = lambda r: "Kjarni" if r["engine"] == "kjarni" else "llama.cpp"
    for key, title, sub, fname in [("gen_tok_s", "Generation speed (higher is better)", "Same GGUF file in both, greedy, CPU · tokens per second", "decode-generation"),
                                   ("prompt_tok_s", "Prompt processing (higher is better)", "512-token prompt, same GGUF file in both, CPU · tokens per second", "decode-prompt")]:
        vals = [[next((r[key] for r in dec if r["model"] == m and eng(r) == e), None) for e in engines] for m in models]
        both(fname, title, sub, [pretty(m) for m in models], engines, vals, lambda v: f"{v:,.0f} tok/s")
# Install size and cold start from run_footprint.py.
fp = load("footprint")
if fp:
    by = {r["stack"]: r for r in fp["install"]}
    stacks = [("kjarni", "Kjarni", "Kjarni"),
              ("onnxruntime", "ONNX Runtime + tokenizers", "ONNX Runtime"),
              ("torch-cpu", "PyTorch, CPU-only build", "PyTorch"),
              ("torch", "PyTorch, default (CUDA)", "PyTorch")]
    engines = ["Kjarni", "ONNX Runtime", "PyTorch"]
    present = [s for s in stacks if s[0] in by]
    size = lambda v: f"{v / 1000:.1f} GB" if v >= 1000 else f"{v:,.0f} MB"
    both("install-size", "Installed size (lower is better)",
         "A clean venv after pip install, site-packages on disk · PyTorch with sentence-transformers",
         [label for _, label, _ in present], engines,
         [[by[key]["installed_mb"] if e == family else None for e in engines] for key, _, family in present], size)
    cs = {(r["engine"], r["mode"]): r for r in fp["coldstart"]}
    both("cold-start", "Start to first embedding (lower is better)",
         "New process, import, load MiniLM-L6, embed one sentence, CPU · median of 5",
         ["Warm OS cache", "After a reboot"], ["Kjarni", "PyTorch"],
         [[value_ms(cs.get((engine, mode))) for engine in ["kjarni", "torch"]] for mode in ["warm", "cold"]],
         lambda v: f"{v / 1000:.2f} s" if v >= 1000 else f"{v:,.0f} ms")

# The CPU and GPU matrix from run_matrix.py.
matrix = load("matrix")
if matrix:
    def cell(**want):
        for r in matrix:
            if "error" not in r and all(r.get(k) == v for k, v in want.items()):
                return r
        return None

    def value(row, key):
        return row[key] if row and key in row else None

    encoders = [("embed-minilm", "MiniLM-L6"), ("embed-bge-m3", "bge-m3")]
    enc_labels = [l for _, l in encoders]
    machine = "i7-13700"
    card = "RTX A2000 12 GB"
    for device, names, where in [("cpu", ["Kjarni", "PyTorch", "ONNX Runtime"], machine),
                                 ("gpu", ["Kjarni WebGPU", "PyTorch CUDA", "ONNX Runtime CUDA"], card)]:
        keys = dict(zip(names, ["kjarni", "torch", "onnx"]))
        both(f"matrix-embeddings-{device}", f"Embeddings on the {device.upper()} (higher is better)",
             f"64 texts per call, {where} · texts per second, each model on its own scale", enc_labels, names,
             [[value(cell(kind="encoder", task=t, engine=keys[n], device=device), "batch_per_s") for n in names] for t, _ in encoders],
             lambda v: f"{v:,.0f}/s" if v >= 100 else f"{v:.1f}/s", per_group=True)

    names = ["Kjarni", "faster-whisper", "faster-whisper int8", "openai-whisper"]
    keys = dict(zip(names, ["kjarni", "faster-whisper", "faster-whisper-int8", "openai-whisper"]))
    both("matrix-whisper", "Speech to text (higher is better)",
         "whisper-small, 58 s of English, greedy · times faster than real time, each device on its own scale",
         [f"CPU, {machine}", f"GPU, {card}"], names,
         [[value(cell(kind="whisper", engine=keys[n], device=d), "realtime_x") for n in names] for d in ["cpu", "gpu"]],
         lambda v: f"{v:.1f}×", per_group=True)

    models = [("qwen2.5-0.5b-instruct", "Qwen2.5 0.5B"), ("llama3.2-1b-instruct", "Llama 3.2 1B")]
    # Every panel compares engines on the same weights: the Q4_K_M GGUF for Kjarni
    # and llama.cpp, the bfloat16 safetensors for Kjarni and PyTorch.
    panels = [(m, w, f"{label} · {'Q4_K_M' if w == 'Q4_K_M' else 'bf16'}") for m, label in models for w in ["Q4_K_M", "safetensors"]]
    panel_labels = [p[2] for p in panels]
    kj = lambda m, w, d, f: value(cell(kind="decoder", model=m, engine="kjarni", weights=w, device=d), f)
    # llama-bench's generation figure starts from an empty context, so the others are
    # shown from a nearly empty one too.
    cpu_gen = lambda m, w: [kj(m, w, "cpu", "gen_short_tok_s"),
                            value(cell(kind="decoder", model=m, engine="llama.cpp", backend="cpu"), "gen_tok_s") if w == "Q4_K_M" else None,
                            value(cell(kind="decoder", model=m, engine="torch", device="cpu"), "gen_short_tok_s") if w != "Q4_K_M" else None]
    gpu_row = lambda m, w, kjarni, llama, torch: [
        kj(m, w, "gpu", kjarni),
        value(cell(kind="decoder", model=m, engine="llama.cpp", backend="vulkan"), llama) if w == "Q4_K_M" else None,
        value(cell(kind="decoder", model=m, engine="llama.cpp", backend="cuda"), llama) if w == "Q4_K_M" else None,
        value(cell(kind="decoder", model=m, engine="torch", device="gpu"), torch) if w != "Q4_K_M" else None]
    tok = lambda v: f"{v:,.0f} tok/s"
    both("matrix-generation-cpu", "Text generation on the CPU (higher is better)",
         f"Same weights in each panel, from a nearly empty context, {machine} · tokens per second",
         panel_labels, ["Kjarni", "llama.cpp", "PyTorch"], [cpu_gen(m, w) for m, w, _ in panels], tok, per_group=True)
    gpu_names = ["Kjarni WebGPU", "llama.cpp Vulkan", "llama.cpp CUDA", "PyTorch CUDA"]
    both("matrix-generation-gpu", "Text generation on the GPU (higher is better)",
         f"Same weights in each panel, from a nearly empty context, {card} · tokens per second", panel_labels, gpu_names,
         [gpu_row(m, w, "gen_short_tok_s", "gen_tok_s", "gen_short_tok_s") for m, w, _ in panels], tok, per_group=True)
    both("matrix-prompt-gpu", "Prompt processing on the GPU (higher is better)",
         f"512-token prompt, same weights in each panel, {card} · tokens per second", panel_labels, gpu_names,
         [gpu_row(m, w, "prompt_tok_s", "prompt_tok_s", "prompt_tok_s") for m, w, _ in panels], tok, per_group=True)

    # Package pages (PyPI, NuGet, crates.io) take a plain image and may not render SVG.
    if shutil.which("rsvg-convert"):
        for name in os.listdir(OUT):
            if name.startswith(("matrix-", "install-size", "cold-start")) and name.endswith("-light.svg"):
                subprocess.run(["rsvg-convert", "--zoom", "2", os.path.join(OUT, name), "-o",
                                os.path.join(OUT, name[:-len(".svg")] + ".png")], check=True)
print("charts written to", os.path.abspath(OUT))
