"""Times one cold start of MiniLM embeddings in a fresh process: import, model
load, first vector. Prints JSON.

    python coldstart_worker.py ENGINE {warm|cold}

ENGINE is kjarni or torch (sentence-transformers), imported from whatever venv
runs this, so run_footprint.py points it at a clean install of each. 'cold'
first drops the model files and every file of the venv's site-packages from the
OS page cache with posix_fadvise(DONTNEED), the state after a reboot.
"""
import glob, json, os, site, sys, time

ENGINE, MODE = sys.argv[1], sys.argv[2]
MODEL_DIR = os.path.expanduser("~/.cache/kjarni/sentence-transformers_all-MiniLM-L6-v2")


def evict(root):
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.isfile(p) and not os.path.islink(p):
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)


if MODE == "cold":
    evict(MODEL_DIR)
    for sp in site.getsitepackages():
        evict(sp)

t0 = time.perf_counter()
if ENGINE == "kjarni":
    import kjarni
    t_import = time.perf_counter()
    model = kjarni.Embedder(model="minilm-l6-v2", device="cpu", quiet=True)
else:
    import torch
    from sentence_transformers import SentenceTransformer
    t_import = time.perf_counter()
    model = SentenceTransformer(MODEL_DIR, device="cpu")
t_load = time.perf_counter()
vector = model.encode("hello world")
t_first = time.perf_counter()

ms = lambda a, b: (b - a) * 1000
print(json.dumps(dict(engine=ENGINE, mode=MODE, import_ms=ms(t0, t_import), load_ms=ms(t_import, t_load),
                      first_ms=ms(t_load, t_first), total_ms=ms(t0, t_first), dim=len(vector))))
