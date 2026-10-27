"""Times CLIP ViT-B/32 (LAION-2B weights) embedding a folder of 64 photos and a
text query, on the CPU, in a fresh process. Prints JSON.

    python clip_worker.py ENGINE PHOTO_DIR      ENGINE: kjarni | open_clip
"""
import json, os, sys, time, resource, glob

ENGINE, PHOTOS = sys.argv[1], sys.argv[2]
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FILES = sorted(glob.glob(os.path.join(PHOTOS, "*.png")))
QUERY = "a red car"

t0 = time.perf_counter()
if ENGINE == "kjarni":
    sys.path.insert(0, os.path.join(REPO, "crates/kjarni-ffi/bindings/python"))
    import kjarni
    idx = kjarni.ImageIndex()
    load_ms = (time.perf_counter() - t0) * 1000
    # A fresh index per run, so each does the same work. Creating one loads the
    # model again, which is outside the timed region.
    def images():
        i2 = kjarni.ImageIndex(); t = time.perf_counter(); i2.add_directory(PHOTOS); return time.perf_counter() - t, i2
    def query(i2):
        t = time.perf_counter(); hits = i2.search(QUERY, 1); return time.perf_counter() - t, os.path.basename(hits[0].path)
else:
    import torch, open_clip
    from PIL import Image
    torch.set_grad_enabled(False)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000
    def images():
        t = time.perf_counter()
        batch = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in FILES])
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return time.perf_counter() - t, feats
    def query(feats):
        t = time.perf_counter()
        q = model.encode_text(tokenizer([QUERY])); q = q / q.norm(dim=-1, keepdim=True)
        best = int((feats @ q.T).argmax())
        return time.perf_counter() - t, os.path.basename(FILES[best])

images()  # warm-up
img_s, state = min((images() for _ in range(3)), key=lambda r: r[0])
q_s, top = min((query(state) for _ in range(10)), key=lambda r: r[0])
print(json.dumps(dict(engine=ENGINE, photos=len(FILES), load_ms=load_ms, images_ms=img_s * 1000,
                      images_per_s=len(FILES) / img_s, query_ms=q_s * 1000, top_hit=top,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)))
