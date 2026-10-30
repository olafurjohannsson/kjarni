"""Times PyTorch generating with the model's safetensors weights: prompt
processing and generation speed, greedy. The counterpart of decoder_worker.py.
Prints JSON.

    python torch_decoder_worker.py MODEL_NAME MODEL_DIR

512 prompt tokens and 128 generated, like llama-bench's pp512 / tg128. float32 on
the CPU; with BENCH_DEVICE=gpu, bfloat16 on CUDA, the usual way to run these there.
"""
import json, os, sys, time, resource
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer

MODEL, DIR = sys.argv[1], sys.argv[2]
GPU = os.environ.get("BENCH_DEVICE") == "gpu"
DEVICE = "cuda" if GPU else "cpu"
PROMPT_TOKENS, GEN_TOKENS = 512, 128

tok = Tokenizer.from_file(os.path.join(DIR, "tokenizer.json"))
filler = ("The committee reviewed the quarterly figures, the supplier contracts and the hiring plan, "
          "and agreed to revisit the budget for the new warehouse next month. ") * 40
ids = torch.tensor([tok.encode(filler).ids[:PROMPT_TOKENS]], device=DEVICE)

t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(DIR, dtype=torch.bfloat16 if GPU else torch.float32).eval().to(DEVICE)
load_ms = (time.perf_counter() - t0) * 1000


class Stamps(BaseStreamer):
    """generate() hands a streamer the prompt once, then each new token."""

    def __init__(self):
        self.calls = []

    def put(self, value):
        if GPU:
            torch.cuda.synchronize()
        self.calls.append(time.perf_counter())

    def end(self):
        pass


def once():
    s = Stamps()
    start = time.perf_counter()
    with torch.no_grad():
        model.generate(ids, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=GEN_TOKENS,
                       min_new_tokens=GEN_TOKENS, streamer=s, pad_token_id=0)
    return start, s.calls[1:]


once()  # warm-up
best = None
for _ in range(3):
    start, stamps = once()
    first = stamps[0] - start
    r = dict(ttft_ms=first * 1000, prompt_tok_s=PROMPT_TOKENS / first,
             gen_tok_s=(len(stamps) - 1) / (stamps[-1] - stamps[0]), generated=len(stamps))
    if best is None or r["gen_tok_s"] > best["gen_tok_s"]:
        best = r
# Generation from a nearly empty context, which is what llama-bench's tg128 measures.
ids = ids[:, :16]
once()
_, stamps = once()
best["gen_short_tok_s"] = (len(stamps) - 1) / (stamps[-1] - stamps[0])
print(json.dumps(dict(engine="torch", device="gpu" if GPU else "cpu", model=MODEL, load_ms=load_ms,
                      threads=torch.get_num_threads(), dtype="bfloat16" if GPU else "float32",
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, **best)))
