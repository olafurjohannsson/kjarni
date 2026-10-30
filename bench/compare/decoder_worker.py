"""Times Kjarni generating from a model's Q4_K_M GGUF weights: prompt processing
and generation speed, greedy, on the CPU. Prints JSON. llama.cpp is timed on the
same file with its own llama-bench (pp512 / tg128); run_rest.py does both.

    python decoder_worker.py MODEL_NAME GGUF TOKENIZER_JSON

KJARNI_QUANTIZED=0 uses the safetensors weights instead; BENCH_DEVICE=gpu runs on
the GPU through WebGPU.
"""
import ctypes, json, os, sys, time, resource

MODEL, GGUF, TOKENIZER = sys.argv[1], sys.argv[2], sys.argv[3]
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "crates/kjarni-ffi/bindings/python"))
from kjarni import chat as kc
from kjarni._ffi import check_error
from tokenizers import Tokenizer

PROMPT_TOKENS, GEN_TOKENS = 512, 128

tok = Tokenizer.from_file(TOKENIZER)
filler = ("The committee reviewed the quarterly figures, the supplier contracts and the hiring plan, "
          "and agreed to revisit the budget for the new warehouse next month. ") * 40
ids = tok.encode(filler).ids[:PROMPT_TOKENS - 24]
prompt = tok.decode(ids) + "\n\nSummarise the text above in detail."

t0 = time.perf_counter()
cfg = kc._lib.kjarni_chat_config_default()
cfg.model_name = MODEL.encode()
cfg.quantized = int(os.environ.get("KJARNI_QUANTIZED", "1"))  # 1: the GGUF weights, the same file llama.cpp is given
cfg.quiet = 1
cfg.device = kc.KjarniDevice.GPU if os.environ.get("BENCH_DEVICE") == "gpu" else kc.KjarniDevice.CPU
handle = ctypes.c_void_p()
check_error(kc._lib.kjarni_chat_new(ctypes.byref(cfg), ctypes.byref(handle)))
load_ms = (time.perf_counter() - t0) * 1000

gen = kc._lib.kjarni_generation_config_default()
gen.temperature = 0.0
gen.do_sample = 0
gen.max_new_tokens = GEN_TOKENS


def once():
    stamps = []
    cb = kc.STREAM_CALLBACK_TYPE(lambda text, _: stamps.append(time.perf_counter()) or True)
    start = time.perf_counter()
    check_error(kc._lib.kjarni_chat_stream(handle, prompt.encode(), ctypes.byref(gen), cb, None, None))
    return start, stamps


once()  # warm-up
best = None
for _ in range(3):
    start, stamps = once()
    first = stamps[0] - start
    decode = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
    r = dict(ttft_ms=first * 1000, prompt_tok_s=(len(ids) + 24) / first, gen_tok_s=decode, generated=len(stamps))
    if best is None or r["gen_tok_s"] > best["gen_tok_s"]:
        best = r
# Generation from a nearly empty context, which is what llama-bench's tg128 measures;
# the figure above follows a 512-token prompt, so attention has far more keys to read.
prompt = "Write a long story about a lighthouse keeper."
once()
start, stamps = once()
best["gen_short_tok_s"] = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
kc._lib.kjarni_chat_free(handle)
print(json.dumps(dict(engine="kjarni", model=os.path.basename(GGUF), load_ms=load_ms,
                      threads=int(os.environ.get("RAYON_NUM_THREADS") or os.cpu_count()),
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, **best)))
