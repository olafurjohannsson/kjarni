"""Writes the PyTorch weight files tests/pytorch_weights.rs reads, and the
values torch.load gives for them.

    python make.py   # in this folder, with torch installed
"""
import collections, json, os

import torch

torch.manual_seed(0)
base = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 7
shared = torch.randn(5, 3)
state = collections.OrderedDict()
state["dense.weight"] = base.clone()
state["dense.bias"] = torch.linspace(-1, 1, 6)
state["transposed"] = base.t()                      # a strided view of the same storage
state["tied.a"] = shared                            # two names, one storage
state["tied.b"] = shared
state["offset_view"] = base.view(-1)[5:11]          # a view at an offset
state["half"] = (torch.randn(3, 4) * 3).half()
state["bf16"] = torch.randn(2, 5).bfloat16()
state["double"] = torch.randn(4, dtype=torch.float64)
state["scalar"] = torch.tensor(2.5)
state["position_ids"] = torch.arange(8)             # int64: present, not a float weight
state["param"] = torch.nn.Parameter(torch.ones(2, 2) * 3)

out = {}
for name, t in state.items():
    if t.dtype == torch.int64:
        out[name] = None
    else:
        out[name] = {"shape": list(t.shape), "values": t.detach().float().flatten().tolist()}

here = os.path.dirname(os.path.abspath(__file__))
torch.save(state, os.path.join(here, "zip.bin"))
torch.save(state, os.path.join(here, "legacy.bin"), _use_new_zipfile_serialization=False)
# A checkpoint that wraps its weights.
torch.save({"state_dict": collections.OrderedDict(w=base[:2].clone())}, os.path.join(here, "nested.bin"))
out["nested"] = {"state_dict.w": {"shape": [2, 6], "values": base[:2].flatten().tolist()}}
json.dump(out, open(os.path.join(here, "expected.json"), "w"), indent=1)
print(torch.__version__)
