from safetensors.torch import load_file
from pathlib import Path
import os

# Find the file in your cache (adjust path if needed)
home = Path.home()
# Look in ~/.cache/kjarni or the specific path from your logs
cache_path = home / ".cache/kjarni/nomic-ai_nomic-embed-text-v1.5" 

if not cache_path.exists():
    print(f"Path not found: {cache_path}")
    exit()

f_path = cache_path / "model.safetensors"
if not f_path.exists():
    print("model.safetensors not found")
    exit()

tensors = load_file(f_path)
print("--- Nomic Keys ---")
for k in tensors.keys():
    #if "embedding" in k or "layer.0" in k:
    print(f"{k}: {tensors[k].shape}")