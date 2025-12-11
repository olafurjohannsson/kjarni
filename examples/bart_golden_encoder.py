import torch
from safetensors.torch import load_file

# Path to your local weights file
path = "/home/olafurj/.cache/edgetransformers/olafuraron_distilbart-cnn-12-6/model.safetensors"

try:
    weights = load_file(path)
    fc1_name = "model.encoder.layers.0.fc1.weight"

    if fc1_name in weights:
        shape = weights[fc1_name].shape
        print(f"Tensor: {fc1_name}")
        print(f"Shape: {shape}")

        if shape == (4096, 1024):
            print("VERDICT: Layout is [Out, In] (Standard PyTorch)")
        elif shape == (1024, 4096):
            print("VERDICT: Layout is [In, Out] (Tensorflow/GPT-2 style)")
        else:
            print("VERDICT: Unknown/Unexpected dimensions")
    else:
        print(f"Could not find {fc1_name}")

except Exception as e:
    print(e)
