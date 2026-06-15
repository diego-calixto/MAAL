import torch

obj = torch.load("checkpoints/Attention/fold_0/best.pt", map_location="cpu")

print(type(obj))

if isinstance(obj, dict):
    print(obj.keys())