import torch
import torch_directml

dml = torch_directml.device()

device="cpu"  

# Perform a matrix multiplication
a = torch.randn(1000, 1000, device=dml)
b = torch.randn(1000, 1000, device=device)
c = torch.matmul(a, b)
print("Matrix multiplication result shape:", c.shape)