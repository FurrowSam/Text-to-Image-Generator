import time
import torch
import torch_directml

# Initialize DirectML device
try:
    dml = torch_directml.device()
except Exception as e:
    print("Error initializing DirectML device:", e)
    dml = None

def matrix_multiplication(device1, device2):
    try:
        # Ensure matrices are on the correct devices
        a = torch.randn(15000, 15000, device=device1)
        b = torch.randn(15000, 15000, device=device2)

        if device1 != device2:
            print(f"Transferring tensor from {device2} to {device1} for compatibility.")
            b = b.to(device1)

        # Measure time for the operation
        start_time = time.time()
        c = torch.matmul(a, b)
        end_time = time.time()

        print(f"Matrix multiplication result shape: {c.shape}")
        print(f"Time taken on {device1.type} device: {end_time - start_time:.6f} seconds")
        return end_time - start_time
    except Exception as e:
        print(f"Error during matrix multiplication on devices {device1.type} and {device2.type}:", e)
        return None

# Compare performance on CPU and DirectML
if dml is not None:
    print("Running matrix multiplication on CPU and DirectML for comparison...")
    time_cpu = matrix_multiplication(torch.device("cpu"), torch.device("cpu"))
    time_dml = matrix_multiplication(dml, dml)
    if time_cpu is not None and time_dml is not None:
        print(f"CPU time: {time_cpu:.6f} seconds")
        print(f"DirectML time: {time_dml:.6f} seconds")
else:
    print("DirectML device is not available. Only CPU computation will be performed.")
    time_cpu = matrix_multiplication(torch.device("cpu"), torch.device("cpu"))
