import cupy as cp

# Define matrices
A_gpu = cp.random.rand(1000, 1000)
B_gpu = cp.random.rand(1000, 1000)

# Perform matrix multiplication
result_gpu = cp.dot(A_gpu, B_gpu)

print(f"Result shape: {result_gpu.shape}")
