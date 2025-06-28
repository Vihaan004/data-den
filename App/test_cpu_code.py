import numpy as np

# Define matrices
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

# Perform matrix multiplication
result = np.dot(A, B)

print(f"Result shape: {result.shape}")
