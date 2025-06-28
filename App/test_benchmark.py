"""
Test script for the benchmark system
"""

import os
from pathlib import Path
from benchmark import wrap_cpu_code, wrap_gpu_code

def test_code_wrapping():
    """Test the code wrapping functions from the benchmark module"""
    
    # Simple test code for CPU
    cpu_test = """
import numpy as np

# Create some random matrices
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

# Perform matrix multiplication
result = np.dot(A, B)

print(f"Result shape: {result.shape}")
"""

    # Simple test code for GPU
    gpu_test = """
import cupy as cp

# Create some random matrices
A_gpu = cp.random.rand(1000, 1000)
B_gpu = cp.random.rand(1000, 1000)

# Perform matrix multiplication
result_gpu = cp.dot(A_gpu, B_gpu)

print(f"Result shape: {result_gpu.shape}")
"""

    # Wrap the code
    cpu_wrapped = wrap_cpu_code(cpu_test)
    gpu_wrapped = wrap_gpu_code(gpu_test)
    
    # Create output directory
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    # Write wrapped code to files
    cpu_file = output_dir / "cpu_test_wrapped.py"
    gpu_file = output_dir / "gpu_test_wrapped.py"
    
    cpu_file.write_text(cpu_wrapped)
    gpu_file.write_text(gpu_wrapped)
    
    print(f"Generated test files in {output_dir}")
    print(f"CPU file: {cpu_file}")
    print(f"GPU file: {gpu_file}")
    print("Please review the generated files to ensure they are properly formatted")

if __name__ == "__main__":
    test_code_wrapping()
