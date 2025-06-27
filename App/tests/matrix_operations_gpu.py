#!/usr/bin/env python3
"""
Simple GPU matrix operations test for Sol supercomputer using RAPIDS 25.02.
This script performs basic matrix operations using CuPy and measures execution time.

Note: If you see CuPy package conflicts, the script will still work but with warnings.
The RAPIDS environment should handle CuPy properly.
"""

import cupy as cp
import time
import sys
import warnings

def main():
    # Suppress CuPy installation warnings for cleaner output
    warnings.filterwarnings("ignore", message=".*multiple CuPy packages.*")
    
    print("=" * 50)
    print("GPU Matrix Operations Test")
    print("Using CuPy (RAPIDS 25.02) for GPU computation")
    print("=" * 50)
    
    # Check GPU availability
    try:
        device_id = cp.cuda.get_device_id()
        device = cp.cuda.Device(device_id)
        with device:
            device_name = device.name.decode('utf-8') if hasattr(device.name, 'decode') else str(device.name)
        print(f"GPU Device: {device_name}")
        print(f"GPU Device ID: {device_id}")
    except Exception as e:
        print(f"GPU Device: Unable to get device name ({e})")
        print(f"GPU Device ID: {cp.cuda.get_device_id()}")
    
    try:
        meminfo = cp.cuda.MemoryInfo()
        print(f"GPU Memory: {meminfo.total / 1024**3:.1f} GB total")
    except Exception as e:
        print(f"GPU Memory: Unable to get memory info ({e})")
    
    try:
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        print(f"CUDA Runtime Version: {cuda_version}")
    except Exception as e:
        print(f"CUDA Runtime Version: Unable to get version ({e})")
    
    print()
    
    # Record start time
    start_time = time.perf_counter()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Set random seed for reproducibility
    cp.random.seed(42)
    
    # Create matrices on GPU
    print("Creating matrices on GPU...")
    matrix_size = 2000
    print(f"Matrix size: {matrix_size} x {matrix_size}")
    
    A = cp.random.randn(matrix_size, matrix_size, dtype=cp.float32)
    B = cp.random.randn(matrix_size, matrix_size, dtype=cp.float32)
    
    print(f"Matrix A shape: {A.shape}, dtype: {A.dtype}")
    print(f"Matrix B shape: {B.shape}, dtype: {B.dtype}")
    print(f"GPU memory usage: ~{(A.nbytes + B.nbytes) / 1024**2:.1f} MB")
    print()
    
    # Matrix multiplication
    print("Performing matrix multiplication (A @ B)...")
    mult_start = time.perf_counter()
    C = cp.dot(A, B)
    cp.cuda.Device().synchronize()  # Wait for GPU computation to complete
    mult_end = time.perf_counter()
    print(f"Matrix multiplication completed in {mult_end - mult_start:.4f} seconds")
    print(f"Result matrix C shape: {C.shape}")
    print()
    
    # Element-wise operations
    print("Performing element-wise operations...")
    elem_start = time.perf_counter()
    D = A * B + cp.sin(A) + cp.cos(B)
    cp.cuda.Device().synchronize()  # Wait for GPU computation to complete
    elem_end = time.perf_counter()
    print(f"Element-wise operations completed in {elem_end - elem_start:.4f} seconds")
    print()
    
    # Statistical operations
    print("Computing statistics...")
    stats_start = time.perf_counter()
    
    # Statistics for matrix C
    mean_C = cp.mean(C)
    std_C = cp.std(C)
    max_C = cp.max(C)
    min_C = cp.min(C)
    
    # Statistics for matrix D
    mean_D = cp.mean(D)
    std_D = cp.std(D)
    max_D = cp.max(D)
    min_D = cp.min(D)
    
    cp.cuda.Device().synchronize()  # Wait for GPU computation to complete
    stats_end = time.perf_counter()
    print(f"Statistical computations completed in {stats_end - stats_start:.4f} seconds")
    print()
    
    # Additional operations
    print("Performing additional operations...")
    additional_start = time.perf_counter()
    
    # Eigenvalue computation (smaller matrix for reasonable time)
    small_matrix = A[:100, :100]
    eigenvals = cp.linalg.eigvals(small_matrix)
    
    # SVD decomposition (smaller matrix)
    U, s, Vt = cp.linalg.svd(small_matrix)
    
    # Matrix inversion (smaller matrix)
    inv_matrix = cp.linalg.inv(small_matrix)
    
    cp.cuda.Device().synchronize()  # Wait for GPU computation to complete
    additional_end = time.perf_counter()
    print(f"Additional operations completed in {additional_end - additional_start:.4f} seconds")
    print()
    
    # Record end time
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Convert GPU results to CPU for printing (if needed)
    mean_C_cpu = float(mean_C)
    std_C_cpu = float(std_C)
    max_C_cpu = float(max_C)
    min_C_cpu = float(min_C)
    
    mean_D_cpu = float(mean_D)
    std_D_cpu = float(std_D)
    max_D_cpu = float(max_D)
    min_D_cpu = float(min_D)
    
    # Print results
    print("=" * 50)
    print("COMPUTATION RESULTS")
    print("=" * 50)
    
    print("Matrix C (A @ B) Statistics:")
    print(f"  Mean: {mean_C_cpu:.6f}")
    print(f"  Std:  {std_C_cpu:.6f}")
    print(f"  Max:  {max_C_cpu:.6f}")
    print(f"  Min:  {min_C_cpu:.6f}")
    print()
    
    print("Matrix D (A*B + sin(A) + cos(B)) Statistics:")
    print(f"  Mean: {mean_D_cpu:.6f}")
    print(f"  Std:  {std_D_cpu:.6f}")
    print(f"  Max:  {max_D_cpu:.6f}")
    print(f"  Min:  {min_D_cpu:.6f}")
    print()
    
    print("Additional Operations:")
    print(f"  Eigenvalues (100x100): {len(eigenvals)} computed")
    print(f"  SVD singular values (100x100): {len(s)} computed")
    print(f"  Matrix inversion (100x100): completed")
    print()
    
    print("=" * 50)
    print("TIMING RESULTS")
    print("=" * 50)
    print(f"Matrix multiplication time: {mult_end - mult_start:.4f} seconds")
    print(f"Element-wise operations time: {elem_end - elem_start:.4f} seconds")
    print(f"Statistical operations time: {stats_end - stats_start:.4f} seconds")
    print(f"Additional operations time: {additional_end - additional_start:.4f} seconds")
    print()
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TOTAL EXECUTION TIME: {total_time:.4f} seconds")
    print("=" * 50)
    
    print()
    print("✅ GPU matrix operations completed successfully!")
    
    # System information
    print()
    print("System Information:")
    print(f"  Python version: {sys.version}")
    print(f"  CuPy version: {cp.__version__}")
    
    try:
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        print(f"  CUDA Runtime version: {cuda_version}")
    except Exception as e:
        print(f"  CUDA Runtime version: Unable to get version ({e})")
    
    try:
        device_id = cp.cuda.get_device_id()
        device = cp.cuda.Device(device_id)
        with device:
            device_name = device.name.decode('utf-8') if hasattr(device.name, 'decode') else str(device.name)
        print(f"  GPU: {device_name}")
    except Exception as e:
        print(f"  GPU: Unable to get device name ({e})")
    
    # GPU Memory info
    try:
        meminfo = cp.cuda.MemoryInfo()
        print(f"  GPU Memory - Total: {meminfo.total / 1024**3:.1f} GB")
        print(f"  GPU Memory - Used: {meminfo.used / 1024**3:.1f} GB")
        print(f"  GPU Memory - Free: {meminfo.free / 1024**3:.1f} GB")
    except Exception as e:
        print(f"  GPU Memory: Unable to get memory info ({e})")

if __name__ == "__main__":
    main()
