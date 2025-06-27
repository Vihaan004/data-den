#!/usr/bin/env python3
"""
Simple CPU matrix operations test for Sol supercomputer.
This script performs basic matrix operations using NumPy and measures execution time.
"""

import numpy as np
import time
import sys

def main():
    print("=" * 50)
    print("CPU Matrix Operations Test")
    print("Using NumPy for CPU computation")
    print("=" * 50)
    
    # Record start time
    start_time = time.perf_counter()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create matrices
    print("Creating matrices...")
    matrix_size = 2000
    print(f"Matrix size: {matrix_size} x {matrix_size}")
    
    A = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    B = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    
    print(f"Matrix A shape: {A.shape}, dtype: {A.dtype}")
    print(f"Matrix B shape: {B.shape}, dtype: {B.dtype}")
    print(f"Memory usage: ~{(A.nbytes + B.nbytes) / 1024**2:.1f} MB")
    print()
    
    # Matrix multiplication
    print("Performing matrix multiplication (A @ B)...")
    mult_start = time.perf_counter()
    C = np.dot(A, B)
    mult_end = time.perf_counter()
    print(f"Matrix multiplication completed in {mult_end - mult_start:.4f} seconds")
    print(f"Result matrix C shape: {C.shape}")
    print()
    
    # Element-wise operations
    print("Performing element-wise operations...")
    elem_start = time.perf_counter()
    D = A * B + np.sin(A) + np.cos(B)
    elem_end = time.perf_counter()
    print(f"Element-wise operations completed in {elem_end - elem_start:.4f} seconds")
    print()
    
    # Statistical operations
    print("Computing statistics...")
    stats_start = time.perf_counter()
    
    # Statistics for matrix C
    mean_C = np.mean(C)
    std_C = np.std(C)
    max_C = np.max(C)
    min_C = np.min(C)
    
    # Statistics for matrix D
    mean_D = np.mean(D)
    std_D = np.std(D)
    max_D = np.max(D)
    min_D = np.min(D)
    
    stats_end = time.perf_counter()
    print(f"Statistical computations completed in {stats_end - stats_start:.4f} seconds")
    print()
    
    # Additional operations
    print("Performing additional operations...")
    additional_start = time.perf_counter()
    
    # Eigenvalue computation (smaller matrix for reasonable time)
    small_matrix = A[:100, :100]
    eigenvals = np.linalg.eigvals(small_matrix)
    
    # SVD decomposition (smaller matrix)
    U, s, Vt = np.linalg.svd(small_matrix)
    
    # Matrix inversion (smaller matrix)
    inv_matrix = np.linalg.inv(small_matrix)
    
    additional_end = time.perf_counter()
    print(f"Additional operations completed in {additional_end - additional_start:.4f} seconds")
    print()
    
    # Record end time
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Print results
    print("=" * 50)
    print("COMPUTATION RESULTS")
    print("=" * 50)
    
    print("Matrix C (A @ B) Statistics:")
    print(f"  Mean: {mean_C:.6f}")
    print(f"  Std:  {std_C:.6f}")
    print(f"  Max:  {max_C:.6f}")
    print(f"  Min:  {min_C:.6f}")
    print()
    
    print("Matrix D (A*B + sin(A) + cos(B)) Statistics:")
    print(f"  Mean: {mean_D:.6f}")
    print(f"  Std:  {std_D:.6f}")
    print(f"  Max:  {max_D:.6f}")
    print(f"  Min:  {min_D:.6f}")
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
    print("✅ CPU matrix operations completed successfully!")
    
    # System information
    print()
    print("System Information:")
    print(f"  Python version: {sys.version}")
    print(f"  NumPy version: {np.__version__}")
    print(f"  NumPy config: {np.show_config()}")

if __name__ == "__main__":
    main()
