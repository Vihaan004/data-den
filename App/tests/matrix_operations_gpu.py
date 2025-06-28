#!/usr/bin/env python3
"""
Simple GPU matrix operations test for Sol supercomputer.
Basic test to verify GPU functionality with minimal output.
"""

import cupy as cp
import time
import warnings

def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    print("GPU Matrix Operations Test - RAPIDS 25.02")
    print("=" * 40)
    
    # Record start time
    start_time = time.perf_counter()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Simple matrix operations
        print("Creating 1000x1000 matrices on GPU...")
        A = cp.random.randn(1000, 1000, dtype=cp.float32)
        B = cp.random.randn(1000, 1000, dtype=cp.float32)
        
        print("Performing matrix multiplication...")
        C = cp.dot(A, B)
        
        print("Performing element-wise operations...")
        D = A * B + cp.sin(A)
        
        # Synchronize to ensure all operations complete
        cp.cuda.Device().synchronize()
        
        # Record end time
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"TOTAL EXECUTION TIME: {total_time:.4f} seconds")
        print("✅ GPU operations completed successfully!")
        
    except Exception as e:
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"❌ GPU operations failed: {str(e)}")
        print(f"TOTAL EXECUTION TIME: {total_time:.4f} seconds")
        exit(1)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
