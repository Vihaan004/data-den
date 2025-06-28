#!/usr/bin/env python3
"""
Simple CPU matrix operations test for Sol supercomputer.
Basic test to verify CPU functionality with minimal output.
"""

import numpy as np
import time
import warnings

def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    print("CPU Matrix Operations Test - NumPy")
    print("=" * 40)
    
    # Record start time
    start_time = time.perf_counter()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Simple matrix operations
        print("Creating 1000x1000 matrices on CPU...")
        A = np.random.randn(1000, 1000).astype(np.float32)
        B = np.random.randn(1000, 1000).astype(np.float32)
        
        print("Performing matrix multiplication...")
        C = np.dot(A, B)
        
        print("Performing element-wise operations...")
        D = A * B + np.sin(A)
        
        # Record end time
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"TOTAL EXECUTION TIME: {total_time:.4f} seconds")
        print("✅ CPU operations completed successfully!")
        
    except Exception as e:
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"❌ CPU operations failed: {str(e)}")
        print(f"TOTAL EXECUTION TIME: {total_time:.4f} seconds")
        exit(1)

if __name__ == "__main__":
    main()
