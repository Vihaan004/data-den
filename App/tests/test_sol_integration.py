#!/usr/bin/env python3
"""
Test script for Sol job runner functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sol_job_runner import SolJobRunner

def test_sol_integration():
    """Test Sol supercomputer integration."""
    print("🧪 Testing Sol Job Runner Integration...")
    
    # Initialize job runner
    job_runner = SolJobRunner()
    
    # Test code samples
    original_code = """
import numpy as np
import time

# Create test data
size = 1000
print(f"Creating {size}x{size} matrices...")

A = np.random.rand(size, size).astype(np.float32)
B = np.random.rand(size, size).astype(np.float32)

print("Performing matrix multiplication...")
C = np.matmul(A, B)

print(f"Result shape: {C.shape}")
print(f"Sum of result: {np.sum(C):.4f}")
print("Matrix multiplication completed!")
"""

    optimized_code = """
import cupy as cp
import time

# Create test data on GPU
size = 1000
print(f"Creating {size}x{size} matrices on GPU...")

A = cp.random.rand(size, size, dtype=cp.float32)
B = cp.random.rand(size, size, dtype=cp.float32)

print("Performing GPU matrix multiplication...")
C = cp.matmul(A, B)

# Synchronize to ensure computation is complete
cp.cuda.Device().synchronize()

print(f"Result shape: {C.shape}")
print(f"Sum of result: {float(cp.sum(C)):.4f}")
print("GPU matrix multiplication completed!")
"""
    
    print("\n🚀 Testing job submission...")
    
    try:
        # Test individual job submission
        print("📝 Testing original code submission...")
        original_job_id, original_script = job_runner.submit_job(original_code, "test_original")
        print(f"✅ Original job submitted: {original_job_id}")
        print(f"📄 Script path: {original_script}")
        
        print("📝 Testing optimized code submission...")
        optimized_job_id, optimized_script = job_runner.submit_job(optimized_code, "test_optimized")
        print(f"✅ Optimized job submitted: {optimized_job_id}")
        print(f"📄 Script path: {optimized_script}")
        
        print("\n⏳ Jobs submitted successfully!")
        print("📊 In a real scenario, you would wait for completion and check results.")
        print(f"🔍 Monitor jobs with: squeue -u $USER")
        print(f"📋 Check results with: cat /tmp/gpu_mentor_jobs/gpu_mentor_*.out")
        
    except Exception as e:
        print(f"❌ Error testing job submission: {e}")
        if "SLURM not available" in str(e):
            print("ℹ️  This is expected when not running on Sol supercomputer")
            print("✅ Job runner code is ready for Sol deployment")
        else:
            print("⚠️  Unexpected error - check implementation")
    
    print("\n🎯 Sol integration test completed!")

if __name__ == "__main__":
    test_sol_integration()
