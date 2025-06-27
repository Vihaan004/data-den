#!/usr/bin/env python3
"""
Simple CPU vs GPU test script for Sol supercomputer with RAPIDS 25.02.

This script submits two lightweight jobs:
1. CPU job using NumPy for basic matrix operations
2. GPU job using CuPy for the same matrix operations

It then waits for both jobs to complete and prints their execution times.
"""

import sys
import time
from pathlib import Path

# Add the current directory to path to import sol_job_runner
sys.path.append(str(Path(__file__).parent))

from sol_job_runner import SolJobRunner

def main():
    """Main function to run the simple CPU vs GPU comparison test."""
    
    print("=" * 60)
    print("GPU Mentor - Simple CPU vs GPU Test (RAPIDS 25.02)")
    print("=" * 60)
    
    # Initialize the job runner
    runner = SolJobRunner()
    
    # Define simple CPU code using NumPy
    cpu_code = """
import numpy as np
import time

print("CPU Code - Using NumPy")
print("-" * 30)

# Set random seed for reproducibility
np.random.seed(42)

# Create matrices for computation
print("Creating matrices...")
size = 2000
A = np.random.randn(size, size).astype(np.float32)
B = np.random.randn(size, size).astype(np.float32)

print(f"Matrix A shape: {A.shape}")
print(f"Matrix B shape: {B.shape}")

# Perform matrix operations
print("Performing matrix multiplication...")
C = np.dot(A, B)

print("Performing element-wise operations...")
D = A * B + np.sin(A) + np.cos(B)

print("Computing statistics...")
mean_C = np.mean(C)
std_C = np.std(C)
max_D = np.max(D)
min_D = np.min(D)

print(f"Results:")
print(f"  Matrix C mean: {mean_C:.6f}")
print(f"  Matrix C std:  {std_C:.6f}")
print(f"  Matrix D max:  {max_D:.6f}")
print(f"  Matrix D min:  {min_D:.6f}")

print("CPU computation completed successfully!")
"""

    # Define simple GPU code using CuPy
    gpu_code = """
import cupy as cp
import time

print("GPU Code - Using CuPy")
print("-" * 30)

# Set random seed for reproducibility
cp.random.seed(42)

# Create matrices for computation
print("Creating matrices on GPU...")
size = 2000
A = cp.random.randn(size, size, dtype=cp.float32)
B = cp.random.randn(size, size, dtype=cp.float32)

print(f"Matrix A shape: {A.shape}")
print(f"Matrix B shape: {B.shape}")

# Perform matrix operations
print("Performing matrix multiplication...")
C = cp.dot(A, B)

print("Performing element-wise operations...")
D = A * B + cp.sin(A) + cp.cos(B)

print("Computing statistics...")
mean_C = cp.mean(C)
std_C = cp.std(C)
max_D = cp.max(D)
min_D = cp.min(D)

# Convert results to CPU for printing (if needed)
mean_C_cpu = float(mean_C)
std_C_cpu = float(std_C)
max_D_cpu = float(max_D)
min_D_cpu = float(min_D)

print(f"Results:")
print(f"  Matrix C mean: {mean_C_cpu:.6f}")
print(f"  Matrix C std:  {std_C_cpu:.6f}")
print(f"  Matrix D max:  {max_D_cpu:.6f}")
print(f"  Matrix D min:  {min_D_cpu:.6f}")

print("GPU computation completed successfully!")
"""
    
    print("🚀 Submitting simple CPU and GPU jobs to Sol supercomputer...")
    print()
    
    # Submit CPU job
    try:
        cpu_job_id, cpu_script = runner.submit_job(cpu_code, "simple_cpu")
        cpu_job_name = Path(cpu_script).stem
        print(f"✅ CPU job submitted: {cpu_job_id}")
    except Exception as e:
        print(f"❌ Failed to submit CPU job: {e}")
        return
    
    # Submit GPU job
    try:
        gpu_job_id, gpu_script = runner.submit_job(gpu_code, "simple_gpu")
        gpu_job_name = Path(gpu_script).stem
        print(f"✅ GPU job submitted: {gpu_job_id}")
    except Exception as e:
        print(f"❌ Failed to submit GPU job: {e}")
        return
    
    print()
    print("⏳ Waiting for jobs to complete...")
    print("   This should take 1-3 minutes for simple matrix operations...")
    print()
    
    # Monitor job status with shorter timeout for simple operations
    max_wait_time = 300  # 5 minutes
    start_wait = time.time()
    check_interval = 5  # Check every 5 seconds
    
    cpu_completed = False
    gpu_completed = False
    
    while (time.time() - start_wait) < max_wait_time:
        current_time = int(time.time() - start_wait)
        
        # Check CPU job
        if not cpu_completed:
            cpu_status = runner.check_job_status(cpu_job_id)
            if cpu_status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
                cpu_completed = True
                print(f"📊 CPU job completed with status: {cpu_status} (after {current_time}s)")
        
        # Check GPU job
        if not gpu_completed:
            gpu_status = runner.check_job_status(gpu_job_id)
            if gpu_status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
                gpu_completed = True
                print(f"📊 GPU job completed with status: {gpu_status} (after {current_time}s)")
        
        # If both completed, break
        if cpu_completed and gpu_completed:
            break
        
        # Show progress every 15 seconds
        if current_time % 15 == 0 and current_time > 0:
            cpu_stat = cpu_status if not cpu_completed else "DONE"
            gpu_stat = gpu_status if not gpu_completed else "DONE"
            print(f"   Status after {current_time}s - CPU: {cpu_stat}, GPU: {gpu_stat}")
        
        # Wait before checking again
        time.sleep(check_interval)
    
    # Check if jobs timed out
    if not cpu_completed or not gpu_completed:
        print(f"⚠️  Jobs did not complete within {max_wait_time//60} minutes")
        if not cpu_completed:
            print(f"   CPU job ({cpu_job_id}) status: {runner.check_job_status(cpu_job_id)}")
        if not gpu_completed:
            print(f"   GPU job ({gpu_job_id}) status: {runner.check_job_status(gpu_job_id)}")
        print("   You can check job status later using: squeue -u $USER")
        return
    
    print()
    print("📋 Retrieving job results...")
    
    # Get job outputs
    cpu_result = runner.get_job_output(cpu_job_id, cpu_job_name)
    gpu_result = runner.get_job_output(gpu_job_id, gpu_job_name)
    
    # Print results
    print()
    print("=" * 60)
    print("EXECUTION RESULTS")
    print("=" * 60)
    
    # CPU Results
    print("🖥️  CPU Job Results (NumPy):")
    print(f"   Job ID: {cpu_job_id}")
    print(f"   Status: {cpu_result.get('status', 'unknown')}")
    if cpu_result.get('execution_time') is not None:
        print(f"   Execution Time: {cpu_result['execution_time']:.4f} seconds")
    else:
        print("   Execution Time: Not available")
    
    if cpu_result.get('status') == 'failed':
        print("   ❌ Job failed - check error output below")
    
    print()
    
    # GPU Results
    print("🔥 GPU Job Results (CuPy/RAPIDS 25.02):")
    print(f"   Job ID: {gpu_job_id}")
    print(f"   Status: {gpu_result.get('status', 'unknown')}")
    if gpu_result.get('execution_time') is not None:
        print(f"   Execution Time: {gpu_result['execution_time']:.4f} seconds")
    else:
        print("   Execution Time: Not available")
    
    if gpu_result.get('status') == 'failed':
        print("   ❌ Job failed - check error output below")
    
    print()
    
    # Performance comparison
    cpu_time = cpu_result.get('execution_time')
    gpu_time = gpu_result.get('execution_time')
    
    if cpu_time is not None and gpu_time is not None:
        print("⚡ PERFORMANCE COMPARISON:")
        print(f"   CPU Time (NumPy):  {cpu_time:.4f} seconds")
        print(f"   GPU Time (CuPy):   {gpu_time:.4f} seconds")
        
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            if speedup > 1:
                print(f"   🚀 GPU is {speedup:.2f}x FASTER than CPU")
            elif speedup < 1:
                print(f"   🐌 GPU is {1/speedup:.2f}x SLOWER than CPU")
                print("   Note: GPU may be slower for small operations due to overhead")
            else:
                print("   ⚖️  CPU and GPU have similar performance")
        
        time_difference = cpu_time - gpu_time
        if time_difference > 0:
            print(f"   ⏱️  Time saved: {time_difference:.4f} seconds")
        else:
            print(f"   ⏱️  Additional time: {abs(time_difference):.4f} seconds")
    else:
        print("⚠️  Could not compare performance - missing execution times")
    
    print()
    print("=" * 60)
    
    # Show job outputs if there were errors or if requested
    show_outputs = False
    if cpu_result.get('status') == 'failed' or gpu_result.get('status') == 'failed':
        show_outputs = True
        print("📄 JOB OUTPUTS (showing due to errors):")
    elif len(sys.argv) > 1 and sys.argv[1] == "--show-output":
        show_outputs = True
        print("📄 JOB OUTPUTS:")
    
    if show_outputs:
        print()
        print("🖥️  CPU Job Output:")
        print("-" * 40)
        if cpu_result.get('stdout'):
            print(cpu_result['stdout'])
        if cpu_result.get('stderr'):
            print("STDERR:")
            print(cpu_result['stderr'])
        
        print()
        print("🔥 GPU Job Output:")
        print("-" * 40)
        if gpu_result.get('stdout'):
            print(gpu_result['stdout'])
        if gpu_result.get('stderr'):
            print("STDERR:")
            print(gpu_result['stderr'])
    
    print()
    print("✅ Simple test completed!")
    print()
    print("To see full job output, run:")
    print(f"   python {__file__} --show-output")
    print()
    print("For more comprehensive testing, run:")
    print("   python test_cpu_gpu_comparison.py")

if __name__ == "__main__":
    main()
