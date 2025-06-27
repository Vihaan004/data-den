#!/usr/bin/env python3
"""
Test script to compare CPU (NumPy/Pandas) vs GPU (cuDF/CuPy) execution times on Sol supercomputer.

This script submits two comprehensive jobs:
1. CPU job using NumPy and Pandas for data processing
2. GPU job using cuDF and CuPy (RAPIDS 25.02) for the same data processing

It then waits for both jobs to complete and prints their execution times.
"""

import sys
import time
from pathlib import Path

# Add the current directory to path to import sol_job_runner
sys.path.append(str(Path(__file__).parent))

from sol_job_runner import SolJobRunner

def main():
    """Main function to run the CPU vs GPU comparison test."""
    
    print("=" * 60)
    print("GPU Mentor - CPU vs GPU Performance Comparison Test")
    print("Using RAPIDS 25.02 for GPU acceleration")
    print("=" * 60)
    
    # Initialize the job runner
    runner = SolJobRunner()
    
    # Define CPU code using NumPy and Pandas
    cpu_code = """
import numpy as np
import pandas as pd
import time

print("CPU Code - Using NumPy and Pandas")
print("-" * 40)

# Create a large dataset for processing
n_rows = 1_000_000
print(f"Creating dataset with {n_rows:,} rows...")

# Generate synthetic data
np.random.seed(42)
data = {
    'id': np.arange(n_rows),
    'value1': np.random.randn(n_rows),
    'value2': np.random.randn(n_rows),
    'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows)
}

# Create DataFrame
df = pd.DataFrame(data)
print(f"DataFrame created with shape: {df.shape}")

# Perform some computations
print("Performing computations...")

# 1. Mathematical operations
df['computed'] = df['value1'] * df['value2'] + np.sqrt(np.abs(df['value1']))

# 2. Aggregations by category
grouped_stats = df.groupby('category').agg({
    'value1': ['mean', 'std', 'min', 'max'],
    'value2': ['mean', 'std', 'min', 'max'],
    'computed': ['mean', 'std', 'min', 'max']
})

# 3. Filtering and sorting
filtered_df = df[df['computed'] > df['computed'].quantile(0.75)]
sorted_df = filtered_df.sort_values('computed', ascending=False)

# 4. More complex operations
df['rolling_mean'] = df['value1'].rolling(window=1000, min_periods=1).mean()
df['cumulative_sum'] = df['value2'].cumsum()

print(f"Final dataset shape: {df.shape}")
print(f"Filtered dataset shape: {filtered_df.shape}")
print(f"Top 5 computed values: {sorted_df['computed'].head().tolist()}")
print("CPU computation completed successfully!")
"""

    # Define GPU code using cuDF and CuPy
    gpu_code = """
import cupy as cp
import cudf
import time

print("GPU Code - Using cuDF and CuPy")
print("-" * 40)

# Create a large dataset for processing
n_rows = 1_000_000
print(f"Creating dataset with {n_rows:,} rows...")

# Generate synthetic data using CuPy
cp.random.seed(42)
data = {
    'id': cp.arange(n_rows),
    'value1': cp.random.randn(n_rows),
    'value2': cp.random.randn(n_rows),
    'category': cp.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows)
}

# Create cuDF DataFrame
df = cudf.DataFrame(data)
print(f"cuDF DataFrame created with shape: {df.shape}")

# Perform some computations
print("Performing computations...")

# 1. Mathematical operations
df['computed'] = df['value1'] * df['value2'] + cp.sqrt(cp.abs(df['value1']))

# 2. Aggregations by category
grouped_stats = df.groupby('category').agg({
    'value1': ['mean', 'std', 'min', 'max'],
    'value2': ['mean', 'std', 'min', 'max'],
    'computed': ['mean', 'std', 'min', 'max']
})

# 3. Filtering and sorting
quantile_75 = df['computed'].quantile(0.75)
filtered_df = df[df['computed'] > quantile_75]
sorted_df = filtered_df.sort_values('computed', ascending=False)

# 4. More complex operations
df['rolling_mean'] = df['value1'].rolling(window=1000, min_periods=1).mean()
df['cumulative_sum'] = df['value2'].cumsum()

print(f"Final dataset shape: {df.shape}")
print(f"Filtered dataset shape: {filtered_df.shape}")
print(f"Top 5 computed values: {sorted_df['computed'].head().to_pandas().tolist()}")
print("GPU computation completed successfully!")
"""
    
    print("🚀 Submitting CPU and GPU jobs to Sol supercomputer...")
    print()
    
    # Submit CPU job
    try:
        cpu_job_id, cpu_script = runner.submit_job(cpu_code, "cpu_test")
        cpu_job_name = Path(cpu_script).stem
        print(f"✅ CPU job submitted: {cpu_job_id}")
    except Exception as e:
        print(f"❌ Failed to submit CPU job: {e}")
        return
    
    # Submit GPU job
    try:
        gpu_job_id, gpu_script = runner.submit_job(gpu_code, "gpu_test")
        gpu_job_name = Path(gpu_script).stem
        print(f"✅ GPU job submitted: {gpu_job_id}")
    except Exception as e:
        print(f"❌ Failed to submit GPU job: {e}")
        return
    
    print()
    print("⏳ Waiting for jobs to complete...")
    print("   This may take a few minutes depending on queue status...")
    print()
    
    # Monitor job status
    max_wait_time = 600  # 10 minutes
    start_wait = time.time()
    check_interval = 10  # Check every 10 seconds
    
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
        
        # Show progress
        if current_time % 30 == 0 and current_time > 0:  # Every 30 seconds
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
    print("🖥️  CPU Job Results (NumPy/Pandas):")
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
    print("🔥 GPU Job Results (cuDF/CuPy - RAPIDS 25.02):")
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
        print(f"   CPU Time: {cpu_time:.4f} seconds")
        print(f"   GPU Time: {gpu_time:.4f} seconds")
        
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            if speedup > 1:
                print(f"   🚀 GPU is {speedup:.2f}x FASTER than CPU")
            elif speedup < 1:
                print(f"   🐌 GPU is {1/speedup:.2f}x SLOWER than CPU")
            else:
                print("   ⚖️  CPU and GPU have similar performance")
        
        time_saved = cpu_time - gpu_time
        if time_saved > 0:
            print(f"   ⏱️  Time saved: {time_saved:.4f} seconds")
        else:
            print(f"   ⏱️  Additional time: {abs(time_saved):.4f} seconds")
    else:
        print("⚠️  Could not compare performance - missing execution times")
    
    print()
    print("=" * 60)
    
    # Show job outputs if requested or if there were errors
    show_outputs = False
    if cpu_result.get('status') == 'failed' or gpu_result.get('status') == 'failed':
        show_outputs = True
        print("📄 JOB OUTPUTS (showing due to errors):")
    
    if show_outputs or len(sys.argv) > 1 and sys.argv[1] == "--show-output":
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
    print("✅ Test completed! Job files are available in:")
    print(f"   {runner.job_dir}")
    print()
    print("To see full job output, run:")
    print(f"   python {__file__} --show-output")

if __name__ == "__main__":
    main()
