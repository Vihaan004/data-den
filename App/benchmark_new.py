import os
import time
import subprocess
import re
from pathlib import Path

def wrap_cpu_code(code):
    """
    Wrap CPU code with timing functionality.
    This preserves the original code structure while adding timing measurements.
    """
    wrapped_code = f"""import time
import warnings
warnings.filterwarnings("ignore")

# ===== CPU BENCHMARK =====
print("="*50)
print("CPU BENCHMARK EXECUTION")
print("="*50)

# Variables for timing
iterations = 5  # Run multiple times for reliable timing
iterations_completed = 0
total_time = 0
print(f"Start time: {{time.strftime('%Y-%m-%d %H:%M:%S')}}")

try:
    # Original code starts here
{indent_code(code, 4)}
    # Original code ends here
    
    # Main timing loop
    for i in range(iterations):
        start_time = time.perf_counter()
        
        # Run computation again
{indent_code(code, 8)}
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        iterations_completed += 1
        print(f"Iteration {{i+1}}: {{elapsed_time:.6f}} seconds")
    
    # Calculate average
    avg_time = total_time / iterations_completed
    print(f"End time: {{time.strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"TOTAL CPU EXECUTION TIME: {{avg_time:.6f}} seconds (averaged over {{iterations_completed}} runs)")
    print("✅ CPU benchmark completed successfully!")
except Exception as e:
    iterations_completed = max(1, iterations_completed)
    if iterations_completed > 0:
        avg_time = total_time / iterations_completed
        print(f"TOTAL CPU EXECUTION TIME: {{avg_time:.6f}} seconds (partial completion, {{iterations_completed}} runs)")
    else:
        print(f"CPU benchmark failed, no successful iterations")
    print(f"❌ Error: {{str(e)}}")
"""
    return wrapped_code

def wrap_gpu_code(code):
    """
    Wrap GPU code with timing functionality plus GPU-specific features.
    This preserves the original code structure while adding timing measurements.
    """
    wrapped_code = f"""import time
import warnings
import cupy as cp
warnings.filterwarnings("ignore")

# ===== GPU BENCHMARK =====
print("="*50)
print("GPU BENCHMARK EXECUTION")
print("="*50)

# Force some computation to ensure GPU is initialized
print("Warming up GPU...")
try:
    warmup_a = cp.random.rand(1000, 1000).astype(cp.float32)
    warmup_b = cp.random.rand(1000, 1000).astype(cp.float32)
    for _ in range(3):
        _ = cp.matmul(warmup_a, warmup_b)
    cp.cuda.stream.get_current_stream().synchronize()
    print("GPU warmup completed")
except Exception as gpu_err:
    print(f"GPU warmup skipped: {{gpu_err}}")

# Variables for timing
iterations = 5  # Run multiple times for reliable timing
iterations_completed = 0
total_time = 0
print(f"Start time: {{time.strftime('%Y-%m-%d %H:%M:%S')}}")

try:
    # Original code starts here (run once to initialize)
{indent_code(code, 4)}
    # Original code ends here
    
    # Make sure all GPU operations have completed
    cp.cuda.stream.get_current_stream().synchronize()
    
    # Main timing loop
    for i in range(iterations):
        start_time = time.perf_counter()
        
        # Run computation again
{indent_code(code, 8)}
        
        # Synchronize before stopping the timer
        cp.cuda.stream.get_current_stream().synchronize()
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        iterations_completed += 1
        print(f"Iteration {{i+1}}: {{elapsed_time:.6f}} seconds")
    
    # Calculate average
    avg_time = total_time / iterations_completed
    print(f"End time: {{time.strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"TOTAL GPU EXECUTION TIME: {{avg_time:.6f}} seconds (averaged over {{iterations_completed}} runs)")
    print("✅ GPU benchmark completed successfully!")
except Exception as e:
    iterations_completed = max(1, iterations_completed)
    if iterations_completed > 0:
        avg_time = total_time / iterations_completed
        print(f"TOTAL GPU EXECUTION TIME: {{avg_time:.6f}} seconds (partial completion, {{iterations_completed}} runs)")
    else:
        print(f"GPU benchmark failed, no successful iterations")
    print(f"❌ Error: {{str(e)}}")
"""
    return wrapped_code

def indent_code(code, spaces=4):
    """
    Properly indent code while preserving its original indentation structure.
    """
    lines = code.split('\n')
    indented_lines = []
    
    for line in lines:
        if line.strip():  # If the line has content
            # Preserve original indentation and add specified spaces
            indented_lines.append(' ' * spaces + line)
        else:
            # Keep empty lines
            indented_lines.append('')
    
    return '\n'.join(indented_lines)

def submit_slurm_job(script_path):
    """
    Submit a job to SLURM and return the job ID.
    """
    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "Submitted batch job" in line:
                return line.split()[-1]
        return result.stdout.strip().split()[-1]
    else:
        raise RuntimeError(f"sbatch failed: {result.stderr}")

def wait_for_job(job_id, timeout=300):
    """
    Wait for a SLURM job to complete, up to the specified timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        status = subprocess.run([
            "squeue", "-j", job_id, "--noheader"
        ], capture_output=True, text=True)
        if not status.stdout.strip():
            return True
        time.sleep(3)
    return False

def extract_execution_time(output):
    """
    Extract execution time from the benchmark output.
    """
    # Look for the TOTAL EXECUTION TIME line in the output
    pattern = r"TOTAL (CPU|GPU) EXECUTION TIME: (\d+\.\d+) seconds"
    match = re.search(pattern, output)
    if match:
        return float(match.group(2))
    return None

def check_job_success(output):
    """
    Check if the job completed successfully.
    """
    # Look for success marker
    return "✅" in output and "completed successfully" in output

def run_benchmark(cpu_code, gpu_code, output_dir):
    """
    Run CPU and GPU benchmarks using SLURM and return the results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique timestamp for this run
    timestamp = int(time.time())
    
    # Define file paths
    cpu_py = Path(output_dir) / f"cpu_{timestamp}.py"
    gpu_py = Path(output_dir) / f"gpu_{timestamp}.py"
    cpu_sh = Path(output_dir) / f"cpu_{timestamp}.sh"
    gpu_sh = Path(output_dir) / f"gpu_{timestamp}.sh"
    cpu_out = Path(output_dir) / f"cpu_{timestamp}.out"
    cpu_err = Path(output_dir) / f"cpu_{timestamp}.err"
    gpu_out = Path(output_dir) / f"gpu_{timestamp}.out"
    gpu_err = Path(output_dir) / f"gpu_{timestamp}.err"

    # Wrap code with timing code
    cpu_code_with_timing = wrap_cpu_code(cpu_code)
    gpu_code_with_timing = wrap_gpu_code(gpu_code)

    # Write Python script files
    cpu_py.write_text(cpu_code_with_timing)
    gpu_py.write_text(gpu_code_with_timing)

    # Create SLURM job scripts
    cpu_sh.write_text(f'''#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-00:05:00
#SBATCH -p general
#SBATCH -q public
#SBATCH -o {cpu_out}
#SBATCH -e {cpu_err}
#SBATCH --export=NONE

module load mamba/latest
source activate scicomp24.11
cd {output_dir}
python {cpu_py.name}
''')

    gpu_sh.write_text(f'''#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-00:05:00
#SBATCH -p general
#SBATCH -q public
#SBATCH -G 1
#SBATCH -o {gpu_out}
#SBATCH -e {gpu_err}
#SBATCH --export=NONE

module load mamba/latest
CUDA_MODULES=$(module avail cuda 2>&1 | grep -E "cuda-[0-9]+\\.[0-9]+\\.[0-9]+-gcc" | grep -v "ont-guppy" | sort -V)
if [ -z "$CUDA_MODULES" ]; then echo "No CUDA modules found!"; exit 1; fi
LATEST_CUDA=$(echo "$CUDA_MODULES" | tail -1 | awk '{{print $1}}')
module load $LATEST_CUDA
source activate rapids25.02
cd {output_dir}
python {gpu_py.name}
''')

    # Make shell scripts executable
    os.chmod(cpu_sh, 0o755)
    os.chmod(gpu_sh, 0o755)
    
    # Submit jobs
    try:
        cpu_jobid = submit_slurm_job(str(cpu_sh))
        print(f"Submitted CPU job with ID: {cpu_jobid}")
        
        gpu_jobid = submit_slurm_job(str(gpu_sh))
        print(f"Submitted GPU job with ID: {gpu_jobid}")
        
        # Wait for jobs to complete
        print("Waiting for CPU job to complete...")
        cpu_done = wait_for_job(cpu_jobid)
        print(f"CPU job {'completed' if cpu_done else 'timed out'}")
        
        print("Waiting for GPU job to complete...")
        gpu_done = wait_for_job(gpu_jobid)
        print(f"GPU job {'completed' if gpu_done else 'timed out'}")
        
    except Exception as e:
        print(f"Error submitting or waiting for jobs: {e}")
    
    # Read outputs
    cpu_out_text = cpu_out.read_text() if cpu_out.exists() else ""
    gpu_out_text = gpu_out.read_text() if gpu_out.exists() else ""
    
    # Log errors but don't include in the result
    if cpu_err.exists() and cpu_err.read_text().strip():
        print(f"CPU Error log: {cpu_err}")
    
    if gpu_err.exists() and gpu_err.read_text().strip():
        print(f"GPU Error log: {gpu_err}")
    
    # Extract timing information
    cpu_time = extract_execution_time(cpu_out_text)
    gpu_time = extract_execution_time(gpu_out_text)
    
    # Check job success
    cpu_success = check_job_success(cpu_out_text)
    gpu_success = check_job_success(gpu_out_text)
    
    # Format the results
    if cpu_time is not None:
        cpu_out_text += f"\nExecution time: {cpu_time:.6f} seconds"
    else:
        cpu_out_text += "\nFailed to extract execution time"
    
    if gpu_time is not None:
        gpu_out_text += f"\nExecution time: {gpu_time:.6f} seconds"
    else:
        gpu_out_text += "\nFailed to extract execution time"
    
    return {
        "cpu": {
            "stdout": cpu_out_text, 
            "stderr": "",
            "success": cpu_success,
            "time": cpu_time
        },
        "gpu": {
            "stdout": gpu_out_text, 
            "stderr": "",
            "success": gpu_success,
            "time": gpu_time
        }
    }

def format_execution_result(result):
    """
    Format the execution results for display in the UI.
    """
    cpu_result = result.get("cpu", {})
    gpu_result = result.get("gpu", {})
    
    cpu_success = cpu_result.get("success", False)
    gpu_success = gpu_result.get("success", False)
    
    cpu_time = cpu_result.get("time")
    gpu_time = gpu_result.get("time")
    
    # Format CPU output
    cpu_status = "Success" if cpu_success else "Failed"
    cpu_output = cpu_result.get("stdout", "")
    
    # Format GPU output
    gpu_status = "Success" if gpu_success else "Failed"
    gpu_output = gpu_result.get("stdout", "")
    
    # Calculate speedup if both runs were successful
    speedup = None
    if cpu_time is not None and gpu_time is not None and cpu_time > 0 and gpu_success and cpu_success:
        speedup = cpu_time / gpu_time
    
    formatted_result = {
        "cpu": {
            "status": cpu_status,
            "output": cpu_output,
            "execution_time": cpu_time
        },
        "gpu": {
            "status": gpu_status,
            "output": gpu_output,
            "execution_time": gpu_time
        },
        "speedup": speedup
    }
    
    return formatted_result

if __name__ == "__main__":
    # For testing purposes
    test_cpu_code = """
import numpy as np

# Create some random matrices
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

# Perform matrix multiplication
C = np.dot(A, B)

print(f"Result shape: {C.shape}")
"""

    test_gpu_code = """
import cupy as cp

# Create some random matrices
A_gpu = cp.random.rand(1000, 1000)
B_gpu = cp.random.rand(1000, 1000)

# Perform matrix multiplication
C_gpu = cp.dot(A_gpu, B_gpu)

print(f"Result shape: {C_gpu.shape}")
"""

    # Run test benchmark
    result = run_benchmark(test_cpu_code, test_gpu_code, "output")
    formatted = format_execution_result(result)
    
    print("\nCPU Status:", formatted["cpu"]["status"])
    print("CPU Time:", formatted["cpu"]["execution_time"])
    print("\nGPU Status:", formatted["gpu"]["status"])
    print("GPU Time:", formatted["gpu"]["execution_time"])
    
    if formatted["speedup"] is not None:
        print(f"\nGPU Speedup: {formatted['speedup']:.2f}x")
