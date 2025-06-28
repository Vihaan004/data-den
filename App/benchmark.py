import os
import time
import subprocess
from pathlib import Path
import re

def wrap_with_timing(code, is_gpu=False):
    """Wrap user code with performance timing code."""
    # Check if the code already has imports for time and warnings
    has_time_import = re.search(r'import\s+time|from\s+time\s+import', code)
    has_warnings_import = re.search(r'import\s+warnings|from\s+warnings\s+import', code)
    
    # Header with imports and setup
    header = ""
    if not has_time_import:
        header += "import time\n"
    if not has_warnings_import:
        header += "import warnings\n"
    
    # We'll split the code into imports/setup and actual computation
    code_lines = code.split('\n')
    import_setup_lines = []
    computation_lines = []
    
    # Patterns to identify import/setup code
    import_pattern = re.compile(r'^\s*(import\s+\w+|from\s+[\w.]+\s+import|#|$)')
    assignment_pattern = re.compile(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=')
    function_def_pattern = re.compile(r'^\s*(def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(|class\s+[a-zA-Z_][a-zA-Z0-9_]*)')
    
    # Extract import statements and setup code
    main_code_started = False
    for line in code_lines:
        stripped = line.strip()
        
        # If we see a main execution block or algorithm start, switch to computation mode
        if (stripped.startswith('if __name__ == "__main__"') or 
            stripped.startswith('# Start computation') or 
            stripped.startswith('# Begin algorithm') or
            stripped.startswith('# START TIMING HERE')):
            main_code_started = True
            computation_lines.append(line)
            continue
            
        # If we're already in main code section, add to computation
        if main_code_started:
            computation_lines.append(line)
            continue
            
        # Check if this is setup code
        if (import_pattern.match(line) or 
            assignment_pattern.match(line) or 
            function_def_pattern.match(line) or
            'print' in stripped or
            'logger' in stripped):
            import_setup_lines.append(line)
        else:
            # This is likely computational code
            computation_lines.append(line)
    
    if is_gpu:
        # Check if cupy is already imported
        has_cupy_import = re.search(r'import\s+cupy|from\s+cupy\s+import|import\s+cupy\s+as\s+cp', code)
        if not has_cupy_import:
            header += "import cupy as cp\n"
        
        # Add all imports and setup code first
        setup_code = "\n".join(import_setup_lines)
        
        # Add benchmark header with timing only around computation code
        header += f"""
{setup_code}

# GPU Benchmark - Added by GPU Mentor
warnings.filterwarnings("ignore")
print("="*50)
print("GPU BENCHMARK EXECUTION")
print("="*50)

# Warm up the GPU to ensure fair timing comparison
if 'cp' in globals() or 'cupy' in globals():
    print("Performing GPU warmup...")
    # Create small arrays and perform operations to warm up the GPU
    warmup_a = cp.ones((1000, 1000))
    warmup_b = cp.ones((1000, 1000))
    for _ in range(5):  # Multiple warmup iterations
        _ = cp.dot(warmup_a, warmup_b)
    # Ensure GPU is synchronized before starting the timer
    cp.cuda.Device().synchronize()
    print("GPU warmup completed")

print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Start timing only for the computation part
start_time = time.perf_counter()

try:
"""
        # Indent only the computation code
        indented_code = "\n".join(["    " + line for line in computation_lines])
        
        # Footer with timing and reporting that ensures GPU synchronization
        footer = """
    # Ensure all GPU operations are complete before stopping the timer
    if 'cp' in globals() or 'cupy' in globals():
        cp.cuda.Device().synchronize()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TOTAL GPU EXECUTION TIME: {total_time:.4f} seconds")
    print("✅ GPU benchmark completed successfully!")
except Exception as e:
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"❌ GPU benchmark failed: {str(e)}")
    print(f"TOTAL GPU EXECUTION TIME: {total_time:.4f} seconds")
"""
    else:
        # Add all imports and setup code first
        setup_code = "\n".join(import_setup_lines)
        
        # CPU benchmark with timing only around computation
        header += f"""
{setup_code}

# CPU Benchmark - Added by GPU Mentor
warnings.filterwarnings("ignore")
print("="*50)
print("CPU BENCHMARK EXECUTION")
print("="*50)
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
# Start timing only for the computation part
start_time = time.perf_counter()

try:
"""
        # Indent only the computation code
        indented_code = "\n".join(["    " + line for line in computation_lines])
        
        # Footer with timing and reporting
        footer = """
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TOTAL CPU EXECUTION TIME: {total_time:.4f} seconds")
    print("✅ CPU benchmark completed successfully!")
except Exception as e:
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"❌ CPU benchmark failed: {str(e)}")
    print(f"TOTAL CPU EXECUTION TIME: {total_time:.4f} seconds")
"""
    
    # Combine everything
    return header + indented_code + footer

def submit_slurm_job(script_path):
    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "Submitted batch job" in line:
                return line.split()[-1]
        return result.stdout.strip().split()[-1]
    else:
        raise RuntimeError(f"sbatch failed: {result.stderr}")

def wait_for_job(job_id, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        status = subprocess.run([
            "squeue", "-j", job_id, "--noheader"
        ], capture_output=True, text=True)
        if not status.stdout.strip():
            return True
        time.sleep(3)
    return False

def run_benchmark(cpu_code, gpu_code, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    cpu_py = Path(output_dir) / f"cpu_{timestamp}.py"
    gpu_py = Path(output_dir) / f"gpu_{timestamp}.py"
    cpu_sh = Path(output_dir) / f"cpu_{timestamp}.sh"
    gpu_sh = Path(output_dir) / f"gpu_{timestamp}.sh"
    cpu_out = Path(output_dir) / f"cpu_{timestamp}.out"
    cpu_err = Path(output_dir) / f"cpu_{timestamp}.err"
    gpu_out = Path(output_dir) / f"gpu_{timestamp}.out"
    gpu_err = Path(output_dir) / f"gpu_{timestamp}.err"

    # Wrap code with timing measurements
    cpu_code_with_timing = wrap_with_timing(cpu_code, is_gpu=False)
    gpu_code_with_timing = wrap_with_timing(gpu_code, is_gpu=True)

    # Write code files
    cpu_py.write_text(cpu_code_with_timing)
    gpu_py.write_text(gpu_code_with_timing)

    # Write SLURM scripts
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

    # Submit jobs
    cpu_jobid = submit_slurm_job(str(cpu_sh))
    gpu_jobid = submit_slurm_job(str(gpu_sh))

    # Wait for jobs
    wait_for_job(cpu_jobid)
    wait_for_job(gpu_jobid)

    # Read outputs - only reading stdout as requested
    cpu_out_text = cpu_out.read_text() if cpu_out.exists() else ""
    gpu_out_text = gpu_out.read_text() if gpu_out.exists() else ""
    
    # Read error files just for logs but don't include in results
    if cpu_err.exists() and cpu_err.read_text().strip():
        print(f"CPU Error log: {cpu_err}")
    if gpu_err.exists() and gpu_err.read_text().strip():
        print(f"GPU Error log: {gpu_err}")

    return {
        "cpu": {"stdout": cpu_out_text, "stderr": ""},
        "gpu": {"stdout": gpu_out_text, "stderr": ""}
    }
