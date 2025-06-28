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
    
    if is_gpu:
        # Check if cupy is already imported
        has_cupy_import = re.search(r'import\s+cupy|from\s+cupy\s+import|import\s+cupy\s+as\s+cp', code)
        if not has_cupy_import:
            header += "import cupy as cp\n"
        
        # Add GPU synchronization if not present
        has_synchronize = "synchronize" in code
        header += """
# GPU Benchmark - Added by GPU Mentor
warnings.filterwarnings("ignore")
print("="*50)
print("GPU BENCHMARK EXECUTION")
print("="*50)
start_time = time.perf_counter()
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

try:
"""
        # Indent the user code
        indented_code = "\n".join(["    " + line for line in code.split("\n")])
        
        # Footer with timing and reporting
        if has_synchronize:
            footer = """
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
            footer = """
    # Ensure all GPU operations are complete
    import cupy as cp
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
        # CPU benchmark
        header += """
# CPU Benchmark - Added by GPU Mentor
warnings.filterwarnings("ignore")
print("="*50)
print("CPU BENCHMARK EXECUTION")
print("="*50)
start_time = time.perf_counter()
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

try:
"""
        # Indent the user code
        indented_code = "\n".join(["    " + line for line in code.split("\n")])
        
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

    # Read outputs
    cpu_out_text = cpu_out.read_text() if cpu_out.exists() else ""
    cpu_err_text = cpu_err.read_text() if cpu_err.exists() else ""
    gpu_out_text = gpu_out.read_text() if gpu_out.exists() else ""
    gpu_err_text = gpu_err.read_text() if gpu_err.exists() else ""

    return {
        "cpu": {"stdout": cpu_out_text, "stderr": cpu_err_text},
        "gpu": {"stdout": gpu_out_text, "stderr": gpu_err_text}
    }
