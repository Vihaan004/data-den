import os
import time
import subprocess
from pathlib import Path

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

    # Write code files
    cpu_py.write_text(cpu_code)
    gpu_py.write_text(gpu_code)

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
