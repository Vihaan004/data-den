import os
import time
import subprocess
import re
from pathlib import Path

def wrap_data_analysis_code(code, dataset_path):
    """
    Wrap data analysis code with dataset loading and GPU setup.
    """
    wrapped_code = f"""import time
import warnings
import io
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import base64
warnings.filterwarnings("ignore")

# Load dataset
dataset_path = "{dataset_path}"
df = pd.read_csv(dataset_path)

# ===== DATA ANALYSIS EXECUTION =====
print("="*50)
print("DATA ANALYSIS EXECUTION")
print("="*50)

print(f"Dataset loaded: {{dataset_path}}")
print(f"Dataset shape: {{df.shape}}")
print(f"Start time: {{time.strftime('%Y-%m-%d %H:%M:%S')}}")

try:
    # GPU warmup if needed
    try:
        import cupy as cp
        warmup_a = cp.random.rand(100, 100).astype(cp.float32)
        warmup_b = cp.random.rand(100, 100).astype(cp.float32)
        _ = cp.matmul(warmup_a, warmup_b)
        cp.cuda.stream.get_current_stream().synchronize()
        print("GPU warmup completed")
    except Exception as gpu_err:
        print(f"GPU warmup skipped: {{gpu_err}}")
    
    # Start timing
    start_time = time.perf_counter()
    
    # Original code starts here
{indent_code(code, 4)}
    # Original code ends here
    
    # Stop timing
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    # Show any plots
    if plt.get_fignums():
        print("\\n----- PLOTS GENERATED -----")
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            # Save plot as base64 for display
            import io
            import base64
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            print(f"PLOT_DATA_START")
            print(plot_data)
            print(f"PLOT_DATA_END")
            buffer.close()
        plt.close('all')
    
    print("\\n" + "="*50)
    print(f"TOTAL EXECUTION TIME: {{execution_time:.6f}} seconds")
    print("✅ Data analysis completed successfully")
    print("="*50)

except Exception as e:
    print(f"\\n❌ Error during execution: {{str(e)}}")
    import traceback
    traceback.print_exc()
    print("="*50)
"""
    return wrapped_code

def indent_code(code, spaces):
    """
    Indent each line of code with the specified number of spaces.
    """
    lines = code.split('\\n')
    indented_lines = []
    
    for line in lines:
        if line.strip():  # If the line has content
            # Preserve original indentation and add specified spaces
            indented_lines.append(' ' * spaces + line)
        else:
            # Keep empty lines
            indented_lines.append('')
    
    return '\\n'.join(indented_lines)

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
    Extract execution time from the analysis output.
    """
    pattern = r"TOTAL EXECUTION TIME: (\\d+\\.\\d+) seconds"
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    return None

def extract_plots(output):
    """
    Extract base64 plot data from the output.
    """
    plots = []
    lines = output.split('\\n')
    i = 0
    while i < len(lines):
        if lines[i].strip() == "PLOT_DATA_START":
            i += 1
            plot_data = ""
            while i < len(lines) and lines[i].strip() != "PLOT_DATA_END":
                plot_data += lines[i]
                i += 1
            if plot_data:
                plots.append(plot_data)
        i += 1
    return plots

def check_job_success(output):
    """
    Check if the job completed successfully.
    """
    return "✅" in output and "completed successfully" in output

def run_data_analysis(code, dataset_path, output_dir):
    """
    Run data analysis code using SLURM and return the results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique timestamp for this run
    timestamp = int(time.time())
    
    # Define file paths
    py_file = Path(output_dir) / f"analysis_{timestamp}.py"
    sh_file = Path(output_dir) / f"analysis_{timestamp}.sh"
    out_file = Path(output_dir) / f"analysis_{timestamp}.out"
    err_file = Path(output_dir) / f"analysis_{timestamp}.err"
    
    # Wrap the code
    wrapped_code = wrap_data_analysis_code(code, dataset_path)
    
    # Write Python file
    with open(py_file, 'w') as f:
        f.write(wrapped_code)
    
    # Create SLURM script
    slurm_script = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-00:10:00
#SBATCH -p general
#SBATCH -q public
#SBATCH -G 1
#SBATCH -o {out_file}
#SBATCH -e {err_file}
#SBATCH --export=NONE

module load mamba/latest
CUDA_MODULES=$(module avail cuda 2>&1 | grep -E "cuda-[0-9]+\\.[0-9]+\\.[0-9]+-gcc" | grep -v "ont-guppy" | sort -V)
if [ -z "$CUDA_MODULES" ]; then echo "No CUDA modules found!"; exit 1; fi
LATEST_CUDA=$(echo "$CUDA_MODULES" | tail -1 | awk '{{print $1}}')
module load $LATEST_CUDA
source activate rapids25.02
cd {output_dir}
python {py_file.name}
"""
    
    with open(sh_file, 'w') as f:
        f.write(slurm_script)
    
    # Make shell script executable
    os.chmod(sh_file, 0o755)
    
    try:
        # Submit job
        job_id = submit_slurm_job(str(sh_file))
        print(f"Submitted job {job_id}")
        
        # Wait for completion
        if wait_for_job(job_id, timeout=600):  # 10 minute timeout
            # Read output
            if out_file.exists():
                with open(out_file, 'r') as f:
                    output = f.read()
            else:
                output = "No output file found."
            
            # Read error
            if err_file.exists():
                with open(err_file, 'r') as f:
                    error = f.read()
            else:
                error = ""
            
            # Extract results
            execution_time = extract_execution_time(output)
            plots = extract_plots(output)
            success = check_job_success(output)
            
            return {
                "success": success,
                "output": output,
                "error": error,
                "execution_time": execution_time,
                "plots": plots,
                "job_id": job_id
            }
        else:
            return {
                "success": False,
                "output": "Job timed out",
                "error": "Job execution exceeded 10 minute timeout",
                "execution_time": None,
                "plots": [],
                "job_id": job_id
            }
    
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "execution_time": None,
            "plots": [],
            "job_id": None
        }
