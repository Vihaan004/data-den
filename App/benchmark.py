import os
import time
import subprocess
from pathlib import Path
import re

def indent_code(code_lines, base_indent=4, loop_indent=0):
    """Helper to indent code with proper levels for repetition loops.
    
    This function preserves existing indentation while adding the specified
    base_indent and loop_indent.
    """
    indented = []
    for line in code_lines:
        if line.strip():  # If line has content
            # Get the raw content of the line without leading whitespace
            stripped_line = line.lstrip()
            # Count leading spaces in original line
            leading_spaces = len(line) - len(stripped_line)
            # Add base indentation plus loop indentation plus original indentation
            indented.append(" " * base_indent + " " * loop_indent + " " * leading_spaces + stripped_line)
        else:
            # For empty lines, just add the base indentation
            indented.append(" " * base_indent + " " * loop_indent)
    return "\n".join(indented)

def wrap_with_timing(code, is_gpu=False):
    """Wrap user code with performance timing code and repetitions for accuracy."""
    # Check if the code already has imports for time and warnings
    has_time_import = re.search(r'import\s+time|from\s+time\s+import', code)
    has_warnings_import = re.search(r'import\s+warnings|from\s+warnings\s+import', code)
    
    # Header with imports and setup
    header = ""
    if not has_time_import:
        header += "import time\n"
    if not has_warnings_import:
        header += "import warnings\n"
    
    # Split code into imports/setup and actual computation
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
        
        # Look for major computation patterns
        if (re.search(r'(np\.dot|np\.matmul|cp\.dot|cp\.matmul|dot\(|matmul\(|\.fit\(|\.predict\()', stripped) or
            re.search(r'for\s+.*\s+in\s+.*:', line)):
            # This looks like computational code
            computation_lines.append(line)
            continue
            
        # Check if this is setup code
        if (import_pattern.match(line) or 
            (assignment_pattern.match(line) and "=" in line and not re.search(r'[+\-*/]=|\+=|\-=|\*=|/=', line)) or
            function_def_pattern.match(line) or
            'print' in stripped or
            'logger' in stripped):
            import_setup_lines.append(line)
        else:
            # This is likely computational code
            computation_lines.append(line)
    
    # If no computation code was identified, use the whole code as computation
    # This ensures timing works even in simple examples
    if not computation_lines:
        computation_lines = code_lines
        import_setup_lines = []
    
    setup_code = "\n".join(import_setup_lines)
    
    if is_gpu:
        # Add GPU-specific code
        has_cupy_import = re.search(r'import\s+cupy|from\s+cupy\s+import|import\s+cupy\s+as\s+cp', code)
        if not has_cupy_import:
            header += "import cupy as cp\n"
        
        # Create the GPU timing wrapper
        gpu_wrapper = f"""
{header}
{setup_code}

# GPU Benchmark - Added by GPU Mentor
warnings.filterwarnings("ignore")
print("="*50)
print("GPU BENCHMARK EXECUTION")
print("="*50)

# Constants for timing control
MIN_RUNS = 5
MAX_RUNS = 100
TARGET_TIME = 1.0  # seconds

# Force some computation to ensure GPU is initialized
print("Warming up GPU...")
try:
    import cupy as cp
    warmup_a = cp.random.rand(1000, 1000).astype(cp.float32)
    warmup_b = cp.random.rand(1000, 1000).astype(cp.float32)
    for _ in range(3):
        _ = cp.matmul(warmup_a, warmup_b)
    cp.cuda.stream.get_current_stream().synchronize()
    print("GPU warmup completed")
except Exception as gpu_err:
    print(f"GPU warmup skipped: {{gpu_err}}")

# Variables for run tracking
iterations_completed = 0
run_start_time = time.perf_counter()
print(f"Start time: {{time.strftime('%Y-%m-%d %H:%M:%S')}}")

try:
    # First do a single run to estimate timing
    start_time = time.perf_counter()
    
{indent_code(computation_lines, 4, 0)}
    
    # Ensure GPU operations complete
    try:
        cp.cuda.stream.get_current_stream().synchronize()
    except:
        pass
        
    end_time = time.perf_counter()
    single_run_time = end_time - start_time
    
    # Calculate needed runs (at least 3, at most 50)
    if single_run_time < 0.001:
        num_runs = MAX_RUNS  # Very fast operation, need many runs
    else:
        num_runs = min(MAX_RUNS, max(MIN_RUNS, int(TARGET_TIME / single_run_time)))
        
    print(f"Using {{num_runs}} iterations for accurate timing")
    
    # Main timing loop
    start_time = time.perf_counter()
    
    for iteration in range(num_runs):
{indent_code(computation_lines, 8, 0)}
        
        # Track iterations
        iterations_completed = iteration + 1
        
        # Synchronize GPU if available
        try:
            cp.cuda.stream.get_current_stream().synchronize()
        except:
            pass
            
    # Final timing
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    
    # Ensure we never report exactly 0
    avg_time = max(0.0001, avg_time)
    
    print(f"End time: {{time.strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"TOTAL GPU EXECUTION TIME: {{avg_time:.6f}} seconds (averaged over {{num_runs}} runs)")
    print("✅ GPU benchmark completed successfully!")
except Exception as e:
    # Handle errors
    end_time = time.perf_counter()
    runs_completed = max(1, iterations_completed)
    total_time = end_time - start_time
    avg_time = total_time / runs_completed
    
    # Ensure we never report exactly 0
    avg_time = max(0.0001, avg_time)
    
    print(f"❌ GPU benchmark failed: {{str(e)}}")
    print(f"TOTAL GPU EXECUTION TIME: {{avg_time:.6f}} seconds (partial completion, {{runs_completed}} runs)")
"""
        return gpu_wrapper
    else:
        # CPU timing wrapper
        cpu_wrapper = f"""
{header}
{setup_code}

# CPU Benchmark - Added by GPU Mentor
warnings.filterwarnings("ignore")
print("="*50)
print("CPU BENCHMARK EXECUTION")
print("="*50)

# Constants for timing control
MIN_RUNS = 5
MAX_RUNS = 100
TARGET_TIME = 1.0  # seconds

# Variables for run tracking
iterations_completed = 0
run_start_time = time.perf_counter()
print(f"Start time: {{time.strftime('%Y-%m-%d %H:%M:%S')}}")

try:
    # First do a single run to estimate timing
    start_time = time.perf_counter()
    
{indent_code(computation_lines, 4, 0)}
    
    end_time = time.perf_counter()
    single_run_time = end_time - start_time
    
    # Calculate needed runs (at least 3, at most 50)
    if single_run_time < 0.001:
        num_runs = MAX_RUNS  # Very fast operation, need many runs
    else:
        num_runs = min(MAX_RUNS, max(MIN_RUNS, int(TARGET_TIME / single_run_time)))
        
    print(f"Using {{num_runs}} iterations for accurate timing")
    
    # Main timing loop
    start_time = time.perf_counter()
    
    for iteration in range(num_runs):
{indent_code(computation_lines, 8, 0)}
        
        # Track iterations
        iterations_completed = iteration + 1
            
    # Final timing
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    
    # Ensure we never report exactly 0
    avg_time = max(0.0001, avg_time)
    
    print(f"End time: {{time.strftime('%Y-%m-%d %H:%M:%S')}}")
    print(f"TOTAL CPU EXECUTION TIME: {{avg_time:.6f}} seconds (averaged over {{num_runs}} runs)")
    print("✅ CPU benchmark completed successfully!")
except Exception as e:
    # Handle errors
    end_time = time.perf_counter()
    runs_completed = max(1, iterations_completed)
    total_time = end_time - start_time
    avg_time = total_time / runs_completed
    
    # Ensure we never report exactly 0
    avg_time = max(0.0001, avg_time)
    
    print(f"❌ CPU benchmark failed: {{str(e)}}")
    print(f"TOTAL CPU EXECUTION TIME: {{avg_time:.6f}} seconds (partial completion, {{runs_completed}} runs)")
"""
        return cpu_wrapper

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
