#!/usr/bin/env python3
"""
Sol Supercomputer Job Runner for GPU Mentor
Handles SLURM job submission and execution timing for code comparison

This module provides functionality to:
1. Generate proper SLURM job scripts for Sol supercomputer
2. Submit jobs with correct partition and resource requests
3. Monitor job status and retrieve results
4. Compare execution times between CPU and GPU code

Follows Sol supercomputer best practices:
- Uses 'general' partition for GPU jobs, 'htc' for CPU jobs
- Proper module loading (mamba/latest, cuda/12.0)
- Uses mamba environments instead of conda
- Correct resource requests and time limits
"""

import os
import subprocess
import time
import tempfile
import uuid
from typing import Dict, Tuple, Optional
from pathlib import Path

class SolJobRunner:
    """Handle job submission and execution on Sol supercomputer."""
    
    def __init__(self):
        self.job_dir = Path("/tmp/gpu_mentor_jobs")
        self.job_dir.mkdir(exist_ok=True)
        self._default_account = None
        
    def _get_default_account(self) -> str:
        """Get the default account to use for job submission."""
        if self._default_account is not None:
            return self._default_account
            
        try:
            # Try to get account information from myfairshare
            result = subprocess.run(
                ['myfairshare'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse myfairshare output to find accounts
                lines = result.stdout.strip().split('\n')
                research_accounts = []
                class_accounts = []
                
                for line in lines[2:]:  # Skip header lines
                    if line.strip() and not line.startswith('-'):
                        parts = line.split()
                        if len(parts) >= 1:
                            account = parts[0]
                            if account.startswith('class_'):
                                class_accounts.append(account)
                            else:
                                research_accounts.append(account)
                
                # For general use, prefer research accounts over class accounts
                # Research accounts typically have public QoS access to htc partition
                if research_accounts:
                    self._default_account = research_accounts[0]
                    print(f"🏷️  Using research account: {self._default_account}")
                    return self._default_account
                elif class_accounts:
                    # If only class accounts available, we need to use general partition
                    # or explicitly specify public QoS
                    print(f"⚠️  Only class accounts found: {class_accounts}")
                    print("   Class accounts may have limited partition access")
                    self._default_account = None  # Don't specify account, let SLURM decide
                    return None
                    
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("ℹ️  Could not retrieve account information, using SLURM defaults")
            pass
        
        # Fallback: don't specify account, let SLURM use default
        self._default_account = None
        return None
        
    def create_job_script(self, code: str, job_name: str, use_gpu: bool = False) -> str:
        """Create a SLURM job script for code execution."""
        
        # Get the appropriate account for job submission
        account = self._get_default_account()
        
        # Determine partition and resources based on GPU usage and account type
        if use_gpu:
            partition = "general"
            # gres = "#SBATCH --gres=gpu:a100:1"
            gres = "#SBATCH -G a100:1"
            qos = "public"
            time_limit = "00:01:00"  # minute for GPU jobs
            modules = """
# Load required modules for GPU computing
module purge
module load mamba/latest
module load cuda/12.0

# Activate or create RAPIDS environment
if mamba info --envs | grep -q 'rapids-25'; then
    echo "Activating existing RAPIDS 25.02 environment..."
    source activate rapids-25
elif mamba info --envs | grep -q 'rapids-22'; then
    echo "Activating older RAPIDS 22 environment (consider upgrading)..."
    source activate rapids-22
else
    echo "RAPIDS environment not found. Please create it first:"
    echo "  module load mamba/latest"
    echo "  mamba create -n rapids-25 -c rapidsai -c conda-forge -c nvidia rapids=25.02 python=3.11 cuda-version=12.0 -y"
    echo "Using base environment with basic packages..."
    source activate base
    mamba install -c conda-forge numpy pandas scikit-learn cupy -y
fi
"""
        else:
            # For CPU jobs, no GPU resources needed
            gres = ""
            
            # For CPU jobs, use htc if we have a research account, general if class account
            if account and account.startswith('class_'):
                partition = "general"  # Class accounts may not have htc access
                qos = "class"  # Use class QoS for class accounts
                time_limit = "00:05:00"  # General partition allows longer times
            else:
                partition = "htc"  # Research accounts can use htc partition
                qos = "public"  # Use public QoS for research accounts
                time_limit = "00:2:00"  # htc partition has 4-hour limit
                
            modules = """
# Load required modules for CPU computing
module purge
module load mamba/latest

# Use existing scicomp environment or create CPU environment
if mamba info --envs | grep -q 'scicomp'; then
    echo "Activating scicomp environment..."
    source activate scicomp
elif mamba info --envs | grep -q 'cpu-env'; then
    echo "Activating CPU environment..."
    source activate cpu-env
else
    echo "Creating CPU environment..."
    mamba create -n cpu-env -c conda-forge python=3.9 numpy pandas scikit-learn matplotlib -y
    source activate cpu-env
fi
"""
        
        # Build account line - only include if we have a specific account
        account_line = f"#SBATCH --account={account}" if account else ""
        
        # Create the job script with timing
        job_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
{account_line}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time={time_limit}
{gres}
#SBATCH --output={self.job_dir}/{job_name}.out
#SBATCH --error={self.job_dir}/{job_name}.err

# Print job information
echo "======================================================"
echo "Job Information:"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: {job_name}"
echo "Partition: {partition}"
echo "Node: $SLURM_NODELIST"
echo "Working Directory: $PWD"
echo "======================================================"

# Set error handling
set -e
set -u

{modules}

# Verify environment
echo "======================================================"
echo "Environment Information:"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
if command -v nvidia-smi &> /dev/null && [[ "{str(use_gpu).lower()}" == "true" ]]; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
fi
echo "Conda/Mamba environment: $CONDA_DEFAULT_ENV"
echo "======================================================"

# Create and execute the Python script
echo "Creating Python script: {job_name}.py"
cat > {self.job_dir}/{job_name}.py << 'EOF'
import time
import sys
import traceback
import os

print("="*50)
print(f"Starting execution: {job_name}")
print(f"Working directory: {{os.getcwd()}}")
print("="*50)

start_time = time.perf_counter()

try:
{self._indent_code(code, 4)}
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    print("="*50)
    print(f"Execution completed successfully")
    print(f"Total execution time: {{execution_time:.4f}} seconds")
    print("="*50)
    
except Exception as e:
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    print("="*50)
    print(f"Execution failed with error:")
    print(f"Error: {{str(e)}}")
    print(f"Execution time before error: {{execution_time:.4f}} seconds")
    print("="*50)
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)
EOF

# Change to job directory and execute
cd {self.job_dir}
echo "Executing Python script..."
python {job_name}.py

echo "======================================================"
echo "Job completed at: $(date)"
echo "======================================================"
"""
        
        return job_script
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code for embedding in job script."""
        lines = code.split('\n')
        indented_lines = [' ' * spaces + line if line.strip() else line for line in lines]
        return '\n'.join(indented_lines)
    
    def submit_job(self, code: str, job_type: str) -> Tuple[str, str]:
        """Submit a job to SLURM and return job_id and script_path."""
        
        # Generate unique job name
        timestamp = int(time.time())
        job_name = f"gpu_mentor_{job_type}_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Determine if this is a GPU job based on imports and libraries
        gpu_indicators = ['cupy', 'cudf', 'cuml', 'cp.', 'rapids', 'gpu', 'cuda', 'torch.cuda', 'tensorflow-gpu']
        use_gpu = any(lib in code.lower() for lib in gpu_indicators)
        
        # Determine partition based on GPU usage and account type
        account = self._get_default_account()
        if use_gpu:
            partition = "general"
        else:
            # For CPU jobs, use htc if we have a research account, general if class account
            if account and account.startswith('class_'):
                partition = "general"  # Class accounts may not have htc access
            else:
                partition = "htc"  # Research accounts can use htc partition
        
        # Create job script
        job_script = self.create_job_script(code, job_name, use_gpu)
        
        # Write job script to file
        script_path = self.job_dir / f"{job_name}.slurm"
        with open(script_path, 'w') as f:
            f.write(job_script)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        try:
            # Submit job using sbatch
            print(f"🔧 Submitting job script: {script_path}")
            print(f"📄 Job type: {'GPU' if use_gpu else 'CPU'} | Partition: {partition}")
            
            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Extract job ID from sbatch output (format: "Submitted batch job XXXXXX")
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if 'Submitted batch job' in line:
                        job_id = line.split()[-1]
                        print(f"✅ Job submitted successfully with ID: {job_id}")
                        return job_id, str(script_path)
                # Fallback: take last word from output
                job_id = result.stdout.strip().split()[-1] if result.stdout.strip() else "unknown"
                print(f"✅ Job submitted with ID: {job_id}")
                return job_id, str(script_path)
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown sbatch error"
                print(f"❌ sbatch failed with return code {result.returncode}")
                print(f"   Error: {error_msg}")
                print(f"   Stdout: {result.stdout}")
                raise Exception(f"sbatch failed (code {result.returncode}): {error_msg}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Job submission timed out after 30 seconds")
        except FileNotFoundError:
            raise Exception("SLURM not available - sbatch command not found. Are you running on Sol supercomputer?")
        except Exception as e:
            raise Exception(f"Job submission failed: {str(e)}")
    
    def check_job_status(self, job_id: str) -> str:
        """Check the status of a SLURM job."""
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '--format=%T', '--noheader'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                # Job might be completed, check sacct
                result = subprocess.run(
                    ['sacct', '-j', job_id, '--format=State', '--noheader', '--parsable2'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    states = result.stdout.strip().split('\n')
                    # Return the last state (job completion state)
                    return states[-1] if states else "UNKNOWN"
                else:
                    return "UNKNOWN"
                    
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "UNKNOWN"
    
    def get_job_output(self, job_id: str, job_name: str) -> Dict[str, str]:
        """Get the output and error files from a completed job."""
        
        output_file = self.job_dir / f"{job_name}.out"
        error_file = self.job_dir / f"{job_name}.err"
        
        result = {
            "stdout": "",
            "stderr": "",
            "execution_time": None,
            "status": "unknown"
        }
        
        # Read output file
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    stdout_content = f.read()
                    result["stdout"] = stdout_content
                    
                    # Extract execution time from output
                    if "Total execution time:" in stdout_content:
                        for line in stdout_content.split('\n'):
                            if "Total execution time:" in line:
                                try:
                                    time_str = line.split(":")[-1].strip().split()[0]
                                    result["execution_time"] = float(time_str)
                                    result["status"] = "completed"
                                    break
                                except (ValueError, IndexError):
                                    pass
                    elif "Execution time before error:" in stdout_content:
                        for line in stdout_content.split('\n'):
                            if "Execution time before error:" in line:
                                try:
                                    time_str = line.split(":")[-1].strip().split()[0]
                                    result["execution_time"] = float(time_str)
                                    result["status"] = "failed"
                                    break
                                except (ValueError, IndexError):
                                    pass
            except Exception as e:
                result["stdout"] = f"Error reading output file: {str(e)}"
        
        # Read error file
        if error_file.exists():
            try:
                with open(error_file, 'r') as f:
                    result["stderr"] = f.read()
            except Exception as e:
                result["stderr"] = f"Error reading error file: {str(e)}"
        
        return result
    
    def run_comparison(self, original_code: str, optimized_code: str) -> Tuple[Dict, Dict]:
        """Run both original and optimized code and return results."""
        
        print("🚀 Submitting jobs to Sol supercomputer...")
        
        # Submit original code job
        try:
            original_job_id, original_script = self.submit_job(original_code, "original")
            # Extract job name from script path
            original_job_name = Path(original_script).stem
            print(f"✅ Original code job submitted: {original_job_id} (script: {original_job_name})")
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Failed to submit original code job: {str(e)}",
                "execution_time": None,
                "output": ""
            }, {}
        
        # Submit optimized code job
        try:
            optimized_job_id, optimized_script = self.submit_job(optimized_code, "optimized")
            # Extract job name from script path
            optimized_job_name = Path(optimized_script).stem
            print(f"✅ Optimized code job submitted: {optimized_job_id} (script: {optimized_job_name})")
        except Exception as e:
            return {}, {
                "status": "failed",
                "error": f"Failed to submit optimized code job: {str(e)}",
                "execution_time": None,
                "output": ""
            }
        
        # Wait for jobs to complete (with timeout)
        max_wait_time = 300  # 5 minutes
        start_wait = time.time()
        
        original_completed = False
        optimized_completed = False
        
        while (time.time() - start_wait) < max_wait_time:
            # Check original job
            if not original_completed:
                original_status = self.check_job_status(original_job_id)
                if original_status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
                    original_completed = True
                    print(f"📊 Original job completed with status: {original_status}")
            
            # Check optimized job
            if not optimized_completed:
                optimized_status = self.check_job_status(optimized_job_id)
                if optimized_status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
                    optimized_completed = True
                    print(f"📊 Optimized job completed with status: {optimized_status}")
            
            # If both completed, break
            if original_completed and optimized_completed:
                break
                
            # Wait a bit before checking again
            time.sleep(5)
        
        # Get results using the job names we extracted earlier
        original_result = self.get_job_output(original_job_id, original_job_name)
        optimized_result = self.get_job_output(optimized_job_id, optimized_job_name)
        
        return original_result, optimized_result
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old job files."""
        try:
            current_time = time.time()
            for file_path in self.job_dir.glob("gpu_mentor_*"):
                if file_path.stat().st_mtime < (current_time - max_age_hours * 3600):
                    file_path.unlink()
        except Exception as e:
            print(f"Warning: Could not clean up old job files: {e}")
