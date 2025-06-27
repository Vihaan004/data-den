#!/usr/bin/env python3
"""
Sol Supercomputer Job Runner for GPU Mentor
Handles SLURM job submission and execution timing for code comparison
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
        
    def create_job_script(self, code: str, job_name: str, use_gpu: bool = False) -> str:
        """Create a SLURM job script for code execution."""
        
        # Determine partition and resources based on GPU usage
        if use_gpu:
            partition = "gpu"
            gres = "#SBATCH --gres=gpu:1"
            modules = """
module load anaconda3
module load cuda
conda activate rapids-env || conda create -n rapids-env python=3.9 -y
conda activate rapids-env
conda install -c rapidsai -c conda-forge -c nvidia rapids=23.10 python=3.9 cudatoolkit=11.8 -y
"""
        else:
            partition = "general"
            gres = ""
            modules = """
module load anaconda3
conda activate cpu-env || conda create -n cpu-env python=3.9 -y
conda activate cpu-env
conda install numpy pandas scikit-learn -y
"""
        
        # Create the job script with timing
        job_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
{gres}
#SBATCH --output={self.job_dir}/{job_name}.out
#SBATCH --error={self.job_dir}/{job_name}.err

# Load modules and activate environment
{modules}

# Create the Python script with timing
cat > {self.job_dir}/{job_name}.py << 'EOF'
import time
import sys
import traceback

print("="*50)
print(f"Starting execution: {{job_name}}")
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

# Execute the Python script
cd {self.job_dir}
python {job_name}.py
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
        
        # Determine if this is a GPU job based on imports
        use_gpu = any(lib in code.lower() for lib in ['cupy', 'cudf', 'cuml', 'cp.', 'cudf.'])
        
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
            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Extract job ID from sbatch output
                job_id = result.stdout.strip().split()[-1]
                return job_id, str(script_path)
            else:
                raise Exception(f"sbatch failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Job submission timed out")
        except FileNotFoundError:
            raise Exception("SLURM not available - not running on Sol supercomputer")
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
            print(f"✅ Original code job submitted: {original_job_id}")
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
            print(f"✅ Optimized code job submitted: {optimized_job_id}")
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
        
        # Get results
        original_job_name = f"gpu_mentor_original_{original_job_id.split('_')[-1] if '_' in original_job_id else original_job_id}"
        optimized_job_name = f"gpu_mentor_optimized_{optimized_job_id.split('_')[-1] if '_' in optimized_job_id else optimized_job_id}"
        
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
