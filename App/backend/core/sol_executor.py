"""
Sol Code Executor - SLURM job submission and management for Sol supercomputer
Extracted from enhanced_agentic_rag_ollama.ipynb
"""

import logging
import subprocess
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class SolCodeExecutor:
    """
    Handles code execution on Sol supercomputer via SLURM job submission.
    Manages job lifecycle, monitoring, and result collection.
    """
    
    def __init__(self, base_work_dir: str = "/tmp/gpu_mentor"):
        self.base_work_dir = Path(base_work_dir)
        self.base_work_dir.mkdir(parents=True, exist_ok=True)
        self.active_jobs = {}
        logger.info(f"Sol executor initialized with work directory: {self.base_work_dir}")
    
    def create_slurm_script(self, code: str, job_type: str = "cpu", 
                           time_limit: str = "00:15:00", 
                           memory: str = "8G") -> Tuple[str, str]:
        """Create a SLURM script for code execution."""
        job_id = str(uuid.uuid4())[:8]
        
        # Configure resources based on job type
        if job_type == "gpu":
            partition = "general"
            gpu_resources = "#SBATCH --gres=gpu:1"
            modules = "module load python/3.11 anaconda3 cuda/12.1\nsource activate rapids-23.08"
        else:
            partition = "general"
            gpu_resources = ""
            modules = "module load python/3.11 anaconda3\nsource activate base"
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name=gpu_mentor_{job_type}_{job_id}
#SBATCH --partition={partition}
#SBATCH --qos=public
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task=4
#SBATCH --mem={memory}
{gpu_resources}
#SBATCH --output={job_type}_output_{job_id}.out
#SBATCH --error={job_type}_error_{job_id}.err

# Change to work directory
cd {self.base_work_dir}

# Load required modules
{modules}

# Create Python script
cat > {job_type}_benchmark_{job_id}.py << 'SCRIPT_EOF'
import time
import sys
import traceback
import json
import os

# GPU-specific imports for GPU jobs
{"# Import GPU libraries" if job_type == "gpu" else ""}
{'''try:
    import cupy as cp
    import cudf
    import cuml
    gpu_available = True
    print("GPU libraries loaded successfully")
except ImportError as e:
    print(f"GPU libraries not available: {e}")
    gpu_available = False''' if job_type == "gpu" else "gpu_available = False"}

# Standard imports
import numpy as np
import pandas as pd

start_time = time.perf_counter()
execution_status = "success"
error_message = ""
result_data = {{}}

try:
{self._indent_code(code)}
except Exception as e:
    execution_status = "error"
    error_message = str(e)
    traceback.print_exc()

end_time = time.perf_counter()
execution_time = end_time - start_time

# Collect system information
result_data.update({{
    "execution_time": execution_time,
    "job_type": "{job_type}",
    "job_id": "{job_id}",
    "status": execution_status,
    "error": error_message,
    "hostname": os.uname().nodename,
    "timestamp": time.time()
}})

# Add GPU-specific information
{'''if gpu_available:
    try:
        mempool = cp.get_default_memory_pool()
        result_data.update({
            "gpu_memory_used": mempool.used_bytes(),
            "gpu_total_memory": mempool.total_bytes(),
            "gpu_available": True
        })
    except:
        result_data["gpu_available"] = False
else:
    result_data["gpu_available"] = False''' if job_type == "gpu" else 'result_data["gpu_available"] = False'}

# Save results
with open("{job_type}_benchmark_{job_id}.json", "w") as f:
    json.dump(result_data, f, indent=2)

print(f"{job_type.upper()} Execution completed in {{execution_time:.4f}} seconds")
if execution_status == "error":
    print(f"Error: {{error_message}}")
    sys.exit(1)
SCRIPT_EOF

# Execute the benchmark script
echo "Starting {job_type.upper()} benchmark execution..."
python {job_type}_benchmark_{job_id}.py

echo "Benchmark completed with exit code: $?"
"""
        
        return script_content, job_id
    
    def _indent_code(self, code: str, indent: str = "    ") -> str:
        """Add proper indentation to user code for embedding in script."""
        return "\n".join(indent + line for line in code.split("\n"))
    
    def submit_job(self, script_content: str, job_id: str) -> Optional[str]:
        """Submit job to SLURM and return SLURM job ID."""
        script_path = self.base_work_dir / f"job_{job_id}.sh"
        
        try:
            # Write script to file
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make script executable
            os.chmod(script_path, 0o755)
            
            # Submit job via sbatch
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                cwd=self.base_work_dir
            )
            
            if result.returncode == 0:
                # Extract SLURM job ID from output (typically "Submitted batch job 12345")
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if "Submitted batch job" in line:
                        slurm_job_id = line.split()[-1]
                        
                        # Store job information
                        self.active_jobs[job_id] = {
                            "slurm_id": slurm_job_id,
                            "script_path": script_path,
                            "submitted_time": time.time(),
                            "status": "PENDING"
                        }
                        
                        logger.info(f"Job {job_id} submitted with SLURM ID: {slurm_job_id}")
                        return slurm_job_id
                
                # If we couldn't parse the job ID, but submission succeeded
                logger.warning(f"Could not parse SLURM job ID from output: {result.stdout}")
                return result.stdout.strip()
            else:
                logger.error(f"Job submission failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return None
    
    def check_job_status(self, slurm_job_id: str) -> str:
        """Check the status of a SLURM job."""
        try:
            # First try squeue (for running/pending jobs)
            result = subprocess.run(
                ["squeue", "-j", slurm_job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                status = result.stdout.strip()
                logger.debug(f"Job {slurm_job_id} status from squeue: {status}")
                return status
            
            # If not in queue, check sacct (for completed jobs)
            result = subprocess.run(
                ["sacct", "-j", slurm_job_id, "-n", "-o", "State", "--parsable2"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # sacct can return multiple lines, take the first non-empty one
                lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                if lines:
                    status = lines[0]
                    logger.debug(f"Job {slurm_job_id} status from sacct: {status}")
                    return status
            
            logger.warning(f"Could not determine status for job {slurm_job_id}")
            return "UNKNOWN"
            
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout checking status for job {slurm_job_id}")
            return "TIMEOUT"
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            return "ERROR"
    
    def get_job_results(self, job_id: str, job_type: str) -> Dict[str, Any]:
        """Retrieve benchmark results from completed job."""
        result_file = self.base_work_dir / f"{job_type}_benchmark_{job_id}.json"
        
        try:
            if result_file.exists():
                with open(result_file, 'r') as f:
                    results = json.load(f)
                logger.info(f"Retrieved results for job {job_id}")
                return results
            else:
                logger.warning(f"Results file not found: {result_file}")
                return {"error": "Results file not found", "job_id": job_id}
                
        except Exception as e:
            logger.error(f"Error reading results for job {job_id}: {e}")
            return {"error": str(e), "job_id": job_id}
    
    def get_job_output(self, job_id: str, job_type: str) -> Dict[str, str]:
        """Get stdout and stderr output from job."""
        output_file = self.base_work_dir / f"{job_type}_output_{job_id}.out"
        error_file = self.base_work_dir / f"{job_type}_error_{job_id}.err"
        
        output_content = ""
        error_content = ""
        
        try:
            if output_file.exists():
                with open(output_file, 'r') as f:
                    output_content = f.read()
        except Exception as e:
            logger.warning(f"Could not read output file: {e}")
        
        try:
            if error_file.exists():
                with open(error_file, 'r') as f:
                    error_content = f.read()
        except Exception as e:
            logger.warning(f"Could not read error file: {e}")
        
        return {
            "stdout": output_content,
            "stderr": error_content
        }
    
    def cleanup_job_files(self, job_id: str):
        """Clean up temporary job files."""
        patterns = [
            f"job_{job_id}.sh",
            f"*_output_{job_id}.out",
            f"*_error_{job_id}.err",
            f"*_benchmark_{job_id}.py",
            f"*_benchmark_{job_id}.json"
        ]
        
        cleaned_files = []
        for pattern in patterns:
            try:
                for file_path in self.base_work_dir.glob(pattern):
                    file_path.unlink()
                    cleaned_files.append(str(file_path))
            except Exception as e:
                logger.warning(f"Error cleaning up files matching {pattern}: {e}")
        
        # Remove from active jobs
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        logger.info(f"Cleaned up {len(cleaned_files)} files for job {job_id}")
    
    def wait_for_job_completion(self, slurm_job_id: str, job_id: str = None, 
                               timeout: int = 900, check_interval: int = 10) -> bool:
        """Wait for a job to complete with timeout."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.check_job_status(slurm_job_id)
            
            # Update active job status
            if job_id and job_id in self.active_jobs:
                self.active_jobs[job_id]["status"] = status
            
            if status in ["COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL"]:
                completion_status = status == "COMPLETED"
                logger.info(f"Job {slurm_job_id} finished with status: {status}")
                return completion_status
            
            time.sleep(check_interval)
        
        logger.warning(f"Job {slurm_job_id} timed out after {timeout} seconds")
        return False
    
    def cancel_job(self, slurm_job_id: str) -> bool:
        """Cancel a running SLURM job."""
        try:
            result = subprocess.run(
                ["scancel", slurm_job_id],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully cancelled job {slurm_job_id}")
                return True
            else:
                logger.error(f"Failed to cancel job {slurm_job_id}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling job {slurm_job_id}: {e}")
            return False
    
    def get_active_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active jobs."""
        # Update status for active jobs
        for job_id, job_info in self.active_jobs.items():
            current_status = self.check_job_status(job_info["slurm_id"])
            job_info["status"] = current_status
            job_info["runtime"] = time.time() - job_info["submitted_time"]
        
        return self.active_jobs.copy()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get Sol system information."""
        system_info = {
            "work_directory": str(self.base_work_dir),
            "active_jobs": len(self.active_jobs),
            "disk_usage": {},
            "queue_info": {}
        }
        
        try:
            # Get disk usage of work directory
            result = subprocess.run(
                ["df", "-h", str(self.base_work_dir)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    fields = lines[1].split()
                    if len(fields) >= 6:
                        system_info["disk_usage"] = {
                            "filesystem": fields[0],
                            "size": fields[1],
                            "used": fields[2],
                            "available": fields[3],
                            "use_percent": fields[4],
                            "mount_point": fields[5]
                        }
        except Exception as e:
            logger.warning(f"Could not get disk usage: {e}")
        
        try:
            # Get queue information
            result = subprocess.run(
                ["sinfo", "-h", "-o", "%P %a %l %D %T"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                partitions = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        fields = line.split()
                        if len(fields) >= 5:
                            partitions.append({
                                "partition": fields[0],
                                "availability": fields[1],
                                "time_limit": fields[2],
                                "nodes": fields[3],
                                "state": fields[4]
                            })
                system_info["queue_info"]["partitions"] = partitions
        except Exception as e:
            logger.warning(f"Could not get queue info: {e}")
        
        return system_info
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Sol and SLURM system."""
        test_results = {
            "work_directory_accessible": False,
            "slurm_available": False,
            "modules_available": False,
            "errors": []
        }
        
        try:
            # Test work directory
            test_file = self.base_work_dir / "connection_test.txt"
            test_file.write_text("GPU Mentor connection test")
            test_file.unlink()
            test_results["work_directory_accessible"] = True
        except Exception as e:
            test_results["errors"].append(f"Work directory not accessible: {e}")
        
        try:
            # Test SLURM
            result = subprocess.run(
                ["squeue", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                test_results["slurm_available"] = True
                test_results["slurm_version"] = result.stdout.strip()
            else:
                test_results["errors"].append("SLURM not available")
        except Exception as e:
            test_results["errors"].append(f"SLURM test failed: {e}")
        
        try:
            # Test module system
            result = subprocess.run(
                ["module", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            # module command returns info on stderr typically
            if "python" in result.stderr.lower() or result.returncode == 0:
                test_results["modules_available"] = True
            else:
                test_results["errors"].append("Module system not available")
        except Exception as e:
            test_results["errors"].append(f"Module test failed: {e}")
        
        test_results["overall_status"] = (
            test_results["work_directory_accessible"] and 
            test_results["slurm_available"]
        )
        
        return test_results
