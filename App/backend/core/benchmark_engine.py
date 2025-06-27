"""
Enhanced Benchmark Engine - Comprehensive CPU vs GPU performance comparison
Extracted from enhanced_agentic_rag_ollama.ipynb
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

from .sol_executor import SolCodeExecutor
from .code_optimizer import CodeOptimizer

logger = logging.getLogger(__name__)


class BenchmarkEngine:
    """
    Comprehensive benchmarking engine for CPU vs GPU performance comparison.
    Integrates with Sol executor for real hardware benchmarking.
    """
    
    def __init__(self, sol_executor: SolCodeExecutor = None, code_optimizer: CodeOptimizer = None):
        self.sol_executor = sol_executor or SolCodeExecutor()
        self.code_optimizer = code_optimizer
        self.benchmark_results = []
        self.predefined_benchmarks = self._setup_predefined_benchmarks()
        logger.info("Benchmark Engine initialized")
    
    def _setup_predefined_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Setup comprehensive predefined benchmarks."""
        return {
            "Matrix Operations": {
                "description": "Compare NumPy vs CuPy for large matrix operations",
                "categories": ["Linear Algebra", "Array Processing"],
                "benchmarks": [
                    {
                        "name": "Matrix Multiplication",
                        "cpu_code": """
import numpy as np
import time

# Setup
size = {size}
A = np.random.rand(size, size).astype(np.float32)
B = np.random.rand(size, size).astype(np.float32)

# Benchmark
start_time = time.perf_counter()
C = np.matmul(A, B)
cpu_time = time.perf_counter() - start_time

result = {{"execution_time": cpu_time, "result_shape": C.shape, "result_sum": float(np.sum(C))}}
""",
                        "gpu_code": """
import cupy as cp
import time

# Setup
size = {size}
A = cp.random.rand(size, size).astype(cp.float32)
B = cp.random.rand(size, size).astype(cp.float32)

# Benchmark
start_time = time.perf_counter()
C = cp.matmul(A, B)
cp.cuda.Device().synchronize()
gpu_time = time.perf_counter() - start_time

result = {{"execution_time": gpu_time, "result_shape": C.shape, "result_sum": float(cp.sum(C))}}
""",
                        "sizes": [256, 512, 1024, 2048],
                        "metric": "execution_time"
                    },
                    {
                        "name": "Singular Value Decomposition",
                        "cpu_code": """
import numpy as np
import time

# Setup
size = {size}
A = np.random.rand(size, size).astype(np.float32)

# Benchmark
start_time = time.perf_counter()
U, s, Vt = np.linalg.svd(A)
cpu_time = time.perf_counter() - start_time

result = {{"execution_time": cpu_time, "singular_values": len(s), "min_sv": float(np.min(s))}}
""",
                        "gpu_code": """
import cupy as cp
import time

# Setup
size = {size}
A = cp.random.rand(size, size).astype(cp.float32)

# Benchmark
start_time = time.perf_counter()
U, s, Vt = cp.linalg.svd(A)
cp.cuda.Device().synchronize()
gpu_time = time.perf_counter() - start_time

result = {{"execution_time": gpu_time, "singular_values": len(s), "min_sv": float(cp.min(s))}}
""",
                        "sizes": [128, 256, 512, 1024],
                        "metric": "execution_time"
                    }
                ]
            },
            
            "DataFrame Operations": {
                "description": "Compare Pandas vs cuDF for data processing tasks",
                "categories": ["Data Processing", "Analytics"],
                "benchmarks": [
                    {
                        "name": "GroupBy Aggregation",
                        "cpu_code": """
import pandas as pd
import numpy as np
import time

# Setup
n = {size}
df = pd.DataFrame({{
    'group': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
    'value1': np.random.randn(n),
    'value2': np.random.randn(n),
    'value3': np.random.randint(1, 100, n)
}})

# Benchmark
start_time = time.perf_counter()
result_df = df.groupby('group').agg({{
    'value1': ['mean', 'std', 'min', 'max'],
    'value2': ['sum', 'count'],
    'value3': ['median']
}})
cpu_time = time.perf_counter() - start_time

result = {{"execution_time": cpu_time, "groups": len(result_df), "total_rows": len(df)}}
""",
                        "gpu_code": """
import cudf
import numpy as np
import time

# Setup
n = {size}
df = cudf.DataFrame({{
    'group': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
    'value1': np.random.randn(n),
    'value2': np.random.randn(n),
    'value3': np.random.randint(1, 100, n)
}})

# Benchmark
start_time = time.perf_counter()
result_df = df.groupby('group').agg({{
    'value1': ['mean', 'std', 'min', 'max'],
    'value2': ['sum', 'count'],
    'value3': ['mean']  # cuDF doesn't support median in groupby
}})
gpu_time = time.perf_counter() - start_time

result = {{"execution_time": gpu_time, "groups": len(result_df), "total_rows": len(df)}}
""",
                        "sizes": [100000, 500000, 1000000, 2000000],
                        "metric": "execution_time"
                    },
                    {
                        "name": "String Operations",
                        "cpu_code": """
import pandas as pd
import numpy as np
import time

# Setup
n = {size}
df = pd.DataFrame({{
    'text': ['sample_text_' + str(i) for i in range(n)],
    'category': np.random.choice(['cat', 'dog', 'bird'], n)
}})

# Benchmark
start_time = time.perf_counter()
df['text_upper'] = df['text'].str.upper()
df['text_length'] = df['text'].str.len()
df['contains_sample'] = df['text'].str.contains('sample')
cpu_time = time.perf_counter() - start_time

result = {{"execution_time": cpu_time, "processed_strings": len(df), "avg_length": df['text_length'].mean()}}
""",
                        "gpu_code": """
import cudf
import numpy as np
import time

# Setup
n = {size}
df = cudf.DataFrame({{
    'text': ['sample_text_' + str(i) for i in range(n)],
    'category': np.random.choice(['cat', 'dog', 'bird'], n)
}})

# Benchmark
start_time = time.perf_counter()
df['text_upper'] = df['text'].str.upper()
df['text_length'] = df['text'].str.len()
df['contains_sample'] = df['text'].str.contains('sample')
gpu_time = time.perf_counter() - start_time

result = {{"execution_time": gpu_time, "processed_strings": len(df), "avg_length": df['text_length'].mean()}}
""",
                        "sizes": [50000, 100000, 250000, 500000],
                        "metric": "execution_time"
                    }
                ]
            },
            
            "Machine Learning": {
                "description": "Compare scikit-learn vs cuML for ML algorithms",
                "categories": ["Machine Learning", "Classification"],
                "benchmarks": [
                    {
                        "name": "K-Means Clustering",
                        "cpu_code": """
import numpy as np
from sklearn.cluster import KMeans
import time

# Setup
n_samples = {size}
n_features = 20
X = np.random.rand(n_samples, n_features).astype(np.float32)

# Benchmark
start_time = time.perf_counter()
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
cpu_time = time.perf_counter() - start_time

result = {{"execution_time": cpu_time, "n_clusters": 8, "inertia": float(kmeans.inertia_)}}
""",
                        "gpu_code": """
import cupy as cp
from cuml.cluster import KMeans
import time

# Setup
n_samples = {size}
n_features = 20
X = cp.random.rand(n_samples, n_features).astype(cp.float32)

# Benchmark
start_time = time.perf_counter()
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
cp.cuda.Device().synchronize()
gpu_time = time.perf_counter() - start_time

result = {{"execution_time": gpu_time, "n_clusters": 8, "inertia": float(kmeans.inertia_)}}
""",
                        "sizes": [10000, 50000, 100000, 200000],
                        "metric": "execution_time"
                    },
                    {
                        "name": "Random Forest Classifier",
                        "cpu_code": """
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import time

# Setup
n_samples = {size}
n_features = 20
X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                          n_informative=10, n_redundant=10, 
                          n_clusters_per_class=1, random_state=42)

# Benchmark
start_time = time.perf_counter()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
cpu_time = time.perf_counter() - start_time

result = {{"execution_time": cpu_time, "n_samples": n_samples, "score": float(rf.score(X, y))}}
""",
                        "gpu_code": """
import cupy as cp
import numpy as np
from cuml.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import time

# Setup
n_samples = {size}
n_features = 20
X_cpu, y_cpu = make_classification(n_samples=n_samples, n_features=n_features, 
                                  n_informative=10, n_redundant=10, 
                                  n_clusters_per_class=1, random_state=42)
X = cp.array(X_cpu)
y = cp.array(y_cpu)

# Benchmark
start_time = time.perf_counter()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
cp.cuda.Device().synchronize()
gpu_time = time.perf_counter() - start_time

result = {{"execution_time": gpu_time, "n_samples": n_samples, "score": float(rf.score(X, y))}}
""",
                        "sizes": [5000, 10000, 25000, 50000],
                        "metric": "execution_time"
                    }
                ]
            },
            
            "Mathematical Functions": {
                "description": "Compare NumPy vs CuPy for mathematical operations",
                "categories": ["Mathematics", "Signal Processing"],
                "benchmarks": [
                    {
                        "name": "FFT Computation",
                        "cpu_code": """
import numpy as np
import time

# Setup
n = {size}
x = np.random.randn(n).astype(np.complex64)

# Benchmark
start_time = time.perf_counter()
fft_result = np.fft.fft(x)
cpu_time = time.perf_counter() - start_time

result = {{"execution_time": cpu_time, "fft_size": len(fft_result), "max_magnitude": float(np.max(np.abs(fft_result)))}}
""",
                        "gpu_code": """
import cupy as cp
import time

# Setup
n = {size}
x = cp.random.randn(n).astype(cp.complex64)

# Benchmark
start_time = time.perf_counter()
fft_result = cp.fft.fft(x)
cp.cuda.Device().synchronize()
gpu_time = time.perf_counter() - start_time

result = {{"execution_time": gpu_time, "fft_size": len(fft_result), "max_magnitude": float(cp.max(cp.abs(fft_result)))}}
""",
                        "sizes": [8192, 16384, 65536, 131072],
                        "metric": "execution_time"
                    },
                    {
                        "name": "Element-wise Operations",
                        "cpu_code": """
import numpy as np
import time

# Setup
n = {size}
x = np.random.rand(n).astype(np.float32)
y = np.random.rand(n).astype(np.float32)

# Benchmark
start_time = time.perf_counter()
z = np.sqrt(x**2 + y**2)
result_sum = np.sum(z)
cpu_time = time.perf_counter() - start_time

result = {{"execution_time": cpu_time, "array_size": n, "result_sum": float(result_sum)}}
""",
                        "gpu_code": """
import cupy as cp
import time

# Setup
n = {size}
x = cp.random.rand(n).astype(cp.float32)
y = cp.random.rand(n).astype(cp.float32)

# Benchmark
start_time = time.perf_counter()
z = cp.sqrt(x**2 + y**2)
result_sum = cp.sum(z)
cp.cuda.Device().synchronize()
gpu_time = time.perf_counter() - start_time

result = {{"execution_time": gpu_time, "array_size": n, "result_sum": float(result_sum)}}
""",
                        "sizes": [1000000, 5000000, 10000000, 50000000],
                        "metric": "execution_time"
                    }
                ]
            }
        }
    
    def get_benchmark_categories(self) -> List[str]:
        """Get list of available benchmark categories."""
        return list(self.predefined_benchmarks.keys())
    
    def get_benchmarks_for_category(self, category: str) -> List[str]:
        """Get list of benchmarks for a specific category."""
        if category in self.predefined_benchmarks:
            return [bench["name"] for bench in self.predefined_benchmarks[category]["benchmarks"]]
        return []
    
    def get_benchmark_sizes(self, category: str, benchmark_name: str) -> List[int]:
        """Get available sizes for a specific benchmark."""
        if category in self.predefined_benchmarks:
            for benchmark in self.predefined_benchmarks[category]["benchmarks"]:
                if benchmark["name"] == benchmark_name:
                    return benchmark["sizes"]
        return []
    
    def run_benchmark(self, category: str, benchmark_name: str, size: int) -> Dict[str, Any]:
        """Run a specific benchmark and return results."""
        try:
            # Find the benchmark
            benchmark_data = None
            if category in self.predefined_benchmarks:
                for benchmark in self.predefined_benchmarks[category]["benchmarks"]:
                    if benchmark["name"] == benchmark_name:
                        benchmark_data = benchmark
                        break
            
            if not benchmark_data:
                return {"error": f"Benchmark {benchmark_name} not found in category {category}"}
            
            if size not in benchmark_data["sizes"]:
                return {"error": f"Size {size} not available for benchmark {benchmark_name}"}
            
            # Format the code with the specific size
            cpu_code = benchmark_data["cpu_code"].format(size=size)
            gpu_code = benchmark_data["gpu_code"].format(size=size)
            
            logger.info(f"Running benchmark: {category} - {benchmark_name} (size: {size})")
            
            # Run benchmarks using Sol executor
            results = self._execute_benchmark(cpu_code, gpu_code, category, benchmark_name, size)
            
            # Store results
            self.benchmark_results.append(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            return {"error": str(e)}
    
    def _execute_benchmark(self, cpu_code: str, gpu_code: str, category: str, 
                          benchmark_name: str, size: int) -> Dict[str, Any]:
        """Execute CPU and GPU benchmarks."""
        results = {
            "category": category,
            "benchmark": benchmark_name,
            "size": size,
            "timestamp": time.time(),
            "cpu_results": {},
            "gpu_results": {},
            "speedup": None,
            "winner": None,
            "error": None
        }
        
        try:
            # Create SLURM scripts for CPU and GPU execution
            cpu_script, cpu_job_id = self.sol_executor.create_slurm_script(
                cpu_code, job_type="cpu", time_limit="00:15:00"
            )
            gpu_script, gpu_job_id = self.sol_executor.create_slurm_script(
                gpu_code, job_type="gpu", time_limit="00:15:00"
            )
            
            # Submit jobs
            cpu_slurm_id = self.sol_executor.submit_job(cpu_script, cpu_job_id)
            gpu_slurm_id = self.sol_executor.submit_job(gpu_script, gpu_job_id)
            
            if not cpu_slurm_id or not gpu_slurm_id:
                results["error"] = "Failed to submit benchmark jobs"
                return results
            
            # Monitor job completion
            cpu_completed = self._wait_for_job_completion(cpu_slurm_id, timeout=900)  # 15 minutes
            gpu_completed = self._wait_for_job_completion(gpu_slurm_id, timeout=900)
            
            if not cpu_completed or not gpu_completed:
                results["error"] = "Benchmark jobs timed out"
                return results
            
            # Collect results
            cpu_results = self.sol_executor.get_job_results(cpu_job_id, "cpu")
            gpu_results = self.sol_executor.get_job_results(gpu_job_id, "gpu")
            
            results["cpu_results"] = cpu_results
            results["gpu_results"] = gpu_results
            
            # Calculate speedup
            if (cpu_results.get("execution_time") and gpu_results.get("execution_time") 
                and cpu_results.get("status") == "success" and gpu_results.get("status") == "success"):
                speedup = cpu_results["execution_time"] / gpu_results["execution_time"]
                results["speedup"] = speedup
                results["winner"] = "GPU" if speedup > 1 else "CPU"
            
            # Cleanup
            self.sol_executor.cleanup_job_files(cpu_job_id)
            self.sol_executor.cleanup_job_files(gpu_job_id)
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Benchmark execution error: {e}")
        
        return results
    
    def _wait_for_job_completion(self, slurm_job_id: str, timeout: int = 900) -> bool:
        """Wait for a SLURM job to complete."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.sol_executor.check_job_status(slurm_job_id)
            if status in ["COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"]:
                return status == "COMPLETED"
            time.sleep(10)  # Check every 10 seconds
        return False
    
    def get_recent_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent benchmark results."""
        return self.benchmark_results[-limit:] if self.benchmark_results else []
    
    def format_benchmark_results(self, results: Dict[str, Any]) -> str:
        """Format benchmark results for display."""
        if results.get("error"):
            return f"❌ **Benchmark Error**: {results['error']}"
        
        category = results.get("category", "Unknown")
        benchmark = results.get("benchmark", "Unknown")
        size = results.get("size", 0)
        
        formatted = f"# 🏁 Benchmark Results: {benchmark}\n\n"
        formatted += f"**Category:** {category}\n"
        formatted += f"**Problem Size:** {size:,}\n\n"
        
        cpu_results = results.get("cpu_results", {})
        gpu_results = results.get("gpu_results", {})
        
        if cpu_results.get("execution_time") and gpu_results.get("execution_time"):
            cpu_time = cpu_results["execution_time"]
            gpu_time = gpu_results["execution_time"]
            speedup = results.get("speedup", 1.0)
            
            formatted += f"## ⏱️ Performance Results\n\n"
            formatted += f"| Implementation | Execution Time | Speedup |\n"
            formatted += f"|----------------|----------------|----------|\n"
            formatted += f"| CPU | {cpu_time:.4f}s | 1.0x |\n"
            formatted += f"| GPU | {gpu_time:.4f}s | {speedup:.2f}x |\n\n"
            
            if speedup > 10:
                formatted += "🔥 **Excellent GPU acceleration!** This workload is highly suitable for GPU computing.\n\n"
            elif speedup > 3:
                formatted += "✅ **Good GPU performance!** Significant speedup achieved.\n\n"
            elif speedup > 1:
                formatted += "⚡ **Moderate GPU benefit.** Some acceleration observed.\n\n"
            else:
                formatted += "⚠️ **GPU overhead detected.** CPU may be better for this workload size.\n\n"
        
        return formatted
    
    def run_user_code_benchmark(self, user_code: str) -> Dict[str, Any]:
        """Benchmark user-provided code against GPU-optimized version."""
        if not self.code_optimizer:
            return {"error": "Code optimizer not available"}
        
        try:
            # Analyze and optimize user code
            analysis = self.code_optimizer.analyze_code(user_code)
            optimized_code = self.code_optimizer.suggest_optimizations(user_code)
            
            # Create benchmark versions
            cpu_benchmark, gpu_benchmark = self.code_optimizer.create_benchmark_code(
                user_code, optimized_code
            )
            
            # Execute benchmarks
            results = self._execute_benchmark(
                cpu_benchmark, gpu_benchmark, "User Code", "Custom", 0
            )
            
            results["original_code"] = user_code
            results["optimized_code"] = optimized_code
            results["analysis"] = analysis
            
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking user code: {e}")
            return {"error": str(e)}
