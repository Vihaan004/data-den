"""
Code Optimizer - GPU acceleration analysis and optimization suggestions
Extracted from enhanced_agentic_rag_ollama.ipynb
"""

import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class CodeOptimizer:
    """
    Analyzes user code and suggests GPU-accelerated alternatives using RAPIDS and CuPy.
    Uses both LLM-powered optimization and pattern-based fallback.
    """
    
    def __init__(self, rag_pipeline=None):
        self.rag_pipeline = rag_pipeline
        self.optimization_patterns = self._setup_optimization_patterns()
        logger.info("Code Optimizer initialized")
    
    def _setup_optimization_patterns(self) -> Dict[str, Dict[str, str]]:
        """Setup optimization patterns for different libraries."""
        return {
            # NumPy to CuPy optimizations
            'numpy': {
                'import numpy as np': 'import cupy as np',
                'np.array(': 'cp.array(',
                'np.random.': 'cp.random.',
                'np.linalg.': 'cp.linalg.',
                'np.fft.': 'cp.fft.',
                '.cpu()': '',  # Remove .cpu() calls
            },
            
            # Pandas to cuDF optimizations
            'pandas': {
                'import pandas as pd': 'import cudf as pd',
                'pd.DataFrame(': 'cudf.DataFrame(',
                'pd.Series(': 'cudf.Series(',
                'pd.read_csv(': 'cudf.read_csv(',
                'pd.read_parquet(': 'cudf.read_parquet(',
                '.to_pandas()': '',  # Remove .to_pandas() calls
            },
            
            # Scikit-learn to cuML optimizations
            'sklearn': {
                'from sklearn.': 'from cuml.',
                'sklearn.': 'cuml.',
            },
            
            # Dask optimizations
            'dask': {
                'import dask.array as da': 'import dask.array as da\\n# Configure Dask to use CuPy backend\\nimport dask\\ndask.config.set({"array.backend": "cupy"})',
                'import dask.dataframe as dd': 'import dask_cudf as dd',
            }
        }
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code for optimization opportunities."""
        analysis = {
            'libraries_detected': [],
            'optimization_opportunities': [],
            'estimated_speedup': 1.0,
            'gpu_compatible': True,
            'warnings': []
        }
        
        # Detect libraries used
        for lib_type, patterns in self.optimization_patterns.items():
            for pattern in patterns.keys():
                if pattern in code:
                    analysis['libraries_detected'].append(lib_type)
                    break
        
        # Check for GPU incompatible operations
        incompatible_patterns = [
            'matplotlib.pyplot',  # Plotting might need CPU arrays
            'pickle.dump',        # Serialization issues
            'multiprocessing',    # GPU memory management conflicts
            'threading',          # GPU context issues
            'open(',              # File I/O operations
            'input(',             # User input
        ]
        
        for pattern in incompatible_patterns:
            if pattern in code:
                analysis['warnings'].append(f"Potential GPU incompatibility: {pattern}")
                analysis['gpu_compatible'] = False
        
        # Estimate optimization opportunities
        optimization_ops = [
            ('np.dot', 'matrix multiplication'),
            ('np.matmul', 'matrix multiplication'),
            ('@', 'matrix multiplication'),
            ('np.sum', 'reduction operations'),
            ('np.mean', 'reduction operations'),
            ('np.std', 'reduction operations'),
            ('pd.groupby', 'dataframe groupby'),
            ('df.groupby', 'dataframe groupby'),
            ('.apply(', 'element-wise operations'),
            ('for ', 'potential vectorization'),
            ('np.random', 'random number generation'),
            ('np.fft', 'FFT operations'),
            ('sklearn.', 'machine learning'),
        ]
        
        speedup_factors = []
        for op, description in optimization_ops:
            if op in code:
                analysis['optimization_opportunities'].append(description)
                # Assign speedup estimates based on operation type
                if 'matrix' in description:
                    speedup_factors.append(10.0)  # Matrix ops
                elif 'fft' in description.lower():
                    speedup_factors.append(15.0)  # FFT ops
                elif 'groupby' in description:
                    speedup_factors.append(8.0)   # DataFrame ops
                elif 'vectorization' in description:
                    speedup_factors.append(5.0)   # Loop vectorization
                elif 'machine learning' in description:
                    speedup_factors.append(12.0)  # ML ops
                else:
                    speedup_factors.append(3.0)   # Other ops
        
        if speedup_factors:
            analysis['estimated_speedup'] = max(speedup_factors)
        
        return analysis
    
    def suggest_optimizations(self, code: str) -> str:
        """Generate GPU-optimized version of the code."""
        # Try LLM-powered optimization first
        if self.rag_pipeline:
            try:
                optimized_code = self._llm_optimize_code(code)
                if optimized_code and optimized_code.strip() != code.strip():
                    return optimized_code
            except Exception as e:
                logger.warning(f"LLM optimization failed, falling back to patterns: {e}")
        
        # Fallback to pattern-based optimization
        return self._pattern_optimize_code(code)
    
    def _llm_optimize_code(self, code: str) -> str:
        """Use LLM to optimize code for GPU acceleration."""
        optimization_prompt = f"""
You are an expert in GPU acceleration using NVIDIA RAPIDS libraries (CuPy, cuDF, cuML).

Please convert the following CPU code to use GPU-accelerated alternatives:

CPU Code:
```python
{code}
```

Provide an optimized GPU version that:
1. Replaces NumPy with CuPy (import cupy as cp)
2. Replaces Pandas with cuDF where possible
3. Replaces scikit-learn with cuML where possible
4. Adds memory management optimizations
5. Maintains the same functionality

Return ONLY the optimized Python code without explanations:
"""
        
        try:
            result = self.rag_pipeline.invoke({
                "messages": [{"role": "user", "content": optimization_prompt}]
            })
            
            response = result["messages"][-1]["content"]
            
            # Extract code from response (handle code blocks)
            code_match = re.search(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            else:
                # If no code block, try to clean up the response
                lines = response.split('\n')
                code_lines = []
                for line in lines:
                    # Skip explanatory text
                    if line.strip() and not line.strip().startswith('#'):
                        code_lines.append(line)
                
                if code_lines:
                    return '\n'.join(code_lines)
                
        except Exception as e:
            logger.error(f"Error in LLM optimization: {e}")
            raise
        
        return code
    
    def _pattern_optimize_code(self, code: str) -> str:
        """Apply pattern-based optimizations as fallback."""
        optimized_code = code
        
        # Apply optimization patterns
        for lib_type, patterns in self.optimization_patterns.items():
            for old_pattern, new_pattern in patterns.items():
                optimized_code = optimized_code.replace(old_pattern, new_pattern)
        
        # Add GPU-specific optimizations
        if 'import cupy' in optimized_code and 'import cupy as cp' not in optimized_code:
            optimized_code = 'import cupy as cp\n' + optimized_code
        
        # Add memory pool for better performance
        if 'cupy' in optimized_code or 'cp.' in optimized_code:
            memory_pool_code = """
# Enable CuPy memory pool for better performance
import cupy
mempool = cupy.get_default_memory_pool()
pinned_mempool = cupy.get_default_pinned_memory_pool()

"""
            optimized_code = memory_pool_code + optimized_code
        
        return optimized_code
    
    def create_benchmark_code(self, original_code: str, optimized_code: str) -> tuple[str, str]:
        """Create benchmark versions of CPU and GPU code."""
        
        # CPU benchmark wrapper
        cpu_benchmark = f"""
import time
import json
import numpy as np
import pandas as pd

start_time = time.perf_counter()
try:
{self._indent_code(original_code)}
    execution_status = "success"
    error_message = ""
except Exception as e:
    execution_status = "error"
    error_message = str(e)
    import traceback
    traceback.print_exc()

end_time = time.perf_counter()
execution_time = end_time - start_time

# Save benchmark results
results = {{
    "execution_time": execution_time,
    "job_type": "cpu",
    "status": execution_status,
    "error": error_message
}}

with open("cpu_benchmark_results.json", "w") as f:
    json.dump(results, f)

print(f"CPU Execution time: {{execution_time:.4f}} seconds")
"""
        
        # GPU benchmark wrapper
        gpu_benchmark = f"""
import time
import json
import cupy as cp
import cudf as pd

# Configure GPU memory
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

start_time = time.perf_counter()
try:
{self._indent_code(optimized_code)}
    cp.cuda.Device().synchronize()  # Ensure GPU operations complete
    execution_status = "success"
    error_message = ""
except Exception as e:
    execution_status = "error"
    error_message = str(e)
    import traceback
    traceback.print_exc()

end_time = time.perf_counter()
execution_time = end_time - start_time

# Save benchmark results
results = {{
    "execution_time": execution_time,
    "job_type": "gpu",
    "status": execution_status,
    "error": error_message,
    "gpu_memory_used": mempool.used_bytes(),
    "gpu_total_memory": mempool.total_bytes()
}}

with open("gpu_benchmark_results.json", "w") as f:
    json.dump(results, f)

print(f"GPU Execution time: {{execution_time:.4f}} seconds")
print(f"GPU Memory used: {{mempool.used_bytes() / 1024**2:.2f}} MB")
"""
        
        return cpu_benchmark, gpu_benchmark
    
    def _indent_code(self, code: str, indent: str = "    ") -> str:
        """Add proper indentation to user code for embedding in script."""
        return "\n".join(indent + line for line in code.split("\n"))
    
    def get_optimization_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Get specific optimization suggestions based on analysis."""
        suggestions = []
        
        libraries = analysis.get("libraries_detected", [])
        opportunities = analysis.get("optimization_opportunities", [])
        
        if "numpy" in libraries:
            suggestions.append("Replace NumPy operations with CuPy for GPU acceleration")
            suggestions.append("Use cupy.fuse decorator for element-wise operations")
            suggestions.append("Keep arrays on GPU to avoid memory transfers")
        
        if "pandas" in libraries:
            suggestions.append("Replace Pandas with cuDF for GPU-accelerated DataFrames")
            suggestions.append("Use GPU-optimized groupby and aggregation operations")
        
        if "sklearn" in libraries:
            suggestions.append("Replace scikit-learn with cuML for GPU-accelerated ML")
        
        if "matrix multiplication" in opportunities:
            suggestions.append("Matrix operations benefit significantly from GPU acceleration")
        
        if "potential vectorization" in opportunities:
            suggestions.append("Consider vectorizing loops for better GPU utilization")
        
        return suggestions
