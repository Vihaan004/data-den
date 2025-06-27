import re
import time
import sys
import io
from typing import Dict, List, Any
from contextlib import redirect_stdout, redirect_stderr

class CodeOptimizer:
    """Optimize Python code for GPU acceleration."""
    
    def __init__(self, rag_agent=None):
        self.rag_agent = rag_agent
        self.optimization_patterns = self._setup_optimization_patterns()
    
    def _setup_optimization_patterns(self):
        """Setup code optimization patterns."""
        return {
            'numpy_to_cupy': [
                (r'import numpy as np', 'import cupy as cp'),
                (r'np\.', 'cp.'),
                (r'numpy\.', 'cupy.'),
            ],
            'pandas_to_cudf': [
                (r'import pandas as pd', 'import cudf as pd'),
                (r'pd\.', 'cudf.'),
                (r'pandas\.', 'cudf.'),
            ],
            'sklearn_to_cuml': [
                (r'from sklearn', 'from cuml'),
                (r'sklearn\.', 'cuml.'),
            ]
        }
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code for GPU optimization opportunities."""
        analysis = {
            "libraries_detected": [],
            "optimization_opportunities": [],
            "estimated_speedup": 1.0,
            "gpu_suitable": False,
            "recommendations": []
        }
        
        # Detect libraries
        if 'numpy' in code or 'np.' in code:
            analysis["libraries_detected"].append("numpy")
            analysis["optimization_opportunities"].append("Convert NumPy operations to CuPy")
            analysis["estimated_speedup"] *= 5.0
        
        if 'pandas' in code or 'pd.' in code:
            analysis["libraries_detected"].append("pandas")
            analysis["optimization_opportunities"].append("Convert Pandas operations to cuDF")
            analysis["estimated_speedup"] *= 3.0
        
        if 'sklearn' in code:
            analysis["libraries_detected"].append("sklearn")
            analysis["optimization_opportunities"].append("Convert scikit-learn to cuML")
            analysis["estimated_speedup"] *= 2.0
        
        # Check for GPU-suitable patterns
        gpu_patterns = [
            r'\.dot\(',
            r'\.matmul\(',
            r'@',  # matrix multiplication
            r'\.sum\(',
            r'\.mean\(',
            r'\.std\(',
            r'for\s+\w+\s+in\s+range\(',  # loops that could be vectorized
        ]
        
        for pattern in gpu_patterns:
            if re.search(pattern, code):
                analysis["gpu_suitable"] = True
                break
        
        # Generate recommendations
        if analysis["gpu_suitable"]:
            analysis["recommendations"].extend([
                "Consider using GPU memory pools for better performance",
                "Profile memory usage to optimize data transfers",
                "Use appropriate data types (float32 vs float64)"
            ])
        
        return analysis
    
    def suggest_optimizations(self, code: str) -> str:
        """Suggest GPU optimizations for the given code."""
        if self.rag_agent:
            # Use LLM-based optimization
            return self._llm_based_optimization(code)
        else:
            # Fallback to pattern-based optimization
            return self._pattern_based_optimization(code)
    
    def _llm_based_optimization(self, code: str) -> str:
        """Use LLM to generate optimized GPU code."""
        try:
            optimization_prompt = f"""
            Optimize the following Python code for GPU acceleration using NVIDIA Rapids libraries (CuPy, cuDF, cuML).
            
            Original code:
            ```python
            {code}
            ```
            
            Please provide:
            1. GPU-optimized version using appropriate Rapids libraries
            2. Brief explanation of the optimizations
            3. Expected performance improvements
            
            Focus on practical, working code that demonstrates GPU acceleration benefits.
            """
            
            response = self.rag_agent.query(optimization_prompt)
            return response
        except Exception as e:
            print(f"LLM optimization failed: {e}")
            return self._pattern_based_optimization(code)
    
    def _pattern_based_optimization(self, code: str) -> str:
        """Apply pattern-based optimizations as fallback."""
        optimized_code = code
        
        # Apply optimization patterns
        for category, patterns in self.optimization_patterns.items():
            for old_pattern, new_pattern in patterns:
                optimized_code = re.sub(old_pattern, new_pattern, optimized_code)
        
        # Add GPU-specific optimizations
        if 'import cupy as cp' in optimized_code:
            optimized_code = self._add_cupy_optimizations(optimized_code)
        
        if 'import cudf' in optimized_code:
            optimized_code = self._add_cudf_optimizations(optimized_code)
        
        return optimized_code
    
    def _add_cupy_optimizations(self, code: str) -> str:
        """Add CuPy-specific optimizations."""
        optimizations = []
        
        # Add memory pool optimization
        if 'import cupy as cp' in code:
            optimizations.append("# GPU memory pool optimization")
            optimizations.append("mempool = cp.get_default_memory_pool()")
            optimizations.append("pinned_mempool = cp.get_default_pinned_memory_pool()")
            optimizations.append("")
        
        # Add synchronization for timing
        if any(op in code for op in ['time.', 'perf_counter']):
            code = code.replace('time.perf_counter()', 
                              'time.perf_counter(); cp.cuda.Device().synchronize()')
        
        if optimizations:
            lines = code.split('\n')
            import_idx = -1
            for i, line in enumerate(lines):
                if 'import cupy as cp' in line:
                    import_idx = i
                    break
            
            if import_idx >= 0:
                lines = lines[:import_idx+1] + [''] + optimizations + lines[import_idx+1:]
                code = '\n'.join(lines)
        
        return code
    
    def _add_cudf_optimizations(self, code: str) -> str:
        """Add cuDF-specific optimizations."""
        # Add dtype optimizations
        if '.read_csv(' in code:
            code = code.replace('.read_csv(', '.read_csv(dtype={"column": "float32"}, ')
        
        return code
    
    def execute_code_safely(self, code: str) -> Dict[str, Any]:
        """Execute code safely and return results."""
        execution_result = {
            "status": "success",
            "stdout": "",
            "stderr": "",
            "execution_time": 0,
            "error": "",
            "variables": {}
        }
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        try:
            start_time = time.perf_counter()
            
            # Redirect output
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Create a safe execution environment
            safe_globals = {
                '__builtins__': __builtins__,
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'type': type,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'time': time,
            }
            
            # Add commonly used libraries
            try:
                import numpy as np
                safe_globals['np'] = np
                safe_globals['numpy'] = np
            except ImportError:
                pass
            
            try:
                import pandas as pd
                safe_globals['pd'] = pd
                safe_globals['pandas'] = pd
            except ImportError:
                pass
            
            try:
                import cupy as cp
                safe_globals['cp'] = cp
                safe_globals['cupy'] = cp
            except ImportError:
                pass
            
            try:
                import cudf
                safe_globals['cudf'] = cudf
            except ImportError:
                pass
            
            local_vars = {}
            
            # Execute the code
            exec(code, safe_globals, local_vars)
            
            end_time = time.perf_counter()
            execution_result["execution_time"] = end_time - start_time
            
            # Capture variables (limit to avoid memory issues)
            for name, value in local_vars.items():
                if not name.startswith('_'):
                    try:
                        # Only store basic info about complex objects
                        if hasattr(value, 'shape'):  # numpy arrays, pandas objects
                            execution_result["variables"][name] = f"{type(value).__name__} with shape {value.shape}"
                        elif hasattr(value, '__len__') and len(value) > 100:
                            execution_result["variables"][name] = f"{type(value).__name__} with {len(value)} elements"
                        elif isinstance(value, (int, float, str, bool, list, dict, tuple)) and len(str(value)) < 1000:
                            execution_result["variables"][name] = str(value)
                        else:
                            execution_result["variables"][name] = f"{type(value).__name__} object"
                    except:
                        execution_result["variables"][name] = f"{type(value).__name__} object"
            
        except Exception as e:
            execution_result["status"] = "error"
            execution_result["error"] = str(e)
            end_time = time.perf_counter()
            execution_result["execution_time"] = end_time - start_time
        
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Capture output
            execution_result["stdout"] = stdout_capture.getvalue()
            execution_result["stderr"] = stderr_capture.getvalue()
        
        return execution_result
