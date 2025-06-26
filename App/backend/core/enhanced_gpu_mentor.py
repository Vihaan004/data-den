"""
Enhanced GPU Mentor Agent - Main orchestration class
Integrates RAG capabilities with code execution and analysis.
"""
import logging
import time
import io
import sys
import contextlib
from typing import Dict, List, Optional, Any
from datetime import datetime

from .rag_pipeline import RAGPipeline
from .benchmark_engine import BenchmarkEngine
from .code_optimizer import CodeOptimizer
from ..config import settings

logger = logging.getLogger(__name__)


class EnhancedGPUMentor:
    """
    Enhanced GPU Mentor that combines RAG capabilities with code execution and analysis.
    Integrates code input directly with LLM for comprehensive responses.
    """
    
    def __init__(self, rag_pipeline: RAGPipeline, benchmark_engine: BenchmarkEngine, code_optimizer: CodeOptimizer):
        self.rag_pipeline = rag_pipeline
        self.benchmark_engine = benchmark_engine
        self.code_optimizer = code_optimizer
        self.conversation_history = []
        self.code_execution_results = []
    
    async def process_user_input(self, user_input: str, code: Optional[str] = None) -> Dict[str, Any]:
        """
        Process user input with optional code, feeding both to LLM for integrated response.
        """
        
        response = {
            "text_response": "",
            "code_analysis": None,
            "code_output": None,
            "optimized_code": None,
            "socratic_questions": [],
            "learning_objectives": [],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Create enhanced prompt that includes code context if provided
            enhanced_prompt = self._create_enhanced_prompt(user_input, code)
            
            # Get RAG response with code context
            rag_result = self.rag_pipeline.invoke({
                "messages": [{"role": "user", "content": enhanced_prompt}]
            })
            response["text_response"] = rag_result["messages"][-1].content
            
            # If code is provided, analyze and execute it
            if code and code.strip():
                logger.info("🔍 Analyzing provided code...")
                
                # Analyze code for optimization opportunities
                analysis = self.code_optimizer.analyze_code(code)
                response["code_analysis"] = analysis
                
                # Generate optimized version
                optimized_code = await self.code_optimizer.suggest_optimizations(code)
                response["optimized_code"] = optimized_code
                
                # Execute code and capture output
                logger.info("⚡ Executing code...")
                try:
                    code_output = self._execute_code_safely(code)
                    response["code_output"] = code_output
                    
                    # Store execution results
                    self.code_execution_results.append({
                        "timestamp": datetime.now().isoformat(),
                        "code": code,
                        "output": code_output,
                        "analysis": analysis
                    })
                    
                except Exception as e:
                    response["code_output"] = {"error": f"Code execution failed: {str(e)}"}
                
                # Generate educational content based on code and context
                response["socratic_questions"] = self._generate_socratic_questions(analysis, user_input, code)
                response["learning_objectives"] = self._generate_learning_objectives(analysis, code)
            
            # Store conversation
            self.conversation_history.append({
                "user_input": user_input,
                "code": code,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            response["text_response"] = f"Sorry, I encountered an error processing your request: {str(e)}"
        
        return response
    
    def _create_enhanced_prompt(self, user_input: str, code: Optional[str] = None) -> str:
        """Create enhanced prompt that includes code context for the LLM."""
        
        if not code or not code.strip():
            return user_input
        
        enhanced_prompt = f"""
User Question: {user_input}

User's Python Code:
```python
{code}
```

Please analyze this code in the context of the user's question. Consider:
1. How the code relates to the question being asked
2. Potential GPU acceleration opportunities in this specific code
3. Any issues, errors, optimizations, or improvements you can suggest
4. Educational insights about GPU acceleration concepts demonstrated in this code
5. Performance characteristics and optimization strategies

Provide a comprehensive response that addresses both the question and the code together.
Include specific recommendations for GPU acceleration using NVIDIA Rapids (CuPy, cuDF, cuML).
"""
        return enhanced_prompt
    
    def _execute_code_safely(self, code: str) -> Dict[str, Any]:
        """Execute code safely and capture output."""
        
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        execution_result = {
            "stdout": "",
            "stderr": "",
            "variables": {},
            "execution_time": 0,
            "status": "success"
        }
        
        try:
            start_time = time.perf_counter()
            
            # Redirect output
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Create a safe execution environment
            safe_globals = self._create_safe_globals()
            local_vars = {}
            
            # Execute the code
            exec(code, safe_globals, local_vars)
            
            end_time = time.perf_counter()
            execution_result["execution_time"] = end_time - start_time
            
            # Capture variables (limit to avoid memory issues)
            execution_result["variables"] = self._extract_variables(local_vars)
            
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
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a safe execution environment with common libraries."""
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
        }
        
        # Add commonly used libraries if available
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
            import matplotlib.pyplot as plt
            safe_globals['plt'] = plt
        except ImportError:
            pass
        
        try:
            import time
            safe_globals['time'] = time
        except ImportError:
            pass
        
        return safe_globals
    
    def _extract_variables(self, local_vars: Dict[str, Any]) -> Dict[str, str]:
        """Extract and format variables from execution context."""
        variables = {}
        
        for name, value in local_vars.items():
            if not name.startswith('_'):
                try:
                    # Only store basic info about complex objects
                    if hasattr(value, 'shape'):  # numpy arrays, pandas objects
                        variables[name] = f"{type(value).__name__} with shape {value.shape}"
                    elif hasattr(value, '__len__') and len(value) > 100:
                        variables[name] = f"{type(value).__name__} with {len(value)} elements"
                    elif isinstance(value, (int, float, str, bool)) and len(str(value)) < 1000:
                        variables[name] = str(value)
                    elif isinstance(value, (list, dict, tuple)) and len(str(value)) < 1000:
                        variables[name] = str(value)
                    else:
                        variables[name] = f"{type(value).__name__} object"
                except Exception:
                    variables[name] = f"{type(value).__name__} object"
        
        return variables
    
    def _generate_socratic_questions(self, analysis: Dict[str, Any], user_context: str, code: str) -> List[str]:
        """Generate Socratic questions based on code analysis and user context."""
        questions = []
        
        libraries = analysis.get("libraries_detected", [])
        estimated_speedup = analysis.get("estimated_speedup", 1.0)
        parallel_ops = analysis.get("parallel_operations", [])
        
        # Library-specific questions
        if "numpy" in libraries:
            questions.extend([
                "Looking at your NumPy operations, which ones do you think would benefit most from GPU acceleration?",
                "How might the memory access patterns in your code affect GPU performance?",
                "What would happen to performance if you increased the array sizes by 10x?"
            ])
        
        if "pandas" in libraries:
            questions.extend([
                "Which pandas operations in your code are most computationally expensive?",
                "How would you modify this code to work with cuDF instead of pandas?",
                "What considerations should you make when transferring data between CPU and GPU?"
            ])
        
        # Operation-specific questions
        if any("matrix" in op.lower() for op in parallel_ops):
            questions.append("What makes matrix operations particularly well-suited for GPU acceleration?")
        
        if "for " in code and "range(" in code:
            questions.append("Could you vectorize any of these loops to improve performance?")
        
        if "def " in code:
            questions.append("How could you modify this function to accept both CPU and GPU arrays?")
        
        # Performance-based questions
        if estimated_speedup > 5:
            questions.append("Your code has high parallelization potential. What characteristics make it suitable for GPU acceleration?")
        elif estimated_speedup < 2:
            questions.append("This code may not benefit much from GPU acceleration. Can you identify why?")
        
        return questions[:3]  # Limit to avoid overwhelming
    
    def _generate_learning_objectives(self, analysis: Dict[str, Any], code: str) -> List[str]:
        """Generate specific learning objectives based on the code and analysis."""
        objectives = []
        
        libraries = analysis.get("libraries_detected", [])
        complexity = analysis.get("complexity_score", 0)
        
        if "numpy" in libraries:
            objectives.extend([
                "Understand when to use CuPy vs NumPy for your specific operations",
                "Learn about GPU memory management for array operations",
                "Master efficient data transfer between CPU and GPU"
            ])
        
        if "pandas" in libraries:
            objectives.extend([
                "Compare cuDF vs pandas for your data processing workflow",
                "Understand GPU memory requirements for dataframe operations",
                "Learn efficient groupby and aggregation patterns on GPU"
            ])
        
        if "sklearn" in libraries:
            objectives.extend([
                "Explore cuML alternatives to scikit-learn algorithms",
                "Understand GPU memory considerations for machine learning",
                "Learn about single precision vs double precision trade-offs"
            ])
        
        # Code-specific objectives
        if "for " in code:
            objectives.append("Explore vectorization techniques to eliminate loops")
        
        if "def " in code:
            objectives.append("Design functions that work efficiently with both CPU and GPU data")
        
        if complexity > 5:
            objectives.append("Analyze computational complexity and parallelization opportunities")
        
        return objectives[:4]  # Limit to key objectives
    
    async def generate_tutorial_content(self, topic: str) -> str:
        """Generate comprehensive tutorial content on specific GPU acceleration topics."""
        
        tutorial_prompt = f"""
        Create a comprehensive tutorial on {topic} for GPU acceleration. Include:
        1. Conceptual explanation of the topic
        2. Code examples comparing CPU vs GPU approaches
        3. Performance considerations and optimization tips
        4. Best practices and common pitfalls to avoid
        5. Hands-on exercises with RAPIDS and CuPy libraries
        6. Real-world use cases and applications
        
        Focus on practical, hands-on learning with clear explanations.
        Include specific code examples that demonstrate the concepts.
        """
        
        try:
            result = self.rag_pipeline.invoke({
                "messages": [{"role": "user", "content": tutorial_prompt}]
            })
            
            return result["messages"][-1].content
        except Exception as e:
            logger.error(f"Error generating tutorial: {e}")
            return f"Error generating tutorial: {str(e)}"
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all code execution results."""
        if not self.code_execution_results:
            return {"message": "No code executed yet"}
        
        summary = {
            "total_executions": len(self.code_execution_results),
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0,
            "common_libraries": [],
            "recent_outputs": [],
            "optimization_opportunities": 0
        }
        
        execution_times = []
        libraries_count = {}
        optimization_count = 0
        
        for result in self.code_execution_results[-10:]:  # Last 10 executions
            if result.get("output", {}).get("status") == "success":
                summary["successful_executions"] += 1
                exec_time = result.get("output", {}).get("execution_time", 0)
                execution_times.append(exec_time)
            else:
                summary["failed_executions"] += 1
            
            # Count libraries
            analysis = result.get("analysis", {})
            for lib in analysis.get("libraries_detected", []):
                libraries_count[lib] = libraries_count.get(lib, 0) + 1
            
            # Count optimization opportunities
            if analysis.get("optimization_opportunities"):
                optimization_count += len(analysis["optimization_opportunities"])
            
            # Add recent output summary
            output = result.get("output", {})
            summary["recent_outputs"].append({
                "timestamp": result.get("timestamp"),
                "status": output.get("status", "unknown"),
                "execution_time": output.get("execution_time", 0),
                "output_length": len(output.get("stdout", ""))
            })
        
        if execution_times:
            summary["average_execution_time"] = sum(execution_times) / len(execution_times)
        
        summary["common_libraries"] = sorted(libraries_count.items(), key=lambda x: x[1], reverse=True)
        summary["optimization_opportunities"] = optimization_count
        
        return summary
    
    def get_conversation_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context."""
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    def clear_history(self) -> None:
        """Clear conversation and execution history."""
        self.conversation_history.clear()
        self.code_execution_results.clear()
        logger.info("Cleared conversation and execution history")
