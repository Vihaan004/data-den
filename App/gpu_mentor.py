from typing import Dict, List, Any
import time
from datetime import datetime

class GPUMentor:
    """Main GPU Mentor class that coordinates RAG agent and code optimization."""
    
    def __init__(self, rag_agent, code_optimizer):
        self.rag_agent = rag_agent
        self.code_optimizer = code_optimizer
        self.conversation_history = []
        self.execution_results = []
    
    def process_user_input(self, question: str, code: str = "") -> Dict[str, Any]:
        """Process user input and provide comprehensive response."""
        timestamp = datetime.now().isoformat()
        
        response = {
            "timestamp": timestamp,
            "question": question,
            "code": code,
            "text_response": "",
            "code_analysis": {},
            "optimized_code": "",
            "code_output": {},
            "socratic_questions": [],
            "learning_objectives": []
        }
        
        try:
            # Get RAG response
            if question or code:
                if code and question:
                    # Both question and code provided
                    combined_query = f"""Question: {question}

Code to analyze and optimize for GPU acceleration:
```python
{code}
```

Please analyze this code and provide:
1. GPU optimization suggestions
2. Specific RAPIDS library recommendations (CuPy, cuDF, cuML)
3. Expected performance improvements
4. Code examples showing the optimized version

Focus on practical GPU acceleration techniques using NVIDIA Rapids libraries."""
                elif code and not question:
                    # Only code provided
                    combined_query = f"""Please analyze the following Python code for GPU acceleration opportunities:

```python
{code}
```

Provide:
1. Analysis of current code and GPU acceleration potential
2. Optimized version using NVIDIA Rapids libraries (CuPy, cuDF, cuML)
3. Expected performance improvements
4. Best practices for GPU optimization

Focus on practical GPU acceleration techniques."""
                else:
                    # Only question provided
                    combined_query = question
                
                response["text_response"] = self.rag_agent.query(combined_query)
            
            # Analyze and optimize code if provided
            if code:
                response["code_analysis"] = self.code_optimizer.analyze_code(code)
                response["optimized_code"] = self.code_optimizer.suggest_optimizations(code)
                
                # Execute original code
                execution_result = self.code_optimizer.execute_code_safely(code)
                response["code_output"] = execution_result
                
                # Generate learning content
                response["socratic_questions"] = self._generate_socratic_questions(
                    response["code_analysis"], question, code
                )
                response["learning_objectives"] = self._generate_learning_objectives(
                    response["code_analysis"], code
                )
            
            # Store in conversation history
            self.conversation_history.append(response)
            
        except Exception as e:
            response["text_response"] = f"Error processing request: {str(e)}"
        
        return response
    
    def chat_interface(self, message: str, code: str, history: List) -> tuple:
        """Interface for chat functionality."""
        if not message.strip() and not code.strip():
            return "", "", history
        
        # Create a meaningful user message that includes both text and code
        user_content = message if message.strip() else "Please analyze this code for GPU optimization opportunities."
        if code.strip():
            user_content += f"\n\nCode to analyze:\n```python\n{code}\n```"
        
        # Process the input
        response = self.process_user_input(message, code)
        
        # Format response for chat
        chat_response = self._format_chat_response(response)
        
        # Add to history in messages format
        user_message = {"role": "user", "content": user_content}
        assistant_message = {"role": "assistant", "content": chat_response}
        
        history.append(user_message)
        history.append(assistant_message)
        
        return "", "", history
    
    def _format_chat_response(self, response: Dict[str, Any]) -> str:
        """Format response for chat interface."""
        formatted = []
        
        if response["text_response"]:
            formatted.append(f"**🤖 AI Response:**\n{response['text_response']}")
        
        if response["code_analysis"]:
            analysis = response["code_analysis"]
            formatted.append(f"\n**🔍 Code Analysis:**")
            if analysis.get("libraries_detected"):
                formatted.append(f"• Libraries: {', '.join(analysis['libraries_detected'])}")
            if analysis.get("estimated_speedup", 1.0) > 1:
                formatted.append(f"• Estimated GPU speedup: {analysis['estimated_speedup']:.1f}x")
            if analysis.get("optimization_opportunities"):
                formatted.append(f"• Optimizations: {', '.join(analysis['optimization_opportunities'])}")
        
        if response["code_output"] and response["code_output"].get("status") == "success":
            output = response["code_output"]
            formatted.append(f"\n**⚡ Execution Results:**")
            formatted.append(f"• Execution time: {output.get('execution_time', 0):.4f}s")
            if output.get("stdout"):
                formatted.append(f"• Output: {output['stdout'][:200]}...")
        
        if response["socratic_questions"]:
            formatted.append(f"\n**🤔 Questions to Consider:**")
            for i, q in enumerate(response["socratic_questions"], 1):
                formatted.append(f"{i}. {q}")
        
        return "\n".join(formatted)
    
    def analyze_code_only(self, code: str) -> tuple:
        """Analyze code and return results for the analysis interface."""
        if not code.strip():
            return "No code provided for analysis.", ""
        
        try:
            # Create structured prompt for LLM to provide both analysis and optimized code
            llm_analysis_prompt = f"""You are GPU Mentor, an expert in NVIDIA Rapids GPU acceleration. Analyze the following Python code and provide GPU optimization recommendations.

**Code to Analyze:**
```python
{code}
```

**Instructions:**
Please provide your response in exactly this format:

## 🔍 AI Analysis & Recommendations

[Provide detailed analysis here including:
- What the code does and its current approach
- Libraries detected and their GPU acceleration potential
- Specific optimization opportunities
- Expected performance improvements
- Memory management considerations
- Best practices for GPU optimization]

## 🚀 GPU-Optimized Code

```python
[Provide the complete GPU-optimized version of the code here using:
- CuPy instead of NumPy
- cuDF instead of Pandas  
- cuML instead of scikit-learn
- Include proper imports and memory management
- Add comments explaining the optimizations]
```

## 💡 Optimization Insights

[Explain the specific changes made:
- Why each optimization was chosen
- How the GPU libraries work differently
- Performance implications
- When to use vs not use GPU acceleration
- Additional tips for scaling]

Focus on practical, working code that demonstrates clear GPU acceleration benefits."""
            
            # Get LLM response
            llm_response = self.rag_agent.query(llm_analysis_prompt)
            
            # Parse the response to separate analysis from optimized code
            analysis_text, optimized_code = self._parse_llm_response(llm_response)
            
            return analysis_text, optimized_code
            
        except Exception as e:
            error_msg = f"Error analyzing code: {str(e)}"
            return error_msg, ""
    
    def _parse_llm_response(self, response: str) -> tuple:
        """Parse LLM response to extract analysis and optimized code."""
        try:
            # Split response into sections
            sections = response.split("## 🚀 GPU-Optimized Code")
            
            if len(sections) >= 2:
                analysis_part = sections[0].strip()
                
                # Extract code from the second section
                code_section = sections[1]
                
                # Look for code blocks in the response
                import re
                code_blocks = re.findall(r'```python\n(.*?)\n```', code_section, re.DOTALL)
                
                if code_blocks:
                    optimized_code = code_blocks[0].strip()
                    
                    # Include insights section in analysis if present
                    if "## 💡 Optimization Insights" in code_section:
                        insights_part = code_section.split("## 💡 Optimization Insights")[1].strip()
                        analysis_part += f"\n\n## 💡 Optimization Insights\n{insights_part}"
                else:
                    # If no code block found, try to extract any code
                    lines = code_section.split('\n')
                    code_lines = []
                    in_code = False
                    for line in lines:
                        if line.strip().startswith('```'):
                            in_code = not in_code
                            continue
                        if in_code or (line.strip() and not line.startswith('#') and ('import' in line or '=' in line or 'def' in line)):
                            code_lines.append(line)
                    optimized_code = '\n'.join(code_lines).strip()
                
                return analysis_part, optimized_code
            else:
                # If response doesn't have expected structure, return as analysis
                return response, ""
                
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return response, ""
    
    def _format_analysis_results(self, analysis: Dict[str, Any], original_code: str) -> str:
        """Format analysis results for display."""
        results = []
        
        results.append("## 🔍 Code Analysis Results\n")
        
        if analysis.get("libraries_detected"):
            results.append(f"**Libraries Detected:** {', '.join(analysis['libraries_detected'])}")
        
        if analysis.get("estimated_speedup", 1.0) > 1:
            speedup = analysis["estimated_speedup"]
            results.append(f"**Estimated GPU Speedup:** {speedup:.1f}x")
        
        if analysis.get("gpu_suitable"):
            results.append("**GPU Suitability:** ✅ Well-suited for GPU acceleration")
        else:
            results.append("**GPU Suitability:** ⚠️ Limited GPU acceleration benefits")
        
        if analysis.get("optimization_opportunities"):
            results.append("\n**Optimization Opportunities:**")
            for opp in analysis["optimization_opportunities"]:
                results.append(f"• {opp}")
        
        if analysis.get("recommendations"):
            results.append("\n**Recommendations:**")
            for rec in analysis["recommendations"]:
                results.append(f"• {rec}")
        
        # Execute original code and show timing
        try:
            execution_result = self.code_optimizer.execute_code_safely(original_code)
            if execution_result["status"] == "success":
                results.append(f"\n**Original Code Execution:**")
                results.append(f"• Execution time: {execution_result['execution_time']:.4f}s")
                if execution_result.get("stdout"):
                    results.append(f"• Output: {execution_result['stdout'][:200]}...")
        except Exception as e:
            results.append(f"\n**Execution Error:** {str(e)}")
        
        return "\n".join(results)
    
    def _format_analysis_results_with_llm(self, analysis: Dict[str, Any], original_code: str, llm_response: str) -> str:
        """Format analysis results including LLM analysis for display."""
        results = []
        
        results.append("## 🔍 Code Analysis Results\n")
        
        # Add LLM analysis first
        results.append("### 🤖 AI Analysis & Recommendations")
        results.append(llm_response)
        results.append("")
        
        # Add technical analysis
        results.append("### 📊 Technical Analysis")
        
        if analysis.get("libraries_detected"):
            results.append(f"**Libraries Detected:** {', '.join(analysis['libraries_detected'])}")
        
        if analysis.get("estimated_speedup", 1.0) > 1:
            speedup = analysis["estimated_speedup"]
            results.append(f"**Estimated GPU Speedup:** {speedup:.1f}x")
        
        if analysis.get("gpu_suitable"):
            results.append("**GPU Suitability:** ✅ Well-suited for GPU acceleration")
        else:
            results.append("**GPU Suitability:** ⚠️ Limited GPU acceleration benefits")
        
        if analysis.get("optimization_opportunities"):
            results.append("\n**Optimization Opportunities:**")
            for opp in analysis["optimization_opportunities"]:
                results.append(f"• {opp}")
        
        if analysis.get("recommendations"):
            results.append("\n**Recommendations:**")
            for rec in analysis["recommendations"]:
                results.append(f"• {rec}")
        
        # Execute original code and show timing
        try:
            execution_result = self.code_optimizer.execute_code_safely(original_code)
            if execution_result["status"] == "success":
                results.append(f"\n**Original Code Execution:**")
                results.append(f"• Execution time: {execution_result['execution_time']:.4f}s")
                if execution_result.get("stdout"):
                    results.append(f"• Output: {execution_result['stdout'][:200]}...")
        except Exception as e:
            results.append(f"\n**Execution Error:** {str(e)}")
        
        return "\n".join(results)
    
    def _generate_socratic_questions(self, analysis: Dict, user_context: str, code: str) -> List[str]:
        """Generate Socratic questions based on code analysis."""
        questions = []
        
        libraries = analysis.get("libraries_detected", [])
        estimated_speedup = analysis.get("estimated_speedup", 1.0)
        
        # Code-specific questions
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
        
        # Context-aware questions
        if "for " in code and "range(" in code:
            questions.append("Could you vectorize any of these loops to improve performance?")
        
        if "def " in code:
            questions.append("How could you modify this function to accept both CPU and GPU arrays?")
        
        if estimated_speedup > 5:
            questions.append("Your code has high parallelization potential. What makes it suitable for GPU acceleration?")
        elif estimated_speedup < 2:
            questions.append("This code may not benefit much from GPU acceleration. Can you identify why?")
        
        return questions[:3]  # Limit to avoid overwhelming
    
    def _generate_learning_objectives(self, analysis: Dict, code: str) -> List[str]:
        """Generate specific learning objectives based on the code and analysis."""
        objectives = []
        
        libraries = analysis.get("libraries_detected", [])
        
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
        
        # Code-specific objectives
        if "for " in code:
            objectives.append("Explore vectorization techniques to eliminate loops")
        
        if "def " in code:
            objectives.append("Design functions that work efficiently with both CPU and GPU data")
        
        return objectives
    
    def get_tutorial_content(self, topic: str) -> str:
        """Generate tutorial content for a specific topic."""
        tutorial_prompt = f"""
        Create a comprehensive tutorial on {topic} for GPU acceleration. Include:
        1. Conceptual explanation
        2. Code examples comparing CPU vs GPU approaches
        3. Performance considerations
        4. Best practices
        5. Common pitfalls to avoid
        
        Focus on practical, hands-on learning with RAPIDS and CuPy libraries.
        """
        
        return self.rag_agent.query(tutorial_prompt)
    
    def get_execution_summary(self) -> str:
        """Get summary of all execution results."""
        if not self.execution_results:
            return "No code executions recorded yet."
        
        summary = []
        summary.append(f"## Execution Summary ({len(self.execution_results)} executions)\n")
        
        successful = sum(1 for r in self.execution_results if r.get("status") == "success")
        summary.append(f"**Successful executions:** {successful}/{len(self.execution_results)}")
        
        if successful > 0:
            avg_time = sum(r.get("execution_time", 0) for r in self.execution_results if r.get("status") == "success") / successful
            summary.append(f"**Average execution time:** {avg_time:.4f}s")
        
        return "\n".join(summary)
