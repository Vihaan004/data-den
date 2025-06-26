#!/usr/bin/env python3
"""
Enhanced Gradio UI for GPU Mentor Backend - Runs on Sol
Matches the advanced interface from enhanced_agentic_rag_ollama.ipynb
"""
import sys
sys.path.append('.')

import gradio as gr
import traceback
import json
import asyncio
from datetime import datetime
from core.enhanced_gpu_mentor import EnhancedGPUMentor
from core.benchmark_engine import BenchmarkEngine
from utils.sample_code_library import SampleCodeLibrary
from utils.educational_content import EducationalContentEnhancer
from utils.performance_visualizer import PerformanceVisualizer

# Initialize all components
try:
    gpu_mentor = EnhancedGPUMentor()
    benchmark_engine = BenchmarkEngine()
    sample_library = SampleCodeLibrary()
    content_enhancer = EducationalContentEnhancer()
    perf_visualizer = PerformanceVisualizer()
    print("✅ Enhanced GPU Mentor backend initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize backend: {e}")
    gpu_mentor = None
    benchmark_engine = None
    content_enhancer = None
    perf_visualizer = None

# Sample code examples for quick testing
sample_codes = {
    "Simple Array Operations": '''import numpy as np

# Create arrays
n = 1000
x = np.random.rand(n)
y = np.random.rand(n)

# Basic operations
result = np.sqrt(x**2 + y**2)
mean_result = np.mean(result)

print(f"Array size: {n}")
print(f"Mean result: {mean_result:.4f}")
print(f"Max result: {np.max(result):.4f}")''',

    "Matrix Multiplication": '''import numpy as np

# Create matrices
n = 500
A = np.random.rand(n, n)
B = np.random.rand(n, n)

# Matrix multiplication
C = np.dot(A, B)

print(f"Matrix size: {n}x{n}")
print(f"Result shape: {C.shape}")
print(f"Result sum: {np.sum(C):.2f}")''',
    
    "DataFrame Operations": '''import pandas as pd
import numpy as np

# Create dataset
n = 10000
df = pd.DataFrame({
    'x': np.random.randn(n),
    'y': np.random.randn(n),
    'group': np.random.choice(['A', 'B', 'C'], n)
})

# Compute statistics
result = df.groupby('group').agg({
    'x': ['mean', 'std'],
    'y': ['sum', 'count']
})

print(f"Dataset size: {len(df)} rows")
print("Grouped results:")
print(result)''',
    
    "Mathematical Functions": '''import numpy as np

# Generate data
n = 5000
x = np.linspace(0, 4*np.pi, n)
y = np.sin(x) * np.exp(-x/10)

# Compute statistics
mean_y = np.mean(y)
std_y = np.std(y)
max_y = np.max(y)

print(f"Data points: {n}")
print(f"Mean: {mean_y:.4f}")
print(f"Std: {std_y:.4f}")
print(f"Max: {max_y:.4f}")''',

    "Data Processing Loop": '''import numpy as np

# Create data
data = np.random.rand(1000, 10)
results = []

# Process data (can be vectorized)
for i in range(len(data)):
    row_sum = np.sum(data[i])
    row_mean = np.mean(data[i])
    results.append(row_sum * row_mean)

final_result = np.array(results)
print(f"Processed {len(data)} rows")
print(f"Final result shape: {final_result.shape}")
print(f"Average result: {np.mean(final_result):.4f}")'''
}

def chat_with_mentor(message, code, chat_history):
    """Handle chat interactions with the GPU Mentor - integrates code with LLM."""
    
    if not gpu_mentor:
        error_msg = "❌ GPU Mentor backend not initialized"
        if chat_history is None:
            chat_history = []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return "", "", chat_history, None, ""
    
    try:
        # Check if process_user_input is async and handle accordingly
        if hasattr(gpu_mentor, 'process_user_input'):
            # Try to run the async method synchronously
            try:
                import inspect
                if inspect.iscoroutinefunction(gpu_mentor.process_user_input):
                    # Create new event loop for async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(gpu_mentor.process_user_input(message, code))
                    loop.close()
                else:
                    # Call synchronously if not async
                    response = gpu_mentor.process_user_input(message, code)
            except Exception as async_error:
                # Fallback: try to get a basic response from RAG system
                print(f"Async error, trying fallback: {async_error}")
                if hasattr(gpu_mentor, 'rag_graph'):
                    rag_result = gpu_mentor.rag_graph.invoke({
                        "messages": [{"role": "user", "content": message}]
                    })
                    response = {
                        "text_response": rag_result["messages"][-1].content,
                        "code_analysis": None,
                        "code_output": None,
                        "optimized_code": "",
                        "socratic_questions": []
                    }
                else:
                    raise async_error
        else:
            # Fallback: use explain_optimization method
            response = {
                "text_response": gpu_mentor.explain_optimization(message),
                "code_analysis": None,
                "code_output": None,
                "optimized_code": "",
                "socratic_questions": []
            }
        
        # Format response for chat
        formatted_response = response["text_response"]
        
        # Add code analysis if available
        if response.get("code_analysis"):
            analysis = response["code_analysis"]
            formatted_response += f"\n\n**📊 Code Analysis:**\n"
            formatted_response += f"• Libraries detected: {', '.join(analysis['libraries_detected'])}\n"
            formatted_response += f"• Estimated speedup potential: {analysis['estimated_speedup']:.1f}x\n"
            formatted_response += f"• GPU compatible: {'✅' if analysis['gpu_compatible'] else '❌'}\n"
            
            if analysis['warnings']:
                formatted_response += f"• ⚠️ Warnings: {'; '.join(analysis['warnings'])}\n"
        
        # Add code execution output
        if response.get("code_output"):
            output = response["code_output"]
            formatted_response += f"\n\n**⚡ Code Execution Results:**\n"
            
            if output.get("status") == "success":
                formatted_response += f"• ✅ Execution successful ({output.get('execution_time', 0):.3f}s)\n"
                
                if output.get("stdout"):
                    formatted_response += f"• 📄 Output:\n```\n{output['stdout']}\n```\n"
                
                if output.get("variables"):
                    formatted_response += f"• 📊 Variables created: {', '.join(output['variables'].keys())}\n"
                    # Show details for important variables
                    for var_name, var_info in list(output['variables'].items())[:3]:
                        formatted_response += f"  - `{var_name}`: {var_info}\n"
            else:
                formatted_response += f"• ❌ Execution failed: {output.get('error', 'Unknown error')}\n"
                if output.get("stderr"):
                    formatted_response += f"• 🚨 Error details:\n```\n{output['stderr']}\n```\n"
        
        # Add Socratic questions
        if response.get("socratic_questions"):
            formatted_response += f"\n\n**🤔 Think About This:**\n"
            for i, question in enumerate(response["socratic_questions"], 1):
                formatted_response += f"{i}. {question}\n"
        
        # Update chat history
        if chat_history is None:
            chat_history = []
        
        # Format user message with code if provided
        user_message = message
        if code and code.strip():
            user_message += f"\n\n```python\n{code}\n```"
        
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": formatted_response})
        
        return "", "", chat_history, response.get("code_output"), response.get("optimized_code", "")
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(f"Chat error: {e}")
        traceback.print_exc()
        if chat_history is None:
            chat_history = []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return "", "", chat_history, None, ""

def analyze_code_only(code):
    """Analyze code for optimization opportunities using LLM."""
    
    if not code.strip():
        return "Please provide code to analyze.", ""
    
    if not gpu_mentor:
        return "❌ GPU Mentor backend not initialized", ""
    
    try:
        # Use LLM for both analysis and optimization
        analysis_prompt = f"""
Please analyze the following Python code for GPU acceleration opportunities using NVIDIA Rapids (cuDF, cuML) and CuPy:

```python
{code}
```

Provide a detailed analysis including:
1. **Libraries Detected**: What Python libraries are being used?
2. **GPU Acceleration Potential**: What operations could benefit from GPU acceleration?
3. **Optimization Opportunities**: Specific suggestions for GPU optimization
4. **Performance Estimate**: Estimated speedup potential
5. **Considerations**: Any potential issues or limitations
6. **Recommendations**: Best practices for GPU optimization

Format your response with clear sections and bullet points for readability.
"""
        
        # Get analysis from LLM
        if hasattr(gpu_mentor, 'rag_graph'):
            analysis_result = gpu_mentor.rag_graph.invoke({
                "messages": [{"role": "user", "content": analysis_prompt}]
            })
            analysis_text = analysis_result["messages"][-1].content
        else:
            analysis_text = "RAG system not available"
        
        # Generate GPU-optimized code using LLM
        optimization_prompt = f"""
Please convert the following Python code to use GPU acceleration with NVIDIA Rapids and CuPy.

Original Code:
```python
{code}
```

Please provide a GPU-optimized version that:
1. Replaces NumPy operations with CuPy where appropriate
2. Replaces Pandas operations with cuDF where beneficial
3. Replaces scikit-learn operations with cuML where possible
4. Adds proper memory management (memory pools, data transfers)
5. Includes necessary imports and setup
6. Maintains the same functionality as the original code

Return ONLY the optimized Python code without explanations or markdown formatting.
"""
        
        # Get optimized code from LLM
        if hasattr(gpu_mentor, 'rag_graph'):
            optimization_result = gpu_mentor.rag_graph.invoke({
                "messages": [{"role": "user", "content": optimization_prompt}]
            })
            optimized_code = optimization_result["messages"][-1].content
        else:
            optimized_code = "# RAG system not available for code optimization"
        
        # Clean up the optimized code (remove markdown formatting if present)
        if "```python" in optimized_code:
            optimized_code = optimized_code.split("```python")[1].split("```")[0].strip()
        elif "```" in optimized_code:
            optimized_code = optimized_code.split("```")[1].strip()
        
        return analysis_text, optimized_code
        
    except Exception as e:
        print(f"Code analysis error: {e}")
        traceback.print_exc()
        return f"Error analyzing code: {str(e)}", ""

def get_tutorial(topic):
    """Generate tutorial content for specific topics."""
    
    if not topic.strip():
        return "Please specify a topic for the tutorial."
    
    if not gpu_mentor:
        return "❌ GPU Mentor backend not initialized"
    
    try:
        if hasattr(gpu_mentor, 'generate_tutorial_content'):
            # Check if it's async
            import inspect
            if inspect.iscoroutinefunction(gpu_mentor.generate_tutorial_content):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tutorial_content = loop.run_until_complete(gpu_mentor.generate_tutorial_content(topic))
                loop.close()
            else:
                tutorial_content = gpu_mentor.generate_tutorial_content(topic)
        else:
            # Fallback: use RAG system directly
            if hasattr(gpu_mentor, 'rag_graph'):
                tutorial_prompt = f"""
Create a comprehensive tutorial on the topic: {topic}

Please include:
1. Introduction and overview
2. Key concepts and definitions  
3. Step-by-step examples with code
4. Best practices and common pitfalls
5. Advanced tips and optimization techniques
6. Further reading and resources

Format the tutorial with clear headings and code examples.
"""
                result = gpu_mentor.rag_graph.invoke({
                    "messages": [{"role": "user", "content": tutorial_prompt}]
                })
                tutorial_content = result["messages"][-1].content
            else:
                tutorial_content = f"Tutorial generation not available. Topic requested: {topic}"
        
        return tutorial_content
    except Exception as e:
        print(f"Tutorial generation error: {e}")
        traceback.print_exc()
        return f"Error generating tutorial: {str(e)}"

def clear_chat():
    """Clear chat history."""
    return [], None, None

# Benchmarking functions
def update_benchmark_options(category):
    """Update benchmark name dropdown based on selected category."""
    if not category or not benchmark_engine:
        return gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None)
    
    benchmarks = benchmark_engine.get_benchmarks_for_category(category)
    return (
        gr.Dropdown(choices=benchmarks, value=benchmarks[0] if benchmarks else None),
        gr.Dropdown(choices=[], value=None)
    )

def update_size_options(category, benchmark_name):
    """Update size dropdown based on selected benchmark."""
    if not category or not benchmark_name or not benchmark_engine:
        return gr.Dropdown(choices=[], value=None)
    
    sizes = benchmark_engine.get_benchmark_sizes(category, benchmark_name)
    return gr.Dropdown(choices=sizes, value=sizes[0] if sizes else None)

def run_selected_benchmark(category, benchmark_name, size):
    """Run the selected benchmark and return enhanced results with visualizations."""
    if not all([category, benchmark_name, size]):
        return "❌ Please select all benchmark parameters", "Please select category, benchmark, and size"
    
    status_msg = f"🏃‍♂️ Running {benchmark_name} benchmark with size {size:,}..."
    
    try:
        results = benchmark_engine.run_benchmark(category, benchmark_name, size)
        
        if not results:
            return "❌ Benchmark failed to execute", "Benchmark execution failed"
        
        # Create comprehensive results with visualizations
        formatted_results = ""
        
        # Add basic benchmark results
        basic_results = benchmark_engine.format_benchmark_results(results)
        formatted_results += basic_results
        
        # Add performance visualization if successful
        if results.get("speedup") and not results.get("error"):
            formatted_results += "\n\n---\n\n"
            formatted_results += perf_visualizer.create_speedup_visualization(results)
            
            # Add educational summary
            formatted_results += "\n\n---\n\n"
            formatted_results += perf_visualizer.create_educational_summary(results)
            
            # Add code examples for this category
            formatted_results += "\n\n---\n\n"
            formatted_results += f"### 💻 Code Examples\n\n"
            
            if "Matrix" in category:
                example = content_enhancer.get_example_for_operation("matrix_multiplication")
            elif "DataFrame" in category:
                example = content_enhancer.get_example_for_operation("dataframe_groupby")
            else:
                example = content_enhancer.get_example_for_operation("element_wise_operations")
            
            if example:
                formatted_results += f"**CPU Implementation:**\n```python\n{example['cpu_code']}\n```\n\n"
                formatted_results += f"**GPU Implementation:**\n```python\n{example['gpu_code']}\n```\n\n"
                formatted_results += f"**Expected Speedup:** {example['expected_speedup']}\n\n"
                formatted_results += "**Key Optimization Points:**\n"
                for point in example['key_points']:
                    formatted_results += f"• {point}\n"
        
        # Success message
        if results and results.get("speedup"):
            success_msg = f"✅ Completed! GPU Speedup: {results['speedup']:.2f}x"
            if results['speedup'] > 10:
                success_msg += " 🔥 Excellent acceleration!"
            elif results['speedup'] > 3:
                success_msg += " ✅ Good performance!"
            elif results['speedup'] > 1:
                success_msg += " ⚡ Moderate improvement"
            else:
                success_msg += " ⚠️ GPU overhead detected"
        else:
            success_msg = "✅ Benchmark completed"
            
        return formatted_results, success_msg
        
    except Exception as e:
        error_msg = f"❌ Benchmark failed: {str(e)}"
        error_details = f"""
## ❌ Benchmark Error

**Error Details:** {str(e)}

**Possible Causes:**
• GPU libraries (CuPy, cuDF, cuML) may not be installed
• Insufficient GPU memory for the selected problem size  
• CUDA driver/runtime issues

**Troubleshooting:**
• Try a smaller problem size
• Check GPU memory availability
• Verify RAPIDS installation: `conda list | grep -E "(cupy|cudf|cuml)"`

**Note:** CPU benchmarks should still work even without GPU libraries.
"""
        return error_details, error_msg

def show_recent_benchmarks():
    """Show recent benchmark results."""
    if not benchmark_engine:
        return {"message": "Benchmark engine not available"}, gr.JSON(visible=True)
        
    recent = benchmark_engine.get_recent_results()
    if not recent:
        return {"message": "No recent benchmarks"}, gr.JSON(visible=True)
    
    summary = []
    for result in recent:
        summary.append({
            "benchmark": result.get("benchmark", "Unknown"),
            "category": result.get("category", "Unknown"),
            "size": result.get("size", 0),
            "speedup": result.get("speedup", "N/A"),
            "winner": result.get("winner", "N/A")
        })
    
    return summary, gr.JSON(visible=True)

def run_quick_benchmark(benchmark_type):
    """Run a quick benchmark for common operations."""
    quick_benchmarks = {
        "Matrix Ops": ("Matrix Operations", "Matrix Multiplication", 1024),
        "DataFrame Ops": ("DataFrame Operations", "GroupBy Aggregation", 500000),
        "ML Algorithms": ("Machine Learning", "K-Means Clustering", 50000),
        "Math Functions": ("Mathematical Functions", "FFT Computation", 131072)
    }
    
    if benchmark_type in quick_benchmarks:
        category, name, size = quick_benchmarks[benchmark_type]
        return run_selected_benchmark(category, name, size)
    
    return "❌ Quick benchmark not found", "Error"

def load_sample_code(sample_name):
    """Load sample code from the predefined samples."""
    if sample_name and sample_name in sample_codes:
        return sample_codes[sample_name]
    return ""

# Create the enhanced Gradio interface
with gr.Blocks(title="GPU Mentor - Enhanced AI Tutor", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🚀 Enhanced GPU Mentor: AI Tutor with Integrated Code Execution")
    
    with gr.Tabs():
        
        # Features Tab (moved from main interface)
        with gr.Tab("ℹ️ Features"):
            gr.Markdown("""
            ## 🚀 Enhanced GPU Mentor Features
            
            **🔗 Integrated LLM + Code**: Ask questions about your code - the AI sees both your question and code together
            
            **⚡ Live Code Execution**: Run your Python code instantly and see the output in the chat
            
            **🔍 Smart Analysis**: Get optimization suggestions and GPU acceleration opportunities
            
            **📚 Educational Guidance**: Socratic questions and learning objectives based on your actual code
            
            **🎯 Multi-Modal Support**: Handles text questions, code analysis, and execution all in one interface
            
            **🚀 GPU Optimization**: Automatic detection of optimization opportunities and GPU-compatible code suggestions
            
            **📊 Performance Insights**: Real-time analysis of code performance and potential speedup estimates
            """)
        
        # Main Chat Playground Tab (Redesigned)
        with gr.Tab("💬 Chat Playground"):
            with gr.Column():
                # Main conversation area
                chatbot = gr.Chatbot(
                    label="GPU Mentor Conversation",
                    height=500,
                    type="messages",
                    show_copy_button=True
                )
                
                # Integrated input area at bottom of conversation
                with gr.Row():
                    with gr.Column(scale=3):
                        message_input = gr.Textbox(
                            label="",
                            placeholder="Ask about GPU acceleration, optimization, or explain your code...",
                            lines=2,
                            show_label=False
                        )
                    with gr.Column(scale=1):
                        submit_btn = gr.Button("💬 Send", variant="primary", size="lg")
                        clear_btn = gr.Button("🧹 Clear", size="sm")
                
                # Code input area (collapsible)
                with gr.Accordion("📝 Python Code (Optional)", open=False):
                    code_input = gr.Code(
                        label="",
                        language="python",
                        lines=8,
                        show_label=False
                    )
                    
                    # Sample code selector
                    with gr.Row():
                        sample_dropdown = gr.Dropdown(
                            choices=list(sample_codes.keys()),
                            label="Load Sample Code",
                            value=None,
                            scale=2
                        )
                        load_sample_btn = gr.Button("📂 Load", scale=1)
        
        # Code Analysis Tab (Updated)
        with gr.Tab("🔍 Code Analysis & Optimization"):
            with gr.Row():
                with gr.Column():
                    analyze_code = gr.Code(
                        label="Code to Analyze",
                        language="python",
                        lines=15
                    )
                    
                    analyze_btn = gr.Button("🔍 Analyze Code", variant="primary")
                    
                    analysis_results = gr.Textbox(
                        label="Analysis Results",
                        lines=15
                    )
                
                with gr.Column():
                    optimized_code = gr.Code(
                        label="GPU-Optimized Version",
                        language="python",
                        lines=20
                    )
        
        # Performance Benchmarking Tab (NEW)
        with gr.Tab("🏁 Performance Benchmarking"):
            gr.Markdown("## 🚀 CPU vs GPU Performance Comparison")
            gr.Markdown("""
            Compare the performance of CPU and GPU implementations across different workloads.
            This interactive benchmarking tool demonstrates real-world GPU acceleration benefits.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Benchmark selection controls
                    benchmark_category = gr.Dropdown(
                        choices=benchmark_engine.get_benchmark_categories() if benchmark_engine else [],
                        label="📂 Benchmark Category",
                        value="Matrix Operations" if benchmark_engine else None
                    )
                    
                    benchmark_name = gr.Dropdown(
                        choices=[],
                        label="🎯 Specific Benchmark",
                        value=None
                    )
                    
                    benchmark_size = gr.Dropdown(
                        choices=[],
                        label="📏 Problem Size",
                        value=None
                    )
                    
                    run_benchmark_btn = gr.Button("🏃‍♂️ Run Benchmark", variant="primary", size="lg")
                    
                    # Benchmark status
                    benchmark_status = gr.Textbox(
                        label="Status",
                        value="Select benchmark parameters and click 'Run Benchmark'",
                        interactive=False,
                        lines=2
                    )
                
                with gr.Column(scale=2):
                    # Results display
                    benchmark_results = gr.Markdown(
                        label="📊 Benchmark Results",
                        value="""
### 🎯 Ready to Benchmark!

Select a category, benchmark, and problem size from the left panel, then click **Run Benchmark** to see CPU vs GPU performance comparison.

**Available Categories:**
- **Matrix Operations**: Linear algebra operations (NumPy vs CuPy)
- **DataFrame Operations**: Data processing tasks (Pandas vs cuDF)  
- **Machine Learning**: ML algorithms (scikit-learn vs cuML)
- **Mathematical Functions**: Mathematical computations (NumPy vs CuPy)

**What You'll Learn:**
- Real-world GPU acceleration benefits
- Performance scaling with problem size
- When GPU acceleration is most effective
- Memory and computational trade-offs

**💡 Pro Tips:**
- Start with Matrix Operations for dramatic speedups
- DataFrame Operations work best with large datasets (>100K rows)
- ML Algorithms show consistent benefits across problem sizes
- Mathematical Functions benefit from kernel fusion techniques
"""
                    )
            
            # Technology comparison section
            with gr.Row():
                gr.Markdown("""
### 🏆 Technology Comparison Guide

| **Operation Type** | **CPU Library** | **GPU Library** | **Typical Speedup** | **Best Use Case** |
|-------------------|-----------------|-----------------|---------------------|-------------------|
| **Matrix Operations** | NumPy | CuPy | 10-50x | Linear algebra, large arrays |
| **DataFrame Operations** | Pandas | cuDF | 5-20x | Data processing, analytics |
| **ML Algorithms** | scikit-learn | cuML | 5-25x | Large datasets, feature engineering |
| **Math Functions** | NumPy | CuPy | 3-15x | Signal processing, numerical computing |

**🎯 Selection Guidelines:**
- **Problem Size**: GPU benefits increase with larger datasets
- **Memory**: Consider GPU memory limitations for very large data  
- **Pipeline**: Keep operations on GPU to avoid transfer overhead
- **Data Type**: Use float32 when possible for better GPU performance
""")
            
            # Recent benchmarks section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📈 Recent Benchmark History")
                    recent_benchmarks_btn = gr.Button("🔍 Show Recent Results")
                    recent_benchmarks = gr.JSON(label="Recent Benchmarks", visible=False)
                    
                    # Quick benchmark buttons for common operations
                    gr.Markdown("### ⚡ Quick Benchmarks")
                    with gr.Row():
                        quick_matrix_btn = gr.Button("Matrix Ops", size="sm")
                        quick_dataframe_btn = gr.Button("DataFrame Ops", size="sm") 
                        quick_ml_btn = gr.Button("ML Algorithms", size="sm")
                        quick_math_btn = gr.Button("Math Functions", size="sm")
        
        # Tutorial Generator Tab  
        with gr.Tab("📚 Personalized Tutorials"):
            with gr.Column():
                tutorial_topic = gr.Textbox(
                    label="Tutorial Topic",
                    placeholder="e.g., 'CuPy memory management', 'cuDF vs pandas performance', 'vectorizing loops'...",
                    lines=1
                )
                
                generate_tutorial_btn = gr.Button("📝 Generate Tutorial", variant="primary")
                
                tutorial_content = gr.Markdown(
                    label="Tutorial Content",
                    value="Enter a topic above to generate a personalized tutorial."
                )
        
        # Execution Summary Tab (Replaces Performance)
        with gr.Tab("📈 Code Execution Summary"):
            with gr.Column():
                gr.Markdown("### Your Code Execution History")
                
                summary_btn = gr.Button("📊 View Execution Summary")
                execution_summary = gr.JSON(label="Execution Summary")
    
    # Wire up the interface
    sample_dropdown.change(load_sample_code, inputs=[sample_dropdown], outputs=[code_input])
    load_sample_btn.click(load_sample_code, inputs=[sample_dropdown], outputs=[code_input])
    
    submit_btn.click(
        chat_with_mentor,
        inputs=[message_input, code_input, chatbot],
        outputs=[message_input, code_input, chatbot, gr.State(), gr.State()]
    )
    
    clear_btn.click(clear_chat, outputs=[chatbot, gr.State(), gr.State()])
    
    analyze_btn.click(
        analyze_code_only,
        inputs=[analyze_code],
        outputs=[analysis_results, optimized_code]
    )
    
    generate_tutorial_btn.click(
        get_tutorial,
        inputs=[tutorial_topic],
        outputs=[tutorial_content]
    )
    
    summary_btn.click(
        lambda: gpu_mentor.get_execution_summary() if gpu_mentor else {"error": "GPU Mentor not available"},
        outputs=[execution_summary]
    )
    
    # Benchmarking event handlers
    if benchmark_engine:
        benchmark_category.change(
            update_benchmark_options,
            inputs=[benchmark_category],
            outputs=[benchmark_name, benchmark_size]
        )
        
        benchmark_name.change(
            update_size_options,
            inputs=[benchmark_category, benchmark_name],
            outputs=[benchmark_size]
        )
        
        run_benchmark_btn.click(
            run_selected_benchmark,
            inputs=[benchmark_category, benchmark_name, benchmark_size],
            outputs=[benchmark_results, benchmark_status]
        )
        
        recent_benchmarks_btn.click(
            show_recent_benchmarks,
            outputs=[recent_benchmarks, recent_benchmarks]
        )
        
        # Quick benchmark buttons
        quick_matrix_btn.click(
            lambda: run_quick_benchmark("Matrix Ops"),
            outputs=[benchmark_results, benchmark_status]
        )
        
        quick_dataframe_btn.click(
            lambda: run_quick_benchmark("DataFrame Ops"),
            outputs=[benchmark_results, benchmark_status]
        )
        
        quick_ml_btn.click(
            lambda: run_quick_benchmark("ML Algorithms"),
            outputs=[benchmark_results, benchmark_status]
        )
        
        quick_math_btn.click(
            lambda: run_quick_benchmark("Math Functions"),
            outputs=[benchmark_results, benchmark_status]
        )

if __name__ == "__main__":
    # Launch the enhanced interface
    # demo.launch(
    #     server_name="0.0.0.0",  # Allow external access on Sol
    #     server_port=7860,
    #     share=False,  # Don't create public link for security
    #     debug=True,
    #     show_error=True
    # )
    demo.launch(share=True)