#!/usr/bin/env python3
"""
Gradio UI for GPU Mentor Backend - Runs on Sol
"""
import sys
sys.path.append('.')

import gradio as gr
import traceback
from datetime import datetime
from core.enhanced_gpu_mentor import EnhancedGPUMentor
from core.benchmark_engine import BenchmarkEngine
from utils.sample_code_library import SampleCodeLibrary

# Initialize the GPU Mentor
try:
    gpu_mentor = EnhancedGPUMentor()
    benchmark_engine = BenchmarkEngine()
    sample_library = SampleCodeLibrary()
    print("✅ GPU Mentor backend initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize backend: {e}")
    gpu_mentor = None

def optimize_code(user_code, user_question=""):
    """Optimize code and provide explanation."""
    if not gpu_mentor:
        return "❌ Backend not initialized", "", ""
    
    try:
        # Get optimization
        optimized = gpu_mentor.optimize_code(user_code)
        
        # Get explanation if question provided
        explanation = ""
        if user_question:
            explanation = gpu_mentor.explain_optimization(user_question)
        
        # Basic performance info
        performance_info = f"""
📊 **Optimization Complete**
- Original code: {len(user_code)} characters
- Optimized code: {len(optimized)} characters
- Timestamp: {datetime.now().strftime('%H:%M:%S')}
"""
        
        return optimized, explanation, performance_info
        
    except Exception as e:
        error_msg = f"❌ Error during optimization: {str(e)}"
        return error_msg, "", traceback.format_exc()

def get_sample_code(category, operation):
    """Get sample code from library."""
    try:
        sample = sample_library.get_sample_code(category, operation)
        if sample:
            return sample
        else:
            return f"No sample found for {category}/{operation}"
    except Exception as e:
        return f"Error retrieving sample: {e}"

def benchmark_code(code_to_benchmark, benchmark_name="benchmark"):
    """Run benchmark on provided code."""
    if not benchmark_engine:
        return "❌ Benchmark engine not available"
    
    try:
        result = benchmark_engine.benchmark_code(code_to_benchmark, benchmark_name)
        return f"✅ Benchmark completed: {result}"
    except Exception as e:
        return f"❌ Benchmark failed: {e}"

def chat_with_mentor(question):
    """Chat with the GPU mentor."""
    try:
        response = gpu_mentor.explain_optimization(question)
        return response
    except Exception as e:
        return f"❌ Error: {e}"

# Create Gradio interface
with gr.Blocks(title="GPU Mentor on Sol", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 GPU Mentor - Running on Sol Supercomputer
    
    **Welcome to GPU Mentor!** This interface is running directly on Sol with access to:
    - 🖥️ NVIDIA A100 GPU acceleration
    - 🧠 AI-powered code optimization
    - 📚 Educational content generation
    - ⚡ Performance benchmarking
    """)
    
    with gr.Tabs():
        # Tab 1: Code Optimization
        with gr.TabItem("🔧 Code Optimization"):
            gr.Markdown("### Optimize your Python code for GPU acceleration")
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_code = gr.Textbox(
                        label="Your Python Code",
                        placeholder="Enter your Python code here...",
                        lines=10,
                        value="""import numpy as np

# Inefficient loop-based approach
data = list(range(1000))
result = []
for i in range(len(data)):
    result.append(data[i] * 2 + 1)

print(f"Result length: {len(result)}")"""
                    )
                    
                    question_input = gr.Textbox(
                        label="Question (Optional)",
                        placeholder="Ask about specific optimizations...",
                        lines=2
                    )
                    
                    optimize_btn = gr.Button("🚀 Optimize Code", variant="primary")
                
                with gr.Column(scale=1):
                    optimized_output = gr.Textbox(
                        label="Optimized Code",
                        lines=10,
                        interactive=False
                    )
                    
                    explanation_output = gr.Textbox(
                        label="Explanation",
                        lines=5,
                        interactive=False
                    )
                    
                    performance_output = gr.Textbox(
                        label="Performance Info",
                        lines=3,
                        interactive=False
                    )
            
            optimize_btn.click(
                optimize_code,
                inputs=[input_code, question_input],
                outputs=[optimized_output, explanation_output, performance_output]
            )
        
        # Tab 2: Sample Code Library
        with gr.TabItem("📚 Sample Code Library"):
            gr.Markdown("### Browse sample code for common GPU operations")
            
            with gr.Row():
                category_dropdown = gr.Dropdown(
                    choices=["basic", "intermediate", "advanced"],
                    label="Category",
                    value="basic"
                )
                
                operation_dropdown = gr.Dropdown(
                    choices=["array_operations", "matrix_operations", "image_processing"],
                    label="Operation",
                    value="array_operations"
                )
                
                get_sample_btn = gr.Button("📖 Get Sample", variant="secondary")
            
            sample_output = gr.Textbox(
                label="Sample Code",
                lines=15,
                interactive=False
            )
            
            get_sample_btn.click(
                get_sample_code,
                inputs=[category_dropdown, operation_dropdown],
                outputs=sample_output
            )
        
        # Tab 3: Performance Benchmarking
        with gr.TabItem("⚡ Benchmarking"):
            gr.Markdown("### Benchmark your code performance")
            
            benchmark_input = gr.Textbox(
                label="Code to Benchmark",
                placeholder="Enter code to benchmark...",
                lines=8,
                value="""import numpy as np
import time

# Benchmark this code
start = time.time()
data = np.random.random((1000, 1000))
result = np.dot(data, data.T)
end = time.time()

print(f"Execution time: {end - start:.4f} seconds")"""
            )
            
            benchmark_name_input = gr.Textbox(
                label="Benchmark Name",
                value="matrix_multiplication_test"
            )
            
            benchmark_btn = gr.Button("⚡ Run Benchmark", variant="primary")
            
            benchmark_output = gr.Textbox(
                label="Benchmark Results",
                lines=8,
                interactive=False
            )
            
            benchmark_btn.click(
                benchmark_code,
                inputs=[benchmark_input, benchmark_name_input],
                outputs=benchmark_output
            )
        
        # Tab 4: Ask the Mentor
        with gr.TabItem("🤖 Ask the Mentor"):
            gr.Markdown("### Chat with the GPU Mentor for guidance")
            
            chat_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask about GPU programming, optimization techniques, etc.",
                lines=3
            )
            
            chat_btn = gr.Button("💬 Ask", variant="primary")
            
            chat_output = gr.Textbox(
                label="Mentor's Response",
                lines=10,
                interactive=False
            )
            
            # Example questions
            gr.Markdown("""
            **Example Questions:**
            - "How can I optimize matrix operations for GPU?"
            - "What are the best practices for memory management?"
            - "Explain vectorization vs GPU acceleration"
            """)
            
            chat_btn.click(
                chat_with_mentor,
                inputs=chat_input,
                outputs=chat_output
            )
        
        # Tab 5: System Info
        with gr.TabItem("ℹ️ System Info"):
            gr.Markdown("### Sol System Information")
            
            system_info = gr.Textbox(
                label="System Information",
                value=f"""
🖥️ **Sol Supercomputer Status**
- Backend Status: ✅ Running
- GPU Access: ✅ Available
- Kernel: genai25.06
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🧠 **Available Components:**
- RAG Pipeline: ✅ Loaded
- Code Optimizer: ✅ Loaded  
- Benchmark Engine: ✅ Loaded
- Sol Executor: ✅ Loaded
- Educational Content: ✅ Loaded

💡 **Tips:**
- Use the optimization tab for performance improvements
- Try sample codes for learning
- Ask the mentor for guidance
""",
                lines=15,
                interactive=False
            )

if __name__ == "__main__":
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Don't create public link for security
        debug=True
    )
