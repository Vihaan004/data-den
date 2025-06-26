#!/usr/bin/env python3
"""
Simple Gradio UI for GPU Mentor Backend - Minimal Interface
Avoids complex streaming features that can cause ASGI issues.
"""
import sys
sys.path.append('.')

import gradio as gr
import asyncio
import traceback
from datetime import datetime

try:
    from core.enhanced_gpu_mentor import EnhancedGPUMentor
    from core.benchmark_engine import BenchmarkEngine
    print("✅ Core imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Global variables
gpu_mentor = None
benchmark_engine = None

def initialize_backend():
    """Initialize the backend components."""
    global gpu_mentor, benchmark_engine
    try:
        print("🚀 Initializing GPU Mentor...")
        gpu_mentor = EnhancedGPUMentor()
        
        # Initialize synchronously for simplicity
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(gpu_mentor.initialize())
        loop.close()
        
        benchmark_engine = BenchmarkEngine()
        return "✅ Backend initialized successfully!"
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return f"❌ Initialization failed: {str(e)}"

def chat_simple(message, history):
    """Simple chat function without complex async handling."""
    if not gpu_mentor:
        return history + [["System", "❌ Backend not initialized. Click 'Initialize Backend' first."]]
    
    if not message.strip():
        return history
    
    try:
        # Handle the response synchronously
        if hasattr(gpu_mentor, 'rag_graph') and gpu_mentor.rag_graph:
            # Use RAG system
            result = gpu_mentor.rag_graph.invoke({
                "messages": [{"role": "user", "content": message}]
            })
            response = result["messages"][-1].content
        else:
            # Fallback response
            response = f"""I can help with GPU acceleration! Here's some guidance:

**Your question:** {message}

**GPU Acceleration Tips:**
• Use CuPy instead of NumPy for array operations
• Use cuDF instead of Pandas for large datasets  
• Use cuML instead of scikit-learn for ML algorithms
• Consider data size - GPU works best with larger datasets
• Keep data on GPU between operations to avoid transfers

**Note:** Full AI analysis is available once the RAG system is properly initialized."""
        
        return history + [[message, response]]
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Chat error: {e}")
        return history + [[message, error_msg]]

def analyze_code_simple(code):
    """Simple code analysis function."""
    if not code.strip():
        return "Please provide code to analyze."
    
    if not gpu_mentor:
        return "❌ Backend not initialized."
    
    try:
        # Basic analysis without complex async
        analysis = f"""**Code Analysis:**

**Input Code:**
```python
{code}
```

**GPU Optimization Suggestions:**
1. **Array Operations**: Replace `numpy` with `cupy` for GPU acceleration
2. **DataFrame Operations**: Replace `pandas` with `cudf` for large datasets
3. **Machine Learning**: Replace `sklearn` with `cuml` for GPU-accelerated ML
4. **Memory Management**: Use memory pools and keep data on GPU
5. **Performance**: Consider data size - GPU benefits increase with larger datasets

**Optimized Example:**
```python
# Replace numpy with cupy
import cupy as cp
# Replace pandas with cudf  
import cudf as df
# Your optimized code here...
```

**Note:** This is a simplified analysis. Full analysis available with complete backend initialization."""
        
        return analysis
        
    except Exception as e:
        return f"Analysis error: {str(e)}"

def run_benchmark_simple(category):
    """Simple benchmark runner."""
    if not benchmark_engine:
        return "❌ Benchmark engine not initialized."
    
    try:
        # Run a simple benchmark
        if category == "Matrix Operations":
            size = 1000
            benchmark_name = "Matrix Multiplication"
        elif category == "DataFrame Operations":
            size = 100000
            benchmark_name = "GroupBy Aggregation"
        else:
            size = 500
            benchmark_name = "Basic Operations"
        
        results = benchmark_engine.run_benchmark(category, benchmark_name, size)
        
        if results:
            return f"""**Benchmark Results:**

**Category:** {category}
**Operation:** {benchmark_name}
**Size:** {size:,}

**CPU Time:** {results.get('cpu_time', 'N/A')}
**GPU Time:** {results.get('gpu_time', 'N/A')}
**Speedup:** {results.get('speedup', 'N/A')}x
**Winner:** {results.get('winner', 'N/A')}

**Analysis:** {results.get('analysis', 'Basic benchmark completed')}
"""
        else:
            return "❌ Benchmark failed to execute."
            
    except Exception as e:
        return f"Benchmark error: {str(e)}"

# Create simple interface
with gr.Blocks(title="GPU Mentor - Simple Interface", theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🚀 GPU Mentor - Simple Interface")
    gr.Markdown("Simplified interface to avoid ASGI streaming issues.")
    
    with gr.Row():
        init_status = gr.Textbox(
            label="Backend Status", 
            value="❌ Not initialized - click Initialize Backend",
            interactive=False
        )
        init_btn = gr.Button("🔄 Initialize Backend")
    
    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(height=400, label="GPU Mentor Chat")
            msg = gr.Textbox(label="Your Message", placeholder="Ask about GPU acceleration...")
            clear = gr.Button("Clear Chat")
            
        with gr.Tab("🔍 Code Analysis"):
            code_input = gr.Code(language="python", label="Python Code")
            analyze_btn = gr.Button("Analyze Code")
            analysis_output = gr.Markdown(label="Analysis Results")
            
        with gr.Tab("🏁 Benchmarks"):
            benchmark_category = gr.Dropdown(
                choices=["Matrix Operations", "DataFrame Operations", "ML Algorithms"],
                label="Benchmark Category",
                value="Matrix Operations"
            )
            benchmark_btn = gr.Button("Run Benchmark")
            benchmark_output = gr.Markdown(label="Benchmark Results")
    
    # Event handlers
    init_btn.click(initialize_backend, outputs=[init_status])
    
    msg.submit(chat_simple, [msg, chatbot], [chatbot]).then(
        lambda: "", None, [msg]
    )
    clear.click(lambda: [], None, [chatbot])
    
    analyze_btn.click(analyze_code_simple, [code_input], [analysis_output])
    benchmark_btn.click(run_benchmark_simple, [benchmark_category], [benchmark_output])

if __name__ == "__main__":
    print("🚀 Starting Simple GPU Mentor UI...")
    print("🌐 This version avoids ASGI streaming issues")
    
    # Simple launch without complex features
    # demo.launch(
    #     server_name="0.0.0.0",
    #     server_port=7861,  # Different port to avoid conflicts
    #     share=False,
    #     debug=False,
    #     quiet=True,
    #     show_tips=False,
    #     enable_queue=False
    # )
    demo.launch(share=True)