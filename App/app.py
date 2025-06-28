import gradio as gr
from document_loader import DocumentLoader, VectorStore
from rag_agent import RAGAgent
from code_optimizer import CodeOptimizer
from gpu_mentor import GPUMentor
from langchain.tools.retriever import create_retriever_tool
from benchmark import run_benchmark

class GPUMentorApp:
    """Main application class for the GPU Mentor."""
    
    def __init__(self):
        self.gpu_mentor = None
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize all components of the GPU Mentor system."""
        print("🚀 Initializing GPU Mentor System...")
        
        try:
            # Load and process documents
            print("📚 Loading documents...")
            doc_loader = DocumentLoader()
            docs = doc_loader.load_documents()
            doc_splits = doc_loader.split_documents(docs)
            
            # Create vector store
            print("🔍 Creating vector store...")
            vector_store = VectorStore()
            retriever = vector_store.create_vectorstore(doc_splits)
            
            # Create retriever tool
            retriever_tool = create_retriever_tool(
                retriever,
                "retrieve_python_gpu_acceleration",
                "Search and return information about accelerating Python code using GPU with RAPIDS and CuPy."
            )
            
            # Initialize RAG agent
            print("🤖 Initializing RAG agent...")
            rag_agent = RAGAgent()
            rag_agent.set_retriever_tool(retriever_tool)
            
            # Initialize code optimizer
            print("⚡ Initializing code optimizer...")
            code_optimizer = CodeOptimizer(rag_agent)
            
            # Initialize GPU mentor
            print("🎓 Initializing GPU mentor...")
            self.gpu_mentor = GPUMentor(rag_agent, code_optimizer)
            
            print("✅ GPU Mentor System initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            self.gpu_mentor = None
    
    def create_interface(self):
        """Create the Gradio interface."""
        if not self.gpu_mentor:
            return gr.Interface(
                fn=lambda: "System not initialized properly. Please check the logs.",
                inputs=[],
                outputs=gr.Textbox(label="Error"),
                title="GPU Mentor - Initialization Error"
            )
        
        # Sample code examples
        sample_codes = {
            "Matrix Multiplication": """import numpy as np

# Create large matrices
size = 1000
A = np.random.rand(size, size).astype(np.float32)
B = np.random.rand(size, size).astype(np.float32)

# Matrix multiplication
C = np.matmul(A, B)

print(f"Result shape: {C.shape}")
print(f"Sum of result: {np.sum(C)}")""",
            
            "DataFrame Operations": """import pandas as pd
import numpy as np

# Create sample dataframe
n_rows = 100000
df = pd.DataFrame({
    'A': np.random.randn(n_rows),
    'B': np.random.randn(n_rows),
    'C': np.random.choice(['X', 'Y', 'Z'], n_rows),
    'D': np.random.randint(1, 100, n_rows)
})

# Perform operations
result = df.groupby('C').agg({
    'A': 'mean',
    'B': 'std',
    'D': 'sum'
})

print(result)""",
            
            "Machine Learning": """from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

# Generate sample data
X, _ = make_blobs(n_samples=10000, centers=10, 
                  n_features=20, random_state=42)

# Perform clustering
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(X)

print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
print(f"Labels shape: {labels.shape}")"""
        }
        
        # Create interface
        # Custom CSS for better layout
        custom_css = """
        .analysis-results {
            min-height: 300px;
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
        }
        .execution-results {
            min-height: 200px;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #d0d0d0;
            border-radius: 6px;
            padding: 12px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 12px;
        }
        .code-analysis-tab {
            height: 100vh;
        }
        .code-input-row {
            margin-bottom: 16px;
        }
        .execution-row {
            margin-bottom: 16px;
        }
        .button-bar {
            margin: 16px 0;
            gap: 8px;
        }
        .gradio-container {
            max-width: 100% !important;
        }
        """
        
        with gr.Blocks(
            title="GPU Mentor - AI-Powered GPU Acceleration Assistant", 
            theme=gr.themes.Soft(),
            css=custom_css
        ) as interface:
            gr.Markdown("""
            # 🚀 GPU Mentor - AI-Powered GPU Acceleration Assistant
            
            Learn how to accelerate your Python code using NVIDIA Rapids libraries (CuPy, cuDF, cuML).
            Get AI-powered code optimization suggestions and educational guidance.
            """)
            
            with gr.Tab("💬 Chat with GPU Mentor"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            height=400, 
                            label="Conversation",
                            type="messages"
                        )
                        
                        with gr.Row():
                            message_input = gr.Textbox(
                                placeholder="Ask about GPU acceleration...",
                                label="Your Question",
                                scale=4
                            )
                            submit_btn = gr.Button("Send", variant="primary", scale=1)
                        
                        code_input = gr.Textbox(
                            placeholder="Paste your Python code here for analysis...",
                            label="Code to Analyze (Optional)",
                            lines=10,
                            max_lines=20
                        )
                        
                        clear_btn = gr.Button("Clear Chat", variant="secondary")
                    
                    with gr.Column(scale=1):
                        sample_dropdown = gr.Dropdown(
                            choices=list(sample_codes.keys()),
                            label="Choose Sample",
                            interactive=True
                        )
                        load_sample_btn = gr.Button("Load Sample", variant="secondary")
            
            with gr.Tab("🔍 Code Analysis & Optimization"):
                # Top row: Code input and GPU optimized code side by side
                with gr.Row(elem_classes=["code-input-row"]):
                    with gr.Column(scale=1):
                        analyze_code = gr.Textbox(
                            placeholder="Paste your Python code here for analysis...",
                            label="💻 Code to Analyze",
                            lines=15,
                            max_lines=25,
                            elem_classes=["code-input"]
                        )
                    
                    with gr.Column(scale=1):
                        optimized_code = gr.Textbox(
                            label="🚀 GPU-Optimized Code",
                            lines=15,
                            max_lines=25,
                            placeholder="Optimized GPU code will appear here after analysis...",
                            interactive=False,
                            elem_classes=["code-output"]
                        )
                
                # Button bar: Sample dropdown, load, analyze, and clear buttons
                with gr.Row(elem_classes=["button-bar"]):
                    analysis_sample_dropdown = gr.Dropdown(
                        choices=list(sample_codes.keys()),
                        label=None,
                        interactive=True,
                        scale=2,
                        elem_classes=["sample-dropdown"]
                    )
                    load_analysis_sample_btn = gr.Button("📥 Load Sample", variant="secondary", scale=1)
                    analyze_btn = gr.Button("🔍 Analyze Code", variant="primary", scale=2)
                    run_comparison_btn = gr.Button("🏃‍♂️ Run on Sol", variant="primary", scale=2)
                    clear_code_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                
                # --- BENCHMARK BUTTON INTEGRATION ---
                with gr.Row(elem_classes=["button-bar"]):
                    benchmark_btn = gr.Button("⚡ Benchmark", variant="primary", scale=2)

                # Output boxes for benchmark results
                with gr.Row(elem_classes=["execution-row"]):
                    cpu_bench_out = gr.Textbox(label="CPU Benchmark Output", lines=8)
                    gpu_bench_out = gr.Textbox(label="GPU Benchmark Output", lines=8)

                def benchmark_handler(cpu_code, gpu_code):
                    if not cpu_code.strip() or not gpu_code.strip():
                        return "Please provide both CPU and GPU code.", ""
                    results = run_benchmark(cpu_code, gpu_code, "/home/vpatel69/R1/App/output")
                    return results["cpu"]["stdout"], results["gpu"]["stdout"]

                # Enable Benchmark only if both code boxes are filled
                def enable_benchmark(cpu, gpu):
                    return bool(cpu.strip()) and bool(gpu.strip())
                analyze_code.change(enable_benchmark, [analyze_code, optimized_code], benchmark_btn)
                optimized_code.change(enable_benchmark, [analyze_code, optimized_code], benchmark_btn)

                benchmark_btn.click(
                    benchmark_handler,
                    inputs=[analyze_code, optimized_code],
                    outputs=[cpu_bench_out, gpu_bench_out]
                )
                
                # Execution results row: Show execution outputs below code boxes
                with gr.Row(elem_classes=["execution-row"]):
                    with gr.Column(scale=1):
                        original_execution_output = gr.Markdown(
                            label="🖥️ Original Code Execution",
                            value="**Original Code Execution Results**\n\nClick 'Run on Sol' to execute the original code on the Sol supercomputer and see timing results.",
                            elem_classes=["execution-results"]
                        )
                    
                    with gr.Column(scale=1):
                        optimized_execution_output = gr.Markdown(
                            label="🚀 GPU Code Execution", 
                            value="**GPU-Optimized Code Execution Results**\n\nClick 'Run on Sol' to execute the GPU-optimized code and compare performance.",
                            elem_classes=["execution-results"]
                        )
                
                # Bottom area: AI insights taking the rest of the space
                with gr.Row():
                    analysis_results = gr.Markdown(
                        label="🧠 AI Analysis & Recommendations",
                        value="**Welcome to GPU Code Analysis!**\n\nSelect code from the samples above or paste your own Python code, then click 'Analyze Code' to see:\n- AI-powered analysis of your code\n- GPU optimization opportunities\n- Performance improvement suggestions\n- Best practices recommendations",
                        elem_classes=["analysis-results"]
                    )
            
            with gr.Tab("📚 Learning Resources"):
                with gr.Row():
                    with gr.Column():
                        tutorial_topic = gr.Textbox(
                            placeholder="e.g., CuPy array operations, cuDF dataframes, cuML machine learning",
                            label="Tutorial Topic"
                        )
                        generate_tutorial_btn = gr.Button("Generate Tutorial", variant="primary")
                    
                    with gr.Column():
                        tutorial_content = gr.Markdown(label="Tutorial Content")
            
            with gr.Tab("📊 Execution Summary"):
                with gr.Row():
                    summary_btn = gr.Button("Get Execution Summary", variant="primary")
                    execution_summary = gr.Markdown(label="Execution Summary")
            
            # Event handlers
            def load_sample_code(sample_name):
                if sample_name and sample_name in sample_codes:
                    return sample_codes[sample_name]
                return ""
            
            def clear_chat():
                return []
            
            def clear_code():
                return ""
            
            def clear_execution_results():
                original_msg = "**Original Code Execution Results**\n\nClick 'Run on Sol' to execute the original code on the Sol supercomputer and see timing results."
                optimized_msg = "**GPU-Optimized Code Execution Results**\n\nClick 'Run on Sol' to execute the GPU-optimized code and compare performance."
                return original_msg, optimized_msg
            
            # Wire up the chat interface
            sample_dropdown.change(load_sample_code, inputs=[sample_dropdown], outputs=[code_input])
            load_sample_btn.click(load_sample_code, inputs=[sample_dropdown], outputs=[code_input])
            
            submit_btn.click(
                self.gpu_mentor.chat_interface,
                inputs=[message_input, code_input, chatbot],
                outputs=[message_input, code_input, chatbot]
            )
            
            message_input.submit(
                self.gpu_mentor.chat_interface,
                inputs=[message_input, code_input, chatbot],
                outputs=[message_input, code_input, chatbot]
            )
            
            clear_btn.click(clear_chat, outputs=[chatbot])
            
            # Wire up the code analysis interface
            analysis_sample_dropdown.change(load_sample_code, inputs=[analysis_sample_dropdown], outputs=[analyze_code])
            load_analysis_sample_btn.click(load_sample_code, inputs=[analysis_sample_dropdown], outputs=[analyze_code])
            clear_code_btn.click(clear_code, outputs=[analyze_code])
            clear_code_btn.click(clear_execution_results, outputs=[original_execution_output, optimized_execution_output])
            
            analyze_btn.click(
                self.gpu_mentor.analyze_code_only,
                inputs=[analyze_code],
                outputs=[analysis_results, optimized_code]
            )
            
            run_comparison_btn.click(
                self.gpu_mentor.run_code_comparison,
                inputs=[analyze_code, optimized_code],
                outputs=[original_execution_output, optimized_execution_output]
            )
            
            generate_tutorial_btn.click(
                self.gpu_mentor.get_tutorial_content,
                inputs=[tutorial_topic],
                outputs=[tutorial_content]
            )
            
            summary_btn.click(
                self.gpu_mentor.get_execution_summary,
                outputs=[execution_summary]
            )
        
        return interface
    
    def launch(self, share=True, **kwargs):
        """Launch the application with proper shutdown handling."""
        interface = self.create_interface()
        
        # Set default parameters for better control - let Gradio find an available port
        launch_params = {
            'share': share,
            'server_name': '0.0.0.0',
            'quiet': False,
            'show_error': True,
            'inbrowser': False,
            'prevent_thread_lock': False
        }
        
        # Override with any user-provided parameters
        launch_params.update(kwargs)
        
        try:
            interface.launch(**launch_params)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Gradio interface...")
            interface.close()
            raise
        except Exception as e:
            print(f"❌ Error in Gradio interface: {e}")
            interface.close()
            raise
    
    def close(self):
        """Close the application gracefully."""
        print("🔄 Closing GPU Mentor application...")
        # Add any cleanup code here if needed

if __name__ == "__main__":
    app = GPUMentorApp()
    app.launch(share=True)
