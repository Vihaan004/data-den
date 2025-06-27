import gradio as gr
from document_loader import DocumentLoader, VectorStore
from rag_agent import RAGAgent
from code_optimizer import CodeOptimizer
from gpu_mentor import GPUMentor
from langchain.tools.retriever import create_retriever_tool

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
        with gr.Blocks(title="GPU Mentor - AI-Powered GPU Acceleration Assistant", theme=gr.themes.Soft()) as interface:
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
                        gr.Markdown("### Sample Code")
                        sample_dropdown = gr.Dropdown(
                            choices=list(sample_codes.keys()),
                            label="Choose Sample",
                            interactive=True
                        )
                        load_sample_btn = gr.Button("Load Sample", variant="secondary")
            
            with gr.Tab("🔍 Code Analysis & Optimization"):
                with gr.Row():
                    with gr.Column():
                        analyze_code = gr.Textbox(
                            placeholder="Paste your Python code here for analysis...",
                            label="Code to Analyze",
                            lines=15,
                            max_lines=25
                        )
                        
                        analyze_btn = gr.Button("Analyze Code", variant="primary")
                    
                    with gr.Column():
                        analysis_results = gr.Markdown(label="Analysis Results")
                        optimized_code = gr.Textbox(
                            label="Optimized GPU Code",
                            lines=15,
                            max_lines=25
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
            
            # Wire up the interface
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
            
            analyze_btn.click(
                self.gpu_mentor.analyze_code_only,
                inputs=[analyze_code],
                outputs=[analysis_results, optimized_code]
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
    
    def launch(self, share=True):
        """Launch the application."""
        interface = self.create_interface()
        interface.launch(share=share)

if __name__ == "__main__":
    app = GPUMentorApp()
    app.launch(share=True)
