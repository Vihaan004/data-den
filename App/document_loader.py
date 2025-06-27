import os
from typing import List, Dict, Any
from langchain_community.document_loaders import WebBaseLoader, NotebookLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from config import KNOWLEDGE_URLS, NOTEBOOK_PATHS, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

class DocumentLoader:
    """Handle loading and processing of documents for the RAG system."""
    
    def __init__(self):
        self.docs = []
        self.doc_splits = []
        
    def load_documents(self) -> List[Document]:
        """Load documents from various sources."""
        # External URLs for GPU acceleration knowledge
        urls = [
            "https://medium.com/cupy-team/announcing-cupy-v13-66979ee7fab0",
            "https://www.unum.cloud/blog/2022-01-26-cupy",
            "https://medium.com/rapids-ai/easy-cpu-gpu-arrays-and-dataframes-run-your-dask-code-where-youd-like-e349d92351d"
        ]

        docs = []
        for url in urls:
            try:
                loaded_docs = WebBaseLoader(url).load()
                docs.extend(loaded_docs)
                print(f"✅ Loaded web content from {url}")
            except Exception as e:
                print(f"⚠️ Could not load {url}: {str(e)}")

        # Load local notebook content if available
        notebook_dir = "../python_notebooks"
        notebook_files = [
            "notebook-1-cupy.ipynb",
            "notebook-2-rapids-cudf.ipynb", 
            "notebook-3-rapids-cuml.ipynb",
            "notebook-4-warp.ipynb"
        ]

        for nb_file in notebook_files:
            nb_path = os.path.join(notebook_dir, nb_file)
            if os.path.exists(nb_path):
                try:
                    nb_loader = NotebookLoader(nb_path, include_outputs=True, max_output_length=1000)
                    nb_docs = nb_loader.load()
                    docs.extend(nb_docs)
                    print(f"✅ Loaded {nb_file}")
                except Exception as e:
                    print(f"⚠️ Could not load {nb_file}: {str(e)}")

        # Add curated GPU acceleration content
        gpu_acceleration_content = """
# GPU Acceleration with NVIDIA Rapids

## CuPy Performance Patterns
- Matrix operations show 5-50x speedup on GPU vs CPU
- Best performance for arrays > 1M elements
- Memory bandwidth is often the bottleneck
- Use .astype() to ensure optimal data types (float32)
- Kernel launch overhead affects small operations

## cuDF Performance Benefits  
- DataFrame operations can achieve 10-100x speedup
- GroupBy operations scale excellently on GPU
- String operations benefit significantly from GPU parallelization
- Best for datasets > 100K rows
- Memory management is crucial for large datasets

## cuML Machine Learning Acceleration
- K-Means clustering: 10-50x speedup typical
- Random Forest: 5-25x speedup
- Logistic Regression: 3-15x speedup
- UMAP/t-SNE: 10-100x speedup for dimensionality reduction

## Best Practices for GPU Acceleration
1. Keep data on GPU between operations
2. Use appropriate data types (prefer float32 over float64)
3. Batch operations to amortize kernel launch overhead
4. Profile memory usage and optimize transfers
5. Use @cupy.fuse for element-wise operations
6. Consider problem size - GPU overhead for small data

## When NOT to use GPU
- Very small datasets (< 10K elements)
- Sequential algorithms that don't parallelize
- Frequent CPU-GPU memory transfers
- Operations dominated by I/O
"""

        docs.append(Document(page_content=gpu_acceleration_content, metadata={"source": "curated_gpu_guide"}))
        
        self.docs = docs
        print(f"📚 Total documents loaded: {len(docs)}")
        return docs
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self.doc_splits = text_splitter.split_documents(docs)
        return self.doc_splits

class VectorStore:
    """Handle vector store operations for document retrieval."""
    
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = None
        self.retriever = None
    
    def create_vectorstore(self, doc_splits: List[Document]):
        """Create vector store from document splits."""
        self.vectorstore = InMemoryVectorStore.from_documents(
            documents=doc_splits, 
            embedding=self.embedding_model
        )
        self.retriever = self.vectorstore.as_retriever()
        return self.retriever
    
    def get_retriever(self):
        """Get the retriever object."""
        return self.retriever
