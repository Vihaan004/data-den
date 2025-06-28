import os
import glob
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import WebBaseLoader, NotebookLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import (EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, 
                   USE_PERSISTENT_VECTORSTORE, VECTORSTORE_PERSIST_DIRECTORY)

class KnowledgeBaseStats:
    """Track statistics about the knowledge base."""
    def __init__(self):
        self.web_sources = 0
        self.web_sources_failed = 0
        self.notebooks = 0
        self.notebooks_failed = 0
        self.curated_content = 0
        self.total_documents = 0
        self.total_chunks = 0
        self.failed_sources = []
        
    def add_failed_source(self, source: str, error: str):
        self.failed_sources.append({"source": source, "error": error})
        
    def get_summary(self) -> str:
        """Get a formatted summary of the knowledge base."""
        summary = f"""
📊 Knowledge Base Summary:
• Web Sources: {self.web_sources} loaded, {self.web_sources_failed} failed
• Notebooks: {self.notebooks} loaded, {self.notebooks_failed} failed  
• Curated Content: {self.curated_content} pieces
• Total Documents: {self.total_documents}
• Total Chunks: {self.total_chunks}
"""
        if self.failed_sources:
            summary += f"\n⚠️ Failed Sources ({len(self.failed_sources)}):\n"
            for fail in self.failed_sources[:3]:  # Show first 3 failures
                summary += f"  • {fail['source']}: {fail['error'][:60]}...\n"
            if len(self.failed_sources) > 3:
                summary += f"  • ... and {len(self.failed_sources) - 3} more\n"
        return summary

class DocumentLoader:
    """Enhanced document loader that uses centralized knowledge directory."""
    
    def __init__(self, knowledge_dir: str = "./knowledge"):
        self.knowledge_dir = knowledge_dir
        self.docs = []
        self.doc_splits = []
        self.stats = KnowledgeBaseStats()
        
    def load_documents(self) -> List[Document]:
        """Load documents from centralized knowledge directory."""
        print(f"📚 Loading documents from knowledge directory: {self.knowledge_dir}")
        docs = []
        
        # Load from sources.txt URLs
        docs.extend(self._load_web_sources())
        
        # Load notebooks from knowledge/python_notebooks
        docs.extend(self._load_notebooks())
        
        # Add curated content
        docs.extend(self._load_curated_content())
        
        # Update stats
        self.stats.total_documents = len(docs)
        self.docs = docs
        
        print(self.stats.get_summary())
        return docs
        
    def _load_web_sources(self) -> List[Document]:
        """Load documents from URLs in sources.txt file."""
        sources_file = os.path.join(self.knowledge_dir, "sources.txt")
        docs = []
        
        if not os.path.exists(sources_file):
            print(f"⚠️ Sources file not found: {sources_file}")
            return docs
            
        print(f"🌐 Loading web sources from {sources_file}")
        urls = self._parse_sources_file(sources_file)
        
        for url in urls:
            try:
                loaded_docs = WebBaseLoader(url).load()
                docs.extend(loaded_docs)
                self.stats.web_sources += 1
                print(f"✅ Loaded web content from {url}")
            except Exception as e:
                error_msg = str(e)[:100]
                self.stats.web_sources_failed += 1
                self.stats.add_failed_source(url, error_msg)
                print(f"⚠️ Could not load {url}: {error_msg}")
                
        return docs
        
    def _parse_sources_file(self, sources_file: str) -> List[str]:
        """Parse the sources.txt file to extract URLs."""
        urls = []
        try:
            with open(sources_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        # Simple URL validation
                        if line.startswith(('http://', 'https://')):
                            urls.append(line)
            print(f"📝 Found {len(urls)} URLs in sources file")
        except Exception as e:
            print(f"⚠️ Error reading sources file: {e}")
            
        return urls
        
    def _load_notebooks(self) -> List[Document]:
        """Load all notebooks from the knowledge/python_notebooks directory."""
        notebooks_dir = os.path.join(self.knowledge_dir, "python_notebooks")
        docs = []
        
        if not os.path.exists(notebooks_dir):
            print(f"⚠️ Notebooks directory not found: {notebooks_dir}")
            return docs
            
        print(f"📓 Loading notebooks from {notebooks_dir}")
        
        # Find all .ipynb files
        notebook_pattern = os.path.join(notebooks_dir, "*.ipynb")
        notebook_files = glob.glob(notebook_pattern)
        
        for nb_path in notebook_files:
            try:
                nb_loader = NotebookLoader(
                    nb_path, 
                    include_outputs=True, 
                    max_output_length=1000,
                    remove_newline=True
                )
                nb_docs = nb_loader.load()
                docs.extend(nb_docs)
                self.stats.notebooks += 1
                nb_name = os.path.basename(nb_path)
                print(f"✅ Loaded notebook: {nb_name}")
            except Exception as e:
                error_msg = str(e)[:100]
                self.stats.notebooks_failed += 1
                self.stats.add_failed_source(nb_path, error_msg)
                print(f"⚠️ Could not load {os.path.basename(nb_path)}: {error_msg}")
                
        return docs
        
    def _load_curated_content(self) -> List[Document]:
        """Load curated GPU acceleration content."""
        docs = []
        
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
        self.stats.curated_content += 1
        
        print(f"📖 Added {len(docs)} curated content piece")
        return docs
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self.doc_splits = text_splitter.split_documents(docs)
        self.stats.total_chunks = len(self.doc_splits)
        return self.doc_splits

class VectorStore:
    """Handle vector store operations for document retrieval."""
    
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = None
        self.retriever = None
        # Track which type of vector store is being used
        self.using_persistent = False
    
    def create_vectorstore(self, doc_splits: List[Document]):
        """Create vector store from document splits."""
        if USE_PERSISTENT_VECTORSTORE:
            # Ensure directory exists
            os.makedirs(VECTORSTORE_PERSIST_DIRECTORY, exist_ok=True)
            
            # Check if we already have a stored database
            if self._check_existing_vectorstore():
                print("🔄 Found existing vector store, loading...")
                return self.load_vectorstore()
            
            # Create new persistent vector store
            self.vectorstore = Chroma.from_documents(
                documents=doc_splits,
                embedding=self.embedding_model,
                persist_directory=VECTORSTORE_PERSIST_DIRECTORY
            )
            
            # Save to disk
            self.vectorstore.persist()
            self.using_persistent = True
            print(f"✅ Created persistent ChromaDB vector store at {VECTORSTORE_PERSIST_DIRECTORY}")
        else:
            # Use in-memory vector store (original implementation)
            self.vectorstore = InMemoryVectorStore.from_documents(
                documents=doc_splits, 
                embedding=self.embedding_model
            )
            self.using_persistent = False
            print("✅ Created in-memory vector store")
        
        # Create retriever (same interface for both implementations)
        self.retriever = self.vectorstore.as_retriever()
        return self.retriever
    
    def _check_existing_vectorstore(self) -> bool:
        """Check if a persistent vector store already exists."""
        return (os.path.exists(VECTORSTORE_PERSIST_DIRECTORY) and 
                len(os.listdir(VECTORSTORE_PERSIST_DIRECTORY)) > 0)
    
    def load_vectorstore(self):
        """Load an existing persistent vector store."""
        if not self._check_existing_vectorstore():
            print("⚠️ No existing vector store found. Creating new one...")
            return None
            
        # Load existing Chroma database
        self.vectorstore = Chroma(
            persist_directory=VECTORSTORE_PERSIST_DIRECTORY,
            embedding_function=self.embedding_model
        )
        self.using_persistent = True
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever()
        print(f"✅ Loaded existing vector store from {VECTORSTORE_PERSIST_DIRECTORY}")
        return self.retriever
    
    def get_retriever(self):
        """Get the retriever object."""
        return self.retriever
        
    def add_documents(self, documents: List[Document]):
        """Add new documents to the existing vector store."""
        if not self.vectorstore:
            print("⚠️ Vector store not initialized. Call create_vectorstore first.")
            return
            
        if self.using_persistent:
            # Add documents to persistent store
            self.vectorstore.add_documents(documents)
            # Persist changes to disk
            self.vectorstore.persist()
            print(f"✅ Added {len(documents)} documents to persistent vector store")
        else:
            # For in-memory store, we need to recreate with combined documents
            # This is a limitation of InMemoryVectorStore
            print("⚠️ Cannot add documents to in-memory vector store. Use persistent store.")
            
    def get_document_count(self):
        """Get the number of documents in the vector store."""
        if not self.vectorstore:
            return 0
            
        if self.using_persistent:
            # For Chroma
            return self.vectorstore._collection.count()
        else:
            # For InMemoryVectorStore
            # Note: InMemoryVectorStore doesn't have a direct way to count documents
            return len(self.vectorstore.embeddings) if hasattr(self.vectorstore, 'embeddings') else 0
