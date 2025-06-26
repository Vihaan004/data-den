"""
RAG Pipeline - Document Loading and Vector Store Management
Handles document loading, processing, and retrieval for GPU Mentor knowledge base.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader, NotebookLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
import socket
import re

try:
    from ..config import settings
except ImportError:
    # Fallback for direct module execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from config import settings
    except ImportError:
        # Create default settings if config is not available
        class DefaultSettings:
            OLLAMA_BASE_URL = "http://localhost:11434"
            OLLAMA_MODEL = "llama2"
            CHROMADB_PATH = "./chromadb"
            TEMP_DIR = "./temp"
            ollama_base_url = "http://localhost:11434"
            ollama_model = "llama2"
            ollama_temperature = 0.0
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            chunk_size = 1000
            chunk_overlap = 200
            gpu_docs_urls = []
            notebook_dir = "../python_notebooks"
            notebook_files = []
        settings = DefaultSettings()

logger = logging.getLogger(__name__)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class RAGPipeline:
    """
    Manages the RAG pipeline for GPU Mentor including document loading,
    vector store management, and graph workflow compilation.
    """
    
    def __init__(self):
        self.documents = []
        self.vectorstore = None
        self.retriever = None
        self.retriever_tool = None
        self.llm_model = None
        self.graph = None
        self.embedding_model = None
        
    async def initialize(self) -> None:
        """Initialize the RAG pipeline components."""
        logger.info("Initializing RAG pipeline...")
        
        try:
            # Load documents
            await self._load_documents()
            
            # Setup embeddings and vector store
            await self._setup_vectorstore()
            
            # Initialize LLM
            await self._setup_llm()
            
            # Compile workflow graph
            await self._compile_workflow()
            
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            raise
    
    async def _load_documents(self) -> None:
        """Load documents from various sources."""
        logger.info("Loading documents...")
        
        # Load web documents
        for url in settings.gpu_docs_urls:
            try:
                loaded_docs = WebBaseLoader(url).load()
                self.documents.extend(loaded_docs)
                logger.info(f"✅ Loaded web content from {url}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load {url}: {str(e)}")
        
        # Load local notebook content
        notebook_dir = Path(settings.notebook_dir)
        for nb_file in settings.notebook_files:
            nb_path = notebook_dir / nb_file
            if nb_path.exists():
                try:
                    nb_loader = NotebookLoader(str(nb_path), include_outputs=True, max_output_length=1000)
                    nb_docs = nb_loader.load()
                    self.documents.extend(nb_docs)
                    logger.info(f"✅ Loaded {nb_file}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load {nb_file}: {str(e)}")
            else:
                logger.warning(f"❌ Notebook not found: {nb_path}")
        
        # Add curated GPU acceleration content
        self._add_curated_content()
        
        logger.info(f"📚 Total documents loaded: {len(self.documents)}")
    
    def _add_curated_content(self) -> None:
        """Add curated GPU acceleration knowledge."""
        gpu_acceleration_content = """
# GPU Acceleration with NVIDIA Rapids - Performance Guide

## CuPy Performance Patterns
- Matrix operations show 5-50x speedup on GPU vs CPU for large arrays (>1M elements)
- Best performance achieved with float32 data types
- Memory bandwidth often becomes the bottleneck for element-wise operations
- Use @cupy.fuse decorator to reduce kernel launch overhead
- Memory pool management crucial for repeated operations
- GPU synchronization needed for accurate timing measurements

## cuDF Performance Benefits  
- DataFrame operations achieve 10-100x speedup for data processing workloads
- GroupBy operations scale excellently with number of groups and data size
- String operations benefit significantly from GPU parallelization
- Best performance with datasets >100K rows
- Memory management critical for large datasets (consider chunking)
- Query operations highly optimized on GPU

## cuML Machine Learning Acceleration
- K-Means clustering: 10-50x speedup typical for large datasets
- Random Forest: 5-25x speedup depending on data characteristics
- Logistic Regression: 3-15x speedup for high-dimensional data
- UMAP/t-SNE: 10-100x speedup for dimensionality reduction
- Single precision (float32) recommended for better performance
- GPU memory considerations for very large feature spaces

## Best Practices for GPU Acceleration
1. Keep data on GPU between operations to minimize CPU-GPU transfers
2. Use appropriate data types (prefer float32 over float64)
3. Batch operations to amortize kernel launch overhead
4. Profile memory usage and optimize memory access patterns
5. Use memory pools for better memory management
6. Consider problem size - GPU overhead significant for small data
7. Validate results between CPU and GPU implementations

## When NOT to use GPU
- Very small datasets (< 10K elements for arrays, < 50K rows for DataFrames)
- Sequential algorithms that don't parallelize well
- Operations requiring frequent CPU-GPU memory transfers
- I/O bound operations where computation is minimal
- Code with many conditional branches that don't vectorize

## Memory Management Best Practices
- Use CuPy memory pools to reduce allocation overhead
- Monitor GPU memory usage with nvidia-smi
- Consider using pinned memory for faster transfers
- Free GPU memory explicitly when working with large datasets
- Use streaming for datasets larger than GPU memory

## Optimization Techniques
- Kernel fusion with @cupy.fuse() for element-wise operations
- Custom CUDA kernels for specialized operations
- Overlapping computation and memory transfers
- Using multiple GPUs with proper data partitioning
- Profiling tools: CuPy profiler, NVIDIA Nsight, nvprof
"""
        
        self.documents.append(Document(
            page_content=gpu_acceleration_content, 
            metadata={"source": "curated_gpu_guide", "type": "performance_guide"}
        ))
    
    async def _setup_vectorstore(self) -> None:
        """Setup embeddings and vector store."""
        logger.info("Setting up vector store...")
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=settings.chunk_size, 
            chunk_overlap=settings.chunk_overlap
        )
        doc_splits = text_splitter.split_documents(self.documents)
        
        # Create vector store
        self.vectorstore = InMemoryVectorStore.from_documents(
            documents=doc_splits, 
            embedding=self.embedding_model
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever()
        
        # Create retriever tool
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_python_gpu_acceleration",
            "Search and return information about accelerating Python code using GPU with RAPIDS, CuPy, and NVIDIA technologies.",
        )
        
        logger.info("Vector store setup completed")
    
    async def _setup_llm(self) -> None:
        """Initialize LLM model."""
        logger.info("Setting up LLM model...")
        
        try:
            host_node = socket.gethostname()
            base_url = settings.ollama_base_url.replace("localhost", host_node)
            
            # Use ChatOllama directly instead of init_chat_model
            self.llm_model = ChatOllama(
                model=settings.ollama_model,
                temperature=settings.ollama_temperature,
                base_url=base_url
            )
            
            logger.info(f"LLM model initialized: {settings.ollama_model}")
            
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
            raise
    
    async def _compile_workflow(self) -> None:
        """Compile the LangGraph workflow."""
        logger.info("Compiling workflow graph...")
        
        workflow = StateGraph(MessagesState)
        
        # Define nodes
        workflow.add_node("generate_query_or_respond", self._generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("rewrite_question", self._rewrite_question)
        workflow.add_node("generate_answer", self._generate_answer)
        
        # Define edges
        workflow.add_edge(START, "generate_query_or_respond")
        
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )
        
        workflow.add_conditional_edges(
            "retrieve",
            self._grade_documents,
        )
        
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        
        # Compile graph
        self.graph = workflow.compile()
        
        logger.info("Workflow graph compiled successfully")
    
    def _generate_query_or_respond(self, state: MessagesState) -> Dict[str, Any]:
        """Generate query or respond based on current state."""
        response = (
            self.llm_model
            .bind_tools([self.retriever_tool])
            .invoke(state["messages"])
        )
        
        # Remove thinking text
        content = re.sub(r"<think>.*</think>", "", response.content, flags=re.DOTALL).strip()
        response.content = content
        
        return {"messages": [response]}
    
    def _grade_documents(self, state: MessagesState) -> str:
        """Determine whether retrieved documents are relevant."""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        GRADE_PROMPT = (
            "You are a grader assessing relevance of a retrieved document to a user question. \n "
            "Here is the retrieved document: \n\n {context} \n\n"
            "Here is the user question: {question} \n"
            "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
            "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
        )
        
        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = (
            self.llm_model
            .with_structured_output(GradeDocuments)
            .invoke([{"role": "user", "content": prompt}])
        )
        
        score = response.binary_score
        return "generate_answer" if score == "yes" else "rewrite_question"
    
    def _rewrite_question(self, state: MessagesState) -> Dict[str, Any]:
        """Rewrite the original user question."""
        messages = state["messages"]
        question = messages[0].content
        
        REWRITE_PROMPT = (
            "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
            "Here is the initial question:"
            "\n ------- \n"
            "{question}"
            "\n ------- \n"
            "Formulate an improved question:"
        )
        
        prompt = REWRITE_PROMPT.format(question=question)
        response = self.llm_model.invoke([{"role": "user", "content": prompt}])
        
        # Remove thinking text
        content = re.sub(r"<think>.*</think>", "", response.content, flags=re.DOTALL).strip()
        response.content = content
        
        return {"messages": [{"role": "user", "content": response.content}]}
    
    def _generate_answer(self, state: MessagesState) -> Dict[str, Any]:
        """Generate an answer based on retrieved context."""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        GENERATE_PROMPT = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \n"
            "Context: {context}"
        )
        
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = self.llm_model.invoke([{"role": "user", "content": prompt}])
        
        # Remove thinking text
        content = re.sub(r"<think>.*</think>", "", response.content, flags=re.DOTALL).strip()
        response.content = content
        
        return {"messages": [response]}
    
    def invoke(self, messages: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the RAG pipeline with user messages."""
        if not self.graph:
            raise RuntimeError("RAG pipeline not initialized. Call initialize() first.")
        
        return self.graph.invoke(messages)
    
    def stream(self, messages: Dict[str, Any]):
        """Stream responses from the RAG pipeline."""
        if not self.graph:
            raise RuntimeError("RAG pipeline not initialized. Call initialize() first.")
        
        return self.graph.stream(messages)
    
    def get_retriever_tool(self):
        """Get the retriever tool for external use."""
        return self.retriever_tool
