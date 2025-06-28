# GPU Mentor Configuration

# LLM Configuration
OLLAMA_PORT = 11434  # Configurable Ollama port - change this to match your Ollama server port
OLLAMA_BASE_URL = f"http://localhost:{OLLAMA_PORT}/"  # Updated to use configurable port

# Model Configuration - Using different models for different tasks
CHAT_MODEL = "qwen2.5-coder:14b"  # General chat and RAG responses
CODE_MODEL = "qwen2.5-coder:14b"  # Code analysis and optimization
LLM_TEMPERATURE = 0

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Application Settings
DEFAULT_PORT = 7860
DEFAULT_HOST = "0.0.0.0"

# Code Execution Settings
MAX_EXECUTION_TIME = 30  # seconds
MAX_OUTPUT_LENGTH = 10000  # characters

# Document Processing
CHUNK_SIZE = 100
CHUNK_OVERLAP = 50

# Vector Store Configuration
USE_PERSISTENT_VECTORSTORE = True  # Set to False to use in-memory
VECTORSTORE_PERSIST_DIRECTORY = "./output/vectorstore_data"  # Directory for persistent vector storage

# External URLs for GPU acceleration knowledge
KNOWLEDGE_URLS = [
    "https://medium.com/cupy-team/announcing-cupy-v13-66979ee7fab0",
    "https://www.unum.cloud/blog/2022-01-26-cupy",
    "https://medium.com/rapids-ai/easy-cpu-gpu-arrays-and-dataframes-run-your-dask-code-where-youd-like-e349d92351d"
]

# Local notebook paths (relative to app directory)
NOTEBOOK_PATHS = [
    "../python_notebooks/notebook-1-cupy.ipynb",
    "../python_notebooks/notebook-2-rapids-cudf.ipynb", 
    "../python_notebooks/notebook-3-rapids-cuml.ipynb",
    "../python_notebooks/notebook-4-warp.ipynb"
]
