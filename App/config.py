# GPU Mentor Configuration

# LLM Configuration
OLLAMA_BASE_URL = "http://localhost:11434/"
OLLAMA_MODEL = "qwen2.5-coder:14b"  # Use the available model
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
