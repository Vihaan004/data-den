"""
Configuration settings for GPU Mentor Backend
"""
import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings
    except ImportError:
        # Fallback for environments without pydantic
        class BaseSettings:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Settings
    app_name: str = "GPU Mentor API"
    app_version: str = "2.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Ollama LLM Settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    ollama_temperature: float = 0.0
    
    # RAG Pipeline Settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Documentation URLs for knowledge base
    gpu_docs_urls: list = [
        "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
        "https://docs.rapids.ai/api/cudf/stable/",
        "https://cupy.dev/en/stable/",
    ]
    
    # Sol Environment Settings
    sol_work_dir: str = "/tmp/gpu_mentor"
    sol_modules_script: str = """
module load python/3.11
module load anaconda3
module load cuda/12.1
source activate rapids-23.08
"""
    
    # SLURM Settings
    slurm_partition: str = "general"
    slurm_qos: str = "public"
    slurm_time_limit: str = "00:15:00"
    slurm_memory: str = "32G"
    slurm_cpus: int = 8
    
    # Vector Store Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 100
    chunk_overlap: int = 50
    
    # Security Settings
    cors_origins: list = ["*"]  # Configure for production
    api_key: Optional[str] = None
    
    # Logging Settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Document Sources
    gpu_docs_urls: list = [
        "https://medium.com/cupy-team/announcing-cupy-v13-66979ee7fab0",
        "https://www.unum.cloud/blog/2022-01-26-cupy",
        "https://medium.com/rapids-ai/easy-cpu-gpu-arrays-and-dataframes-run-your-dask-code-where-youd-like-e349d92351d"
    ]
    
    notebook_dir: str = "../python_notebooks"
    notebook_files: list = [
        "notebook-1-cupy.ipynb",
        "notebook-2-rapids-cudf.ipynb", 
        "notebook-3-rapids-cuml.ipynb",
        "notebook-4-warp.ipynb"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
