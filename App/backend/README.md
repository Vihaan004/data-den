# GPU Mentor Backend - Deployment Guide for Sol Supercomputer

## 🎯 Overview

This directory contains the modularized backend code extracted from the `enhanced_agentic_rag_ollama.ipynb` notebook. All core functionality has been ported to production-ready Python modules suitable for deployment on the Sol supercomputer.

## 📁 Directory Structure

```
App/backend/
├── core/                          # Core GPU Mentor components
│   ├── rag_pipeline.py           # RAG system with LangGraph workflow
│   ├── code_optimizer.py         # Code analysis and GPU optimization
│   ├── benchmark_engine.py       # Performance comparison engine  
│   ├── sol_executor.py           # SLURM job submission and monitoring
│   └── enhanced_gpu_mentor.py    # Main orchestration class
├── utils/                         # Utility and educational components
│   ├── educational_content.py    # Code examples and learning content
│   ├── performance_visualizer.py # Benchmark result visualization
│   └── sample_code_library.py    # Comprehensive test code samples
├── models/                        # Pydantic models for API
│   └── api_models.py             # Data models and validation
├── config.py                     # Configuration management
├── requirements.txt              # Python dependencies
└── verify_backend.py            # Verification script
```

## ✅ Verified Components Ported from Notebook

### Core Components (Sections 1-14)
- **RAG Pipeline** ✅ - Document loading, vector store, LangGraph workflow
- **Sol Code Executor** ✅ - SLURM job submission, monitoring, file management
- **Code Optimizer** ✅ - Pattern detection, LLM-based optimization, GPU suggestions
- **Benchmark Engine** ✅ - CPU vs GPU performance comparison, visualization
- **Enhanced GPU Mentor** ✅ - Main orchestration, Socratic questioning, learning objectives

### Utility Components  
- **Educational Content Enhancer** ✅ - Code examples, optimization tips, learning materials
- **Performance Visualizer** ✅ - Benchmark result formatting, educational summaries
- **Sample Code Library** ✅ - 12+ test codes across 6 categories (Matrix, DataFrame, ML, etc.)

### Not Ported (Frontend Components)
- Gradio Interface (Section 15) - Frontend UI component, not needed for backend API
- Testing Examples (Section 16) - Interactive examples, replaced by verification script

## 🚀 Deployment Instructions

### 1. Environment Setup on Sol

```bash
# Load the GenAI kernel
module load genai25.06

# Navigate to backend directory
cd App/backend/
```

### 2. Backend Initialization (REQUIRED)

**⚠️ IMPORTANT**: The RAG pipeline must be initialized before use. Choose one method:

#### Option A: Interactive Initialization Script
```bash
python initialize_backend.py
```

#### Option B: Quick Start Menu
```bash
chmod +x quick_start.sh
./quick_start.sh
# Select option 1: "Initialize Backend Only"
```

#### Option C: Initialize from Gradio UI
1. Start the UI normally: `python gradio_ui.py`
2. Click the "🔄 Initialize Backend" button in the UI
3. Wait for "✅ Backend Status: Successfully initialized and ready"

### 3. Launch Options
# Load required modules
module load python/3.11 anaconda3 cuda/12.1

# Activate RAPIDS environment
source activate rapids-23.08

# OR create new environment
conda create -n gpu-mentor-backend python=3.11
conda activate gpu-mentor-backend
```

### 2. Install Dependencies

```bash
# Navigate to backend directory
cd /path/to/App/backend

# Install Python dependencies
pip install -r requirements.txt

# Install GPU libraries (on Sol with GPU access)
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=23.08 cuml=23.08 cupy>=12.0 dask-cudf=23.08
```

### 3. Configure Ollama

```bash
# Install Ollama (if not available)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the required model
ollama pull qwen2:14b

# Verify model is available
ollama list
```

### 4. Verification

```bash
# Run verification script
python verify_backend.py
```

Expected output:
```
🚀 GPU Mentor Backend Verification
==================================================
🔍 Verifying Backend Structure...
✅ core/rag_pipeline.py
✅ core/code_optimizer.py
✅ core/benchmark_engine.py
✅ core/sol_executor.py
✅ core/enhanced_gpu_mentor.py
...
✅ All components successfully ported from notebook!
```

### 5. Testing on Sol

```bash
# Test basic imports
python -c "
from core.enhanced_gpu_mentor import EnhancedGPUMentor
from core.rag_pipeline import RAGPipeline
from core.code_optimizer import CodeOptimizer
print('✅ All imports successful')
"

# Test initialization (REQUIRED)
python initialize_backend.py

# Test Sol SLURM integration
python -c "
from core.sol_executor import SolCodeExecutor
executor = SolCodeExecutor()
print('✅ Sol executor initialized')
"
```

## ⚠️ Important Notes

### RAG Pipeline Initialization
- **MUST call `initialize()` before using EnhancedGPUMentor**
- Initialization sets up document loading, vector store, and LLM components
- Can be done via script, UI button, or programmatically
- Only needs to be done once per session

### Error Resolution
If you see "RAG pipeline not initialized. Call initialize() first.":
1. Run `python initialize_backend.py` from backend directory
2. OR click "🔄 Initialize Backend" button in Gradio UI
3. OR ensure your code calls `await gpu_mentor.initialize()` before use

If you see "Connection refused" errors:
1. Start Ollama service: `./start_ollama.sh`
2. Or manually: `ollama serve` and `ollama pull qwen2:14b`
3. The system will fall back to basic responses if Ollama is unavailable

For other issues, see `TROUBLESHOOTING.md` for comprehensive solutions.

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2:14b

# Sol Configuration  
SOL_WORK_DIR=/tmp/gpu_mentor
SOL_PARTITION=gpu
SOL_DEFAULT_TIME=00:15:00
SOL_DEFAULT_MEMORY=32G

# RAG Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Logging
LOG_LEVEL=INFO
```

### SLURM Job Templates

The `sol_executor.py` includes optimized SLURM job templates for:
- CPU benchmarking jobs (base partition)
- GPU benchmarking jobs (gpu partition)
- Automatic environment loading
- Error handling and cleanup

## 🧪 API Integration

### Basic Usage

```python
from core.enhanced_gpu_mentor import EnhancedGPUMentor
from core.rag_pipeline import RAGPipeline
from core.benchmark_engine import BenchmarkEngine
from core.code_optimizer import CodeOptimizer

# Initialize components
rag_pipeline = RAGPipeline()
await rag_pipeline.initialize()

code_optimizer = CodeOptimizer(rag_pipeline.get_graph())
benchmark_engine = BenchmarkEngine()

# Create GPU Mentor
gpu_mentor = EnhancedGPUMentor(
    rag_pipeline=rag_pipeline,
    benchmark_engine=benchmark_engine, 
    code_optimizer=code_optimizer
)

# Process user request
response = await gpu_mentor.process_user_input(
    user_input="How can I optimize this matrix multiplication?",
    code="import numpy as np\nA = np.random.rand(1000, 1000)\nB = np.random.rand(1000, 1000)\nC = np.dot(A, B)"
)
```

## 📊 Performance Benchmarking

The system supports real-world benchmarking across multiple categories:

### Benchmark Categories
1. **Matrix Operations** - NumPy vs CuPy (10-50x speedup)
2. **DataFrame Operations** - Pandas vs cuDF (5-20x speedup)  
3. **Machine Learning** - scikit-learn vs cuML (5-25x speedup)
4. **Mathematical Functions** - NumPy vs CuPy (3-15x speedup)

### Benchmark Workflow
```python
# Run comprehensive benchmark
benchmark_results = await benchmark_engine.run_comprehensive_benchmark(
    user_code="import numpy as np\nresult = np.matmul(A, B)",
    timeout=300
)

# Get educational insights
from utils.performance_visualizer import PerformanceVisualizer
visualizer = PerformanceVisualizer()
summary = visualizer.create_educational_summary(benchmark_results)
```

## 🎓 Educational Features

### Sample Code Library
- 12+ curated examples across 6 categories
- CPU and GPU optimized versions
- Educational annotations and expected speedups

### Learning Content
- Code optimization patterns
- Performance insights and guidelines
- Socratic questioning for interactive learning
- Learning objective generation

## 🛡️ Error Handling & Monitoring

- Comprehensive logging throughout all components
- SLURM job failure detection and recovery
- GPU memory management and cleanup
- Timeout handling for long-running operations

## 🚀 Next Steps

1. **API Development** - Create FastAPI endpoints wrapping the core components
2. **Frontend Integration** - Build UI that consumes the backend API
3. **Monitoring** - Add Prometheus metrics and health checks
4. **Scaling** - Implement job queuing for concurrent requests
5. **Security** - Add authentication and rate limiting

## 📋 Validation Checklist

- [ ] All core components import successfully
- [ ] Ollama qwen2:14b model is available
- [ ] RAPIDS libraries (cudf, cuml, cupy) are installed
- [ ] SLURM commands are accessible (`sbatch`, `squeue`, `scancel`)
- [ ] Sol GPU partition is available
- [ ] Verification script passes all tests
- [ ] Sample benchmark jobs submit and complete successfully

---

**Status**: ✅ Complete - All notebook components successfully ported and ready for Sol deployment

**Kernel Compatibility**: Designed for `genai25.06` kernel environment on Sol supercomputer
