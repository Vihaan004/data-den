# GPU Mentor Backend Testing Guide for Sol Supercomputer

This guide provides step-by-step instructions for testing the modularized GPU Mentor backend on Sol using the genai25.06 kernel.

## Prerequisites

- Access to Sol supercomputer
- genai25.06 kernel available
- SLURM job scheduler access
- Git access to upload the codebase

## 1. Upload and Setup on Sol

### Transfer the Backend Code
```bash
# Upload the entire App/backend directory to Sol
scp -r App/backend/ username@sol.asu.edu:~/gpu-mentor-backend/

# Or clone from your repository if pushed to Git
git clone <your-repo-url> ~/gpu-mentor-backend/
cd ~/gpu-mentor-backend/App/backend/
```

### Load the genai25.06 Kernel
```bash
# Load the required kernel
module load genai25.06

# Verify Python environment
python --version
which python
```

## 2. Install Dependencies

### Install Required Packages
```bash
# Install all requirements
pip install -r requirements.txt

# Install additional dependencies if needed
pip install ollama chromadb sentence-transformers matplotlib seaborn plotly

# Verify critical imports
python -c "import ollama, chromadb, sentence_transformers; print('Core dependencies installed')"
```

## 3. Environment Configuration

### Create Configuration File
```bash
# Copy and edit the config file
cp config.py config_sol.py
```

Edit `config_sol.py` with Sol-specific settings:
```python
# Sol-specific configuration
OLLAMA_BASE_URL = "http://localhost:11434"  # Adjust if different on Sol
CHROMADB_PATH = "/scratch/your_username/chromadb"  # Use scratch space
TEMP_DIR = "/scratch/your_username/gpu_mentor_temp"
SLURM_PARTITION = "gpu"  # Check available partitions
DEFAULT_GPU_TYPE = "a100"  # Check available GPU types
```

## 4. Basic Functionality Tests

### Test 1: Import Verification
Create `test_imports.py`:
```python
#!/usr/bin/env python3
"""Test all imports on Sol"""

import sys
sys.path.append('.')

def test_imports():
    try:
        from core.rag_pipeline import RAGPipeline
        from core.code_optimizer import CodeOptimizer
        from core.benchmark_engine import BenchmarkEngine
        from core.sol_executor import SolCodeExecutor
        from core.enhanced_gpu_mentor import EnhancedGPUMentor
        from utils.educational_content import EducationalContentEnhancer
        from utils.performance_visualizer import PerformanceVisualizer
        from utils.sample_code_library import SampleCodeLibrary
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()
```

Run the test:
```bash
python test_imports.py
```

### Test 2: Component Initialization
Create `test_components.py`:
```python
#!/usr/bin/env python3
"""Test component initialization"""

import sys
sys.path.append('.')

def test_component_init():
    try:
        # Test RAG Pipeline
        from core.rag_pipeline import RAGPipeline
        rag = RAGPipeline()
        print("✅ RAG Pipeline initialized")
        
        # Test Code Optimizer
        from core.code_optimizer import CodeOptimizer
        optimizer = CodeOptimizer()
        print("✅ Code Optimizer initialized")
        
        # Test Benchmark Engine
        from core.benchmark_engine import BenchmarkEngine
        benchmark = BenchmarkEngine()
        print("✅ Benchmark Engine initialized")
        
        # Test Sol Executor
        from core.sol_executor import SolCodeExecutor
        executor = SolCodeExecutor()
        print("✅ Sol Executor initialized")
        
        # Test Enhanced GPU Mentor
        from core.enhanced_gpu_mentor import EnhancedGPUMentor
        mentor = EnhancedGPUMentor()
        print("✅ Enhanced GPU Mentor initialized")
        
        return True
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_component_init()
```

## 5. SLURM Job Testing

### Create SLURM Test Script
Create `test_job.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=gpu_mentor_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=gpu_mentor_test_%j.out
#SBATCH --error=gpu_mentor_test_%j.err

# Load modules
module load genai25.06

# Change to project directory
cd ~/gpu-mentor-backend/App/backend/

# Set environment variables
export PYTHONPATH=$PWD:$PYTHONPATH

# Run tests
echo "Starting GPU Mentor Backend Tests..."
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Test imports
echo "=== Testing Imports ==="
python test_imports.py

# Test component initialization
echo "=== Testing Component Initialization ==="
python test_components.py

# Test with a simple optimization task
echo "=== Testing Code Optimization ==="
python -c "
from core.enhanced_gpu_mentor import EnhancedGPUMentor
mentor = EnhancedGPUMentor()
result = mentor.optimize_code('import numpy as np\nx = np.array([1,2,3])\nprint(x.sum())')
print('Optimization test completed')
"

echo "Tests completed!"
```

Submit the job:
```bash
sbatch test_job.slurm
```

Monitor the job:
```bash
squeue -u $USER
cat gpu_mentor_test_*.out
cat gpu_mentor_test_*.err
```

## 6. Interactive Testing

### Start Interactive Session
```bash
# Request interactive GPU node
srun --partition=gpu --gres=gpu:1 --mem=16G --time=02:00:00 --pty bash

# Load modules and navigate
module load genai25.06
cd ~/gpu-mentor-backend/App/backend/
```

### Test Ollama Integration
```python
# Start Python and test Ollama connection
python3
```

```python
from core.enhanced_gpu_mentor import EnhancedGPUMentor
mentor = EnhancedGPUMentor()

# Test basic functionality
test_code = """
import numpy as np
def slow_function(data):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    return result

data = list(range(1000))
result = slow_function(data)
"""

# Test optimization
optimized = mentor.optimize_code(test_code)
print("Optimization completed!")
```

## 7. Benchmark Testing

### Test Performance Benchmarking
Create `test_benchmark.py`:
```python
#!/usr/bin/env python3
"""Test benchmark functionality"""

import sys
sys.path.append('.')

from core.benchmark_engine import BenchmarkEngine
import numpy as np

def test_benchmark():
    engine = BenchmarkEngine()
    
    # Test CPU vs GPU comparison (if available)
    test_code = """
import numpy as np
import time

start_time = time.time()
data = np.random.random((1000, 1000))
result = np.dot(data, data.T)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")
"""
    
    print("Running benchmark test...")
    result = engine.benchmark_code(test_code, "numpy_matrix_multiply")
    print(f"Benchmark completed: {result}")

if __name__ == "__main__":
    test_benchmark()
```

## 8. Full Integration Test

### Create Comprehensive Test
Create `integration_test.py`:
```python
#!/usr/bin/env python3
"""Full integration test of GPU Mentor backend"""

import sys
sys.path.append('.')

from core.enhanced_gpu_mentor import EnhancedGPUMentor
from core.rag_pipeline import RAGPipeline
from utils.sample_code_library import SampleCodeLibrary

def integration_test():
    print("=== GPU Mentor Integration Test ===")
    
    # Initialize components
    mentor = EnhancedGPUMentor()
    rag = RAGPipeline()
    samples = SampleCodeLibrary()
    
    # Test workflow
    print("1. Testing code optimization...")
    test_code = samples.get_sample_code("basic", "matrix_operations")
    if test_code:
        optimized = mentor.optimize_code(test_code)
        print("   ✅ Code optimization completed")
    
    print("2. Testing educational content...")
    explanation = mentor.explain_optimization("vectorization")
    print("   ✅ Educational content generated")
    
    print("3. Testing performance analysis...")
    # Add performance test if GPU available
    
    print("=== Integration test completed ===")

if __name__ == "__main__":
    integration_test()
```

## 9. Troubleshooting Common Issues

### Module Import Issues
```bash
# If imports fail, check Python path
export PYTHONPATH=$PWD:$PYTHONPATH
python -c "import sys; print('\n'.join(sys.path))"
```

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed (check Sol documentation)
ollama serve &
```

### GPU Access Issues
```bash
# Check GPU availability
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Verify in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Memory Issues
```bash
# Monitor memory usage
free -h
nvidia-smi

# Increase job memory if needed
#SBATCH --mem=32G
```

## 10. Performance Monitoring

### Create Monitoring Script
Create `monitor_performance.py`:
```python
#!/usr/bin/env python3
"""Monitor backend performance"""

import psutil
import time
import GPUtil

def monitor_system():
    print("=== System Performance Monitor ===")
    
    # CPU and Memory
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    
    # GPU (if available)
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.load*100:.1f}% load, {gpu.memoryUtil*100:.1f}% memory")
    except:
        print("GPU monitoring not available")

if __name__ == "__main__":
    monitor_system()
```

## 11. Expected Test Results

### Successful Output Should Show:
- ✅ All module imports successful
- ✅ All components initialize without errors
- ✅ Code optimization produces improved code
- ✅ Benchmark engine generates performance metrics
- ✅ Educational content is generated appropriately
- ✅ SLURM jobs complete successfully

### Next Steps After Testing:
1. API endpoint development for web interface
2. Frontend integration
3. Advanced job queueing implementation
4. Monitoring and logging setup
5. Production deployment configuration

## Contact Information
- Report issues with specific error messages and logs
- Include SLURM job IDs for debugging failed jobs
- Provide system information (module versions, GPU types)

---
*This guide ensures comprehensive testing of the GPU Mentor backend on Sol infrastructure using the genai25.06 kernel.*
