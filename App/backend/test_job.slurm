#!/bin/bash
#SBATCH --job-name=gpu_mentor_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=gpu_mentor_test_%j.out
#SBATCH --error=gpu_mentor_test_%j.err

echo "========================================================"
echo "GPU MENTOR BACKEND TESTING ON SOL"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "========================================================"

# Load required modules
echo "Loading genai25.06 kernel..."
module load genai25.06

# Verify environment
echo "Python version: $(python --version)"
echo "Node name: $(hostname)"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "Job working directory: $(pwd)"

# Change to project directory
cd ~/gpu-mentor-backend/App/backend/ || {
    echo "❌ Failed to change to backend directory"
    exit 1
}

echo "Current directory: $(pwd)"

# Set Python path
export PYTHONPATH=$PWD:$PYTHONPATH
echo "Python path configured"

# Check if required files exist
echo ""
echo "=== Checking File Structure ==="
required_files=(
    "core/rag_pipeline.py"
    "core/code_optimizer.py"
    "core/benchmark_engine.py"
    "core/sol_executor.py"
    "core/enhanced_gpu_mentor.py"
    "utils/educational_content.py"
    "utils/performance_visualizer.py"
    "utils/sample_code_library.py"
    "requirements.txt"
    "test_imports.py"
    "test_components.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file - MISSING"
    fi
done

# Install any missing dependencies
echo ""
echo "=== Installing Dependencies ==="
pip install -r requirements.txt --user

# Test 1: Import verification
echo ""
echo "=== TEST 1: IMPORT VERIFICATION ==="
python test_imports.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Import test passed"
else
    echo "❌ Import test failed"
fi

# Test 2: Component initialization
echo ""
echo "=== TEST 2: COMPONENT INITIALIZATION ==="
python test_components.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Component test passed"
else
    echo "❌ Component test failed"
fi

# Test 3: Simple optimization workflow
echo ""
echo "=== TEST 3: OPTIMIZATION WORKFLOW ==="
python -c "
import sys
sys.path.append('.')
try:
    from core.enhanced_gpu_mentor import EnhancedGPUMentor
    print('🧪 Creating GPU Mentor instance...')
    mentor = EnhancedGPUMentor()
    print('✅ GPU Mentor created successfully')
    
    # Test with simple code
    test_code = '''
import numpy as np
result = []
for i in range(1000):
    result.append(i * 2)
print(len(result))
'''
    
    print('🧪 Testing code optimization...')
    optimized = mentor.optimize_code(test_code)
    
    if optimized and len(optimized) > 10:
        print('✅ Code optimization successful')
        print(f'   Original: {len(test_code)} chars')
        print(f'   Optimized: {len(optimized)} chars')
    else:
        print('⚠️  Code optimization returned minimal result')
        
except Exception as e:
    print(f'❌ Optimization workflow failed: {e}')
    import traceback
    traceback.print_exc()
"

# Test 4: Check GPU access (if available)
echo ""
echo "=== TEST 4: GPU ACCESS CHECK ==="
python -c "
import os
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"Not set\")}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
except ImportError:
    print('PyTorch not available - GPU tests skipped')
except Exception as e:
    print(f'GPU check failed: {e}')
"

# Test 5: Memory and performance check
echo ""
echo "=== TEST 5: SYSTEM RESOURCES ==="
python -c "
import psutil
import os

print(f'Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB')
print(f'CPU cores: {psutil.cpu_count()}')
print(f'CPU usage: {psutil.cpu_percent()}%')

# Check disk space
import shutil
total, used, free = shutil.disk_usage('.')
print(f'Disk space: {free / (1024**3):.1f} GB free of {total / (1024**3):.1f} GB total')
"

echo ""
echo "========================================================"
echo "GPU MENTOR BACKEND TESTING COMPLETED"
echo "Job ID: $SLURM_JOB_ID"
echo "Check output files for detailed results:"
echo "  - gpu_mentor_test_${SLURM_JOB_ID}.out"
echo "  - gpu_mentor_test_${SLURM_JOB_ID}.err"
echo "========================================================"
