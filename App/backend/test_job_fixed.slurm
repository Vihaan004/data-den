#!/bin/bash
#SBATCH --job-name=gpu_mentor_fixed_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=gpu_mentor_fixed_test_%j.out
#SBATCH --error=gpu_mentor_fixed_test_%j.err

echo "========================================================"
echo "GPU MENTOR BACKEND - FIXED IMPORT TESTING ON SOL"
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
cd ~/R1/App/backend/ || {
    echo "❌ Failed to change to backend directory"
    exit 1
}

echo "Current directory: $(pwd)"

# Set Python path explicitly
export PYTHONPATH=$PWD:$PYTHONPATH
echo "Python path configured: $PYTHONPATH"

# Install any missing dependencies
echo ""
echo "=== Installing Dependencies ==="
pip install -r requirements.txt --user

# Test 1: Fixed import verification
echo ""
echo "=== TEST 1: FIXED IMPORT VERIFICATION ==="
python test_imports_fixed.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Fixed import test passed"
else
    echo "❌ Fixed import test failed"
fi

# Test 2: Original import test for comparison
echo ""
echo "=== TEST 2: ORIGINAL IMPORT TEST (for comparison) ==="
python test_imports.py

# Test 3: Component initialization test
echo ""
echo "=== TEST 3: COMPONENT INITIALIZATION ==="
python test_components.py

# Test 4: Simple functionality test
echo ""
echo "=== TEST 4: SIMPLE FUNCTIONALITY TEST ==="
python -c "
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    print('🧪 Testing EnhancedGPUMentor initialization...')
    from core.enhanced_gpu_mentor import EnhancedGPUMentor
    mentor = EnhancedGPUMentor()
    print('✅ EnhancedGPUMentor created successfully')
    
    print('🧪 Testing simple optimization...')
    test_code = '''
result = []
for i in range(100):
    result.append(i * 2)
print(len(result))
'''
    
    try:
        optimized = mentor.optimize_code(test_code)
        if optimized:
            print('✅ Code optimization successful')
            print(f'   Original: {len(test_code)} chars')
            print(f'   Result: {len(str(optimized))} chars')
        else:
            print('⚠️  Code optimization returned empty result')
    except Exception as opt_e:
        print(f'⚠️  Code optimization failed: {opt_e}')
        
except Exception as e:
    print(f'❌ Functionality test failed: {e}')
    import traceback
    traceback.print_exc()
"

# Test 5: Configuration test
echo ""
echo "=== TEST 5: CONFIGURATION TEST ==="
python -c "
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    print('🧪 Testing configuration import...')
    import config
    settings = config.settings
    print(f'✅ Config loaded successfully')
    print(f'   App name: {settings.app_name}')
    print(f'   Ollama model: {settings.ollama_model}')
    print(f'   Debug mode: {settings.debug}')
except Exception as e:
    print(f'❌ Configuration test failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "========================================================"
echo "GPU MENTOR BACKEND - FIXED TESTING COMPLETED"
echo "Job ID: $SLURM_JOB_ID"
echo "Check output files for detailed results:"
echo "  - gpu_mentor_fixed_test_${SLURM_JOB_ID}.out"
echo "  - gpu_mentor_fixed_test_${SLURM_JOB_ID}.err"
echo "========================================================"
