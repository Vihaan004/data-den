#!/bin/bash
# GPU Mentor Backend Setup Script for Sol
# Run this script after uploading the backend to Sol

echo "=================================================="
echo "GPU MENTOR BACKEND SETUP FOR SOL"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Are you in the backend directory?"
    exit 1
fi

echo "✅ Backend directory confirmed"

# Load the genai25.06 kernel
echo "📦 Loading genai25.06 kernel..."
module load genai25.06

# Verify Python
echo "🐍 Python version: $(python --version)"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p temp
mkdir -p logs
mkdir -p data

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt --user

# Make test scripts executable
echo "🔧 Making test scripts executable..."
chmod +x test_imports.py
chmod +x test_components.py
chmod +x integration_test.py
chmod +x test_job.slurm

# Create a Sol-specific configuration
echo "⚙️  Creating Sol configuration..."
cat > config_sol.py << EOF
"""
Sol-specific configuration for GPU Mentor Backend
"""
import os

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"

# ChromaDB configuration  
CHROMADB_PATH = "/scratch/\$USER/gpu_mentor_chromadb"
CHROMADB_COLLECTION = "gpu_mentor_knowledge"

# Temporary directories (use scratch space)
TEMP_DIR = "/scratch/\$USER/gpu_mentor_temp" 
LOG_DIR = "/scratch/\$USER/gpu_mentor_logs"

# SLURM configuration
DEFAULT_PARTITION = "gpu"
DEFAULT_GPU_TYPE = "a100"
DEFAULT_MEMORY = "16G"
DEFAULT_TIME = "01:00:00"

# Performance settings
MAX_CONCURRENT_JOBS = 5
TIMEOUT_SECONDS = 300

# Educational content settings
ENABLE_DETAILED_EXPLANATIONS = True
INCLUDE_PERFORMANCE_TIPS = True

# Debug settings
DEBUG_MODE = os.environ.get('GPU_MENTOR_DEBUG', 'false').lower() == 'true'
VERBOSE_LOGGING = True

print("Sol configuration loaded successfully")
EOF

# Create scratch directories
echo "📁 Creating scratch space directories..."
mkdir -p "/scratch/$USER/gpu_mentor_chromadb"
mkdir -p "/scratch/$USER/gpu_mentor_temp"
mkdir -p "/scratch/$USER/gpu_mentor_logs"

# Test basic imports
echo "🧪 Testing basic imports..."
python -c "
import sys
sys.path.append('.')

try:
    from core.enhanced_gpu_mentor import EnhancedGPUMentor
    print('✅ Core imports successful')
except Exception as e:
    print(f'❌ Import test failed: {e}')
"

# Check GPU availability
echo "🖥️  Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  nvidia-smi not available (this is normal on login nodes)"
fi

# Create a quick test job submission script
echo "📝 Creating quick test submission script..."
cat > quick_test.sh << 'EOF'
#!/bin/bash
# Quick test submission for GPU Mentor Backend

echo "Submitting test job to SLURM..."
sbatch test_job.slurm

echo "Checking job status..."
sleep 2
squeue -u $USER

echo ""
echo "To monitor the job output:"
echo "  tail -f gpu_mentor_test_*.out"
echo ""
echo "To check for errors:"
echo "  tail -f gpu_mentor_test_*.err"
EOF

chmod +x quick_test.sh

# Display final instructions
echo ""
echo "=================================================="
echo "SETUP COMPLETE!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Submit test job:     ./quick_test.sh"
echo "2. Run import test:     python test_imports.py"
echo "3. Run component test:  python test_components.py"
echo "4. Run integration:     python integration_test.py"
echo ""
echo "Interactive testing:"
echo "1. Request GPU node:    srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash"
echo "2. Load kernel:         module load genai25.06"
echo "3. Navigate here:       cd ~/gpu-mentor-backend/App/backend/"
echo "4. Run tests:           python integration_test.py"
echo ""
echo "Configuration files created:"
echo "  - config_sol.py (Sol-specific settings)"
echo "  - quick_test.sh (job submission script)"
echo ""
echo "Scratch directories created:"
echo "  - /scratch/$USER/gpu_mentor_chromadb"
echo "  - /scratch/$USER/gpu_mentor_temp" 
echo "  - /scratch/$USER/gpu_mentor_logs"
echo ""
echo "=================================================="
