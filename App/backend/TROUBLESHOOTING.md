# GPU Mentor Troubleshooting Guide

## 🚨 Common Errors and Solutions

### 1. "RAG pipeline not initialized. Call initialize() first."

**Problem**: The backend RAG pipeline needs initialization before use.

**Solutions**:
```bash
# Option A: Run initialization script
python initialize_backend.py

# Option B: Use quick start menu
./quick_start.sh
# Select option 1: "Initialize Backend Only"

# Option C: Use Gradio UI button
# Start UI and click "🔄 Initialize Backend" button
```

### 2. "Connection refused" / Ollama not responding

**Problem**: Ollama service is not running or model is not available.

**Solutions**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
chmod +x start_ollama.sh
./start_ollama.sh

# Or manually:
ollama serve &
ollama pull qwen2:14b
```

### 3. Gradio ASGI Application Error

**Problem**: Gradio streaming response issues, often related to async/sync handling.

**Solutions**:
```bash
# Restart the Gradio UI
pkill -f gradio_ui.py
python gradio_ui.py

# Or use different launch method
python gradio_ui.py --server-name 0.0.0.0 --server-port 7860
```

### 4. Import Errors

**Problem**: Missing dependencies or incorrect Python path.

**Solutions**:
```bash
# Check Python environment
module load genai25.06
python --version

# Install missing packages
pip install -r requirements.txt

# Set Python path
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 5. GPU Libraries Not Found

**Problem**: RAPIDS, CuPy, or other GPU libraries not installed.

**Solutions**:
```bash
# Check if GPU libraries are available
python -c "import cupy; print('CuPy OK')"
python -c "import cudf; print('cuDF OK')"
python -c "import cuml; print('cuML OK')"

# If not available, benchmarking will fall back to CPU-only
# This is expected behavior on systems without GPU support
```

## 🔧 Debugging Steps

### Step 1: Environment Check
```bash
# Check current environment
echo "Python: $(python --version)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

# Check key imports
python -c "
try:
    from core.enhanced_gpu_mentor import EnhancedGPUMentor
    print('✅ EnhancedGPUMentor import OK')
except Exception as e:
    print(f'❌ Import error: {e}')
"
```

### Step 2: Ollama Check
```bash
# Test Ollama connection
curl -s http://localhost:11434/api/tags || echo "Ollama not running"

# List available models
ollama list

# Test model
ollama run qwen2:14b "Hello, can you help with Python?"
```

### Step 3: Backend Initialization
```bash
# Test initialization
python test_e2e.py

# Manual initialization test
python -c "
import asyncio
from core.enhanced_gpu_mentor import EnhancedGPUMentor

async def test():
    mentor = EnhancedGPUMentor()
    await mentor.initialize()
    print('✅ Initialization successful')

asyncio.run(test())
"
```

### Step 4: UI Testing
```bash
# Test UI startup
python gradio_ui.py &
UI_PID=$!

# Wait and test
sleep 5
curl -s http://localhost:7860 > /dev/null && echo "✅ UI responding" || echo "❌ UI not responding"

# Clean up
kill $UI_PID
```

## 🎯 Quick Fixes

### Reset Everything
```bash
# Kill all processes
pkill -f "ollama serve"
pkill -f "gradio_ui.py"

# Restart services
./start_ollama.sh
python initialize_backend.py
python gradio_ui.py
```

### Fallback Mode
If you can't get Ollama working, the system will run in fallback mode:
- Basic responses without full AI capabilities
- Code analysis still works
- Benchmarking still works
- Educational content still available

### Sol-Specific Issues
```bash
# Check Sol modules
module list

# Reload environment
module purge
module load genai25.06

# Check GPU availability
nvidia-smi

# Check SLURM
squeue -u $USER
```

## 📋 Verification Checklist

- [ ] Python environment loaded (`module load genai25.06`)
- [ ] Working directory is correct (App/backend/)
- [ ] All imports successful (`python verify_backend.py`)
- [ ] Ollama service running (`curl http://localhost:11434/api/tags`)
- [ ] Model available (`ollama list | grep qwen2:14b`)
- [ ] Backend initialized (`python initialize_backend.py`)
- [ ] UI accessible (`curl http://localhost:7860`)

## 🆘 When All Else Fails

1. **Check the logs**: Look for error messages in the terminal
2. **Restart everything**: Kill all processes and start fresh
3. **Use fallback mode**: The system will work without Ollama, just with limited AI features
4. **Test components individually**: Use the test scripts to isolate issues
5. **Check Sol documentation**: Ensure you're using the correct modules and environment

## 📞 Getting Help

If you're still having issues:
1. Run `python test_e2e.py` and share the output
2. Check `python verify_backend.py` results
3. Share the exact error messages you're seeing
4. Mention which Sol node you're using
5. Confirm which modules you have loaded
