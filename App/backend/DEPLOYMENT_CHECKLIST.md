# GPU Mentor Backend - Sol Deployment Checklist

## Pre-Deployment Checklist (Local)

### ✅ File Structure Verification
- [ ] All core modules present in `core/` directory
  - [ ] `rag_pipeline.py`
  - [ ] `code_optimizer.py`
  - [ ] `benchmark_engine.py`
  - [ ] `sol_executor.py`
  - [ ] `enhanced_gpu_mentor.py`
- [ ] All utility modules present in `utils/` directory
  - [ ] `educational_content.py`
  - [ ] `performance_visualizer.py`
  - [ ] `sample_code_library.py`
  - [ ] `__init__.py`
- [ ] Support files present
  - [ ] `requirements.txt`
  - [ ] `config.py`
  - [ ] `models/api_models.py`
  - [ ] `README.md`

### ✅ Test Files Created
- [ ] `test_imports.py` - Import verification
- [ ] `test_components.py` - Component initialization
- [ ] `integration_test.py` - Full workflow testing
- [ ] `test_job.slurm` - SLURM job script
- [ ] `setup_sol.sh` - Setup automation script
- [ ] `SOL_TESTING_GUIDE.md` - Comprehensive guide

## Sol Deployment Steps

### Step 1: Upload Backend to Sol
```bash
# Option A: SCP upload
scp -r App/backend/ username@sol.asu.edu:~/gpu-mentor-backend/

# Option B: Git clone (if repository is set up)
ssh username@sol.asu.edu
git clone <repository-url> ~/gpu-mentor-backend/
```

### Step 2: Initial Setup on Sol
```bash
# Navigate to backend directory
cd ~/gpu-mentor-backend/App/backend/

# Run setup script
chmod +x setup_sol.sh
./setup_sol.sh
```

### Step 3: Verify Environment
```bash
# Load required kernel
module load genai25.06

# Check Python and environment
python --version
which python
echo $PYTHONPATH
```

### Step 4: Run Basic Tests
```bash
# Test imports
python test_imports.py

# Test component initialization
python test_components.py

# Quick functionality test
python -c "
from core.enhanced_gpu_mentor import EnhancedGPUMentor
mentor = EnhancedGPUMentor()
print('✅ Basic functionality works')
"
```

### Step 5: Submit SLURM Test Job
```bash
# Submit test job
sbatch test_job.slurm

# Monitor job
squeue -u $USER

# Check results
tail -f gpu_mentor_test_*.out
tail -f gpu_mentor_test_*.err
```

### Step 6: Interactive Testing
```bash
# Request interactive GPU node
srun --partition=gpu --gres=gpu:1 --mem=16G --time=02:00:00 --pty bash

# Load environment and test
module load genai25.06
cd ~/gpu-mentor-backend/App/backend/
python integration_test.py
```

## Post-Deployment Verification

### ✅ Core Functionality Tests
- [ ] All modules import successfully
- [ ] All components initialize without errors
- [ ] Code optimization produces valid results
- [ ] Educational content generation works
- [ ] Sample code library functions correctly
- [ ] RAG pipeline retrieves relevant information
- [ ] Benchmark engine measures performance

### ✅ Sol-Specific Features
- [ ] SLURM job submission works
- [ ] GPU access is available when requested
- [ ] Scratch space directories are created
- [ ] Module loading (genai25.06) works correctly
- [ ] Performance monitoring functions

### ✅ Integration Tests
- [ ] Full workflow test passes
- [ ] Error handling works appropriately
- [ ] Memory usage is reasonable
- [ ] Execution time is acceptable
- [ ] No critical errors in logs

## Troubleshooting Common Issues

### Import Errors
```bash
# Check Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Verify module structure
ls -la core/
ls -la utils/

# Test individual imports
python -c "from core.enhanced_gpu_mentor import EnhancedGPUMentor"
```

### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt --user --force-reinstall

# Check specific packages
python -c "import ollama, chromadb, sentence_transformers"
```

### SLURM Job Issues
```bash
# Check job status
squeue -u $USER
sacct -u $USER

# Review job output
cat gpu_mentor_test_*.out
cat gpu_mentor_test_*.err

# Check resource limits
sinfo -p gpu
```

### GPU Access Issues
```bash
# Verify GPU allocation
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

# Test GPU in Python
python -c "
try:
    import torch
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name()}')
except ImportError:
    print('PyTorch not available')
"
```

## Performance Benchmarks

### Expected Performance Metrics
- Import time: < 5 seconds
- Component initialization: < 10 seconds
- Code optimization: < 30 seconds for simple code
- Educational content generation: < 15 seconds
- Memory usage: < 2GB for basic operations

### Performance Monitoring
```bash
# Monitor during testing
python -c "
import psutil
import time

print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
"
```

## Success Criteria

The deployment is successful when:
- [ ] All test scripts complete without critical errors
- [ ] SLURM jobs run successfully on GPU nodes
- [ ] Core optimization functionality produces improved code
- [ ] Educational features generate appropriate content
- [ ] Performance is within acceptable ranges
- [ ] Error handling prevents system crashes

## Next Steps After Successful Deployment

1. **API Development**: Create REST API endpoints for web interface
2. **Frontend Integration**: Connect to React/Vue.js frontend
3. **Job Queue System**: Implement advanced job management
4. **Monitoring Setup**: Add logging and performance monitoring
5. **Production Configuration**: Optimize for high-load scenarios

## Support and Documentation

- **Sol Documentation**: Check ASU Research Computing guides
- **Module Information**: `module avail` and `module show genai25.06`
- **SLURM Help**: `man sbatch`, `man srun`
- **Debug Mode**: Set `GPU_MENTOR_DEBUG=true` for verbose output

---

**Date Created**: June 26, 2025  
**Last Updated**: June 26, 2025  
**Version**: 1.0  
**Target Platform**: Sol Supercomputer (ASU Research Computing)  
**Kernel**: genai25.06
