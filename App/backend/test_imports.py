#!/usr/bin/env python3
"""Test all imports on Sol"""

import sys
import os
sys.path.append('.')

def test_imports():
    """Test all critical imports for GPU Mentor backend."""
    print("🔍 Testing imports on Sol...")
    
    imports_to_test = [
        ('core.rag_pipeline', 'RAGPipeline'),
        ('core.code_optimizer', 'CodeOptimizer'),
        ('core.benchmark_engine', 'BenchmarkEngine'),
        ('core.sol_executor', 'SolCodeExecutor'),
        ('core.enhanced_gpu_mentor', 'EnhancedGPUMentor'),
        ('utils.educational_content', 'EducationalContentEnhancer'),
        ('utils.performance_visualizer', 'PerformanceVisualizer'),
        ('utils.sample_code_library', 'SampleCodeLibrary'),
        ('models.api_models', None),  # Just test module import
    ]
    
    success_count = 0
    total_count = len(imports_to_test)
    
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name] if class_name else [])
            if class_name:
                getattr(module, class_name)
                print(f"✅ {module_name}.{class_name}")
            else:
                print(f"✅ {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module_name}: Import Error - {e}")
        except AttributeError as e:
            print(f"❌ {module_name}.{class_name}: Class Not Found - {e}")
        except Exception as e:
            print(f"❌ {module_name}: Unexpected Error - {e}")
    
    print(f"\n📊 Import Test Results: {success_count}/{total_count} successful")
    
    if success_count == total_count:
        print("🎉 All imports successful! Backend is ready for testing.")
        return True
    else:
        print("⚠️  Some imports failed. Check dependencies and Python path.")
        return False

def check_environment():
    """Check the Sol environment setup."""
    print("\n🔍 Checking Sol environment...")
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # Working directory
    print(f"Working directory: {os.getcwd()}")
    
    # Python path
    print(f"Python path includes: {sys.path[:3]}...")
    
    # Check for GPU
    try:
        import torch
        print(f"PyTorch available: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("PyTorch not available")
    
    # Check environment variables
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_devices}")
    
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'Not running in SLURM')
    print(f"SLURM Job ID: {slurm_job_id}")

if __name__ == "__main__":
    print("=" * 60)
    print("GPU MENTOR BACKEND - SOL IMPORT TEST")
    print("=" * 60)
    
    check_environment()
    test_imports()
    
    print("\n" + "=" * 60)
    print("Import test completed!")
    print("=" * 60)
