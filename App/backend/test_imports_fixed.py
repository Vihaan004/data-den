#!/usr/bin/env python3
"""Final import verification test for Sol"""

import sys
import os
sys.path.append('.')

def test_fixed_imports():
    """Test all imports after fixes."""
    print("🔍 Testing final fixed imports on Sol...")
    
    imports_to_test = [
        ('core.rag_pipeline', 'RAGPipeline'),
        ('core.code_optimizer', 'CodeOptimizer'),
        ('core.benchmark_engine', 'BenchmarkEngine'),
        ('core.sol_executor', 'SolCodeExecutor'),
        ('core.enhanced_gpu_mentor', 'EnhancedGPUMentor'),
        ('utils.educational_content', 'EducationalContentEnhancer'),
        ('utils.performance_visualizer', 'PerformanceVisualizer'),
        ('utils.sample_code_library', 'SampleCodeLibrary'),
        ('models.api_models', None),
        ('config', 'settings'),
    ]
    
    success_count = 0
    total_count = len(imports_to_test)
    
    for module_name, class_or_attr in imports_to_test:
        try:
            if class_or_attr:
                module = __import__(module_name, fromlist=[class_or_attr])
                getattr(module, class_or_attr)
                print(f"✅ {module_name}.{class_or_attr}")
            else:
                __import__(module_name)
                print(f"✅ {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module_name}: Import Error - {e}")
        except AttributeError as e:
            print(f"❌ {module_name}.{class_or_attr}: Not Found - {e}")
        except Exception as e:
            print(f"❌ {module_name}: Error - {e}")
    
    print(f"\n📊 Final Import Test Results: {success_count}/{total_count} successful")
    
    if success_count == total_count:
        print("🎉 ALL IMPORTS SUCCESSFUL! Backend ready for full testing.")
        return True
    else:
        print("⚠️  Some imports still failing.")
        return False

def test_component_creation():
    """Test that components can be created."""
    print("\n🧪 Testing component creation...")
    
    try:
        from core.enhanced_gpu_mentor import EnhancedGPUMentor
        mentor = EnhancedGPUMentor()
        print("✅ EnhancedGPUMentor created successfully")
        
        # Test a simple method call
        if hasattr(mentor, 'conversation_history'):
            print("✅ Component has expected attributes")
        
        return True
    except Exception as e:
        print(f"❌ Component creation failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("GPU MENTOR BACKEND - FINAL IMPORT VERIFICATION")
    print("=" * 70)
    
    import_success = test_fixed_imports()
    creation_success = test_component_creation()
    
    if import_success and creation_success:
        print("\n🎉 ALL TESTS PASSED! Ready for Sol deployment!")
    else:
        print("\n❌ Some tests failed. Check error messages above.")
    
    print("\n" + "=" * 70)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def test_fixed_imports():
    """Test imports with fixed relative import issues."""
    print("🔍 Testing fixed imports on Sol...")
    
    imports_to_test = [
        ('core.rag_pipeline', 'RAGPipeline'),
        ('core.code_optimizer', 'CodeOptimizer'),
        ('core.benchmark_engine', 'BenchmarkEngine'),
        ('core.sol_executor', 'SolCodeExecutor'),
        ('core.enhanced_gpu_mentor', 'EnhancedGPUMentor'),
        ('utils.educational_content', 'EducationalContentEnhancer'),
        ('utils.performance_visualizer', 'PerformanceVisualizer'),
        ('utils.sample_code_library', 'SampleCodeLibrary'),
        ('models.api_models', None),
        ('config', 'settings'),  # Test config import
    ]
    
    success_count = 0
    total_count = len(imports_to_test)
    
    for module_name, class_name in imports_to_test:
        try:
            if module_name == 'config' and class_name == 'settings':
                # Special case for settings
                import config
                getattr(config, 'settings')
                print(f"✅ {module_name}.{class_name}")
            else:
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
    
    print(f"\n📊 Fixed Import Test Results: {success_count}/{total_count} successful")
    
    return success_count == total_count

def test_component_initialization():
    """Test component initialization with fixed imports."""
    print("\n🧪 Testing component initialization with fixed imports...")
    
    try:
        # Test enhanced GPU mentor initialization
        from core.enhanced_gpu_mentor import EnhancedGPUMentor
        mentor = EnhancedGPUMentor()
        print("✅ EnhancedGPUMentor initialized successfully")
        
        # Test a simple optimization
        test_code = "x = [i*2 for i in range(100)]"
        result = mentor.optimize_code(test_code)
        if result:
            print("✅ Code optimization test completed")
        else:
            print("⚠️ Code optimization returned empty result")
            
        return True
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_sol_environment():
    """Check Sol-specific environment."""
    print("\n🔍 Checking Sol environment...")
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # SLURM info
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'Not in SLURM')
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"SLURM Job ID: {slurm_job_id}")
    print(f"CUDA Devices: {cuda_devices}")
    
    # GPU check
    try:
        import torch
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("PyTorch not available for GPU check")
    
    # Working directory
    print(f"Working directory: {os.getcwd()}")
    print(f"Backend directory in path: {backend_dir in sys.path}")

if __name__ == "__main__":
    print("=" * 70)
    print("GPU MENTOR BACKEND - FIXED SOL IMPORT TEST")
    print("=" * 70)
    
    check_sol_environment()
    
    import_success = test_fixed_imports()
    
    if import_success:
        print("\n🎉 All imports successful! Testing component initialization...")
        init_success = test_component_initialization()
        
        if init_success:
            print("\n🎉 All tests passed! Backend is ready for Sol deployment.")
        else:
            print("\n⚠️ Component initialization had issues, but imports work.")
    else:
        print("\n❌ Some imports still failing. Check dependencies.")
    
    print("\n" + "=" * 70)
    print("Fixed import test completed!")
    print("=" * 70)
