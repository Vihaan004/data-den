#!/usr/bin/env python3
"""Final import verification test for Sol"""

import sys
import os
sys.path.append('.')

def test_final_imports():
    """Test all imports after all fixes."""
    print("🔍 Testing final imports on Sol...")
    
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
        print("🎉 ALL IMPORTS SUCCESSFUL!")
        return True
    else:
        print("⚠️  Some imports still failing.")
        return False

def test_component_creation():
    """Test component creation."""
    print("\n🧪 Testing component creation...")
    
    try:
        from core.enhanced_gpu_mentor import EnhancedGPUMentor
        mentor = EnhancedGPUMentor()
        print("✅ EnhancedGPUMentor created successfully")
        
        if hasattr(mentor, 'conversation_history'):
            print("✅ Component has expected attributes")
        
        return True
    except Exception as e:
        print(f"❌ Component creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("GPU MENTOR BACKEND - FINAL VERIFICATION")
    print("=" * 70)
    
    # Environment info
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests
    import_success = test_final_imports()
    creation_success = test_component_creation()
    
    print("\n" + "=" * 70)
    if import_success and creation_success:
        print("🎉 ALL TESTS PASSED! Backend is ready for Sol!")
    else:
        print("❌ Some tests failed. Check error messages above.")
    print("=" * 70)
