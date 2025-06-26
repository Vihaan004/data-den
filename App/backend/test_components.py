#!/usr/bin/env python3
"""Test component initialization on Sol"""

import sys
import traceback
sys.path.append('.')

def test_component_init():
    """Test initialization of all GPU Mentor components."""
    print("🔍 Testing component initialization...")
    
    components_to_test = [
        ('core.rag_pipeline', 'RAGPipeline'),
        ('core.code_optimizer', 'CodeOptimizer'),
        ('core.benchmark_engine', 'BenchmarkEngine'),
        ('core.sol_executor', 'SolCodeExecutor'),
        ('core.enhanced_gpu_mentor', 'EnhancedGPUMentor'),
        ('utils.educational_content', 'EducationalContentEnhancer'),
        ('utils.performance_visualizer', 'PerformanceVisualizer'),
        ('utils.sample_code_library', 'SampleCodeLibrary'),
    ]
    
    success_count = 0
    total_count = len(components_to_test)
    
    for module_name, class_name in components_to_test:
        try:
            print(f"\n🧪 Testing {class_name}...")
            module = __import__(module_name, fromlist=[class_name])
            component_class = getattr(module, class_name)
            
            # Try to initialize with default parameters
            instance = component_class()
            print(f"✅ {class_name} initialized successfully")
            
            # Test basic method existence
            if hasattr(instance, '__dict__'):
                attrs = len(instance.__dict__)
                print(f"   📝 Component has {attrs} attributes")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ {class_name} initialization failed: {e}")
            print(f"   📋 Error details:")
            traceback.print_exc()
    
    print(f"\n📊 Component Initialization Results: {success_count}/{total_count} successful")
    
    if success_count == total_count:
        print("🎉 All components initialized successfully!")
        return True
    else:
        print("⚠️  Some components failed initialization. Check dependencies.")
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\n🔍 Testing basic functionality...")
    
    try:
        # Test Enhanced GPU Mentor with a simple task
        from core.enhanced_gpu_mentor import EnhancedGPUMentor
        mentor = EnhancedGPUMentor()
        
        # Simple optimization test
        test_code = """
import numpy as np
x = []
for i in range(1000):
    x.append(i * 2)
"""
        
        print("🧪 Testing code optimization...")
        result = mentor.optimize_code(test_code)
        if result:
            print("✅ Code optimization test passed")
        else:
            print("⚠️  Code optimization returned empty result")
            
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        # Test Sample Code Library
        from utils.sample_code_library import SampleCodeLibrary
        library = SampleCodeLibrary()
        
        print("🧪 Testing sample code library...")
        sample = library.get_sample_code("basic", "array_operations")
        if sample:
            print("✅ Sample code library test passed")
        else:
            print("⚠️  Sample code library returned empty result")
            
    except Exception as e:
        print(f"❌ Sample code library test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("GPU MENTOR BACKEND - SOL COMPONENT INITIALIZATION TEST")
    print("=" * 70)
    
    init_success = test_component_init()
    
    if init_success:
        func_success = test_basic_functionality()
        
        if func_success:
            print("\n🎉 All tests passed! Backend is ready for full testing.")
        else:
            print("\n⚠️  Some functionality tests failed.")
    else:
        print("\n❌ Component initialization failed. Check setup.")
    
    print("\n" + "=" * 70)
    print("Component test completed!")
    print("=" * 70)
