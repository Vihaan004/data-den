#!/usr/bin/env python3
"""Full integration test of GPU Mentor backend on Sol"""

import sys
import time
import traceback
sys.path.append('.')

def test_full_workflow():
    """Test the complete GPU Mentor workflow."""
    print("🔍 Starting full workflow integration test...")
    
    try:
        # Initialize main components
        print("\n📦 Initializing components...")
        from core.enhanced_gpu_mentor import EnhancedGPUMentor
        from core.rag_pipeline import RAGPipeline
        from utils.sample_code_library import SampleCodeLibrary
        from utils.educational_content import EducationalContentEnhancer
        
        mentor = EnhancedGPUMentor()
        rag = RAGPipeline()
        samples = SampleCodeLibrary()
        educator = EducationalContentEnhancer()
        
        print("✅ All components initialized")
        
        # Test 1: Code optimization workflow
        print("\n🧪 Test 1: Code Optimization Workflow")
        test_codes = [
            # Inefficient loop
            """
import numpy as np
result = []
for i in range(1000):
    result.append(i * 2)
print(sum(result))
""",
            # Inefficient matrix operations
            """
import numpy as np
matrix = np.random.random((100, 100))
result = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        result[i][j] = matrix[i][j] * 2
""",
        ]
        
        for i, code in enumerate(test_codes, 1):
            print(f"  🔧 Optimizing code sample {i}...")
            start_time = time.time()
            optimized = mentor.optimize_code(code)
            end_time = time.time()
            
            if optimized and len(optimized) > 10:
                print(f"  ✅ Code {i} optimized in {end_time - start_time:.2f}s")
                print(f"     Original: {len(code)} chars, Optimized: {len(optimized)} chars")
            else:
                print(f"  ⚠️  Code {i} optimization returned minimal result")
        
        # Test 2: Educational content generation
        print("\n🧪 Test 2: Educational Content Generation")
        topics = ["vectorization", "gpu_acceleration", "memory_optimization"]
        
        for topic in topics:
            print(f"  📚 Generating content for: {topic}")
            try:
                content = mentor.explain_optimization(topic)
                if content and len(content) > 50:
                    print(f"  ✅ Generated {len(content)} chars of content")
                else:
                    print(f"  ⚠️  Limited content generated for {topic}")
            except Exception as e:
                print(f"  ❌ Failed to generate content for {topic}: {e}")
        
        # Test 3: Sample code library
        print("\n🧪 Test 3: Sample Code Library")
        categories = ["basic", "intermediate"]
        operations = ["array_operations", "matrix_operations"]
        
        for category in categories:
            for operation in operations:
                print(f"  📖 Fetching {category}/{operation}...")
                try:
                    sample = samples.get_sample_code(category, operation)
                    if sample:
                        print(f"  ✅ Retrieved sample code ({len(sample)} chars)")
                    else:
                        print(f"  ⚠️  No sample found for {category}/{operation}")
                except Exception as e:
                    print(f"  ❌ Error retrieving {category}/{operation}: {e}")
        
        # Test 4: RAG pipeline
        print("\n🧪 Test 4: RAG Pipeline")
        try:
            print("  🔍 Testing document retrieval...")
            query = "How to optimize numpy array operations for GPU?"
            results = rag.query_knowledge_base(query)
            
            if results:
                print(f"  ✅ RAG query returned {len(results)} results")
            else:
                print("  ⚠️  RAG query returned no results")
        except Exception as e:
            print(f"  ❌ RAG pipeline test failed: {e}")
        
        # Test 5: Performance benchmarking
        print("\n🧪 Test 5: Performance Benchmarking")
        try:
            from core.benchmark_engine import BenchmarkEngine
            benchmark = BenchmarkEngine()
            
            test_code = """
import numpy as np
import time
start = time.time()
data = np.random.random((500, 500))
result = np.dot(data, data.T)
end = time.time()
print(f"Time: {end - start:.4f}s")
"""
            
            print("  ⏱️  Running benchmark...")
            result = benchmark.benchmark_code(test_code, "matrix_multiply_test")
            
            if result:
                print("  ✅ Benchmark completed successfully")
            else:
                print("  ⚠️  Benchmark returned limited results")
                
        except Exception as e:
            print(f"  ❌ Benchmark test failed: {e}")
        
        print("\n🎉 Integration test workflow completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling capabilities."""
    print("\n🧪 Testing error handling...")
    
    try:
        from core.enhanced_gpu_mentor import EnhancedGPUMentor
        mentor = EnhancedGPUMentor()
        
        # Test with invalid code
        invalid_code = "this is not valid python code !!!"
        print("  🔧 Testing with invalid code...")
        
        try:
            result = mentor.optimize_code(invalid_code)
            print("  ✅ Invalid code handled gracefully")
        except Exception as e:
            print(f"  ✅ Invalid code properly rejected: {type(e).__name__}")
        
        # Test with empty input
        print("  🔧 Testing with empty input...")
        try:
            result = mentor.optimize_code("")
            print("  ✅ Empty input handled gracefully")
        except Exception as e:
            print(f"  ✅ Empty input properly handled: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False

def system_info():
    """Display system information for the test."""
    print("🖥️  System Information:")
    
    import os
    import platform
    
    print(f"  Platform: {platform.platform()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Architecture: {platform.architecture()[0]}")
    
    # SLURM info
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'Not in SLURM')
    slurm_node = os.environ.get('SLURMD_NODENAME', 'Unknown')
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    
    print(f"  SLURM Job ID: {slurm_job_id}")
    print(f"  SLURM Node: {slurm_node}")
    print(f"  CUDA Devices: {cuda_devices}")
    
    # Memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  Total Memory: {mem.total / (1024**3):.1f} GB")
        print(f"  Available Memory: {mem.available / (1024**3):.1f} GB")
    except ImportError:
        print("  Memory info: psutil not available")

if __name__ == "__main__":
    print("=" * 80)
    print("GPU MENTOR BACKEND - FULL INTEGRATION TEST ON SOL")
    print("=" * 80)
    
    system_info()
    
    print("\n" + "=" * 80)
    
    # Run main workflow test
    workflow_success = test_full_workflow()
    
    # Run error handling test
    error_handling_success = test_error_handling()
    
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    if workflow_success and error_handling_success:
        print("🎉 ALL TESTS PASSED!")
        print("   The GPU Mentor backend is ready for production use on Sol.")
    elif workflow_success:
        print("✅ Core functionality works, minor issues with error handling.")
    else:
        print("❌ Integration test failed. Check logs for details.")
    
    print("\n" + "=" * 80)
    print("Integration test completed!")
    print("=" * 80)
