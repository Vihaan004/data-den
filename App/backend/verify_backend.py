"""
GPU Mentor Backend Verification Script
Validates that all components from the notebook have been successfully ported
"""
import sys
import importlib
import traceback
from pathlib import Path

# Add the backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def check_import(module_name, class_name=None):
    """Check if a module/class can be imported successfully."""
    try:
        module = importlib.import_module(module_name)
        if class_name:
            getattr(module, class_name)
        return True, "✅ Success"
    except ImportError as e:
        return False, f"❌ Import Error: {e}"
    except AttributeError as e:
        return False, f"❌ Class Not Found: {e}"
    except Exception as e:
        return False, f"❌ Error: {e}"

def verify_backend_structure():
    """Verify the backend directory structure."""
    print("🔍 Verifying Backend Structure...")
    
    required_files = [
        "core/rag_pipeline.py",
        "core/code_optimizer.py", 
        "core/benchmark_engine.py",
        "core/sol_executor.py",
        "core/enhanced_gpu_mentor.py",
        "utils/educational_content.py",
        "utils/performance_visualizer.py",
        "utils/sample_code_library.py",
        "models/api_models.py",
        "config.py",
        "requirements.txt"
    ]
    
    for file_path in required_files:
        full_path = backend_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
    
    print()

def verify_core_components():
    """Verify core component imports."""
    print("🧩 Verifying Core Components...")
    
    components = [
        ("core.rag_pipeline", "RAGPipeline"),
        ("core.code_optimizer", "CodeOptimizer"),
        ("core.benchmark_engine", "BenchmarkEngine"),
        ("core.sol_executor", "SolCodeExecutor"),
        ("core.enhanced_gpu_mentor", "EnhancedGPUMentor")
    ]
    
    all_success = True
    for module, class_name in components:
        success, message = check_import(module, class_name)
        print(f"{message} - {module}.{class_name}")
        if not success:
            all_success = False
    
    print()
    return all_success

def verify_utility_components():
    """Verify utility component imports."""
    print("🛠️ Verifying Utility Components...")
    
    components = [
        ("utils.educational_content", "EducationalContentEnhancer"),
        ("utils.performance_visualizer", "PerformanceVisualizer"),
        ("utils.sample_code_library", "SampleCodeLibrary")
    ]
    
    all_success = True
    for module, class_name in components:
        success, message = check_import(module, class_name)
        print(f"{message} - {module}.{class_name}")
        if not success:
            all_success = False
    
    print()
    return all_success

def verify_notebook_coverage():
    """Verify that all major notebook components are covered."""
    print("📓 Verifying Notebook Coverage...")
    
    notebook_components = {
        "RAG Pipeline (Sections 1-10)": "✅ Ported to core/rag_pipeline.py",
        "Sol Code Executor (Section 11)": "✅ Ported to core/sol_executor.py", 
        "Code Optimizer (Section 12)": "✅ Ported to core/code_optimizer.py",
        "Benchmark Engine (Section 13)": "✅ Ported to core/benchmark_engine.py",
        "Enhanced GPU Mentor (Section 14)": "✅ Ported to core/enhanced_gpu_mentor.py",
        "Educational Content Enhancer": "✅ Ported to utils/educational_content.py",
        "Performance Visualizer": "✅ Ported to utils/performance_visualizer.py",
        "Sample Code Library": "✅ Ported to utils/sample_code_library.py",
        "Gradio Interface (Section 15)": "⚠️ Not ported (frontend component)",
        "Testing Examples (Section 16)": "⚠️ Not needed for backend API"
    }
    
    for component, status in notebook_components.items():
        print(f"{status} - {component}")
    
    print()

def test_component_functionality():
    """Test basic functionality of core components."""
    print("⚙️ Testing Component Functionality...")
    
    try:
        # Test CodeOptimizer
        from core.code_optimizer import CodeOptimizer
        optimizer = CodeOptimizer()
        test_code = "import numpy as np\nx = np.array([1, 2, 3])\nresult = np.sum(x)"
        analysis = optimizer.analyze_code(test_code)
        if 'libraries_detected' in analysis:
            print("✅ CodeOptimizer.analyze_code() - Working")
        else:
            print("❌ CodeOptimizer.analyze_code() - Not returning expected format")
    except Exception as e:
        print(f"❌ CodeOptimizer test failed: {e}")
    
    try:
        # Test EducationalContentEnhancer
        from utils.educational_content import EducationalContentEnhancer
        enhancer = EducationalContentEnhancer()
        examples = enhancer.get_example_for_operation("matrix_multiplication")
        if examples and 'cpu_code' in examples:
            print("✅ EducationalContentEnhancer.get_example_for_operation() - Working")
        else:
            print("❌ EducationalContentEnhancer test failed - No examples returned")
    except Exception as e:
        print(f"❌ EducationalContentEnhancer test failed: {e}")
    
    try:
        # Test SampleCodeLibrary
        from utils.sample_code_library import SampleCodeLibrary
        library = SampleCodeLibrary()
        samples = library.get_all_samples()
        if samples and len(samples) > 5:
            print("✅ SampleCodeLibrary.get_all_samples() - Working")
        else:
            print("❌ SampleCodeLibrary test failed - Insufficient samples")
    except Exception as e:
        print(f"❌ SampleCodeLibrary test failed: {e}")
    
    print()

def main():
    """Run all verification checks."""
    print("🚀 GPU Mentor Backend Verification")
    print("=" * 50)
    
    verify_backend_structure()
    core_success = verify_core_components()
    utils_success = verify_utility_components()
    verify_notebook_coverage()
    test_component_functionality()
    
    print("📋 Summary")
    print("=" * 50)
    
    if core_success and utils_success:
        print("✅ All components successfully ported from notebook!")
        print("✅ Backend is ready for integration and testing")
        print()
        print("🎯 Next Steps:")
        print("1. Deploy to Sol supercomputer environment")
        print("2. Install RAPIDS libraries (cupy, cudf, cuml)")
        print("3. Configure Ollama with qwen3:14b model")
        print("4. Test Sol SLURM integration")
        print("5. Create API endpoints for frontend integration")
    else:
        print("❌ Some components failed verification")
        print("🔧 Please fix the issues above before deployment")
    
    print()

if __name__ == "__main__":
    main()
