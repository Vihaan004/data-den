#!/usr/bin/env python3
"""
Sol Environment Information Extractor
This script gathers comprehensive information about the Sol GPU environment
for optimizing LLM prompts and code generation.
"""

import sys
import os
import platform
import subprocess
import pandas as pd
import numpy as np

print("="*60)
print("SOL ENVIRONMENT INFORMATION EXTRACTOR")
print("="*60)

# 1. System Information
print("\n1. SYSTEM INFORMATION")
print("-" * 30)
print(f"Python Version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")
print(f"Machine: {platform.machine()}")
print(f"Node: {platform.node()}")

# 2. GPU Information
print("\n2. GPU INFORMATION")
print("-" * 30)
try:
    gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits'], 
                             capture_output=True, text=True)
    print("GPU Details:")
    print(gpu_info.stdout)
except Exception as e:
    print(f"Error getting GPU info: {e}")

# 3. CUDA Information
print("\n3. CUDA INFORMATION")
print("-" * 30)
try:
    cuda_version = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    print("CUDA Version:")
    print(cuda_version.stdout)
except Exception as e:
    print(f"Error getting CUDA info: {e}")

# 4. Python Package Versions
print("\n4. PYTHON PACKAGE VERSIONS")
print("-" * 30)

packages_to_check = [
    'pandas', 'numpy', 'matplotlib', 'sklearn', 'scipy',
    'cudf', 'cuml', 'cupy', 'rmm', 'dask', 'dask-cudf',
    'numba', 'xgboost', 'lightgbm', 'seaborn', 'plotly'
]

for package in packages_to_check:
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'Unknown')
        print(f"{package}: {version}")
    except ImportError:
        print(f"{package}: Not installed")
    except Exception as e:
        print(f"{package}: Error - {e}")

# 5. RAPIDS Environment
print("\n5. RAPIDS ENVIRONMENT")
print("-" * 30)
try:
    import cudf
    import cuml
    import cupy
    
    print(f"cuDF version: {cudf.__version__}")
    print(f"cuML version: {cuml.__version__}")
    print(f"CuPy version: {cupy.__version__}")
    
    # Test basic cuDF operations
    print("\nTesting cuDF operations...")
    df = pd.DataFrame({
        'numeric1': [1, 2, 3, 4, 5],
        'numeric2': [10, 20, 30, 40, 50],
        'string_col': ['a', 'b', 'c', 'd', 'e'],
        'mixed_col': [1, 'text', 3.14, None, 'end']
    })
    
    try:
        gdf = cudf.from_pandas(df)
        print("✓ cuDF DataFrame creation: SUCCESS")
        print(f"  Data types: {gdf.dtypes.to_dict()}")
        
        # Test numeric operations
        numeric_gdf = gdf.select_dtypes(include=['number'])
        print(f"  Numeric columns: {numeric_gdf.columns.tolist()}")
        
        # Test correlation
        corr_matrix = numeric_gdf.corr()
        print("✓ Correlation on numeric columns: SUCCESS")
        
        # Test full DataFrame correlation (should fail)
        try:
            full_corr = gdf.corr()
            print("✗ Full DataFrame correlation: UNEXPECTED SUCCESS")
        except Exception as e:
            print(f"✓ Full DataFrame correlation fails as expected: {type(e).__name__}")
            
        # Test plotting data conversion
        x_numpy = gdf['numeric1'].to_numpy()
        y_numpy = gdf['numeric2'].to_numpy()
        print("✓ cuDF to numpy conversion: SUCCESS")
        
        # Test cuML data conversion
        x_cupy = gdf['numeric1'].to_cupy()
        y_cupy = gdf['numeric2'].to_cupy()
        print("✓ cuDF to cupy conversion: SUCCESS")
        
    except Exception as e:
        print(f"✗ cuDF operations failed: {e}")
        
except ImportError as e:
    print(f"RAPIDS not available: {e}")

# 6. Machine Learning Libraries
print("\n6. MACHINE LEARNING LIBRARIES")
print("-" * 30)

try:
    from cuml.linear_model import LinearRegression
    from cuml.cluster import KMeans
    from cuml.decomposition import PCA
    from cuml.metrics import r2_score
    print("✓ cuML imports: SUCCESS")
    
    # Test basic cuML operations
    try:
        import cupy as cp
        X = cp.random.rand(100, 5)
        y = cp.random.rand(100)
        
        # Test Linear Regression
        lr = LinearRegression()
        lr.fit(X, y)
        pred = lr.predict(X)
        r2 = r2_score(y, pred)
        print(f"✓ cuML Linear Regression: SUCCESS (R2: {r2:.4f})")
        
        # Test KMeans
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        print("✓ cuML KMeans: SUCCESS")
        
        # Test PCA
        pca = PCA(n_components=3)
        pca.fit(X)
        transformed = pca.transform(X)
        print("✓ cuML PCA: SUCCESS")
        
    except Exception as e:
        print(f"✗ cuML operations failed: {e}")
        
except ImportError as e:
    print(f"cuML not available: {e}")

# 7. Memory and Performance Constraints
print("\n7. MEMORY AND PERFORMANCE")
print("-" * 30)

try:
    import cupy as cp
    mempool = cp.get_default_memory_pool()
    print(f"GPU Memory Pool - Used: {mempool.used_bytes() / 1024**3:.2f} GB")
    print(f"GPU Memory Pool - Total: {mempool.total_bytes() / 1024**3:.2f} GB")
    
    # Test large array creation
    try:
        large_array = cp.random.rand(10000, 100)
        print("✓ Large array creation (10K x 100): SUCCESS")
        del large_array
    except Exception as e:
        print(f"✗ Large array creation failed: {e}")
        
except ImportError:
    print("CuPy not available for memory testing")

# 8. Common Error Patterns
print("\n8. COMMON ERROR PATTERNS")
print("-" * 30)

error_tests = [
    "DataFrame with mixed types correlation",
    "matplotlib with cuDF Series",
    "cuML.DataFrame usage",
    "Large dataset memory issues"
]

try:
    import cudf
    import matplotlib.pyplot as plt
    
    # Test mixed types correlation
    df_mixed = pd.DataFrame({
        'num1': [1, 2, 3],
        'num2': [4, 5, 6], 
        'str_col': ['a', 'b', 'c']
    })
    gdf_mixed = cudf.from_pandas(df_mixed)
    
    try:
        corr_mixed = gdf_mixed.corr()
        print("✗ Mixed types correlation: UNEXPECTED SUCCESS")
    except Exception as e:
        print(f"✓ Mixed types correlation fails: {type(e).__name__}")
    
    # Test matplotlib with cuDF
    try:
        plt.figure()
        plt.scatter(gdf_mixed['num1'], gdf_mixed['num2'])  # Should fail
        print("✗ matplotlib with cuDF: UNEXPECTED SUCCESS")
        plt.close()
    except Exception as e:
        print(f"✓ matplotlib with cuDF fails: {type(e).__name__}")
        
    # Test cuML.DataFrame
    try:
        import cuml
        fake_df = cuml.DataFrame(gdf_mixed[['num1', 'num2']])
        print("✗ cuML.DataFrame: UNEXPECTED SUCCESS")
    except AttributeError:
        print("✓ cuML.DataFrame not available (expected)")
    except Exception as e:
        print(f"✓ cuML.DataFrame fails: {type(e).__name__}")
        
except Exception as e:
    print(f"Error testing patterns: {e}")

# 9. Recommended Workflows
print("\n9. RECOMMENDED WORKFLOWS")
print("-" * 30)

print("Based on testing, recommended patterns:")
print("1. Always use gdf.select_dtypes(include=['number']) before correlations")
print("2. Convert cuDF to numpy for matplotlib: gdf['col'].to_numpy()")
print("3. Convert cuDF to cupy for cuML: gdf['col'].to_cupy()")
print("4. Import specific cuML algorithms, not cuML.DataFrame")
print("5. Handle string/categorical columns separately from numeric GPU operations")

# 10. Environment Variables and Modules
print("\n10. ENVIRONMENT VARIABLES")
print("-" * 30)

env_vars = ['CUDA_VISIBLE_DEVICES', 'RAPIDS_NO_INITIALIZE', 'CUPY_CACHE_DIR', 
           'NUMBA_CUDA_DRIVER', 'PYTHONPATH', 'LD_LIBRARY_PATH']

for var in env_vars:
    value = os.environ.get(var, 'Not set')
    print(f"{var}: {value}")

print("\n" + "="*60)
print("ENVIRONMENT EXTRACTION COMPLETE")
print("="*60)
