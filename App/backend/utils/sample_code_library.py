"""
Sample Code Library - Comprehensive collection of sample codes for testing and learning
"""
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class SampleCodeLibrary:
    """Comprehensive collection of sample codes for testing GPU acceleration."""
    
    def __init__(self):
        self.sample_codes = self._load_sample_codes()
    
    def _load_sample_codes(self):
        """Load comprehensive sample code library."""
        return {
            "Simple Array Operations": '''import numpy as np

# Create arrays
n = 50000
x = np.random.rand(n)
y = np.random.rand(n)

# Perform operations
result = np.sqrt(x**2 + y**2)
print(f"Array size: {n}")
print(f"Max result: {np.max(result):.4f}")''',

            "Matrix Multiplication": '''import numpy as np

# Create matrices
n = 500
A = np.random.rand(n, n)
B = np.random.rand(n, n)

# Matrix multiplication
C = np.dot(A, B)

print(f"Matrix size: {n}x{n}")
print(f"Result shape: {C.shape}")
print(f"Result sum: {np.sum(C):.2f}")''',
            
            "DataFrame Operations": '''import pandas as pd
import numpy as np

# Create dataset
n = 10000
df = pd.DataFrame({
    'x': np.random.randn(n),
    'y': np.random.randn(n),
    'group': np.random.choice(['A', 'B', 'C'], n)
})

# Compute statistics
result = df.groupby('group').agg({
    'x': ['mean', 'std'],
    'y': ['sum', 'count']
})

print(f"Dataset size: {len(df)} rows")
print("Grouped results:")
print(result)''',
            
            "Mathematical Functions": '''import numpy as np

# Generate data
n = 5000
x = np.linspace(0, 4*np.pi, n)
y = np.sin(x) * np.exp(-x/10)

# Compute statistics
mean_y = np.mean(y)
std_y = np.std(y)
max_y = np.max(y)

print(f"Data points: {n}")
print(f"Mean: {mean_y:.4f}")
print(f"Std: {std_y:.4f}")
print(f"Max: {max_y:.4f}")''',

            "Data Processing Loop": '''import numpy as np

# Create data
data = np.random.rand(1000, 10)
results = []

# Process data (can be vectorized)
for i in range(len(data)):
    row_sum = np.sum(data[i])
    row_mean = np.mean(data[i])
    results.append(row_sum * row_mean)

final_result = np.array(results)
print(f"Processed {len(data)} rows")
print(f"Final result shape: {final_result.shape}")
print(f"Average result: {np.mean(final_result):.4f}")''',

            "Machine Learning - K-Means": '''import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
n_samples = 1000
n_features = 20
n_centers = 5

X, _ = make_blobs(n_samples=n_samples, centers=n_centers, 
                  n_features=n_features, random_state=42)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_centers, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

print(f"Dataset: {n_samples} samples, {n_features} features")
print(f"Clusters: {n_centers}")
print(f"Inertia: {kmeans.inertia_:.2f}")''',

            "Large DataFrame GroupBy": '''import pandas as pd
import numpy as np

# Create large dataset
n = 100000
df = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
    'subcategory': np.random.choice(['X', 'Y', 'Z'], n),
    'value1': np.random.randn(n),
    'value2': np.random.exponential(2, n),
    'value3': np.random.uniform(0, 100, n)
})

# Complex aggregation
result = df.groupby(['category', 'subcategory']).agg({
    'value1': ['mean', 'std', 'count'],
    'value2': ['sum', 'max', 'min'],
    'value3': ['median', 'var']
})

print(f"Original dataset: {len(df):,} rows")
print(f"Grouped results shape: {result.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")''',

            "Signal Processing": '''import numpy as np

# Generate signal
duration = 2.0  # seconds
sample_rate = 44100  # Hz
t = np.linspace(0, duration, int(sample_rate * duration))

# Create complex signal
freq1, freq2, freq3 = 440, 880, 1320  # Hz
signal = (np.sin(2 * np.pi * freq1 * t) + 
          0.5 * np.sin(2 * np.pi * freq2 * t) + 
          0.25 * np.sin(2 * np.pi * freq3 * t))

# Add noise
noise = np.random.normal(0, 0.1, signal.shape)
noisy_signal = signal + noise

# Apply FFT
fft_result = np.fft.fft(noisy_signal)
freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)

print(f"Signal length: {len(signal):,} samples")
print(f"Duration: {duration} seconds")
print(f"Sample rate: {sample_rate:,} Hz")
print(f"FFT peaks at: {freqs[np.argsort(np.abs(fft_result))[-3:]]}")''',

            "Image Convolution": '''import numpy as np

# Create sample image
height, width = 512, 512
image = np.random.rand(height, width).astype(np.float32)

# Define convolution kernel (edge detection)
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]], dtype=np.float32)

# Manual convolution (can be optimized with GPU)
kh, kw = kernel.shape
output = np.zeros_like(image)

for i in range(kh//2, height - kh//2):
    for j in range(kw//2, width - kw//2):
        region = image[i-kh//2:i+kh//2+1, j-kw//2:j+kw//2+1]
        output[i, j] = np.sum(region * kernel)

print(f"Image size: {height}x{width}")
print(f"Kernel size: {kh}x{kw}")
print(f"Output range: [{np.min(output):.3f}, {np.max(output):.3f}]")''',

            "Monte Carlo Simulation": '''import numpy as np

# Monte Carlo estimation of Pi
n_samples = 1000000

# Generate random points in unit square
x = np.random.uniform(-1, 1, n_samples)
y = np.random.uniform(-1, 1, n_samples)

# Check if points are inside unit circle
inside_circle = (x**2 + y**2) <= 1
pi_estimate = 4 * np.sum(inside_circle) / n_samples

# Statistical analysis
true_pi = np.pi
error = abs(pi_estimate - true_pi)
error_percent = (error / true_pi) * 100

print(f"Samples: {n_samples:,}")
print(f"Pi estimate: {pi_estimate:.6f}")
print(f"True Pi: {true_pi:.6f}")
print(f"Error: {error:.6f} ({error_percent:.3f}%)")''',
            
            "GPU-Optimized CuPy Example": '''import cupy as cp
import time

# Enable memory pool for better performance
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

# Create large arrays on GPU
n = 10000
A = cp.random.rand(n, n, dtype=cp.float32)
B = cp.random.rand(n, n, dtype=cp.float32)

# GPU matrix multiplication
start_time = time.perf_counter()
C = cp.matmul(A, B)
cp.cuda.Stream.null.synchronize()  # Ensure computation completes
gpu_time = time.perf_counter() - start_time

print(f"Matrix size: {n}x{n}")
print(f"GPU computation time: {gpu_time:.4f} seconds")
print(f"Memory usage: {mempool.used_bytes() / 1024**3:.2f} GB")

# Clean up
del A, B, C
mempool.free_all_blocks()''',

            "GPU-Optimized cuDF Example": '''import cudf
import cupy as cp
import time

# Create large dataset on GPU
n = 1000000
df = cudf.DataFrame({
    'id': cp.arange(n),
    'category': cp.random.choice(['A', 'B', 'C', 'D'], n),
    'value1': cp.random.normal(0, 1, n, dtype=cp.float32),
    'value2': cp.random.exponential(1, n).astype(cp.float32),
    'value3': cp.random.uniform(0, 100, n, dtype=cp.float32)
})

# GPU DataFrame operations
start_time = time.perf_counter()
result = df.groupby('category').agg({
    'value1': ['mean', 'std'],
    'value2': ['sum', 'count'],
    'value3': ['min', 'max']
})
gpu_time = time.perf_counter() - start_time

print(f"DataFrame size: {len(df):,} rows")
print(f"GPU processing time: {gpu_time:.4f} seconds")
print(f"Result shape: {result.shape}")
print("\\nSample results:")
print(result.head())'''
        }
    
    def get_all_samples(self) -> Dict[str, str]:
        """Get all sample codes."""
        return self.sample_codes
    
    def get_sample(self, name: str) -> str:
        """Get a specific sample code by name."""
        return self.sample_codes.get(name, "")
    
    def get_sample_names(self) -> List[str]:
        """Get list of all sample code names."""
        return list(self.sample_codes.keys())
    
    def get_samples_by_category(self, category: str) -> Dict[str, str]:
        """Get samples filtered by category."""
        category_filters = {
            "basic": ["Simple Array Operations", "Mathematical Functions"],
            "matrix": ["Matrix Multiplication", "Signal Processing"],
            "dataframe": ["DataFrame Operations", "Large DataFrame GroupBy"],
            "ml": ["Machine Learning - K-Means"],
            "advanced": ["Image Convolution", "Monte Carlo Simulation"],
            "gpu": ["GPU-Optimized CuPy Example", "GPU-Optimized cuDF Example"]
        }
        
        if category.lower() not in category_filters:
            return {}
        
        names = category_filters[category.lower()]
        return {name: self.sample_codes[name] for name in names if name in self.sample_codes}
