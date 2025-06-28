"""
Code examples for GPU acceleration demonstrations.

These examples showcase various computational patterns that benefit from GPU acceleration
using NVIDIA Rapids libraries (CuPy, cuDF, cuML).
"""

SAMPLE_CODES = {
    "Matrix Multiplication": """import numpy as np

# Create large matrices
size = 1000
A = np.random.rand(size, size).astype(np.float32)
B = np.random.rand(size, size).astype(np.float32)

# Matrix multiplication
C = np.matmul(A, B)

print(f"Result shape: {C.shape}")
print(f"Sum of result: {np.sum(C)}")""",
    
    "DataFrame Operations": """import pandas as pd
import numpy as np

# Create sample dataframe
n_rows = 100000
df = pd.DataFrame({
    'A': np.random.randn(n_rows),
    'B': np.random.randn(n_rows),
    'C': np.random.choice(['X', 'Y', 'Z'], n_rows),
    'D': np.random.randint(1, 100, n_rows)
})

# Perform operations
result = df.groupby('C').agg({
    'A': 'mean',
    'B': 'std',
    'D': 'sum'
})

print(result)""",
    
    "Machine Learning": """from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

# Generate sample data
X, _ = make_blobs(n_samples=10000, centers=10, 
                  n_features=20, random_state=42)

# Perform clustering
kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(X)

print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
print(f"Labels shape: {labels.shape}")""",

    "Large Scale Data Aggregation": """import pandas as pd
import numpy as np
from time import perf_counter

# Create large dataset similar to NVIDIA examples
n = 50000000
data = {
    'key': np.random.randint(0, 1000, n),
    'value1': np.random.randn(n),
    'value2': np.random.randn(n),
    'category': np.random.choice(['A', 'B', 'C', 'D'], n),
    'timestamp': pd.date_range('2020-01-01', periods=n, freq='1S')
}

df = pd.DataFrame(data)

# Complex aggregation operations
start_time = perf_counter()
result = df.groupby(['key', 'category']).agg({
    'value1': ['sum', 'mean', 'std'],
    'value2': ['min', 'max', 'median']
})
end_time = perf_counter()

print(f"Processing time: {end_time - start_time:.4f} seconds")
print(f"Result shape: {result.shape}")""",

    "Linear Regression Analysis": """import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from time import perf_counter

# Load California housing dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train linear regression model
start_time = perf_counter()
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
training_time = perf_counter() - start_time

# Evaluate performance
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"Training time: {training_time:.4f} seconds")
print(f"R-squared score: {r2:.4f}")
print(f"Mean squared error: {mse:.4f}")""",

    "K-Means Clustering Large Dataset": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from time import perf_counter

# Generate large synthetic dataset
np.random.seed(42)
n_samples = 1500000
n_features = 8
n_clusters = 5

# Create dataset with natural clusters
X, true_labels = make_blobs(
    n_samples=n_samples,
    centers=n_clusters,
    n_features=n_features,
    cluster_std=1.5,
    random_state=42
)

print(f"Dataset shape: {X.shape}")

# Perform K-means clustering
start_time = perf_counter()
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)
clustering_time = perf_counter() - start_time

print(f"Clustering time: {clustering_time:.4f} seconds")
print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
print(f"Inertia: {kmeans.inertia_:.2f}")

# Show cluster distribution
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"Cluster {cluster}: {count} points")""",

    "Time Series Analysis": """import pandas as pd
import numpy as np
from time import perf_counter

# Create large time series dataset
np.random.seed(42)
n_points = 10000000
date_range = pd.date_range('2020-01-01', periods=n_points, freq='1min')

# Generate synthetic time series data
trend = np.linspace(100, 200, n_points)
seasonal = 10 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 60))  # Daily pattern
noise = np.random.normal(0, 5, n_points)
values = trend + seasonal + noise

df = pd.DataFrame({
    'timestamp': date_range,
    'value': values,
    'sensor_id': np.random.randint(1, 100, n_points),
    'location': np.random.choice(['North', 'South', 'East', 'West'], n_points)
})

print(f"Dataset shape: {df.shape}")

# Perform time-based aggregations
start_time = perf_counter()

# Resample to hourly data with multiple statistics
hourly_stats = df.set_index('timestamp').groupby(['sensor_id', 'location']).resample('1H')['value'].agg([
    'mean', 'std', 'min', 'max', 'count'
]).reset_index()

processing_time = perf_counter() - start_time

print(f"Processing time: {processing_time:.4f} seconds")
print(f"Hourly stats shape: {hourly_stats.shape}")
print(hourly_stats.head())""",

    "Advanced Array Operations": """import numpy as np
from time import perf_counter

# Create large multi-dimensional arrays
np.random.seed(42)
size = (2000, 2000, 50)
array1 = np.random.randn(*size).astype(np.float32)
array2 = np.random.randn(*size).astype(np.float32)

print(f"Array shape: {array1.shape}")
print(f"Memory usage: {array1.nbytes / 1024**3:.2f} GB per array")

# Perform complex mathematical operations
start_time = perf_counter()

# Element-wise operations
result1 = np.sin(array1) * np.cos(array2)

# Reductions along different axes
mean_axis0 = np.mean(result1, axis=0)
std_axis1 = np.std(result1, axis=1)
sum_axis2 = np.sum(result1, axis=2)

# Matrix operations on slices
final_result = np.dot(mean_axis0.T, mean_axis0)

computation_time = perf_counter() - start_time

print(f"Computation time: {computation_time:.4f} seconds")
print(f"Final result shape: {final_result.shape}")
print(f"Mean of final result: {np.mean(final_result):.6f}")""",

    "Feature Engineering Pipeline": """import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from time import perf_counter

# Create synthetic dataset with mixed data types
np.random.seed(42)
n_samples = 500000
n_features = 50

# Generate mixed feature types
data = {}

# Numerical features
for i in range(30):
    data[f'num_feature_{i}'] = np.random.randn(n_samples)

# Categorical features
categories = ['category_A', 'category_B', 'category_C', 'category_D', 'category_E']
for i in range(10):
    data[f'cat_feature_{i}'] = np.random.choice(categories, n_samples)

# Text-like features (simplified)
for i in range(10):
    data[f'text_feature_{i}'] = np.random.choice(
        ['short', 'medium_length', 'very_long_text_feature'], n_samples
    )

# Target variable
data['target'] = np.random.choice([0, 1, 2], n_samples)

df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")

# Feature engineering pipeline
start_time = perf_counter()

# Encode categorical variables
label_encoders = {}
for col in [c for c in df.columns if 'cat_' in c or 'text_' in c]:
    if col != 'target':
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le

# Create interaction features
df['interaction_1'] = df['num_feature_0'] * df['num_feature_1']
df['interaction_2'] = df['num_feature_2'] / (df['num_feature_3'] + 1e-8)

# Scaling numerical features
numerical_cols = [c for c in df.columns if 'num_' in c or 'interaction_' in c]
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

pipeline_time = perf_counter() - start_time

print(f"Feature engineering time: {pipeline_time:.4f} seconds")
print(f"Final dataset shape: {df.shape}")
print(f"Feature types: {df.dtypes.value_counts()}")"""
}

# Additional metadata for samples
SAMPLE_METADATA = {
    "Matrix Multiplication": {
        "description": "Basic matrix operations using NumPy",
        "complexity": "Beginner",
        "gpu_benefit": "High",
        "memory_usage": "1GB",
        "libraries": ["numpy"],
        "gpu_libraries": ["cupy"]
    },
    "DataFrame Operations": {
        "description": "Pandas DataFrame groupby and aggregation operations",
        "complexity": "Beginner",
        "gpu_benefit": "High",
        "memory_usage": "800MB",
        "libraries": ["pandas", "numpy"],
        "gpu_libraries": ["cudf"]
    },
    "Machine Learning": {
        "description": "K-means clustering with scikit-learn",
        "complexity": "Intermediate",
        "gpu_benefit": "Medium",
        "memory_usage": "160MB",
        "libraries": ["sklearn", "numpy"],
        "gpu_libraries": ["cuml"]
    },
    "Large Scale Data Aggregation": {
        "description": "Complex aggregations on 50M row dataset",
        "complexity": "Advanced",
        "gpu_benefit": "Very High",
        "memory_usage": "12GB",
        "libraries": ["pandas", "numpy"],
        "gpu_libraries": ["cudf"]
    },
    "Linear Regression Analysis": {
        "description": "Complete ML pipeline with California housing data",
        "complexity": "Intermediate",
        "gpu_benefit": "High",
        "memory_usage": "100MB",
        "libraries": ["sklearn", "pandas", "numpy"],
        "gpu_libraries": ["cuml", "cudf"]
    },
    "K-Means Clustering Large Dataset": {
        "description": "Clustering 1.5M samples with performance timing",
        "complexity": "Advanced",
        "gpu_benefit": "Very High",
        "memory_usage": "480MB",
        "libraries": ["sklearn", "numpy", "matplotlib"],
        "gpu_libraries": ["cuml", "cupy"]
    },
    "Time Series Analysis": {
        "description": "Time series processing with 10M data points",
        "complexity": "Advanced",
        "gpu_benefit": "Very High",
        "memory_usage": "8GB",
        "libraries": ["pandas", "numpy"],
        "gpu_libraries": ["cudf"]
    },
    "Advanced Array Operations": {
        "description": "Complex mathematical operations on large 3D arrays",
        "complexity": "Advanced",
        "gpu_benefit": "Very High",
        "memory_usage": "16GB",
        "libraries": ["numpy"],
        "gpu_libraries": ["cupy"]
    },
    "Feature Engineering Pipeline": {
        "description": "Complete preprocessing pipeline with mixed data types",
        "complexity": "Advanced",
        "gpu_benefit": "High",
        "memory_usage": "4GB",
        "libraries": ["pandas", "sklearn", "numpy"],
        "gpu_libraries": ["cudf", "cuml"]
    }
}
