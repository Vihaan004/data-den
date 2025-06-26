"""
Performance Visualizer - Create visualizations and insights for benchmark results
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class PerformanceVisualizer:
    """Create visualizations and insights for benchmark results."""
    
    def __init__(self):
        self.visualization_templates = self._setup_visualization_templates()
    
    def _setup_visualization_templates(self):
        """Setup templates for different types of performance visualizations."""
        return {
            "speedup_chart": """
## 📈 Performance Speedup Analysis

```
Benchmark: {benchmark_name}
Category: {category}
Problem Size: {size:,}

CPU Time:    {cpu_time:.4f}s  ████████████████████████
GPU Time:    {gpu_time:.4f}s  {gpu_bar}
Speedup:     {speedup:.2f}x   {speedup_indicator}
```

**Performance Insights:**
{insights}
""",
            
            "scaling_analysis": """
## 📊 Performance Scaling Analysis

**How performance changes with problem size:**

{scaling_data}

**Key Observations:**
- GPU advantage increases with larger problem sizes
- Overhead is more significant for smaller datasets
- Memory bandwidth becomes the limiting factor at large scales
""",
            
            "comparison_matrix": """
## 🏆 Technology Comparison

| Operation Type | CPU Library | GPU Library | Typical Speedup | Best Use Case |
|----------------|-------------|-------------|-----------------|---------------|
| Matrix Ops     | NumPy       | CuPy        | 10-50x         | Linear algebra, large arrays |
| DataFrame Ops  | Pandas      | cuDF        | 5-20x          | Data processing, analytics |
| ML Algorithms  | scikit-learn| cuML        | 5-25x          | Large datasets, feature engineering |
| Math Functions | NumPy       | CuPy        | 3-15x          | Signal processing, numerical computing |

**💡 Selection Guidelines:**
- **Problem Size**: GPU benefits increase with larger datasets
- **Memory**: Consider GPU memory limitations for very large data
- **Pipeline**: Keep operations on GPU to avoid transfer overhead
"""
        }
    
    def create_speedup_visualization(self, benchmark_result):
        """Create a text-based speedup visualization."""
        if not benchmark_result or benchmark_result.get("error"):
            return "❌ Benchmark data not available"
        
        speedup = benchmark_result.get("speedup", 1)
        cpu_time = benchmark_result.get("cpu_time", 0)
        gpu_time = benchmark_result.get("gpu_time", 0)
        
        # Create visual bar representation
        max_bar_length = 24
        if cpu_time > 0:
            gpu_bar_length = max(1, int((gpu_time / cpu_time) * max_bar_length))
            gpu_bar = "█" * gpu_bar_length
        else:
            gpu_bar = "█"
        
        # Speedup indicator
        if speedup > 10:
            speedup_indicator = "🔥 Excellent acceleration!"
        elif speedup > 3:
            speedup_indicator = "✅ Good performance gain"
        elif speedup > 1:
            speedup_indicator = "⚡ Moderate improvement"
        else:
            speedup_indicator = "⚠️ GPU overhead dominates"
        
        # Generate insights
        insights = self._generate_performance_insights(benchmark_result)
        
        return self.visualization_templates["speedup_chart"].format(
            benchmark_name=benchmark_result.get("benchmark", "Unknown"),
            category=benchmark_result.get("category", "Unknown"),
            size=benchmark_result.get("size", 0),
            cpu_time=cpu_time,
            gpu_time=gpu_time,
            gpu_bar=gpu_bar,
            speedup=speedup,
            speedup_indicator=speedup_indicator,
            insights=insights
        )
    
    def _generate_performance_insights(self, benchmark_result):
        """Generate specific insights based on benchmark results."""
        insights = []
        speedup = benchmark_result.get("speedup", 1)
        category = benchmark_result.get("category", "")
        size = benchmark_result.get("size", 0)
        
        # Size-based insights
        if size < 1000:
            insights.append("• Small problem size - GPU overhead may limit benefits")
        elif size < 100000:
            insights.append("• Medium problem size - good balance of performance and overhead")
        else:
            insights.append("• Large problem size - excellent candidate for GPU acceleration")
        
        # Category-specific insights
        if "Matrix" in category:
            if speedup > 10:
                insights.append("• Matrix operations scale excellently on GPU due to high parallelism")
            else:
                insights.append("• Consider larger matrices or different algorithms for better GPU utilization")
        
        elif "DataFrame" in category:
            if speedup > 5:
                insights.append("• DataFrame operations benefit from GPU's parallel processing capabilities")
            else:
                insights.append("• Try larger datasets or more complex operations for better GPU benefits")
        
        elif "ML" in category:
            insights.append("• Machine learning algorithms often show consistent GPU acceleration")
            if speedup < 3:
                insights.append("• Consider using larger datasets or different algorithms")
        
        # General speedup insights
        if speedup < 1.5:
            insights.append("• GPU overhead exceeds benefits - consider CPU for this workload")
        elif speedup > 20:
            insights.append("• Exceptional GPU performance - this workload is highly parallel")
        
        return "\n".join(insights) if insights else "• Standard GPU acceleration performance"
    
    def create_educational_summary(self, benchmark_result):
        """Create educational summary explaining the results."""
        category = benchmark_result.get("category", "Unknown")
        speedup = benchmark_result.get("speedup", 1)
        
        summary = f"""
### 🎓 Educational Summary

**What happened in this benchmark:**
"""
        
        if "Matrix" in category:
            summary += f"""
1. **CPU Processing**: NumPy used optimized BLAS libraries for matrix operations
2. **GPU Processing**: CuPy leveraged thousands of GPU cores for parallel computation
3. **Key Factor**: Matrix multiplication is highly parallelizable, leading to {speedup:.1f}x speedup
"""
        elif "DataFrame" in category:
            summary += f"""
1. **CPU Processing**: Pandas processed data sequentially with optimized C extensions
2. **GPU Processing**: cuDF distributed operations across GPU cores
3. **Key Factor**: Parallel aggregation and grouping operations showed {speedup:.1f}x improvement
"""
        elif "ML" in category:
            summary += f"""
1. **CPU Processing**: scikit-learn used optimized CPU algorithms
2. **GPU Processing**: cuML leveraged GPU parallelism for distance calculations and updates
3. **Key Factor**: ML algorithms with many data points benefit from massive parallelization
"""
        
        # Add learning objectives
        summary += f"""
**Learning Objectives Achieved:**
• Demonstrated {speedup:.1f}x performance improvement with GPU acceleration
• Showed real-world application of NVIDIA Rapids ecosystem
• Illustrated when GPU acceleration provides significant benefits
• Experienced hands-on performance comparison

**Next Steps to Explore:**
• Try different problem sizes to see how speedup scales
• Experiment with different data types (float32 vs float64)
• Profile memory usage between CPU and GPU implementations
• Explore memory usage patterns between CPU and GPU implementations
"""
        
        return summary
    
    def create_comparison_matrix(self):
        """Create a technology comparison matrix."""
        return self.visualization_templates["comparison_matrix"]
