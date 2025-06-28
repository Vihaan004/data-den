#!/usr/bin/env python3
"""
Test script to verify plot creation and extraction works locally.
"""

import os
import sys
import glob
import base64
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from job_executor import wrap_data_analysis_code

def test_plot_extraction():
    """Test the plot extraction functionality locally."""
    
    # Create a simple test analysis code that generates a plot
    test_code = """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('../datasets/employee_sample.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Create a simple scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['salary'])
plt.xlabel('Age') 
plt.ylabel('Salary')
plt.title('Age vs Salary')
plt.tight_layout()
plt.savefig('age_salary_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Scatter plot saved as 'age_salary_scatter.png'")

# Create a histogram
plt.figure(figsize=(10, 6))
plt.hist(df['salary'], bins=10, alpha=0.7, edgecolor='black')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.title('Salary Distribution')
plt.tight_layout()
plt.savefig('salary_histogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("Histogram saved as 'salary_histogram.png'")

print("Analysis completed successfully")
"""
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing in directory: {temp_dir}")
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Copy the dataset to the temp directory structure
            datasets_dir = Path(temp_dir) / "datasets"
            datasets_dir.mkdir(exist_ok=True)
            
            # Copy the sample dataset
            original_dataset = Path(original_cwd) / "datasets" / "employee_sample.csv"
            temp_dataset = datasets_dir / "employee_sample.csv"
            shutil.copy2(original_dataset, temp_dataset)
            print(f"Copied dataset to: {temp_dataset}")
            
            # Generate wrapped code
            wrapped_code = wrap_data_analysis_code(test_code, str(temp_dataset))
            
            # Write and execute the wrapped code
            test_script = Path(temp_dir) / "test_analysis.py"
            with open(test_script, 'w') as f:
                f.write(wrapped_code)
            
            print("Executing test script...")
            
            # Execute the script and capture output
            import subprocess
            result = subprocess.run([sys.executable, str(test_script)], 
                                  capture_output=True, text=True, cwd=temp_dir)
            
            print("STDOUT:")
            print(result.stdout)
            
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            # Check for plot files
            png_files = list(Path(temp_dir).glob("*.png"))
            print(f"Found PNG files: {[f.name for f in png_files]}")
            
            # Test plot data extraction
            if "PLOT_DATA_START" in result.stdout:
                print("✅ Plot data markers found in output")
                
                # Extract plot data
                lines = result.stdout.split('\n')
                plot_count = 0
                for i, line in enumerate(lines):
                    if line.strip() == "PLOT_DATA_START":
                        plot_count += 1
                        print(f"✅ Found plot data block {plot_count}")
                
                print(f"Total plot data blocks found: {plot_count}")
            else:
                print("❌ No plot data markers found in output")
            
            return result.returncode == 0
            
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    print("Testing plot extraction functionality...")
    success = test_plot_extraction()
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
