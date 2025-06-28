#!/usr/bin/env python3
"""
Test script to verify that the job-specific plot naming works correctly.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from job_executor import wrap_data_analysis_code, extract_plots_from_directory

def test_job_specific_plots():
    """Test that plots are correctly named with job timestamps."""
    
    # Create a simple test analysis code that generates multiple plots
    test_code = """
import matplotlib.pyplot as plt
import numpy as np

# Generate some test data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create first plot
plt.figure(figsize=(8, 6))
plt.plot(x, y1)
plt.title('Sine Wave')
plt.savefig('sine_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Sine plot saved")

# Create second plot
plt.figure(figsize=(8, 6))
plt.plot(x, y2)
plt.title('Cosine Wave')
plt.savefig('cosine_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Cosine plot saved")

print("Test analysis completed successfully")
"""
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing in directory: {temp_dir}")
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Generate a test timestamp
            test_timestamp = int(time.time())
            print(f"Test timestamp: {test_timestamp}")
            
            # Generate wrapped code with timestamp
            wrapped_code = wrap_data_analysis_code(test_code, "dummy_dataset.csv", test_timestamp)
            
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
            
            # Check for renamed PNG files
            png_files = list(Path(temp_dir).glob("*.png"))
            job_specific_files = list(Path(temp_dir).glob(f"job_{test_timestamp}_*.png"))
            
            print(f"All PNG files: {[f.name for f in png_files]}")
            print(f"Job-specific PNG files: {[f.name for f in job_specific_files]}")
            
            # Test plot extraction
            plots = extract_plots_from_directory(temp_dir, test_timestamp)
            print(f"Extracted {len(plots)} plots using timestamp {test_timestamp}")
            
            # Test that we can distinguish between different job timestamps
            fake_timestamp = test_timestamp + 1000
            fake_plots = extract_plots_from_directory(temp_dir, fake_timestamp)
            print(f"Extracted {len(fake_plots)} plots using fake timestamp {fake_timestamp}")
            
            # Verify results
            success = True
            if len(job_specific_files) == 0:
                print("❌ No job-specific files found")
                success = False
            
            if len(plots) == 0:
                print("❌ No plots extracted with correct timestamp")
                success = False
            
            if len(fake_plots) > 0:
                print("❌ Plots extracted with wrong timestamp (should be 0)")
                success = False
            
            if success:
                print("✅ Job-specific plot naming works correctly!")
            
            return success
            
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    print("Testing job-specific plot naming...")
    success = test_job_specific_plots()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
