import os
from pathlib import Path
import sys

# Add App directory to path so we can import benchmark module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark import wrap_with_timing

# Create output dir if it doesn't exist
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Read CPU and GPU test code
cpu_code_path = Path(__file__).parent / "test_cpu_code.py"
gpu_code_path = Path(__file__).parent / "test_gpu_code.py"

cpu_code = cpu_code_path.read_text()
gpu_code = gpu_code_path.read_text()

# Generate wrapped code
cpu_wrapped = wrap_with_timing(cpu_code, is_gpu=False)
gpu_wrapped = wrap_with_timing(gpu_code, is_gpu=True)

# Write wrapped code to output dir for inspection
output_cpu = output_dir / "test_cpu_wrapped.py"
output_gpu = output_dir / "test_gpu_wrapped.py"

output_cpu.write_text(cpu_wrapped)
output_gpu.write_text(gpu_wrapped)

print(f"Generated wrapped code in {output_dir}")
print(f"CPU code: {output_cpu}")
print(f"GPU code: {output_gpu}")
