#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 4            # number of cores 
#SBATCH -t 0-00:05:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -G 1    # request 1 GPU
#SBATCH -o /home/vpatel69/R1/App/tests/jobs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e /home/vpatel69/R1/App/tests/jobs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Load required software
module load mamba/latest

# Automatically find and load the most recent CUDA module
echo "Searching for available CUDA modules..."
CUDA_MODULES=$(module avail cuda 2>&1 | grep -E "cuda-[0-9]+\.[0-9]+\.[0-9]+-gcc" | grep -v "ont-guppy" | sort -V)

if [ -z "$CUDA_MODULES" ]; then
    echo "❌ No CUDA modules found!"
    exit 1
fi

# Get the latest CUDA version (extract just the module name)
LATEST_CUDA=$(echo "$CUDA_MODULES" | tail -1 | awk '{print $1}')

echo "Available CUDA modules:"
echo "$CUDA_MODULES"
echo "Loading latest CUDA module: $LATEST_CUDA"

module load $LATEST_CUDA

# Verify CUDA was loaded successfully
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA module failed to load properly"
    exit 1
fi

echo "✅ Successfully loaded CUDA module: $LATEST_CUDA"
echo "CUDA Compiler version:"
nvcc --version | head -4

#Activate our environment (RAPIDS 25.02)
source activate rapids25.02

#Change to the directory of our script
cd ~/R1/App/tests

#Run the software/python script
python matrix_operations_gpu.py
