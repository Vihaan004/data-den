#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 0-00:10:00
#SBATCH -p general
#SBATCH -q public
#SBATCH -G 1
#SBATCH -o /home/vpatel69/R1/App/sol_environment_info.out
#SBATCH -e /home/vpatel69/R1/App/sol_environment_info.err
#SBATCH --export=NONE

echo "Starting Sol Environment Information Extraction..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Date: $(date)"
echo ""

# Load required modules (same as your job executor)
module load mamba/latest

# Load CUDA module
CUDA_MODULES=$(module avail cuda 2>&1 | grep -E "cuda-[0-9]+\.[0-9]+\.[0-9]+-gcc" | grep -v "ont-guppy" | sort -V)
if [ -z "$CUDA_MODULES" ]; then 
    echo "No CUDA modules found!"
    exit 1
fi
LATEST_CUDA=$(echo "$CUDA_MODULES" | tail -1 | awk '{print $1}')
echo "Loading CUDA module: $LATEST_CUDA"
module load $LATEST_CUDA

# Activate RAPIDS environment  
echo "Activating RAPIDS environment: rapids25.02"
source activate rapids25.02

# Show loaded modules
echo ""
echo "Loaded modules:"
module list

echo ""
echo "Environment activated. Running Python environment extraction..."
echo ""

# Run the Python script
python sol_environment_info.py

echo ""
echo "Environment extraction completed."
echo "Job finished at: $(date)"
