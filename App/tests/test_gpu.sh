#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 4            # number of cores 
#SBATCH -t 0-00:05:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -G 1    # request 1 GPU
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Load required software
module load mamba/latest
module load cuda/12.0

#Activate our environment (RAPIDS 25.02)
source activate rapids25.02

#Change to the directory of our script
cd ~/R1/App/tests

#Run the software/python script
python matrix_operations_gpu.py
