#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 4            # number of cores 
#SBATCH -t 0-00:05:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o /home/vpatel69/R1/App/tests/lurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e /home/vpatel69/R1/App/tests/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Load required software
module load mamba/latest

#Activate our enviornment
source activate scicomp24.11 

#Change to the directory of our script
cd ~/R1/App

#Run the software/python script
python matrix_operations_cpu.py