#!/bin/bash
#SBATCH --job-name=gpu_mentor_ui
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=gpu_mentor_ui_%j.out
#SBATCH --error=gpu_mentor_ui_%j.err

echo "============================================================"
echo "GPU MENTOR - GRADIO UI ON SOL"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "============================================================"

# Load required modules
module load genai25.06

# Change to project directory
cd ~/R1/App/backend/ || {
    echo "❌ Failed to change to backend directory"
    exit 1
}

# Set Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Get the compute node name and IP
NODE_NAME=$(hostname)
NODE_IP=$(hostname -i)

echo "🖥️  Compute Node: $NODE_NAME"
echo "🌐 IP Address: $NODE_IP"
echo "🚀 Starting GPU Mentor UI..."

# Start the Gradio interface
python gradio_ui.py

echo ""
echo "============================================================"
echo "UI session ended"
echo "============================================================"
