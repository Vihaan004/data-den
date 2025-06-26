#!/bin/bash
# Quick Start Script for GPU Mentor UI on Sol

echo "🚀 GPU MENTOR - QUICK START DEPLOYMENT"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "gradio_ui.py" ]; then
    echo "❌ Error: gradio_ui.py not found. Run from backend directory."
    exit 1
fi

echo "✅ Backend directory confirmed"

# Load required modules
echo "📦 Loading genai25.06 kernel..."
module load genai25.06

# Check Python environment
echo "🐍 Python version: $(python --version)"

# Install Gradio if not available
echo "📦 Checking Gradio installation..."
python -c "import gradio" 2>/dev/null || {
    echo "Installing Gradio..."
    pip install gradio --user
}

# Set Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Get node information
NODE_NAME=$(hostname)
NODE_IP=$(hostname -i 2>/dev/null || echo "IP not available")

echo ""
echo "🖥️  Deployment Information:"
echo "   Node: $NODE_NAME"
echo "   IP: $NODE_IP"
echo "   Port: 7860"
echo ""

# Choice menu
echo "🎯 Choose deployment option:"
echo "   1) Quick Test (Interactive - 30 minutes)"
echo "   2) SLURM Job (Background - 4 hours)"
echo "   3) FastAPI Server (Production)"
echo ""

read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "🚀 Starting interactive Gradio UI..."
        echo ""
        echo "📡 Access Instructions:"
        echo "   1. From your local machine, run:"
        echo "      ssh -L 7860:$NODE_NAME:7860 \$USER@sol.asu.edu"
        echo "   2. Open browser to: http://localhost:7860"
        echo ""
        echo "⏰ Starting in 5 seconds... (Ctrl+C to cancel)"
        sleep 5
        python gradio_ui.py
        ;;
    2)
        echo "📋 Submitting SLURM job..."
        sbatch start_ui.slurm
        echo ""
        echo "✅ Job submitted! Monitor with:"
        echo "   squeue -u \$USER"
        echo "   tail -f gpu_mentor_ui_*.out"
        ;;
    3)
        echo "🌐 Starting FastAPI server..."
        echo ""
        echo "📡 Access at: http://$NODE_NAME:8000"
        echo "   API docs: http://$NODE_NAME:8000/docs"
        echo ""
        python fastapi_ui.py
        ;;
    *)
        echo "❌ Invalid choice. Please run again and select 1, 2, or 3."
        exit 1
        ;;
esac

echo ""
echo "✨ GPU Mentor deployment completed!"
