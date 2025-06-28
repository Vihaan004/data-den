#!/bin/bash

# GPU Mentor Environment Setup Script
# This script loads the required modules and starts Ollama

echo "🔧 Loading mamba..."
module load mamba/latest

echo "🐍 Activating genai25.06 environment..."
source activate genai25.06

echo "✅ Environment setup complete!"
