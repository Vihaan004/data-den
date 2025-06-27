#!/bin/bash

# GPU Mentor Safe Launcher
# This script provides better process control for the GPU Mentor application

echo "🚀 GPU Mentor Safe Launcher"
echo "=========================="

# Function to cleanup on exit
cleanup() {
    echo
    echo "🛑 Cleaning up GPU Mentor processes..."
    
    # Kill any Python processes related to our app
    pkill -f "launch.py" 2>/dev/null
    pkill -f "app.py" 2>/dev/null
    pkill -f "gpu_mentor" 2>/dev/null
    
    # Wait a moment for cleanup
    sleep 2
    
    echo "✅ Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo "💡 Starting GPU Mentor application..."
echo "💡 Use Ctrl+C to stop cleanly"
echo

# Run the application
python launch.py

# If we get here, the app exited normally
echo "👋 GPU Mentor application stopped"
