#!/bin/bash
# Check and start Ollama service for GPU Mentor

echo "🔍 Checking Ollama service status..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama service not running"
    echo "🚀 Starting Ollama service..."
    
    # Start Ollama in background
    ollama serve &
    OLLAMA_PID=$!
    
    # Wait for service to start
    echo "⏳ Waiting for Ollama to start..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama service started successfully"
            break
        fi
        sleep 1
        echo -n "."
    done
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "❌ Failed to start Ollama service"
        exit 1
    fi
else
    echo "✅ Ollama service is already running"
fi

# Check if required model is available
echo ""
echo "🔍 Checking for required model..."
if ollama list | grep -q "qwen2:14b"; then
    echo "✅ Model qwen2:14b is available"
else
    echo "❌ Model qwen2:14b not found"
    echo "📥 Pulling model (this may take several minutes)..."
    ollama pull qwen2:14b
    
    if [ $? -eq 0 ]; then
        echo "✅ Model qwen2:14b downloaded successfully"
    else
        echo "❌ Failed to download model qwen2:14b"
        exit 1
    fi
fi

echo ""
echo "🎉 Ollama is ready for GPU Mentor!"
echo "🚀 You can now run: python gradio_ui.py"
