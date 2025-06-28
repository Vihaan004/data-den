# GPU Mentor - AI-Powered GPU Acceleration Assistant

A modular application that helps developers learn and implement GPU acceleration for Python code using NVIDIA Rapids libraries (CuPy, cuDF, cuML).

## Features

- **AI-Powered Code Analysis**: Analyze Python code for GPU acceleration opportunities
- **Intelligent Code Optimization**: Get GPU-optimized versions of your CPU code
- **Interactive Chat Interface**: Ask questions about GPU acceleration techniques
- **Educational Content**: Generate tutorials on specific GPU acceleration topics
- **Code Execution**: Safely execute and time your code

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama** (required for LLM functionality):
   ```bash
   ollama serve
   ollama pull qwen2.5:14b
   ```

3. **Run the Application**:
   ```bash
   python run_app.py
   ```

4. **Access the Interface**:
   Open your browser to `http://localhost:7860`

## Usage

### Chat Interface
- Ask questions about GPU acceleration
- Paste code for analysis and optimization suggestions
- Get educational guidance and learning resources

### Code Analysis
- Analyze Python code for GPU acceleration potential
- Get optimized GPU versions using Rapids libraries
- See performance estimates and recommendations

### Learning Resources
- Generate custom tutorials on GPU acceleration topics
- Learn best practices for CuPy, cuDF, and cuML

## Command Line Options

```bash
python run_app.py --help
```

- `--port PORT`: Set custom port (default: 7860)
- `--host HOST`: Set custom host (default: 0.0.0.0)
- `--share`: Create public shareable link
- `--skip-checks`: Skip requirement and Ollama checks

## Requirements

- Python 3.8+
- Ollama with qwen2.5:14b model
- All packages listed in requirements.txt

## Architecture

- `app.py`: Main Gradio application
- `gpu_mentor.py`: Core mentor logic
- `rag_agent.py`: RAG system for knowledge retrieval
- `code_optimizer.py`: Code analysis and optimization
- `document_loader.py`: Document loading and vector store
- `run_app.py`: Application runner with checks
