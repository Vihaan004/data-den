#!/usr/bin/env python3
"""
FastAPI Web Interface for GPU Mentor Backend
"""
import sys
sys.path.append('.')

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import traceback
import os

from core.enhanced_gpu_mentor import EnhancedGPUMentor
from core.benchmark_engine import BenchmarkEngine
from utils.sample_code_library import SampleCodeLibrary

# Initialize FastAPI app
app = FastAPI(title="GPU Mentor API", version="2.0.0")

# Initialize backend components
try:
    gpu_mentor = EnhancedGPUMentor()
    benchmark_engine = BenchmarkEngine()
    sample_library = SampleCodeLibrary()
    print("✅ GPU Mentor backend initialized")
except Exception as e:
    print(f"❌ Backend initialization failed: {e}")
    gpu_mentor = None

# Pydantic models for API
class CodeOptimizationRequest(BaseModel):
    code: str
    question: Optional[str] = ""

class CodeOptimizationResponse(BaseModel):
    optimized_code: str
    explanation: str
    performance_info: str
    success: bool

class BenchmarkRequest(BaseModel):
    code: str
    name: str = "benchmark"

class SampleCodeRequest(BaseModel):
    category: str
    operation: str

class ChatRequest(BaseModel):
    question: str

@app.get("/")
async def read_root():
    """Serve the main HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GPU Mentor - Sol Supercomputer</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: #2d3748;
                color: white;
                padding: 20px;
                text-align: center;
            }
            .content {
                padding: 30px;
            }
            .tab-container {
                display: flex;
                border-bottom: 1px solid #e2e8f0;
                margin-bottom: 20px;
            }
            .tab {
                padding: 12px 20px;
                cursor: pointer;
                border: none;
                background: none;
                font-size: 16px;
                color: #666;
                border-bottom: 2px solid transparent;
            }
            .tab.active {
                color: #667eea;
                border-bottom-color: #667eea;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: 600;
                color: #2d3748;
            }
            textarea, input, select {
                width: 100%;
                padding: 12px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                font-family: 'Monaco', 'Consolas', monospace;
                font-size: 14px;
                box-sizing: border-box;
            }
            textarea {
                resize: vertical;
                min-height: 200px;
            }
            button {
                background: #667eea;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
            }
            button:hover {
                background: #5a6fd8;
            }
            .response {
                margin-top: 20px;
                padding: 15px;
                background: #f7fafc;
                border-radius: 6px;
                border-left: 4px solid #667eea;
            }
            .error {
                border-left-color: #e53e3e;
                background: #fed7d7;
            }
            .system-info {
                background: #e6fffa;
                padding: 20px;
                border-radius: 6px;
                border-left: 4px solid #38b2ac;
            }
            .two-column {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            @media (max-width: 768px) {
                .two-column {
                    grid-template-columns: 1fr;
                }
                .tab-container {
                    flex-wrap: wrap;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 GPU Mentor</h1>
                <p>AI-Powered GPU Code Optimization on Sol Supercomputer</p>
            </div>
            
            <div class="content">
                <div class="tab-container">
                    <button class="tab active" onclick="showTab('optimize')">🔧 Optimize</button>
                    <button class="tab" onclick="showTab('samples')">📚 Samples</button>
                    <button class="tab" onclick="showTab('benchmark')">⚡ Benchmark</button>
                    <button class="tab" onclick="showTab('chat')">🤖 Chat</button>
                    <button class="tab" onclick="showTab('system')">ℹ️ System</button>
                </div>
                
                <!-- Code Optimization Tab -->
                <div id="optimize" class="tab-content active">
                    <h2>Code Optimization</h2>
                    <div class="two-column">
                        <div>
                            <div class="form-group">
                                <label for="input-code">Your Python Code:</label>
                                <textarea id="input-code" placeholder="Enter your Python code here...">import numpy as np

# Inefficient approach
data = list(range(1000))
result = []
for i in range(len(data)):
    result.append(data[i] * 2 + 1)

print(f"Result length: {len(result)}")</textarea>
                            </div>
                            <div class="form-group">
                                <label for="question">Question (Optional):</label>
                                <input type="text" id="question" placeholder="Ask about specific optimizations...">
                            </div>
                            <button onclick="optimizeCode()">🚀 Optimize Code</button>
                        </div>
                        <div>
                            <div class="form-group">
                                <label>Optimized Code:</label>
                                <textarea id="optimized-output" readonly></textarea>
                            </div>
                        </div>
                    </div>
                    <div id="optimize-response" class="response" style="display:none;"></div>
                </div>
                
                <!-- Sample Code Tab -->
                <div id="samples" class="tab-content">
                    <h2>Sample Code Library</h2>
                    <div class="form-group">
                        <label for="category">Category:</label>
                        <select id="category">
                            <option value="basic">Basic</option>
                            <option value="intermediate">Intermediate</option>
                            <option value="advanced">Advanced</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="operation">Operation:</label>
                        <select id="operation">
                            <option value="array_operations">Array Operations</option>
                            <option value="matrix_operations">Matrix Operations</option>
                            <option value="image_processing">Image Processing</option>
                        </select>
                    </div>
                    <button onclick="getSample()">📖 Get Sample Code</button>
                    <div class="form-group">
                        <label>Sample Code:</label>
                        <textarea id="sample-output" readonly></textarea>
                    </div>
                </div>
                
                <!-- Benchmark Tab -->
                <div id="benchmark" class="tab-content">
                    <h2>Performance Benchmarking</h2>
                    <div class="form-group">
                        <label for="benchmark-code">Code to Benchmark:</label>
                        <textarea id="benchmark-code">import numpy as np
import time

start = time.time()
data = np.random.random((1000, 1000))
result = np.dot(data, data.T)
end = time.time()

print(f"Execution time: {end - start:.4f} seconds")</textarea>
                    </div>
                    <div class="form-group">
                        <label for="benchmark-name">Benchmark Name:</label>
                        <input type="text" id="benchmark-name" value="matrix_multiplication_test">
                    </div>
                    <button onclick="runBenchmark()">⚡ Run Benchmark</button>
                    <div id="benchmark-response" class="response" style="display:none;"></div>
                </div>
                
                <!-- Chat Tab -->
                <div id="chat" class="tab-content">
                    <h2>Ask the GPU Mentor</h2>
                    <div class="form-group">
                        <label for="chat-input">Your Question:</label>
                        <textarea id="chat-input" rows="3" placeholder="Ask about GPU programming, optimization techniques, etc."></textarea>
                    </div>
                    <button onclick="askMentor()">💬 Ask Mentor</button>
                    <div id="chat-response" class="response" style="display:none;"></div>
                    
                    <div style="margin-top: 20px;">
                        <h3>Example Questions:</h3>
                        <ul>
                            <li>How can I optimize matrix operations for GPU?</li>
                            <li>What are the best practices for memory management?</li>
                            <li>Explain vectorization vs GPU acceleration</li>
                        </ul>
                    </div>
                </div>
                
                <!-- System Info Tab -->
                <div id="system" class="tab-content">
                    <h2>System Information</h2>
                    <div class="system-info">
                        <h3>🖥️ Sol Supercomputer Status</h3>
                        <p><strong>Backend Status:</strong> ✅ Running</p>
                        <p><strong>GPU Access:</strong> ✅ Available</p>
                        <p><strong>Kernel:</strong> genai25.06</p>
                        <p><strong>Components:</strong> All loaded ✅</p>
                        
                        <h3>💡 Usage Tips</h3>
                        <ul>
                            <li>Use the optimization tab for performance improvements</li>
                            <li>Browse sample codes for learning common patterns</li>
                            <li>Benchmark your code to measure improvements</li>
                            <li>Ask the mentor for guidance and explanations</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            function showTab(tabName) {
                // Hide all tab contents
                const tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Remove active class from all tabs
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach(tab => tab.classList.remove('active'));
                
                // Show selected tab content
                document.getElementById(tabName).classList.add('active');
                
                // Add active class to clicked tab
                event.target.classList.add('active');
            }
            
            async function optimizeCode() {
                const code = document.getElementById('input-code').value;
                const question = document.getElementById('question').value;
                
                try {
                    const response = await fetch('/optimize', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ code, question })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        document.getElementById('optimized-output').value = result.optimized_code;
                        showResponse('optimize-response', result.explanation + '\\n\\n' + result.performance_info);
                    } else {
                        showResponse('optimize-response', 'Error: ' + result.explanation, true);
                    }
                } catch (error) {
                    showResponse('optimize-response', 'Network error: ' + error.message, true);
                }
            }
            
            async function getSample() {
                const category = document.getElementById('category').value;
                const operation = document.getElementById('operation').value;
                
                try {
                    const response = await fetch('/sample', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ category, operation })
                    });
                    
                    const result = await response.json();
                    document.getElementById('sample-output').value = result.code || result.error;
                } catch (error) {
                    document.getElementById('sample-output').value = 'Error: ' + error.message;
                }
            }
            
            async function runBenchmark() {
                const code = document.getElementById('benchmark-code').value;
                const name = document.getElementById('benchmark-name').value;
                
                try {
                    const response = await fetch('/benchmark', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ code, name })
                    });
                    
                    const result = await response.json();
                    showResponse('benchmark-response', result.result || result.error);
                } catch (error) {
                    showResponse('benchmark-response', 'Error: ' + error.message, true);
                }
            }
            
            async function askMentor() {
                const question = document.getElementById('chat-input').value;
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question })
                    });
                    
                    const result = await response.json();
                    showResponse('chat-response', result.response || result.error);
                } catch (error) {
                    showResponse('chat-response', 'Error: ' + error.message, true);
                }
            }
            
            function showResponse(elementId, message, isError = false) {
                const element = document.getElementById(elementId);
                element.innerHTML = '<pre>' + message + '</pre>';
                element.style.display = 'block';
                element.className = 'response' + (isError ? ' error' : '');
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/optimize", response_model=CodeOptimizationResponse)
async def optimize_code(request: CodeOptimizationRequest):
    """Optimize code endpoint."""
    if not gpu_mentor:
        raise HTTPException(status_code=500, detail="Backend not initialized")
    
    try:
        optimized = gpu_mentor.optimize_code(request.code)
        explanation = ""
        
        if request.question:
            explanation = gpu_mentor.explain_optimization(request.question)
        
        performance_info = f"Original: {len(request.code)} chars, Optimized: {len(optimized)} chars"
        
        return CodeOptimizationResponse(
            optimized_code=optimized,
            explanation=explanation,
            performance_info=performance_info,
            success=True
        )
    except Exception as e:
        return CodeOptimizationResponse(
            optimized_code="",
            explanation=str(e),
            performance_info="",
            success=False
        )

@app.post("/sample")
async def get_sample_code(request: SampleCodeRequest):
    """Get sample code."""
    try:
        sample = sample_library.get_sample_code(request.category, request.operation)
        return {"code": sample or f"No sample found for {request.category}/{request.operation}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/benchmark")
async def benchmark_code(request: BenchmarkRequest):
    """Benchmark code."""
    try:
        result = benchmark_engine.benchmark_code(request.code, request.name)
        return {"result": f"Benchmark completed: {result}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat_with_mentor(request: ChatRequest):
    """Chat with mentor."""
    try:
        response = gpu_mentor.explain_optimization(request.question)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "backend_initialized": gpu_mentor is not None,
        "components": {
            "gpu_mentor": gpu_mentor is not None,
            "benchmark_engine": benchmark_engine is not None,
            "sample_library": sample_library is not None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
