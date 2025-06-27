# GPU Mentor API Specification v2.0

## 🎯 Project Overview

**Project Name:** GPU Mentor - AI-Powered GPU Acceleration Tutor  
**Architecture:** Microservices with RAG Agent Backend + React Frontend  
**Target Environment:** ASU Sol Supercomputer + External Frontend  
**Date:** June 26, 2025  

## 🏗️ System Architecture

```
┌─────────────────┐    HTTP/WebSocket    ┌─────────────────┐
│   React Frontend│ ◄─────────────────► │   FastAPI       │
│   (Local/Remote)│                     │   Backend       │
│                 │                     │   (Sol Server)  │
│   - Chat UI     │                     │   - RAG Agent   │
│   - Code Editor │                     │   - LangGraph   │
│   - Benchmarks  │                     │   - GPU Mentor  │
│   - Tutorials   │                     │   - Ollama LLM  │
└─────────────────┘                     └─────────────────┘
                                                 │
                                        ┌─────────────────┐
                                        │   Sol Resources │
                                        │   - GPU Nodes   │
                                        │   - CUDA/Rapids │
                                        │   - Benchmarking│
                                        └─────────────────┘
```

## 🎯 Core Objectives

### Primary Goals
1. **Separation of Concerns**: Isolate RAG agent logic from UI components
2. **Scalability**: Support multiple concurrent frontend clients
3. **Flexibility**: Enable different frontend frameworks/platforms
4. **Performance**: Leverage Sol's GPU resources optimally
5. **Accessibility**: Allow external access via VPN/SSH tunneling

### Success Metrics
- API response time < 2 seconds for chat queries
- Code analysis completion < 10 seconds
- Benchmark execution tracking with real-time updates
- Support for 10+ concurrent users
- 99.9% uptime during Sol job allocation

## 🔧 Technical Stack

### Backend (Sol Supercomputer)
- **Framework**: FastAPI with async support
- **LLM Engine**: Ollama (llama3.1, phi3, codellama)
- **RAG Framework**: LangGraph for agent workflows
- **GPU Acceleration**: CUDA, Rapids, CuPy, CuML
- **Database**: Vector store for document embeddings
- **Job Management**: SLURM integration for resource allocation

### Frontend (External)
- **Framework**: React 18+ with TypeScript
- **UI Library**: Material-UI (MUI) v5
- **State Management**: React Context + useReducer
- **HTTP Client**: Axios with interceptors
- **Real-time**: WebSocket integration
- **Code Editor**: Monaco Editor (VS Code engine)

### Communication
- **REST API**: Synchronous operations
- **WebSocket**: Real-time chat and progress updates
- **Authentication**: Bearer token (future enhancement)
- **CORS**: Configured for cross-origin requests

## 📡 API Endpoints Specification

### Core Chat & Analysis
```http
POST /api/chat
Content-Type: application/json

{
  "message": "How can I optimize this CUDA code?",
  "code": "// CUDA code here",
  "session_id": "optional-session-identifier"
}

Response:
{
  "success": true,
  "response": "I can help you optimize this CUDA code...",
  "code_analysis": {...},
  "code_output": "Execution results",
  "optimized_code": "// Optimized version",
  "socratic_questions": ["What do you think about memory coalescing?"],
  "learning_objectives": ["Understand GPU memory patterns"],
  "timestamp": "2025-06-26T10:30:00Z"
}
```

### Code Analysis
```http
POST /api/analyze-code
Content-Type: application/json

{
  "code": "import cupy as cp\n# Your code here"
}

Response:
{
  "success": true,
  "analysis": {
    "performance_issues": [...],
    "optimization_opportunities": [...],
    "gpu_compatibility": "high",
    "estimated_speedup": "5-10x"
  },
  "optimized_code": "// GPU-optimized version",
  "timestamp": "2025-06-26T10:30:00Z"
}
```

### Benchmarking
```http
POST /api/benchmark
Content-Type: application/json

{
  "category": "linear_algebra",
  "benchmark_name": "matrix_multiplication",
  "size": 1000
}

Response:
{
  "success": true,
  "results": {
    "cpu_time": 2.45,
    "gpu_time": 0.12,
    "speedup": 20.4,
    "memory_usage": {...},
    "throughput": {...}
  },
  "timestamp": "2025-06-26T10:30:00Z"
}
```

### Tutorial Generation
```http
POST /api/tutorial
Content-Type: application/json

{
  "topic": "CUDA memory management"
}

Response:
{
  "success": true,
  "content": {
    "title": "CUDA Memory Management Fundamentals",
    "sections": [...],
    "code_examples": [...],
    "exercises": [...],
    "estimated_time": "30 minutes"
  },
  "timestamp": "2025-06-26T10:30:00Z"
}
```

### Metadata Endpoints
```http
GET /api/benchmark-categories
GET /api/benchmarks/{category}
GET /api/benchmark-sizes/{category}/{benchmark_name}
GET /api/execution-summary
GET /api/health
```

## 🔌 WebSocket Protocol

### Connection
```javascript
ws://sol-node:8000/ws
```

### Message Format
```json
// Client → Server
{
  "type": "chat_message",
  "data": {
    "message": "User input",
    "code": "Optional code",
    "session_id": "session-123"
  },
  "timestamp": "2025-06-26T10:30:00Z"
}

// Server → Client
{
  "type": "response",
  "data": {
    "response": "AI response",
    "code_analysis": {...},
    "progress": 0.75  // For long-running operations
  },
  "timestamp": "2025-06-26T10:30:00Z"
}

// Progress Updates
{
  "type": "progress",
  "data": {
    "operation": "benchmark_execution",
    "progress": 0.60,
    "status": "Running matrix multiplication benchmark...",
    "eta_seconds": 45
  },
  "timestamp": "2025-06-26T10:30:00Z"
}
```

## 🔒 Security & Networking

### Sol Access Requirements
1. **VPN Connection**: ASU VPN required for direct node access
2. **SSH Tunneling**: Alternative for development
   ```bash
   ssh -L 8000:compute-node:8000 asurite@sol.asu.edu
   ```
3. **SLURM Job**: API runs within allocated compute job
4. **Firewall**: Sol nodes typically block external connections

### Security Measures
- CORS policy configuration
- Input sanitization for code execution
- Rate limiting on API endpoints
- Request size limits
- Execution timeouts
- Resource usage monitoring

## 🚀 Performance Requirements

### Response Time Targets
- **Health Check**: < 100ms
- **Simple Chat**: < 2 seconds
- **Code Analysis**: < 10 seconds
- **Tutorial Generation**: < 15 seconds
- **Small Benchmarks**: < 30 seconds
- **Large Benchmarks**: < 5 minutes with progress updates

### Resource Allocation
- **CPU**: 4-8 cores per API instance
- **Memory**: 16-32GB RAM
- **GPU**: 1 GPU node for acceleration
- **Storage**: 10GB temporary space for outputs
- **Network**: Low latency internal Sol network

### Scalability
- Support 10-20 concurrent users
- Horizontal scaling via multiple API instances
- Load balancing across Sol compute nodes
- Session management for multi-turn conversations

## 📊 Data Flow

### Chat Interaction Flow
1. User inputs message + optional code in React frontend
2. Frontend sends HTTP POST to `/api/chat`
3. Backend processes through RAG agent pipeline:
   - Query understanding and intent classification
   - Document retrieval from knowledge base
   - Code analysis (if provided)
   - LLM response generation
   - Socratic question formulation
4. Response sent back to frontend
5. Frontend updates UI with response and suggestions

### Code Analysis Flow
1. User submits code via frontend editor
2. Backend receives code at `/api/analyze-code`
3. Code analysis pipeline:
   - Syntax and structure analysis
   - GPU optimization opportunity detection
   - Performance prediction
   - Optimized code generation
4. Results returned with recommendations
5. Frontend displays analysis in structured format

### Benchmark Execution Flow
1. User selects benchmark parameters
2. Frontend initiates benchmark via `/api/benchmark`
3. Backend executes on Sol GPU resources:
   - CPU baseline measurement
   - GPU accelerated execution
   - Performance comparison
   - Resource utilization tracking
4. Real-time progress via WebSocket
5. Final results displayed in frontend dashboard

## 🏗️ Component Breakdown

### Backend Components

#### 1. Enhanced GPU Mentor Core
- **Purpose**: Main RAG agent orchestration
- **Dependencies**: LangGraph, Ollama, vector store
- **Key Methods**:
  - `process_user_input(message, code)`
  - `generate_tutorial_content(topic)`
  - `get_execution_summary()`

#### 2. Code Optimizer
- **Purpose**: Analyze and optimize user code
- **Dependencies**: AST parsing, CUDA/Rapids knowledge
- **Key Methods**:
  - `analyze_code(code_string)`
  - `suggest_optimizations(code_string)`
  - `estimate_performance(code_string)`

#### 3. Benchmark Engine
- **Purpose**: Execute performance benchmarks
- **Dependencies**: CuPy, CuML, Sol compute resources
- **Key Methods**:
  - `run_benchmark(category, name, size)`
  - `get_benchmark_categories()`
  - `get_available_benchmarks()`

#### 4. Sol Code Executor
- **Purpose**: Safe code execution on Sol resources
- **Dependencies**: SLURM, CUDA runtime
- **Key Methods**:
  - `execute_code(code_string, timeout)`
  - `get_system_info()`
  - `monitor_resources()`

### Frontend Components

#### 1. Chat Interface
- **Purpose**: Conversational interaction with AI
- **Features**: Message history, code highlighting, copy/paste
- **State**: Chat history, typing indicators, response status

#### 2. Code Analyzer
- **Purpose**: Interactive code analysis and optimization
- **Features**: Monaco editor, split view (original/optimized), syntax highlighting
- **State**: Current code, analysis results, optimization suggestions

#### 3. Benchmark Runner
- **Purpose**: Performance benchmarking interface
- **Features**: Parameter selection, progress tracking, results visualization
- **State**: Available benchmarks, execution status, historical results

#### 4. Tutorial Generator
- **Purpose**: AI-generated learning content
- **Features**: Topic selection, structured content display, interactive examples
- **State**: Available topics, generated content, user progress

## 📁 Project Structure

```
gpu-mentor-project/
├── backend/                    # Sol deployment
│   ├── gpu_mentor_api.py      # FastAPI application
│   ├── enhanced_gpu_mentor.py # Core RAG agent
│   ├── code_optimizer.py     # Code analysis engine
│   ├── benchmark_engine.py   # Performance testing
│   ├── sol_executor.py       # Code execution on Sol
│   ├── requirements.txt      # Python dependencies
│   ├── run_api.sh           # Sol startup script
│   └── api_job.slurm        # SLURM job definition
├── frontend/                  # React application
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── services/         # API communication
│   │   ├── hooks/           # Custom React hooks
│   │   ├── utils/           # Utility functions
│   │   └── types/           # TypeScript definitions
│   ├── public/              # Static assets
│   └── package.json         # Node.js dependencies
└── docs/                     # Documentation
    ├── API_SPECIFICATION.md
    ├── DEVELOPMENT_PLAN.md
    ├── DEPLOYMENT_GUIDE.md
    └── USER_MANUAL.md
```

## 🎯 Future Enhancements

### Phase 2 Features
- User authentication and session persistence
- Advanced tutorial progress tracking
- Custom benchmark creation
- Code collaboration features
- Integration with popular IDEs (VS Code extension)

### Phase 3 Features
- Multi-language support (C++, Rust, Python)
- Advanced visualization of GPU performance
- Integration with academic course management
- Mobile app for tutorial consumption
- Community features (code sharing, discussions)

## 📋 Technical Constraints

### Sol Environment Limitations
- Network access requires VPN or SSH tunneling
- SLURM job time limits (typically 2-8 hours)
- GPU resource availability subject to scheduling
- Limited internet access on compute nodes

### API Design Constraints
- Stateless design for horizontal scaling
- Timeout handling for long-running operations
- Error recovery for network interruptions
- Resource cleanup after job completion

### Frontend Constraints
- Cross-origin request handling
- Real-time connection management
- Offline capability considerations
- Mobile responsiveness requirements

This specification provides the foundation for developing a production-ready GPU Mentor system with clear separation between the AI backend and user interface components.
