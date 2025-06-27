"""
Pydantic models for API requests and responses
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User message")
    code: Optional[str] = Field(None, description="Optional Python code")
    session_id: Optional[str] = Field(None, description="Session identifier")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    success: bool
    response: str
    code_analysis: Optional[Dict[str, Any]] = None
    code_output: Optional[Dict[str, Any]] = None
    optimized_code: Optional[str] = None
    socratic_questions: List[str] = []
    learning_objectives: List[str] = []
    timestamp: datetime


class CodeAnalysisRequest(BaseModel):
    """Request model for code analysis"""
    code: str = Field(..., description="Python code to analyze")


class CodeAnalysisResponse(BaseModel):
    """Response model for code analysis"""
    success: bool
    analysis: Dict[str, Any]
    optimized_code: str
    timestamp: datetime


class BenchmarkRequest(BaseModel):
    """Request model for benchmark execution"""
    category: str = Field(..., description="Benchmark category")
    benchmark_name: str = Field(..., description="Specific benchmark name")
    size: int = Field(..., description="Problem size")


class BenchmarkResponse(BaseModel):
    """Response model for benchmark execution"""
    success: bool
    results: Dict[str, Any]
    timestamp: datetime


class TutorialRequest(BaseModel):
    """Request model for tutorial generation"""
    topic: str = Field(..., description="Tutorial topic")


class TutorialResponse(BaseModel):
    """Response model for tutorial generation"""
    success: bool
    content: str
    timestamp: datetime


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    node: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Response model for errors"""
    success: bool = False
    error: str
    details: Optional[str] = None
    timestamp: datetime


class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime


class ProgressUpdate(BaseModel):
    """Progress update for long-running operations"""
    operation: str
    progress: float = Field(..., ge=0.0, le=1.0)
    status: str
    eta_seconds: Optional[int] = None
