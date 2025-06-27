#!/usr/bin/env python3
"""Test script to verify both LLM models are working properly."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_agent import RAGAgent
from config import CHAT_MODEL, CODE_MODEL

def test_models():
    """Test both models to ensure they're working."""
    print("🧪 Testing GPU Mentor Models...")
    
    # Initialize RAG agent
    rag_agent = RAGAgent()
    
    # Test Chat Model
    print(f"\n📞 Testing Chat Model ({CHAT_MODEL})...")
    try:
        if rag_agent.chat_llm_model:
            response = rag_agent.query("Hello, who are you?")
            print(f"✅ Chat model response: {response[:100]}...")
        else:
            print("❌ Chat model not initialized")
    except Exception as e:
        print(f"❌ Chat model error: {e}")
    
    # Test Code Model
    print(f"\n💻 Testing Code Model ({CODE_MODEL})...")
    try:
        if rag_agent.code_llm_model:
            test_code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
result = np.sum(arr * 2)
print(result)
"""
            response = rag_agent.query_code_analysis(f"Analyze this Python code for GPU optimization:\n\n```python{test_code}```")
            print(f"✅ Code model response: {response[:200]}...")
        else:
            print("❌ Code model not initialized")
    except Exception as e:
        print(f"❌ Code model error: {e}")
    
    print("\n🏁 Model testing complete!")

if __name__ == "__main__":
    test_models()
